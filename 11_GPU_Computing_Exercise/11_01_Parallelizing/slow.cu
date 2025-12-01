// slow.cu — simple CUDA version of vector add + matrix-vector product
// Usage: ./slow [numElements] [threadsPerBlock]
// Defaults: numElements = 32768, threadsPerBlock = 256

#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cstdint>

// Vector addition kernel
// Grid: grid1 = ( (numElements + threadsPerBlock - 1) / threadsPerBlock )blocks.
// Block: threadsPerBlock threads.
// Each thread handles one element.
__global__ void vecAdd(const int32_t* a, const int32_t* b, int32_t* c, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElements)
        c[i] = a[i] + b[i];
}

/*
Matrix-vector multiplication kernel
Grid: grid2 = numElements (There is totally numElements block per grid. It indicate that one block process one matrix row).
Block: block2 = threadsPerBlock.
Shared memory per block: threadsPerBlock * sizeof(long long) bytes.

Threads in the block sum this row with striding, then reduce.
Inside each block: 
  row = blockIdx.x selects the current matrix row.
  Each thread does a strided partial dot product over that row:


Block-wide reduction in shared memory:
- Each thread writes localSum to sharedData[tid].
- A simple power-of-two reduction halves the active threads each step until tid == 0 holds the row’s dot product.
- Thread 0 writes out[row].


This design is easy to read: “one row per block, block threads cooperate to sum the row.”
*/

__global__ void matVec(const int32_t* __restrict__ mat,
                       const int32_t* __restrict__ vec,
                       int32_t* __restrict__ out,
                       int numElements) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int threadsPerBlock = blockDim.x;

    if (row >= numElements)
        return;

    // Each thread does a strided partial dot product over that row:
    // Using 64-bit long long reduces overflow risk during accumulation (final cast back to int32_t matches your original output type).
    long long localSum = 0;
    for (int j = tid; j < numElements; j += threadsPerBlock)
        localSum += (long long)mat[row * numElements + j] * (long long)vec[j];

    extern __shared__ long long sharedData[];
    sharedData[tid] = localSum;
    __syncthreads();

    // reduction
    for (int step = threadsPerBlock >> 1; step > 0; step >>= 1) {
        if (tid < step)
            sharedData[tid] += sharedData[tid + step];
        __syncthreads();
    }

    if (tid == 0)
        out[row] = (int32_t)sharedData[0];
}

static void init_host(int numElements, int32_t* a, int32_t* b, int32_t* mat) {
    std::mt19937 rng(2024);
    std::uniform_int_distribution<int32_t> dist(-16, 16);
    for (int i = 0; i < numElements; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }
    for (long long i = 0; i < 1LL * numElements * numElements; ++i)
        mat[i] = dist(rng);
}


/*
variable tpb: threads per block
numElements: size of vectors (n) and matrix (n x n)
*/
int main(int argc, char** argv) {
    int numElements = (argc >= 2 ? std::stoi(argv[1]) : 32768);
    int threadsPerBlock = (argc >= 3 ? std::stoi(argv[2]) : 256);
    if (threadsPerBlock <= 0)
        threadsPerBlock = 256;

    std::cout << "numElements = " << numElements
              << ", threadsPerBlock = " << threadsPerBlock << "\n";

    // host memory
    // Vectors: numElements * 4 bytes each (because int32_t).
    // Matrix: numElements * numElements * 4 bytes.
    size_t vectorBytes = sizeof(int32_t) * (size_t)numElements;
    size_t matrixBytes = sizeof(int32_t) * (size_t)numElements * (size_t)numElements;
    int32_t *h_a=nullptr, *h_b=nullptr, *h_mat=nullptr, *h_out=nullptr;

    try {
        h_a   = new int32_t[numElements];
        h_b   = new int32_t[numElements];
        h_mat = new int32_t[(size_t)numElements * (size_t)numElements];
        h_out = new int32_t[numElements];
    } catch (...) {
        std::cerr << "Host memory allocation failed. Try smaller numElements.\n";
        return 1;
    }

    init_host(numElements, h_a, h_b, h_mat);

    // device memory
    int32_t *d_a=nullptr, *d_b=nullptr, *d_tmp=nullptr, *d_mat=nullptr, *d_out=nullptr;
    if (cudaMalloc(&d_a, vectorBytes) ||
        cudaMalloc(&d_b, vectorBytes) ||
        cudaMalloc(&d_tmp, vectorBytes) ||
        cudaMalloc(&d_out, vectorBytes) ||
        cudaMalloc(&d_mat, matrixBytes)) {
        std::cerr << "Device memory allocation failed. Try smaller numElements.\n";
        return 1;
    }

    // copy to device
    cudaMemcpy(d_a, h_a, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, h_mat, matrixBytes, cudaMemcpyHostToDevice);

    // kernel timing
    // Timing: CUDA events measure GPU time only (from right before vecAdd to right after matVec). 
    // Host↔Device copies are not included in that time; you could measure end-to-end with std::chrono if you need total wall time.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // kernel 1: tmp = a + b
    dim3 grid1((numElements + threadsPerBlock - 1) / threadsPerBlock);
    dim3 block1(threadsPerBlock);
    vecAdd<<<grid1, block1>>>(d_a, d_b, d_tmp, numElements);

    // kernel 2: out = mat * tmp
    dim3 grid2(numElements);  // one block per matrix row
    dim3 block2(threadsPerBlock);
    size_t sharedBytes = threadsPerBlock * sizeof(long long);   // Shared memory per block: threadsPerBlock * sizeof(long long) bytes.
    matVec<<<grid2, block2, sharedBytes>>>(d_mat, d_tmp, d_out, numElements);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernelMs = 0.0f;
    cudaEventElapsedTime(&kernelMs, start, stop);

    // copy back
    cudaMemcpy(h_out, d_out, vectorBytes, cudaMemcpyDeviceToHost);

    std::cout << "First 3 entries of Out Vec:\n";
    for (int i = 0; i < 3 && i < numElements; ++i)
        std::cout << h_out[i] << "\n";

    std::cout << "GPU time (kernels): " << (kernelMs / 1000.0) << " s\n";

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_tmp); cudaFree(d_mat); cudaFree(d_out);
    delete[] h_a; delete[] h_b; delete[] h_mat; delete[] h_out;

    return 0;
}
