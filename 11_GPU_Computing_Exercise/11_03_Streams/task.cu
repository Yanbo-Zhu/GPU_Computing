#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#define CHECK_CUDA(call)                                        \
    if ((call) != cudaSuccess)                                  \
    {                                                           \
        std::cerr << "CUDA error at " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                     \
    }


// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const int NUM_MATRICES = 10; // Number of matrix multiplications。 The result of each multiplication is independent and is stored in a separate matrix C. 
const int MATRIX_SIZE = 4096;
const int TILE_SIZE = 32;

// ---------------------------------------------------------------------------
// Naive matrix multiplication kernel
// C = A * B, all matrices N x N (row-major)
// ---------------------------------------------------------------------------

__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C, int n)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Tiled matrix multiplication kernel using shared memory
// Memory Coalescing
// Threads within the same warp should access consecutive global memory addresses, so the hardware can combine these loads/stores into fewer, more efficient memory transactions.
//
// Shared Memory Usage
// Load the sub-tiles of A and B that are needed for the current computation into shared memory first, and then reuse those values from shared memory instead of repeatedly reading from slow global memory.
// 
// Avoiding Warp Divergence
// Ensure that all threads in the same warp follow the same execution path.
// Avoid if conditions inside loops whenever possible, so threads in a warp do not diverge into different branches.
// ---------------------------------------------------------------------------
__global__ void matrixMultiplyKernelTiled(const float *A, const float *B, float *C, int n)
{
    // TODO: allocate shared memory for two tiles (one for A and one for B)
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Each thread computes C[row][col]
    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // Number of tiles along the K dimension.  Assume n is divisible by TILE_SIZE → no boundary checks → no warp divergence

    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    // TODO: iterate over tiles
    // TODO: copy tiles from global memory into shared memory
    // TODO: compute the matrix multiplication of the two tiles
    for (int t = 0; t < numTiles; ++t)
    {
        // ---------------------------
        // Memory Coalescing:
        // Each thread loads one contiguous element of A and B
        // ---------------------------
        int aCol = t * TILE_SIZE + tx;
        int bRow = t * TILE_SIZE + ty;

        // Load tiles from global memory (with bounds checking)
        // Load A and B tiles into shared memory
        if (row < n && aCol < n)
            tileA[threadIdx.y][threadIdx.x] = A[row * n + aCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (bRow < n && col < n)
            tileB[threadIdx.y][threadIdx.x] = B[bRow * n + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // Ensure all data is loaded before computation

        // ---------------------------
        // Compute partial sum for this tile
        // Multiply tileA × tileB
        // Shared memory → fast reuse
        // No divergence inside loop
        // ---------------------------
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads(); // Ensure all threads are done before loading new tiles
    }

    // TODO: write back the results into the matrix C
    if (row < n && col < n)
    {
        C[row * n + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// no streams, everything in the default stream (sequential)
// ---------------------------------------------------------------------------
void matrixMultiplyNoStreams()
{
    // Host and device pointers
    float *h_A[NUM_MATRICES], *h_B[NUM_MATRICES], *h_C[NUM_MATRICES];
    float *d_A[NUM_MATRICES], *d_B[NUM_MATRICES], *d_C[NUM_MATRICES];

    for (int i = 0; i < NUM_MATRICES; i++)
    {
        // Allocate host memory
        h_A[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        h_B[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        h_C[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

        // Initialize example matrices with random numbers
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++)
        {
            // pick testing values, that allow us to compute the expected result on the CPU cheaply
            h_A[i][j] = 1.0f;
            h_B[i][j] = 0.01f;
            h_C[i][j] = 0.0f;
        }

        // Allocate device memory
        CHECK_CUDA(cudaMalloc(&d_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));

        // Copy matrices A and B to the device
        CHECK_CUDA(cudaMemcpy(d_A[i], h_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B[i], h_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        // Launch matrix multiplication kernel
        // Define block and grid sizes: two-dimensional grid of two-dimensional blocks
        // there are 128x128x1024 = 16,777,216 threads in total in grid 
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 blocksPerGrid(MATRIX_SIZE/TILE_SIZE, MATRIX_SIZE/TILE_SIZE);

        std::cout << "Launch kernel with " << blocksPerGrid.x * blocksPerGrid.y << " blocks each with " << threadsPerBlock.x * threadsPerBlock.y << " threads\n";
        matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        // matrixMultiplyKernelTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        CHECK_CUDA(cudaGetLastError());

        // Copy results back to the host
        CHECK_CUDA(cudaMemcpy(h_C[i], d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        // Verify results
        double eps = 1.e-6;  // machine zero
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++) {
            double abs_err = fabs(h_C[i][j] - (MATRIX_SIZE * 0.01f));
            double dot_length = MATRIX_SIZE;
            double abs_val = fabs(h_C[i][j]);
            double rel_err = abs_err / abs_val / dot_length;

            if (rel_err > eps) {
                printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    j, h_C[i][j], MATRIX_SIZE * 0.01f, eps);
            }
        }

        // Cleanup
        free(h_A[i]);
        free(h_B[i]);
        free(h_C[i]);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
}

// ---------------------------------------------------------------------------
// With streams: overlap H2D/D2H copies and computation
// each matrix multiplication is assigned to its own CUDA stream, allowing for concurrent execution of memory transfers and kernel computations across multiple streams.
// ---------------------------------------------------------------------------
void matrixMultiplyWithStreams()
{
    // Host and device pointers
    float *h_A[NUM_MATRICES], *h_B[NUM_MATRICES], *h_C[NUM_MATRICES];
    float *d_A[NUM_MATRICES], *d_B[NUM_MATRICES], *d_C[NUM_MATRICES];

    // cudaStream_t streams[NUM_STREAMS]; It is used to hold multiple CUDA streams for concurrent execution.
    cudaStream_t streams[NUM_MATRICES];

    // TODO: Allocate memory, initialize data, create streams and copy data asynchronously
    for (int i = 0; i < NUM_MATRICES; ++i)
    {
        // Allocate host memory
        h_A[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        h_B[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        h_C[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

        // Initialize example matrices with random numbers. Same as in matrixMultiplyNoStreams
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; ++j)
        {
            h_A[i][j] = 1.0f;
            h_B[i][j] = 0.01f;
            h_C[i][j] = 0.0f;
        }
        // Allocate device memory
        CHECK_CUDA(cudaMalloc(&d_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));

        // Create stream
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

        // Asynchronously copy A and B to device
        // Asynchronous memory copy operations are issued to the specific stream, allowing them to overlap with computation in other streams. Asynchronous memory copy operations do not block the host thread, enabling concurrent execution in the CPU.
        CHECK_CUDA(cudaMemcpyAsync(d_A[i], h_A[i],
                                   MATRIX_SIZE * MATRIX_SIZE * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_B[i], h_B[i],
                                   MATRIX_SIZE * MATRIX_SIZE * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[i]));
    }

    // TODO: Launch matrix multiplication kernel for each stream
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(MATRIX_SIZE / TILE_SIZE, MATRIX_SIZE / TILE_SIZE);

    for (int i = 0; i < NUM_MATRICES; ++i)
    {
        // Launch matrix multiplication kernel for each stream
        matrixMultiplyKernelTiled<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        //matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        CHECK_CUDA(cudaGetLastError());
    }

    // TODO: Copy results back to the host asynchronously
    for (int i = 0; i < NUM_MATRICES; ++i)
    {
        CHECK_CUDA(cudaMemcpyAsync(h_C[i], d_C[i],
                                   MATRIX_SIZE * MATRIX_SIZE * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[i]));
    }

    // TODO: Synchronize all streams
    for (int i = 0; i < NUM_MATRICES; ++i)
    {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }


    // Verify results (slow! use only for debugging)
    for (int i = 0; i < NUM_MATRICES; i++)
    {
        std::cout << "Matrix C[" << i << "]:" << std::endl;
        for (int row = 0; row < MATRIX_SIZE; row++)
        {
            for (int col = 0; col < MATRIX_SIZE; col++)
            {
                std::cout << h_C[i][row * MATRIX_SIZE + col] << " ";
            }
            std::cout << std::endl;
        }
    }

    // TODO: Cleanup
    for (int i = 0; i < NUM_MATRICES; ++i)
    {
        if (h_A[i]) free(h_A[i]);
        if (h_B[i]) free(h_B[i]);
        if (h_C[i]) free(h_C[i]);

        if (d_A[i]) cudaFree(d_A[i]);
        if (d_B[i]) cudaFree(d_B[i]);
        if (d_C[i]) cudaFree(d_C[i]);

        cudaStreamDestroy(streams[i]);
    }
}

int main()
{
    // matrixMultiplyWithStreams();
    matrixMultiplyNoStreams();
    return EXIT_SUCCESS;
}
