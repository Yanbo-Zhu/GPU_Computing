#include <chrono>
#include <curand.h>
#include <iostream>
#include <stdlib.h>

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#ifndef CUDA_CALL
#define CUDA_CALL(x) do { cudaError_t err = (x); if (err != cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); return EXIT_FAILURE; }} while(0)
#endif

/*
Parallel inclusive scan (prefix product) over complex numbers using Kogge–Stone.

This function treats the input array as pairs of adjacent float values representing complex numbers — [re0, im0, re1, im1, ...] — and
It performs an inclusive prefix multiplication, where the k-th complex output equals (z0 * z1 * ... * zk）

Optimizations: (1) reduce divergence, (2) shared memory, (3) thread coarsening, (4) Memory Coalescing.
*/

/*
Summary & Key Points

Data layout and Memory Coalescing
Each complex number is stored as two consecutive floats ([real, imag]). Inside the kernel, data is loaded/stored using float2 to ensure coalesced memory access.

Operation:
The scan operator is complex multiplication, and the identity element is (1, 0) (complex 1).

Kogge–Stone:
An in-place inclusive Kogge–Stone scan is performed in shared memory within each block to avoid long dependency chains. Control flow is uniform with minimal divergence.

Shared memory:
Each thread block processes one tile (BLOCK_SIZE * ITEMS_PER_THREAD) and completes a full scan of this tile entirely in shared memory.

Thread coarsening and Memory Coalescing:
ITEMS_PER_THREAD = 4 — each thread processes four consecutive complex numbers to reduce scheduling overhead and increase throughput. This also helps with memory coalescing, as each thread accesses contiguous memory locations which stores consecutive complex numbers.

Multi-block hierarchy:
Each block outputs its final inclusive result as the block’s block sum. These block sums are recursively scanned on the GPU. The resulting prefix (the inclusive result of all previous blocks) is multiplied back into each block’s elements to produce the global inclusive scan.
*/

/* hierarchical scan

The Hierarchical Scan used here is a multi-stage strategy for implementing parallel prefix sums (or prefix products) on a GPU when the input array is larger than what a single thread block can handle.

A GPU thread block can process only a limited chunk of data (for example, 1024 complex numbers).
If your array is very large — say, 10⁶ complex numbers — a single block cannot scan it.
So we split the large array into multiple smaller tiles,
let multiple thread blocks scan their own subarrays in parallel,
and then combine these partial results together.
This multi-stage approach is what we call a Hierarchical Scan.

A hierarchical scan works by first letting each block perform its own small-range scan in parallel, then recursively aggregating these block results and propagating them back, ultimately enabling large-scale parallel prefix sum/product computation.

1 Block Scan
Perform a prefix product inside each block
→ Local parallelism

2 Block-Sum Scan
Perform a prefix product on each block’s total product
→ Propagate prefixes across blocks

3 Apply Prefix
Multiply the block-prefix back into each block’s results
→ Form the global result



Phase 1 Block scan: Block-level scan with Kogge–Stone. Each Block processes a tile of complex numbers, performs an inclusive Kogge–Stone scan in shared memory, and writes the block’s total product to block_sums for the next phase.

Phase 2 Block Sum Scan: Hierarchical scan of block sums. A separate kernel or host code performs an inclusive scan on block_sums to obtain each block’s prefix product.

Phase 3 Apply Prefix: Distribution of block prefixes, Apply Prefix. A final kernel multiplies each element in a block by the prefix product of all previous blocks to complete the global inclusive scan.

*/

/*
Represent complex as float2: x = real, y = imag. with float2 , it ensure coalesced memory access.
__device__ __host__ inline: the function with those modifier can be called from both host and device code. inline suggests to the compiler to insert the function's code directly at each call site to reduce function call overhead.
*/

// compute complex multiplication
__device__ __host__ inline float2 c_mul(float2 a, float2 b)
{
    float2 r;
    r.x = a.x * b.x - a.y * b.y;
    r.y = a.x * b.y + a.y * b.x;
    return r;
}

// identity element for complex multiplication
__device__ __host__ inline float2 c_id()
{
    float2 r;
    r.x = 1.f;
    r.y = 0.f;
    return r;
}

// load i-th two floats from array and save them as a complex number into a float2 variable. idx_c is index in complex units, not float units.
__device__ inline float2 ld_c(const float *base, size_t idx_c)
{ // idx_c in complex units
    float2 v;
    v.x = base[2 * idx_c + 0];
    v.y = base[2 * idx_c + 1];
    return v;
}

// write a float2 variable as two floats representing a complex number into array. idx_c is index in complex units, not float units.
__device__ inline void st_c(float *base, size_t idx_c, float2 v)
{
    base[2 * idx_c + 0] = v.x;
    base[2 * idx_c + 1] = v.y;
}

/* template <int ITEMS_PER_THREAD, int BLOCK_SIZE>

ITEMS_PER_THREAD and BLOCK_SIZE are integer constant parameters fixed at compile time.
- Thread coarsening: each thread processes ITEMS_PER_THREAD consecutive complex numbers.
- BLOCK_SIZE: Threads number per Block

It allows a kernel to be written in a parametric form so that the compiler can automatically generate specialized versions of the code for different configurations (e.g., different block sizes or different numbers of items per thread) without having to write multiple versions manually.

when you call the kernel function like
block_scan_ks<4, 256><<<numBlocks, 256, shmem_bytes>>>(...);
Compiler generates a version of block_scan_ks where ITEMS_PER_THREAD is replaced with 4 and BLOCK_SIZE with 256 throughout the code.

The compiler substitutes these constants during compilation,
so the kernel can use them directly inside the function body
without consuming extra registers or memory.
*/

/* 
block_scan_ks: performs an inclusive Kogge–Stone scan over a tile of complex numbers in shared memory.
-block_sums: an  output array to store the total product of each block for hierarchical scanning.
-N_complex: total number of complex numbers to process. (not float numbers. each complex numbers contains two float numbers)

Goal: Each block processes one tile (Demension of tile = BLOCK_SIZE * ITEMS_PER_THREAD complex numbers), performs an inclusive Kogge–Stone scan on that tile in shared memory.

Then outputs the block’s total product (the last element of the inclusive result) to block_sums[blockIdx.x] for use in the next hierarchy level.
*/
template <int ITEMS_PER_THREAD, int BLOCK_SIZE>
__global__ void block_scan_ks(const float *in, float *out, float2 *block_sums, int N_complex)
{
    // sh[]: Dynamically allocated shared memory (its size is provided as the third argument in the kernel launch).kernel<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(...); sharedMemBytes is the size in bytes of the shared memory to allocate for this kernel launch.
    // size = BLOCK_SIZE* ITEMS_PER_THREAD. 
    extern __shared__ float2 sh[]; 
    const int tid = threadIdx.x;
    const int blockItemBase = (blockIdx.x * BLOCK_SIZE * ITEMS_PER_THREAD); // The global starting index of the complex-number segment handled by the current block.
    const int laneBase = blockItemBase + tid * ITEMS_PER_THREAD; // The global index of the first complex number handled by the current thread (since each thread processes ITEMS_PER_THREAD elements, it multiplies by laneBase).

    /*
    in Kernel function, each for loop  is excuted by all threads in parallel.. Each thread has its own copy of the loop variable i and executes the loop independently.

    laneBase decides the starting index for each thread to process its ITEMS_PER_THREAD consecutive complex numbers. Each thread process ITEMS_PER_THREAD consecutive complex numbers starting from laneBase.
    */


    /*
    Coalesced load: threads load consecutive float2s
    
    The following loop provides two benefits at the same time:
    - Memory Coalescing
        Each thread accesses neighboring addresses, allowing the GPU to combine these requests into a single large read/write.
    - Thread Coarsening
        Each thread processes a larger chunk of data (e.g., 4 items instead of 1), reducing GPU scheduling overhead and increasing throughput
    
    Each thread loads its own ITEMS_PER_THREAD consecutive complex numbers from global memory into shared memory.
    
    This “complex-number–contiguous + thread-sequential” layout enables coalesced memory access.
    
    When out of bounds (i.e., the tile does not fully cover a complete block), the missing elements are filled with the multiplicative identity c_id() = (1, 0) to keep the scan logic correct and avoid branching.
    
    __syncthreads() ensures that all data has been written to shared memory before the scan begins.

    #pragma unroll // Its purpose is to tell the compiler: Please unroll this loop at compile time instead of executing it as a real loop at runtime. it allows the GPU to avoid loop-condition checks and branch jumps during execution, which can improve performance.

    */
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        int gIdx = laneBase + i;
        float2 v = (gIdx < N_complex) ? ld_c(in, gIdx) : c_id(); // identity padding
        sh[tid * ITEMS_PER_THREAD + i] = v;
    }
    __syncthreads();


    /*
    In-place Kogge–Stone (inclusive) over the tile in shared memory.
    Goal: replace each element in sh[0..M-1] with the product of all previous elements (inclusive). 

    Each thread in the same block process its own data form offset 1 to M-1. But localIdx of each thread is different. So they process different data in shared memory sh[localIdx].
    
    The Kogge–Stone pattern uses a sequence of rounds with doubling step sizes to propagate prefix information:
    - Round 1: offset = 1, indices start from i = 1, update sh[i] using sh[i - 1]
    - Round 2: offset = 2, indices start from i = 2, update sh[i] using sh[i - 2]
    - Round 3: offset = 4, indices start from i = 4, update sh[i] using sh[i - 4]
    - Continue until offset >= M.
    Each round multiplies in the prefix from farther to the left. After log₂(M) rounds, each index i has accumulated all the elements it should include.

    Why it is “parallel with very little divergence”
    - In each round of for loop, all threads execute the same steps (only their localIdx values differ). The loop control is uniform, and the only branch is a simple boundary check (localIdx >= offset).
    
    Because ITEMS_PER_THREAD is a compile-time constant, #pragma unroll can fully unroll the small inner loop, further reducing control-flow overhead and register dependencies.

    */
    
    const int M = BLOCK_SIZE * ITEMS_PER_THREAD;// The thread’s private linear index into shared memory (from 0 to M−1).
    for (int offset = 1; offset < M; offset <<= 1)
    {
        __syncthreads(); // ensure all threads have updated shared memory before next step. Let all threads can get the latest data from shared memory from previous step.
        
        #pragma unroll // Its purpose is to tell the compiler: Please unroll this loop at compile time instead of executing it as a real loop at runtime. it allows the GPU to avoid loop-condition checks and branch jumps during execution, which can improve performance.

        for (int i = 0; i < ITEMS_PER_THREAD; i++)
        {
            int localIdx = tid * ITEMS_PER_THREAD + i; // each thread processes ITEMS_PER_THREAD consecutive elements in shared memory.

            // only update if there are element in the left
            if (localIdx - offset >= 0)
            {
                float2 a = sh[localIdx];
                float2 b = sh[localIdx - offset];
                sh[localIdx] = c_mul(b, a); // inclusive update: prev * curr. the new value = “previous prefix” × “old current value”.  .Write the product back to shared memory in the same location. 
            }
        }
        __syncthreads(); // ensure all threads have updated shared memory before next step. Let all threads can get the latest data from shared memory from previous step.
    }

    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        int gIdx = laneBase + i;
        if (gIdx < N_complex)
        {
            st_c(out, gIdx, sh[tid * ITEMS_PER_THREAD + i]);
        }
    }

    /* Write block total (inclusive last) for hierarchical scan

    When block_sums is not null (needed in the first phase), the block’s final inclusive result — i.e., the total product of this tile — is written to block_sums[blockIdx.x].

    Using tid == BLOCK_SIZE - 1 ensures that only the last thread performs this write, avoiding race conditions.

    valid computes the last valid element index within this block’s range (handling the case where the final tile is smaller than M). If there are no valid elements, the identity value is written instead. 

    Later, the host or another kernel performs the same scan on block_sums to obtain each block’s prefix product. Then the prefix product of all previous blocks is multiplied back into every element of the current block, completing the global inclusive scan—this is the standard hierarchical scan framework.

    */
    if (block_sums)
    {
        if (tid == BLOCK_SIZE - 1)
        {
            int valid = min(N_complex - blockItemBase, M) - 1;
            block_sums[blockIdx.x] = (valid >= 0) ? sh[valid] : c_id();
        }
    }
}


/*
Apply scanned block prefixes (exclusive) to each element of a block's tile.

why exclusive? because the first block has no previous blocks, so its prefix is the identity element (1, 0). For subsequent blocks, we want to multiply by the product of all preceding blocks only, not including the current block's own total.

Arguments:
- scanned_block_prefix[k]: The product of all blocks before block k (already inclusively scanned). It provides the exclusive prefix for the current block — i.e., the accumulated result of all preceding blocks.
- inout: The array containing the per-block scan results (output from the previous stage). Here, it is updated in place by multiplying in the cross-block prefix

*/
template<int ITEMS_PER_THREAD, int BLOCK_SIZE>
__global__ void apply_prefix(const float2* scanned_block_prefix, // size = numBlocks
                             float* inout, int N_complex){

    // the index of the first complex number in the global array that the current thread block is responsible for.
    const int blockBase = blockIdx.x * BLOCK_SIZE * ITEMS_PER_THREAD;

    // the 0-th block has no previous blocks, so its prefix is the identity element (1, 0). For subsequent blocks, we want to use scanned_block_prefix[blockIdx.x - 1], the product by multiplying the product of all preceding blocks only, not including the current block's own total.
    float2 pref = (blockIdx.x == 0) ? c_id() : scanned_block_prefix[blockIdx.x - 1];
    const int tid = threadIdx.x;
    const int laneBase = blockBase + tid * ITEMS_PER_THREAD;

    #pragma unroll
    for(int i=0;i<ITEMS_PER_THREAD;i++){
        int gIdx = laneBase + i;
        if(gIdx < N_complex){ // boundary check in case the last block is not full
            float2 v = ld_c(inout, gIdx); // load current value
            v = c_mul(pref, v);
            st_c(inout, gIdx, v); // store updated value into inout in the same location
        }
    }
}

// Integer division with rounding up. used to compute the number of blocks or tiles needed. For example, splitting N_complex elements into Block by Tile (through ceil(N/Tile)). The Dimension of Tile is BLOCK_SIZE * ITEMS_PER_THREAD.
static inline int iDivUp(int a, int b){ return (a + b - 1) / b; }

// Main function for GPU inclusive scan over complex numbers.
int gpu_inclusive_scan_complex(size_t size_floats, const float* in_d, float* out_d){
    const int N_complex = (int)(size_floats / 2);
    if (N_complex <= 0) return EXIT_SUCCESS;

    constexpr int BLOCK_SIZE = 256;
    constexpr int ITEMS_PER_THREAD = 4; // thread coarsening
    const int TILE = BLOCK_SIZE * ITEMS_PER_THREAD;

    const int numBlocks = iDivUp(N_complex, TILE); // number of blocks needed to cover all complex numbers

    // 1) Per-block scan and collect block totals
    float2* d_block_sums = nullptr;
    CUDA_CALL(cudaMalloc(&d_block_sums, max(1, numBlocks) * (int)sizeof(float2)));

    size_t shmem_bytes = TILE * sizeof(float2);
    block_scan_ks<ITEMS_PER_THREAD, BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, shmem_bytes>>>(
        in_d, out_d, d_block_sums, N_complex);
    CUDA_CALL(cudaGetLastError());

    // 2) Recursively scan the block sums on GPU
    float2 *d_in = d_block_sums;
    float2 *d_out = nullptr;
    
    int n = numBlocks;
    if(n > 1) CUDA_CALL(cudaMalloc(&d_out, n * sizeof(float2)));

    while (n > 1) {
        int blocks2 = iDivUp(n, TILE);
        if (blocks2 > 1) {
            float2* d_next = nullptr;
            CUDA_CALL(cudaMalloc(&d_next, blocks2 * sizeof(float2)));

            block_scan_ks<ITEMS_PER_THREAD, BLOCK_SIZE>
                <<<blocks2, BLOCK_SIZE, shmem_bytes>>>(
                    reinterpret_cast<const float*>(d_in),
                    reinterpret_cast<float*>(d_out),
                    d_next, n);
            CUDA_CALL(cudaGetLastError());

            // only free d_in when it is not d_block_sums to avoid double free
            float2* old_in = d_in;
            d_in = d_out;
            d_out = nullptr;
            n = blocks2;

            CUDA_CALL(cudaMalloc(&d_out, n * sizeof(float2)));
            CUDA_CALL(cudaFree(d_next));

            if (old_in != d_block_sums) {
                CUDA_CALL(cudaFree(old_in));
            }
        } else {
            // In the last level, only one block is needed, so block_sums is no longer required.
            block_scan_ks<ITEMS_PER_THREAD, BLOCK_SIZE>
                <<<blocks2, BLOCK_SIZE, shmem_bytes>>>(
                    reinterpret_cast<const float*>(d_in),
                    reinterpret_cast<float*>(d_out),
                    /*block_sums*/ nullptr, n);
            CUDA_CALL(cudaGetLastError());
            break;
        }
    }


    // 3) Apply scanned block prefixes
    if(numBlocks > 1){
        const float2* d_scanned = (n == 1) ? d_out : d_in;
        apply_prefix<ITEMS_PER_THREAD, BLOCK_SIZE><<<numBlocks, BLOCK_SIZE>>>(
            d_scanned, out_d, N_complex);
        CUDA_CALL(cudaGetLastError());

        // free the unused one (make sure not to free d_scanned)
        if (d_scanned == d_in) {
            if (d_out && d_out != d_block_sums) CUDA_CALL(cudaFree(d_out));
        } else { // d_scanned == d_out
            if (d_in && d_in != d_block_sums) CUDA_CALL(cudaFree(d_in));
        }
    }

    CUDA_CALL(cudaFree(d_block_sums));
    return EXIT_SUCCESS;
}
