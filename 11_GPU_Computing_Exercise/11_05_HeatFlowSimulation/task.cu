#include <iostream>
#include <iomanip>
#include <cuda.h>

#define N 128
#define M 128
#define ITERATIONS 100000
#define DIFFUSION_FACTOR 0.5f
#define CELL_SIZE 0.01f

// ------------------------------------------------------------
// Initialize the grid on the CPU (same as original CPU version)
// ------------------------------------------------------------
void initializeGrid(float *grid, int n, int m)
{
    for (int y = 0; y < m; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            // Set one quadrant to a high temperature (100.0f)
            if (y > m / 2 && x > n / 2)
                grid[y * n + x] = 100.0f;
            else
                grid[y * n + x] = 0.0f;
        }
    }
}

// ------------------------------------------------------------
// CUDA Kernel: Shared-memory accelerated 2D heat diffusion stencil
// ------------------------------------------------------------
__global__ void heatKernel(float *curr, float *next, int n, int m,
                           float dt, float dx2, float dy2)
{
    // Tile/block size for CUDA threads
    const int BLOCK_SIZE = 16;

    // Shared memory tile (BLOCK_SIZE + 2 halo cells on each side)
    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    // Global coordinates of the current thread
    int global_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int global_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Local coordinates inside shared memory (+1 for halo shift)
    int local_x = threadIdx.x + 1;
    int local_y = threadIdx.y + 1;

    // ---------------------------
    // Load center cell into tile
    // ---------------------------
    if (global_x < n && global_y < m)
    {
        tile[local_y][local_x] = curr[global_y * n + global_x];
    }

    // ---------------------------
    // Load halo regions (left/right/up/down)
    // ---------------------------
    if (threadIdx.x == 0 && global_x > 0)
        tile[local_y][0] = curr[global_y * n + global_x - 1];

    if (threadIdx.x == BLOCK_SIZE - 1 && global_x < n - 1)
        tile[local_y][BLOCK_SIZE + 1] = curr[global_y * n + global_x + 1];

    if (threadIdx.y == 0 && global_y > 0)
        tile[0][local_x] = curr[(global_y - 1) * n + global_x];

    if (threadIdx.y == BLOCK_SIZE - 1 && global_y < m - 1)
        tile[BLOCK_SIZE + 1][local_x] = curr[(global_y + 1) * n + global_x];

    // Make sure all threads have loaded shared memory
    __syncthreads();

    // Skip border cells (boundary conditions)
    if (global_x == 0 || global_x == n - 1 ||
        global_y == 0 || global_y == m - 1)
        return;

    // Fetch neighbor values from shared memory tile
    float center = tile[local_y][local_x];
    float left   = tile[local_y][local_x - 1];
    float right  = tile[local_y][local_x + 1];
    float below  = tile[local_y - 1][local_x];
    float above  = tile[local_y + 1][local_x];

    // Apply diffusion formula (5-point stencil)
    next[global_y * n + global_x] =
        center + DIFFUSION_FACTOR * dt *
                     ((left - 2.0f * center + right) / dy2 +
                      (above - 2.0f * center + below) / dx2);
}

// ------------------------------------------------------------
// Main function: Run the heat simulation on GPU
// ------------------------------------------------------------
int main()
{
    // Allocate host memory (CPU)
    float *h_curr = (float *)malloc(N * M * sizeof(float));
    float *h_next = (float *)malloc(N * M * sizeof(float));

    // Initialize both grids
    initializeGrid(h_curr, N, M);
    initializeGrid(h_next, N, M);

    // Precompute values used in diffusion equation
    float dx2 = CELL_SIZE * CELL_SIZE;
    float dy2 = CELL_SIZE * CELL_SIZE;
    float dt = dx2 * dy2 / (2.0f * DIFFUSION_FACTOR * (dx2 + dy2));

    // Allocate device memory (GPU)
    float *d_curr, *d_next;
    cudaMalloc(&d_curr, N * M * sizeof(float));
    cudaMalloc(&d_next, N * M * sizeof(float));

    // Copy initial states from host to device
    cudaMemcpy(d_curr, h_curr, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_next, h_next, N * M * sizeof(float), cudaMemcpyHostToDevice);

    // Configure block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    // -----------------------------------
    // Run simulation (double buffering)
    // -----------------------------------
    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        heatKernel<<<grid, block>>>(d_curr, d_next, N, M, dt, dx2, dy2);

        // Swap buffers (current <-> next)
        std::swap(d_curr, d_next);
    }

    // Copy final grid back to host
    cudaMemcpy(h_curr, d_curr, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    // Print a small portion of the result (top-left 16x16)
    std::cout << "Final grid values (top-left corner):" << std::endl;
    for (int y = 0; y < 16; ++y)
    {
        for (int x = 0; x < 16; ++x)
        {
            std::cout << std::setw(6)
                      << std::fixed << std::setprecision(2)
                      << h_curr[y * N + x] << " ";
        }
        std::cout << std::endl;
    }

    // Free GPU and CPU memory
    cudaFree(d_curr);
    cudaFree(d_next);
    free(h_curr);
    free(h_next);

    return 0;
}
