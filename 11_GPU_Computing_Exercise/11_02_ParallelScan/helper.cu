#include <iostream>
#include <stdlib.h>
#include <curand.h>
#include <chrono>


// Robustness: The CUDA_CALL and CURAND_CALL macros help you quickly locate errors during development and debugging, such as memory out-of-bounds or failed CUDA API calls.
// The Macro of examing error. Once a failure occurs, an error message is printed and the program exits — preventing a “silent failure.”
#define CUDA_CALL(x)                                                                                          \
    do                                                                                                        \
    {                                                                                                         \
        cudaError_t error = x;                                                                                \
        if (error != cudaSuccess)                                                                             \
        {                                                                                                     \
            const char *cuda_err_str = cudaGetErrorString(error);                                             \
            std::cerr << "Cuda Error at" << __FILE__ << ":" << __LINE__ << ": " << cuda_err_str << std::endl; \
            return EXIT_FAILURE;                                                                              \
        }                                                                                                     \
    } while (0)

// The Macro of examing error which occurs in cuRAND
// cuRAND（CUDA Random Number Generation library）
#define CURAND_CALL(x)                                                                                    \
    do                                                                                                    \
    {                                                                                                     \
        curandStatus_t error = x;                                                                         \
        if (error != CURAND_STATUS_SUCCESS)                                                               \
        {                                                                                                 \
            std::cerr << "CudaRand Error " << error << " at" << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                                          \
        }                                                                                                 \
    } while (0)

#define CHECK_ALLOC(x)                                                                 \
    do                                                                                 \
    {                                                                                  \
        if ((x) == NULL)                                                               \
        {                                                                              \
            std::cerr << "Alloc Error at" << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                       \
        }                                                                              \
    } while (0)

// scale_vec provides a minimal working kernel example — including one-dimensional thread mapping, boundary checks, and sensible grid configuration. This serves as a useful reference when you implement your own Kogge–Stone scan kernel, especially for thread indexing and boundary handling.
__global__ void scale_vec(float *in_d, float x, size_t size)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)  // boundary checks
    {
        in_d[i] *= x;   // in_d[i] = in_d[i] * s;
    }
}

// ensure in_d has been allocated on device and in_h on host
int random_init(size_t size, float *in_d, float *in_h)
{
    curandGenerator_t gen;
    // Create PRNG = Pseudo-Random Number Generator
    CURAND_CALL(curandCreateGenerator(&gen,
                                      CURAND_RNG_PSEUDO_DEFAULT));
    // Set seed
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,
                                                   2048ULL));

    // Generate size floats in range (0,1] on device ( range is set by "Uniform")
    CURAND_CALL(curandGenerateUniform(gen, in_d, size));

    // Scale by two so it does not get boring. Multiply by 2.0 on each floats
    scale_vec<<<ceil(size / 256.0), 256>>>(in_d, 2.0, size);

    // Copy device memory to host
    CUDA_CALL(cudaMemcpy(in_h, in_d, size * sizeof(float),
                         cudaMemcpyDeviceToHost));

    CURAND_CALL(curandDestroyGenerator(gen));

    return EXIT_SUCCESS;
}