#include <chrono>
#include <curand.h>
#include <iostream>
#include <stdlib.h>

#include "helper.cu"
#include <vector>

int gpu_inclusive_scan_complex(size_t size_floats, const float* in_d, float* out_d);


/*
This function treats the input array as pairs of adjacent float values representing complex numbers — [re0, im0, re1, im1, ...] — and 
It performs an inclusive prefix multiplication, where the k-th complex output equals (z0 * z1 * ... * zk）

Initialization: 
out_h[0..1] = in_h[0..1] — the prefix product of the first complex number is itself.

Iteration:
Then, for every step of two floats (one complex number), it takes the previous prefix result (real_prev, im_prev) and the current input (real_cur, im_cur) and performs complex multiplication:
(a+bi)⋅(c+di)=(ac−bd)+(ad+bc)i

size represents the number of floats, not the number of complex numbers. Therefore, it must be an even number (twice the number of complex entries).

This sequential version serves as the CPU ground truth for verifying the correctness of your future parallel Kogge–Stone GPU implementation.
*/
void sequential_scan(size_t size, float *in_h, float *out_h) {

  out_h[0] = in_h[0];
  out_h[1] = in_h[1];
  
  for (auto i = 2; i < size; i += 2) {
    float real_prev = out_h[i - 2];
    float real_cur = in_h[i];
    float im_prev = out_h[i - 1];
    float im_cur = in_h[i + 1];

    out_h[i] = real_prev * real_cur - im_prev * im_cur;
    out_h[i + 1] = real_prev * im_cur + real_cur * im_prev;
  }
}

int main() {
  size_t size = 33554432 * 2;
  float *in_d, *in_h, *out_d, *out_h;

  // Allocate on host
  in_h = (float *)calloc(size, sizeof(float));
  CHECK_ALLOC(in_h);
  out_h = (float *)calloc(size, sizeof(float));
  CHECK_ALLOC(out_h);
  // Allocate on device
  CUDA_CALL(cudaMalloc((void **)&in_d, size * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&out_d, size * sizeof(float)));

  // Initialize
  int e = random_init(size, in_d, in_h);
  if (e == EXIT_FAILURE)
    return EXIT_FAILURE;

  auto start = std::chrono::system_clock::now();
  sequential_scan(size, in_h, out_h);
  auto end = std::chrono::system_clock::now();

  std::cout << "=== Computation in CPU using sequential_scan" << std::endl;
  std::cout << "First 3 entries of In Vec:" << std::endl;
  for (int32_t i = 0; i < 5 * 2; i += 2)
    std::cout << in_h[i] << "," << in_h[i + 1] << std::endl;
  std::cout << "First 3 entries of Out Vec:" << std::endl;
  for (int32_t i = 0; i < 5 * 2; i += 2)
    std::cout << out_h[i] << " + " << out_h[i + 1] << std::endl;

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;


  std::cout << "=== Computation in GPU using Parallel inclusive scan (prefix product) over complex numbers using Kogge–Stone" << std::endl;

  CUDA_CALL(cudaMemset(out_d, 0, size * sizeof(float))); // clear output buffer


  auto start_GPU = std::chrono::system_clock::now();
  int rc = gpu_inclusive_scan_complex(size, in_d, out_d);
  auto end_GPU  = std::chrono::system_clock::now();

  if (rc != EXIT_SUCCESS) {
      std::cerr << "GPU scan failed\n";
      return rc;
  }
  CUDA_CALL(cudaDeviceSynchronize());

  std::vector<float> out_h_gpu(size);
  CUDA_CALL(cudaMemcpy(out_h_gpu.data(), out_d, size * sizeof(float), cudaMemcpyDeviceToHost));

  std::cout << "First 3 entries of In Vec:" << std::endl;
  for (int32_t i = 0; i < 5 * 2; i += 2)
    std::cout << in_h[i] << "," << in_h[i + 1] << std::endl;
  
  std::cout << "First 3 entries of Out Vec:" << std::endl;
  for (int32_t i = 0; i < 5 * 2; i += 2)
    std::cout << out_h_gpu[i] << " + " << out_h_gpu[i + 1] << std::endl;

  std::chrono::duration<double> elapsed_seconds_GPU = end_GPU - start_GPU;
  std::cout << "Elapsed time : " << elapsed_seconds_GPU.count() << "s" << std::endl;


  // compare the result
  std::cout << "=== Compare the result using Parallel inclusive scan (prefix product)  and using sequential_scan" << std::endl;

  bool ok = true;
  for (size_t i = 0; i < size; i++) {
      if (fabs(out_h_gpu[i] - out_h[i]) > 1e-4) {
          std::cerr << "Mismatch at index" << i << ": "
                    << out_h[i] << " vs " << out_h_gpu[i] << std::endl;
          ok = false;
          break;
      }
  }
  if (ok) std::cout << "GPU scan matches sequential result!" << std::endl;



  CUDA_CALL(cudaFree(in_d));
  CUDA_CALL(cudaFree(out_d));
  free(in_h);
  free(out_h);
  return EXIT_SUCCESS;
}
