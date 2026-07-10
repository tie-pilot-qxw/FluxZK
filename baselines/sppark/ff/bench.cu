#define __CUDA_ARCH__
#include "alt_bn128.hpp"

using namespace alt_bn128;

#include <iostream>

const int BATCH = 128;
const int THREADS = 512;
const int ITERS = 2;

__global__ void bench(fr_mont *r, const fr_mont *a)
{
  fr_mont v = *a;
  for (int i = 0; i < BATCH; i++)
  {
    v = v * v;
  }
  *r = v;
}

int main()
{
  float total_time = 0;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  int grid_size = 32 * deviceProp.multiProcessorCount;

  for (int i = 0; i < ITERS; i++)
  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    fr_mont *r, *a;
    cudaMalloc(&r, sizeof(fr_mont));
    cudaMalloc(&a, sizeof(fr_mont));

    auto ha = fr_mont::host_random();
    cudaMemcpy(a, &ha, sizeof(fr_t), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    bench<<<grid_size, THREADS>>>(r, a);
    cudaEventRecord(stop);

    auto err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
      return 1;
    }

    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    total_time += elapsed_time;
  }

  std::cout << THREADS * BATCH * ITERS * grid_size / total_time * 1000 << std::endl;

  return 0;
}


