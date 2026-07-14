#include "../src/4step_ntt.cuh"
#if defined(NTT_FIELD_MNT4753)
#include "../../mont/src/mnt4753_fr.cuh"
#else
#include "../../mont/src/bn254_fr.cuh"
#endif
#include "bench_args.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <stdexcept>

#if defined(NTT_FIELD_MNT4753)
typedef mnt4753_fr::Element Field;
#else
typedef bn254_fr::Element Field;
#endif

int main(int argc, char **argv) {
    try {
        BenchArgs args = parse_bench_args(argc, argv, 20, 30, 2, 1, 0);
        for (int k : args.ks) {
            auto omega = Field::host_random();

            long long length = 1ll << k;

            printf("preparing k = %d, length = %lld\n", k, length);
            fflush(stdout);

            Field *src_gpu, *dst_gpu;
            cudaError_t err = cudaHostAlloc(&src_gpu, length * sizeof(Field), cudaHostAllocDefault);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaHostAlloc src failed: ") + cudaGetErrorString(err));
            }
            err = cudaHostAlloc(&dst_gpu, length * sizeof(Field), cudaHostAllocDefault);
            if (err != cudaSuccess) {
                cudaFreeHost(src_gpu);
                throw std::runtime_error(std::string("cudaHostAlloc dst failed: ") + cudaGetErrorString(err));
            }

            for (long long i = 0; i < length; i++) {
                src_gpu[i] = Field::host_random();
            }

            for (int sample = 0; sample < args.samples; sample++) {
                printf("running k = %d, sample = %d\n", k, sample);
                fflush(stdout);
                ntt::offchip_ntt<Field>((uint*)src_gpu, (uint*)dst_gpu, k, (uint*)&omega);
                fflush(stdout);
            }

            cudaFreeHost(dst_gpu);
            cudaFreeHost(src_gpu);
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
}
