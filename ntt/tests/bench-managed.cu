#include "../src/self_sort_in_place_ntt.cuh"
#if defined(NTT_FIELD_BN254)
#include "../../mont/src/bn254_fr.cuh"
#else
#include "../../mont/src/mnt4753_fr.cuh"
#endif
#include "bench_args.hpp"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace ntt;
#if defined(NTT_FIELD_BN254)
typedef bn254_fr::Element Field;
#else
typedef mnt4753_fr::Element Field;
#endif

int main(int argc, char **argv) {
    try {
        BenchArgs args = parse_bench_args(argc, argv, 20, 30, 2, 1, 0);
        for (int k : args.ks) {
            long long length = 1ll << k;
            printf("preparing k = %d, length = %lld\n", k, length);
            fflush(stdout);

            auto omega = Field::host_random();
            auto config = self_sort_in_place_ntt<Field>::SSIP_config();
            config.max_threads_stage1_log = 7;
            config.max_threads_stage2_log = 7;
            self_sort_in_place_ntt<Field> ntt(reinterpret_cast<u32*>(&omega), k, false, 1, false, false, nullptr, nullptr, config);
            ntt.to_gpu();

            Field *data, *data_d;
            data = (Field*)malloc(length * sizeof(Field));
            if (data == nullptr) {
                throw std::runtime_error("host allocation failed");
            }
            cudaError_t err = cudaMallocManaged(&data_d, length * sizeof(Field));
            if (err != cudaSuccess) {
                free(data);
                throw std::runtime_error(std::string("cudaMallocManaged failed: ") + cudaGetErrorString(err));
            }
            for (long long i = 0; i < length; i++) {
                data_d[i] = Field::host_random();
            }

            for (int sample = 0; sample < args.samples; sample++) {
                printf("running k = %d, sample = %d\n", k, sample);
                fflush(stdout);

                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                cudaEventRecord(start);
                ntt.ntt(reinterpret_cast<u32*>(data_d), 0, 0, true);
                memcpy(data, data_d, length * sizeof(Field));
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                printf("k = %d, sample = %d, time = %f ms\n", k, sample, milliseconds);
                fflush(stdout);

                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }

            free(data);
            cudaFree(data_d);
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
}
