#include "../src/self_sort_in_place_ntt.cuh"
#include "../../mont/src/bn254_fr.cuh"
#include "bench_args.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace ntt;
typedef bn254_fr::Element Field;

int main(int argc, char **argv) {
    try {
        BenchArgs args = parse_bench_args(argc, argv, 20, 28, 2, 10, 10);
        for (int k : args.ks) {
            long long length = 1ll << k;
            printf("preparing k = %d, length = %lld\n", k, length);
            fflush(stdout);

            auto omega = Field::host_random();
            auto config = self_sort_in_place_ntt<Field>::SSIP_config();
            self_sort_in_place_ntt<Field> ntt(reinterpret_cast<u32*>(&omega), k, false, 1, false, false, nullptr, nullptr, config);
            ntt.to_gpu();

            Field *data, *data_d;
            cudaMalloc(&data_d, length * sizeof(Field));
            data = (Field*)malloc(length * sizeof(Field));
            if (data == nullptr) {
                throw std::runtime_error("host allocation failed");
            }
            for (long long i = 0; i < length; i++) {
                data[i] = Field::host_random();
            }
            cudaMemcpy(data_d, data, length * sizeof(Field), cudaMemcpyHostToDevice);

            for (int i = 0; i < args.warmups; i++) {
                ntt.ntt(reinterpret_cast<u32*>(data_d), 0, 0, true);
            }
            cudaDeviceSynchronize();

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            for (int i = 0; i < args.samples; i++) {
                cudaEventRecord(start);
                ntt.ntt(reinterpret_cast<u32*>(data_d), 0, 0, true);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                printf("k = %d, sample = %d, time = %f ms\n", k, i, milliseconds);
                fflush(stdout);
            }

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            free(data);
            cudaFree(data_d);
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
}
