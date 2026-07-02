#include "./msm_c_api.h"
#include "../../../msm/src/msm.cuh"
#include "../../../msm/src/bn254.cuh"
#include "../../../mont/src/bn254_scalar.cuh"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>

using mont::u32;
using bn254::Point;
using bn254::PointAffine;
using bn254_scalar::Number;
using bn254_scalar::Element;

bool cuda_msm(unsigned int len, const unsigned int* scalers, const unsigned int* points, unsigned int* res) {

    using Config = msm::MsmConfig<255, 16, 16, false>;

    bool success = true;
    bool scalers_registered = false;
    bool points_registered = false;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::array<u32*, Config::n_precompute> h_points{};
    h_points[0] = (u32*)points;

    auto cleanup = [&](bool result) {
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
        if (stream != nullptr) cudaStreamDestroy(stream);
        for (u32 i = 1; i < Config::n_precompute; i++) {
            if (h_points[i] != nullptr) cudaFreeHost(h_points[i]);
        }
        if (points_registered) cudaHostUnregister((void*)points);
        if (scalers_registered) cudaHostUnregister((void*)scalers);
        return result;
    };

    auto check_cuda = [](cudaError_t err, const char *file, int line) {
        if (err == cudaSuccess) return true;
        std::cerr << "CUDA Error [" << file << ":" << line << "]: " << cudaGetErrorString(err) << "\n";
        return false;
    };

#define CHECK_CUDA_OR_RETURN(call) \
    do { if (!check_cuda((call), __FILE__, __LINE__)) return cleanup(false); } while (0)

    CHECK_CUDA_OR_RETURN(cudaHostRegister((void*)scalers, len * sizeof(Number), cudaHostRegisterDefault));
    scalers_registered = true;
    CHECK_CUDA_OR_RETURN(cudaHostRegister((void*)points, len * sizeof(PointAffine), cudaHostRegisterDefault));
    points_registered = true;

    u32 batch_size = 1;
    u32 batch_per_run = 1;
    u32 parts = 2;
    u32 stage_scalers = 2;
    u32 stage_points = 2;

    for (u32 i = 1; i < Config::n_precompute; i++) {
        CHECK_CUDA_OR_RETURN(cudaHostAlloc(&h_points[i], len * sizeof(PointAffine), cudaHostAllocDefault));
    }

    
    std::vector<u32*> scalers_batches;
    for (int i = 0; i < batch_size; i++) {
        scalers_batches.push_back((u32*)scalers);
    }

    std::vector<Point> r(batch_size);

    std::vector<u32> cards;
    int card_count = 0;
    CHECK_CUDA_OR_RETURN(cudaGetDeviceCount(&card_count));
    if (card_count <= 0) {
        std::cerr << "CUDA Error: no CUDA devices available\n";
        return cleanup(false);
    }
    for (int i = 0; i < card_count; i++) {
        cards.push_back(i);
    }

    msm::MultiGPUMSM<Config, Number, Point, PointAffine> msm_solver(len, batch_per_run, parts, stage_scalers, stage_points, cards);

    // std::cout << "start precompute" << std::endl;

    CHECK_CUDA_OR_RETURN(cudaStreamCreate(&stream));
    if (!check_cuda(msm::MSMPrecompute<Config, Point, PointAffine>::precompute(len, h_points), __FILE__, __LINE__)) {
        return cleanup(false);
    }
    msm_solver.set_points(h_points);

    // std::cout << "Precompute done" << std::endl;
    CHECK_CUDA_OR_RETURN(msm_solver.alloc_gpu());
    // std::cout << "Alloc GPU done" << std::endl;
    float elapsedTime = 0.0;

    CHECK_CUDA_OR_RETURN(cudaEventCreate(&start));
    CHECK_CUDA_OR_RETURN(cudaEventCreate(&stop));
    CHECK_CUDA_OR_RETURN(cudaEventRecord(start, 0));

    CHECK_CUDA_OR_RETURN(msm_solver.msm(scalers_batches, r));

    CHECK_CUDA_OR_RETURN(cudaEventRecord(stop, 0));
    CHECK_CUDA_OR_RETURN(cudaEventSynchronize(stop));
    CHECK_CUDA_OR_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
    // std::cout << "Run done" << std::endl;

    // for (int i = 0; i < batch_size; i++) {
    //     std::cout << r[i].to_affine() << std::endl;
    // }

    // std::cout << "Total cost time:" << elapsedTime << std::endl;

    auto r_affine = r[0].to_affine();

    auto x = r_affine.x;
    auto y = r_affine.y;
    auto z = Element::one();

    if (r_affine.is_identity()) { // identity
        x = Element::zero();
        y = Element::one();
        z = Element::zero();
    }

    for(int i=0;i<Element::LIMBS;++i) {
        res[i] = x.n.limbs[i];
    }
    for(int i = 0; i < Element::LIMBS; ++i) {
        res[i+Element::LIMBS] = y.n.limbs[i];
    }
    for(int i = 0; i < Element::LIMBS; ++i) {
        res[i + Element::LIMBS * 2] = z.n.limbs[i];
    }

    return cleanup(success);

#undef CHECK_CUDA_OR_RETURN
}
