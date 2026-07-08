#include "../src/bn254.cuh"
#include "../src/msm.cuh"
#include "../../mont/src/bn254_scalar.cuh"

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using bn254::Point;
using bn254::PointAffine;
using bn254_scalar::Element;
using bn254_scalar::Number;
using mont::u32;
using mont::u64;

struct MsmProblem
{
  u64 len = 0;
  PointAffine *points = nullptr;
  Element *scalers = nullptr;

  ~MsmProblem()
  {
    delete[] points;
    delete[] scalers;
  }

  MsmProblem() = default;
  MsmProblem(const MsmProblem &) = delete;
  MsmProblem &operator=(const MsmProblem &) = delete;
};

std::istream &
operator>>(std::istream &is, MsmProblem &msm)
{
  u64 base_len;
  is >> base_len;
  if (msm.len == 0) {
    msm.len = base_len;
  }
  if (base_len == 0 || base_len > msm.len) {
    throw std::runtime_error("invalid MSM input length");
  }
  msm.scalers = new Element[msm.len];
  msm.points = new PointAffine[msm.len];
  for (u64 i = 0; i < base_len; i++)
  {
    char _;
    is >> msm.scalers[i].n >> _ >> msm.points[i];
  }
  for (u64 i = base_len; i < msm.len; i++) {
    msm.scalers[i] = msm.scalers[i - base_len];
    msm.points[i] = msm.points[i - base_len];
  }
  return is;
}

std::ostream &
operator<<(std::ostream &os, const MsmProblem &msm)
{
  for (u64 i = 0; i < msm.len; i++)
  {
    os << msm.scalers[i].n << '|' << msm.points[i] << std::endl;
  }
  return os;
}

int main(int argc, char *argv[])
{
  if (argc != 2 && argc != 3)
  {
    std::cout << "usage: <prog> input_file [log_len]" << std::endl;
    return 2;
  }

  std::ifstream rf(argv[1]);
  if (!rf.is_open())
  {
    std::cout << "open file " << argv[1] << " failed" << std::endl;
    return 3;
  }

  MsmProblem msm;
  if (argc == 3) {
    int log_len = std::stoi(argv[2]);
    msm.len = 1ull << log_len;
  }

  rf >> msm;

  cudaHostRegister((void*)msm.scalers, msm.len * sizeof(Element), cudaHostRegisterDefault);
  cudaHostRegister((void*)msm.points, msm.len * sizeof(PointAffine), cudaHostRegisterDefault);

#ifndef MSM_WINDOW_SIZE
#define MSM_WINDOW_SIZE 22
#endif

#ifndef MSM_PRECOMPUTE
#define MSM_PRECOMPUTE 2
#endif

#ifndef MSM_BATCH_SIZE
#define MSM_BATCH_SIZE 4
#endif

#ifndef MSM_BATCH_PER_RUN
#define MSM_BATCH_PER_RUN 2
#endif

#ifndef MSM_PARTS
#define MSM_PARTS 8
#endif

#ifndef MSM_STAGE_SCALERS
#define MSM_STAGE_SCALERS 2
#endif

#ifndef MSM_STAGE_POINTS
#define MSM_STAGE_POINTS 2
#endif

  using Config = msm::MsmConfig<255, MSM_WINDOW_SIZE, MSM_PRECOMPUTE, false>;
  u32 batch_size = MSM_BATCH_SIZE;
  u32 batch_per_run = MSM_BATCH_PER_RUN;
  u32 parts = MSM_PARTS;
  u32 stage_scalers = MSM_STAGE_SCALERS;
  u32 stage_points = MSM_STAGE_POINTS;

  std::array<u32*, Config::n_precompute> h_points;
  h_points[0] = (u32*)msm.points;
  for (u32 i = 1; i < Config::n_precompute; i++) {
    cudaHostAlloc(&h_points[i], msm.len * sizeof(PointAffine), cudaHostAllocDefault);
  }

  
  std::vector<u32*> scalers_batches;
  for (int i = 0; i < batch_size; i++) {
    scalers_batches.push_back((u32*)msm.scalers);
  }

  std::vector<Point> r(batch_size);

  std::vector<u32> cards;
  int card_count;
  cudaGetDeviceCount(&card_count);
  for (int i = 0; i < card_count; i++) {
    cards.push_back(i);
  }

  msm::MultiGPUMSM<Config, Number, Point, PointAffine> msm_solver(msm.len, batch_per_run, parts, stage_scalers, stage_points, cards);

  std::cout << "start precompute" << std::endl;

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  msm::MSMPrecompute<Config, Point, PointAffine>::precompute(msm.len, h_points, 4);
  msm_solver.set_points(h_points);

  std::cout << "Precompute done" << std::endl;
  msm_solver.alloc_gpu();
  std::cout << "Alloc GPU done" << std::endl;
  cudaEvent_t start, stop;
  float elapsedTime = 0.0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  msm_solver.msm(scalers_batches, r);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Run done" << std::endl;

  cudaStreamDestroy(stream);

  for (int i = 0; i < batch_size; i++) {
    std::cout << r[i].to_affine() << std::endl;
  }

  std::cout << std::dec;
  std::cout << "Config: curve=bn254"
            << " len=" << msm.len
            << " window_size=" << Config::s
            << " precompute=" << Config::n_windows
            << " batch_size=" << batch_size
            << " batch_per_run=" << batch_per_run
            << " parts=" << parts
            << std::endl;
  std::cout << "Total cost time:" << elapsedTime << std::endl;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaHostUnregister((void*)msm.scalers);
  cudaHostUnregister((void*)msm.points);
  for (u32 i = 1; i < Config::n_precompute; i++) {
    cudaFreeHost(h_points[i]);
  }

  return 0;
}
