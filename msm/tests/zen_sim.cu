#include "../src/bls12381.cuh"
#include "../src/msm.cuh"

#include <iostream>
#include <fstream>

using mont::u32;
using mont::u64;
template <typename G>
struct MsmPoints
{
  u32 len;
  G *ptr;

  MsmPoints extend_to(u32 new_len) {
    G *new_ptr = new G[new_len];
    u32 cnt = 0;

    while (cnt < new_len) {
      u32 copy_len = std::min(this->len, new_len - cnt);
      memcpy(new_ptr + cnt, this->ptr, copy_len * sizeof(G));
      cnt += copy_len;
    }

    delete [] this->ptr;

    return MsmPoints {
      .len = new_len,
      .ptr = new_ptr
    };
  }

  static MsmPoints random(u32 total_length, u32 random_length) {
    MsmPoints points;
    points.len = random_length;
    points.ptr = new G[random_length];

    for (u32 i = 0; i < random_length; i ++)
      points.ptr[i] = G::false_host_random();

    return points.extend_to(total_length);
  }
};

struct MsmScalars
{
  u32 len;
  bls12381::Number *ptr;

  MsmScalars extend_to(u32 new_len) {
    MsmScalars new_scalars;
    new_scalars.len = new_len;
    new_scalars.ptr = new bls12381_fr::Number[new_len];

    u32 cnt = 0;
    while (cnt < new_len) {
      u32 copy_len = std::min(this->len, new_len - cnt);
      memcpy(new_scalars.ptr + cnt, this->ptr, copy_len * sizeof(bls12381_fr::Number));
      cnt += copy_len;
    }

    delete [] this->ptr;

    return new_scalars;
  }

  static MsmScalars random(u32 total_length, u32 random_length) {
    MsmScalars scalars;
    scalars.len = random_length;
    scalars.ptr = new bls12381_fr::Number[random_length];

    for (u32 i = 0; i < random_length; i ++)
      scalars.ptr[i] = bls12381_fr::Element::host_random().to_number();

    return scalars.extend_to(total_length);
  }
};

template <typename G, typename GXYZZ>
float run_msm(MsmPoints<G> &points, MsmScalars &scalars, u32 len)
{

  cudaHostRegister((void *)scalars.ptr, scalars.len * sizeof(bls12381::Number), cudaHostRegisterDefault);
  cudaHostRegister((void *)points.ptr, points.len * sizeof(G), cudaHostRegisterDefault);

  using Config = msm::MsmConfig<382, 16, 16, true>;
  u32 batch_size = 1;
  u32 batch_per_run = 1;
  u32 parts = 8;
  u32 stage_scalers = 2;
  u32 stage_points = 2;

  std::array<u32 *, Config::n_precompute> h_points;
  h_points[0] = (u32 *)points.ptr;
  for (u32 i = 1; i < Config::n_precompute; i++)
  {
    cudaHostAlloc(&h_points[i], len * sizeof(G), cudaHostAllocDefault);
  }

  std::vector<u32 *> scalers_batches;
  for (int i = 0; i < batch_size; i++)
  {
    scalers_batches.push_back((u32 *)scalars.ptr);
  }

  std::vector<GXYZZ> r(batch_size);

  std::vector<u32> cards;
  int card_count = 1;
  for (int i = 0; i < card_count; i++)
  {
    cards.push_back(i);
  }

  msm::MultiGPUMSM<Config, bls12381::Number, GXYZZ, G> msm_solver(len, batch_per_run, parts, stage_scalers, stage_points, cards);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  msm::MSMPrecompute<Config, GXYZZ, G>::precompute(len, h_points, 4);
  msm_solver.set_points(h_points);

  msm_solver.alloc_gpu();
  cudaEvent_t start, stop;
  float elapsedTime = 0.0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  msm_solver.msm(scalers_batches, r);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  cudaStreamDestroy(stream);

  for (int i = 0; i < batch_size; i++)
  {
    std::cout << r[i].to_affine() << std::endl;
  }

  std::cout << "Total cost time:" << elapsedTime << std::endl;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaHostUnregister((void *)scalars.ptr);
  cudaHostUnregister((void *)points.ptr);
  for (u32 i = 1; i < Config::n_precompute; i++)
  {
    cudaFreeHost(h_points[i]);
  }

  return elapsedTime;
}

int main(int argc, char *argv[]) {
  if (argc != 3)
  {
    std::cout << "usage: <prog> len1 len2" << std::endl;
    return 2;
  }

  u32 len1 = atoi(argv[1]);
  u32 len2 = atoi(argv[2]);

  std::cout << "Generating " << std::max(len1, len2) << " scalars" << std::endl;
  auto scalars = MsmScalars::random(len1, 1024);

  std::cout << "Generating " << std::max(len1, len2) << " G1 points" << std::endl;
  auto points1 = MsmPoints<bls12381::PointAffine>::random(std::max(len1, len2), 1024);
  std::cout << "Generating " << len1 << " G2 points" << std::endl;
  auto points2 = MsmPoints<bls12381_g2::PointAffine>::random(std::max(len1, len2), 1024);

  std::cout << "points1.length = " << points1.len << std::endl;
  std::cout << "points2.length = " << points2.len << std::endl;

  // std::cout << "Compute MSM on G1 with length " << len1 << std::endl;
  // auto time1 = run_msm<bls12381::PointAffine, bls12381::Point>(points1, scalars, len1);
  // std::cout << "Finished in " << time1 << " s" << std::endl;
  //
  std::cout << "PointAffine::N_WORDS = " << bls12381_g2::PointAffine::N_WORDS << std::endl;
  std::cout << "Point::N_WORDS = " << bls12381_g2::Point::N_WORDS << std::endl;
  std::cout << "sizeof(PointAffine) = " << sizeof(bls12381_g2::PointAffine) << std::endl;
  std::cout << "sizeof(Point) = " << sizeof(bls12381_g2::Point) << std::endl;

  std::cout << "Compute MSM on G2 with length" << len1 << std::endl;
  auto time2 = run_msm<bls12381_g2::PointAffine, bls12381_g2::Point>(points2, scalars, len1);
  std::cout << "Finished in " << time2 << " s" << std::endl;

  std::cout << "Compute MSM on G1 with length " << len2 << std::endl;
  auto time3 = run_msm<bls12381::PointAffine, bls12381::Point>(points1, scalars, len2);
  std::cout << "Finished in " << time3 << " s" << std::endl;

  return 0;
}
