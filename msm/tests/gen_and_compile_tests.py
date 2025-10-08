import os
import concurrent.futures
import time

# C++ 模板代码保持不变
run_template = """
#include "../src/bn254.cuh"
#include "../src/msm.cuh"
#include "../../mont/src/bn254_scalar.cuh"

#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <array>

using bn254::Point;
using bn254::PointAffine;
using bn254_scalar::Element;
using bn254_scalar::Number;
using mont::u32;
using mont::u64;

struct MsmProblem
{{
  u32 len;
  PointAffine *points;
  Element *scalers;
}};

std::istream &
operator>>(std::istream &is, MsmProblem &msm)
{{
  int base_len;
  is >> base_len;
  msm.scalers = new Element[msm.len];
  msm.points = new PointAffine[msm.len];
  assert(base_len <= msm.len);
  for (u32 i = 0; i < base_len; i++)
  {{
    char _;
    is >> msm.scalers[i].n >> _ >> msm.points[i];
  }}
  for (u32 i = base_len; i < msm.len; i++) {{
    msm.scalers[i] = msm.scalers[i - base_len];
    msm.points[i] = msm.points[i - base_len];
  }}
  return is;
}}

std::ostream &
operator<<(std::ostream &os, const MsmProblem &msm)
{{

  for (u32 i = 0; i < msm.len; i++)
  {{
    os << msm.scalers[i].n << '|' << msm.points[i] << std::endl;
  }}
  return os;
}}

int main(int argc, char *argv[])
{{
  // if (argc != 2)
  // {{
  //   std::cout << "usage: <prog> input_file" << std::endl;
  //   return 2;
  // }}

  u32 len = 1 << {log_len};
  constexpr u32 window_size = {window_size};
  constexpr u32 precompute = {precompute};
  constexpr bool debug = {debug};
  u32 parts = {parts};

  char filename[] = "/state/partition/xwqiang/zk0.99c/msm/tests/msm20.input";

  std::ifstream rf(filename);
  if (!rf.is_open())
  {{
    std::cout << "open file " << filename << " failed" << std::endl;
    return 3;
  }}

  MsmProblem msm;
  msm.len = len;

  rf >> msm;

  cudaHostRegister((void*)msm.scalers, msm.len * sizeof(Element), cudaHostRegisterDefault);
  cudaHostRegister((void*)msm.points, msm.len * sizeof(PointAffine), cudaHostRegisterDefault);

  using Config = msm::MsmConfig<255, window_size, precompute, debug>;
  u32 batch_size = 4;
  u32 batch_per_run = 4;
  u32 stage_scalers = 3;
  u32 stage_points = 3;

  std::array<u32*, Config::n_precompute> h_points;
  h_points[0] = (u32*)msm.points;
  for (u32 i = 1; i < Config::n_precompute; i++) {{
    cudaHostAlloc(&h_points[i], msm.len * sizeof(PointAffine), cudaHostAllocDefault);
  }}

  
  std::vector<u32*> scalers_batches;
  for (int i = 0; i < batch_size; i++) {{
    scalers_batches.push_back((u32*)msm.scalers);
  }}

  std::vector<Point> r(batch_size);

  std::vector<u32> cards;
  int card_count;
  cudaGetDeviceCount(&card_count);
  for (int i = 0; i < card_count; i++) {{
    cards.push_back(i);
  }}

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

  for (int i = 0; i < batch_size; i++) {{
    std::cout << r[i].to_affine() << std::endl;
  }}

  std::cout << "Total cost time:" << elapsedTime << std::endl;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaHostUnregister((void*)msm.scalers);
  cudaHostUnregister((void*)msm.points);
  for (u32 i = 1; i < Config::n_precompute; i++) {{
    cudaFreeHost(h_points[i]);
  }}

  return 0;
}}
"""

def generate_file_name(log_len, window_size, precompute, debug, parts):
    return f"msm_bn254_len{log_len}_w{window_size}_p{precompute}_debug{debug}_parts{parts}.cu"

def generate_run_file(log_len, window_size, precompute, debug, parts):
    instance = run_template.format(log_len=log_len, window_size=window_size, precompute=precompute, debug=str(debug).lower(), parts=parts)
    # print(f"Generating run file for log_len={log_len}, window_size={window_size}, precompute={precompute}, debug={debug}, parts={parts}")
    path = "msm/tests/"
    file_name = generate_file_name(log_len, window_size, precompute, debug, parts)
    with open(path + file_name, "w") as f:
        f.write(instance)
    return file_name

def compile_run_file(file_name):
    file_path = "msm/tests/"
    output_path = "build/"
    # 确保 build 目录存在
    os.makedirs(output_path, exist_ok=True)
    output_name = file_name.replace(".cu", "")
    command = f"/usr/local/cuda/bin/nvcc  -Xcompiler -fPIE -O3 -L/state/partition/xwqiang/zk0.99c/lib/ -lmsm_lib -lcudadevrt -lcudart_static  -I/usr/local/cuda/include --std c++20 -m64 -rdc=true -ccbin=g++-11 -gencode arch=compute_89,code=sm_89 -DNDEBUG -o {output_path}{output_name}  {file_path}{file_name}"
    # print(f"Executing: {command}")
    exit_code = os.system(command)
    if exit_code != 0:
        return f"Error compiling {file_name}"
    return f"Successfully compiled {file_name}"

def process_config(config):
    """
    为单个配置生成并编译文件。这是并行任务的执行单元。
    """
    try:
        file_name = generate_run_file(*config)
        result = compile_run_file(file_name)
        return result
    except Exception as e:
        return f"An error occurred while processing {config}: {e}"

def generate_and_compile_all_parallel(configs):
    """
    使用进程池并行地生成和编译所有配置文件。
    """
    # ProcessPoolExecutor 会自动使用您机器上的所有可用 CPU 核心
    with concurrent.futures.ProcessPoolExecutor() as executor:
        print(f"Starting parallel generation and compilation for {len(configs)} configs...")
        
        # executor.map 会将 configs 中的每个元素传递给 process_config 函数
        # 并且并行地执行它们。它会保持原始顺序返回结果。
        results = list(executor.map(process_config, configs))
        
        # 打印出每个任务的结果
        for result in results:
            print(result)

# 程序的入口点
if __name__ == "__main__":
    start_time = time.time()
    
    # 定义要测试的参数范围
    window_sizes = [8, 12, 16, 20, 24]
    precomputes = [1, 2, 4, 8, 16, 32]
    lengths = [24]
    parts = [1, 2, 4, 8, 16]
    
    # 生成所有配置组合
    configs = []
    for length in lengths:
        for window_size in window_sizes:
            for precompute in precomputes:
                for debug in ["true", "false"]:
                    for part in parts:
                        configs.append((length, window_size, precompute, debug, part))
    
    # 调用并行处理函数
    generate_and_compile_all_parallel(configs)
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")