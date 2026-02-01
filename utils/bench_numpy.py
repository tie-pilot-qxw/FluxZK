import numpy as np
import timeit
import time
import torch

def benchmark_numpy_transpose(n, m, num_runs=10):
    # Create a simple test array instead of random values
    setup_code = f"""
import numpy as np
# Use a simple shape to avoid random-init overhead
arr = np.ones(({n}, {m}, 8), dtype=np.int32)
# Slightly modify values to validate the transpose
for i in range(min(10, {n})):
    for j in range(min(10, {m})):
        arr[i, j] = np.arange(i*10+j, i*10+j+8)
"""
    
    # Test code - use .copy() to force the transpose
    test_code = "np.transpose(arr, axes=(1, 0, 2)).copy()"
    
    # Compute array size in bytes
    element_size = np.dtype(np.int32).itemsize
    array_size_bytes = n * m * 8 * element_size
    
    # Theoretical memory operations
    theoretical_memory_operations = 2 * array_size_bytes  # read + write
    
    print(f"NumPy Array dimensions: {n}x{m}x8 (int32)")
    print(f"Total array size: {array_size_bytes / (1024*1024):.2f} MB")
    
    # Run once beforehand
    locals_dict = {}
    exec(setup_code, globals(), locals_dict)
    
    # First warm-up (not measured)
    exec(test_code, globals(), locals_dict)
    
    # Measure execution time
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        exec(test_code, globals(), locals_dict)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / num_runs
    
    # Compute theoretical memory bandwidth
    bandwidth = theoretical_memory_operations / (avg_time * 1024 * 1024 * 1024)  # GB/s
    
    print(f"Average time to transpose a {n}x{m}x8 NumPy array over {num_runs} runs: {avg_time:.6f} seconds")
    print(f"Theoretical memory bandwidth: {bandwidth:.2f} GB/s")

def benchmark_pytorch_transpose(n, m, num_runs=10, device='cpu'):
    # Create a PyTorch tensor - no random initialization
    setup_code = f"""
import torch
device = '{device}'
# Use a simple shape to avoid random-init overhead
tensor = torch.ones(({n}, {m}, 8), dtype=torch.int32, device=device)
# Slightly modify values to validate the transpose
for i in range(min(10, {n})):
    for j in range(min(10, {m})):
        tensor[i, j] = torch.arange(i*10+j, i*10+j+8, dtype=torch.int32, device=device)
"""
    
    # Test code - use .contiguous() to force the transpose
    test_code = "tensor.permute(1, 0, 2).contiguous()"
    
    # Compute tensor size in bytes
    element_size = 4  # Bytes per torch.int32
    tensor_size_bytes = n * m * 8 * element_size
    
    # Theoretical memory operations
    theoretical_memory_operations = 2 * tensor_size_bytes
    
    print(f"PyTorch Tensor dimensions: {n}x{m}x8 (int32) on {device}")
    print(f"Total tensor size: {tensor_size_bytes / (1024*1024):.2f} MB")
    
    # Run once to set up the environment
    exec(setup_code)
    locals_dict = {}
    exec(setup_code, globals(), locals_dict)
    
    # First warm-up (not measured)
    exec(test_code, globals(), locals_dict)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Manual timing to avoid timeit overhead
    times = []
    for _ in range(num_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        exec(test_code, globals(), locals_dict)
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / num_runs
    
    # Compute theoretical memory bandwidth
    bandwidth = theoretical_memory_operations / (avg_time * 1024 * 1024 * 1024)  # GB/s
    
    print(f"Average time to transpose a {n}x{m}x8 PyTorch tensor over {num_runs} runs: {avg_time:.6f} seconds")
    print(f"Theoretical memory bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    print("Benchmark started by: anonymous")
    print(f"Date and Time: 2025-03-30 08:19:22 (UTC)")
    print("\n")
    
    # Use fewer runs for large-scale tests
    small_runs = 20
    medium_runs = 10
    large_runs = 5
    
    print("=" * 50)
    print("NUMPY BENCHMARK")
    print("=" * 50)
    
    # print("Small array test (100x100x8):")
    # benchmark_numpy_transpose(100, 100, small_runs)
    
    # print("\nMedium array test (1000x1000x8):")
    # benchmark_numpy_transpose(1000, 1000, medium_runs)
    
    print("\nLarge array test (2000x2000x8):")
    # benchmark_numpy_transpose(2**4, 2**26, large_runs)
    
    print("\n" + "=" * 50)
    print("PYTORCH CPU BENCHMARK")
    print("=" * 50)
    
    # print("Small tensor test (100x100x8):")
    # benchmark_pytorch_transpose(100, 100, small_runs)
    
    # print("\nMedium tensor test (1000x1000x8):")
    # benchmark_pytorch_transpose(1000, 1000, medium_runs)
    
    print("\nLarge tensor test (2000x2000x8):")
    benchmark_pytorch_transpose(2**4, 2**26, large_runs)
    
    # # If a CUDA device is available, also test GPU
    # if torch.cuda.is_available():
    #     print("\n" + "=" * 50)
    #     print("PYTORCH CUDA BENCHMARK")
    #     print("=" * 50)
        
    #     print("Small tensor test (100x100x8):")
    #     benchmark_pytorch_transpose(100, 100, small_runs, device='cuda')
        
    #     print("\nMedium tensor test (1000x1000x8):")
    #     benchmark_pytorch_transpose(1000, 1000, medium_runs, device='cuda')
        
    #     print("\nLarge tensor test (2000x2000x8):")
    #     benchmark_pytorch_transpose(2000, 2000, large_runs, device='cuda')
