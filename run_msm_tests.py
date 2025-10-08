def generate_file_name(log_len, window_size, precompute, debug, parts):
    return f"msm_bn254_len{log_len}_w{window_size}_p{precompute}_debug{debug}_parts{parts}"

import os

if __name__ == "__main__":    
    # 定义要测试的参数范围
    log_file = "logs/runs.log"
    
    window_sizes = [12, 16, 20]
    precomputes = [1, 2, 4, 8, 16]
    lengths = [24]
    parts = [1, 2, 4, 8]
    
    # 生成所有配置组合
    configs = []
    for length in lengths:
        for window_size in window_sizes:
            for precompute in precomputes:
                for debug in ["false"]:
                        for part in parts:
                            configs.append((length, window_size, precompute, debug, part))

    print(f"Running {len(configs)} configurations...")
    for config in configs:
        print(f"Running with config: len {config[0]} window {config[1]} precompute {config[2]} debug {config[3]} parts {config[4]}")
        # add to log file what is being run
        with open(log_file, "a") as f:
            f.write(f"Running with config: len {config[0]} window {config[1]} precompute {config[2]} debug {config[3]} parts {config[4]}\n")
        file_name = generate_file_name(*config)
        os.system(f"CUDA_VISIBLE_DEVICES=0 ./build/{file_name} >> {log_file} 2>&1")

        # pause for 1 seconds between runs
        os.system("sleep 1")


    