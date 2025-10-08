

template_config = """
#include "../msm_impl.cuh"
#include "../bn254.cuh"

namespace msm {{
    using Config = MsmConfig<255, {window_size}, {precompute}, {debug}>;
    template class MSM<Config, bn254::Number, bn254::Point, bn254::PointAffine>;
    template class MSMPrecompute<Config, bn254::Point, bn254::PointAffine>;
    template class MultiGPUMSM<Config, bn254::Number, bn254::Point, bn254::PointAffine>;
}}
"""

window_sizes = [8, 12, 16, 20, 24]
precomputes = [1, 2, 4, 8, 16, 32]

configs = []

for window_size in window_sizes:
    for p in precomputes:
        for debug in ["true", "false"]:
            configs.append((window_size, p, debug))

for config in configs:
    window_size, precompute, debug = config
    instance = template_config.format(window_size=window_size, precompute=precompute, debug=debug)
    file_path = "msm/src/fast_compile/"
    file_name = f"msm_bn254_w{window_size}_p{precompute}_d{debug}.cu"
    with open(file_path + file_name, "w") as f:
        f.write(instance)
