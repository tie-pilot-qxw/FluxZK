

template_config = """
#include "../msm_impl.cuh"
#include "../mnt4753.cuh"

namespace msm {{
    using Config = MsmConfig<754, {window_size}, {precompute}, {debug}>;
    template class MSM<Config, mnt4753::Number, mnt4753::Point, mnt4753::PointAffine>;
    template class MSMPrecompute<Config, mnt4753::Point, mnt4753::PointAffine>;
    template class MultiGPUMSM<Config, mnt4753::Number, mnt4753::Point, mnt4753::PointAffine>;
}}
"""

window_sizes = [20]
precomputes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]

configs = []

for window_size in window_sizes:
    for p in precomputes:
        for debug in ["true"]:
            configs.append((window_size, p, debug))

configs = []
configs.append((16, 2, "true"))
configs.append((22, 2, "true"))
configs.append((18, 42, "true"))
configs.append((20, 100, "true"))

for config in configs:
    window_size, precompute, debug = config
    instance = template_config.format(window_size=window_size, precompute=precompute, debug=debug)
    file_path = "msm/src/fast_compile/"
    file_name = f"msm_mnt4753_w{window_size}_p{precompute}_d{debug}.cu"
    with open(file_path + file_name, "w") as f:
        f.write(instance)
