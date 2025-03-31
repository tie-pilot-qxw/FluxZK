#include "../msm_impl.cuh"
#include "../alt_bn128.cuh"
#include "../../../mont/src/alt_bn128_fr.cuh"

namespace msm {
    using Config = MsmConfig<255, 20, 16, true>;
    template class MSM<Config, alt_bn128_fr::Number, alt_bn128_g1::Point, alt_bn128_g1::PointAffine>;
    template class MSMPrecompute<Config, alt_bn128_g1::Point, alt_bn128_g1::PointAffine>;
    template class MultiGPUMSM<Config, alt_bn128_fr::Number, alt_bn128_g1::Point, alt_bn128_g1::PointAffine>;
}