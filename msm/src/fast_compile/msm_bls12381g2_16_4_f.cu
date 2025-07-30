
#include "../msm_impl.cuh"
#include "../bls12381.cuh"

namespace msm {
    using Config = MsmConfig<382, 16, 16, true>;
    template class MSM<Config, bls12381::Number, bls12381_g2::Point, bls12381_g2::PointAffine>;
    template class MSMPrecompute<Config, bls12381_g2::Point, bls12381_g2::PointAffine>;
    template class MultiGPUMSM<Config, bls12381::Number, bls12381_g2::Point, bls12381_g2::PointAffine>;
}
