// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use sppark::NTTInputOutputOrder;

const DEFAULT_GPU: usize = 0;

#[cfg(any(
    feature = "bls12_377",
    feature = "bls12_381",
    feature = "bn254",
    feature = "pallas",
    feature = "vesta",
    feature = "mnt4753",
))]
fn main() {
    #[cfg(feature = "bls12_377")]
    use ark_bls12_377::Fr;
    #[cfg(feature = "bls12_381")]
    use ark_bls12_381::Fr;
    #[cfg(feature = "bn254")]
    use ark_bn254::Fr;
    #[cfg(feature = "pallas")]
    use ark_pallas::Fr;
    #[cfg(feature = "vesta")]
    use ark_vesta::Fr;
    #[cfg(feature = "mnt4753")]
    use ark_mnt4_753::Fr;

    use ark_ff::{PrimeField, UniformRand};
    use ark_poly::{
        domain::DomainCoeff, EvaluationDomain, GeneralEvaluationDomain,
    };
    use ark_std::test_rng;

    fn test_ntt<
        F: PrimeField,
        T: DomainCoeff<F> + UniformRand + core::fmt::Debug + Eq,
        R: ark_std::rand::Rng,
        D: EvaluationDomain<F>,
    >(
        rng: &mut R,
    ) {
        for lg_domain_size in (20..=30).step_by(2) {
            let domain_size = 1usize << lg_domain_size;

            println!("Testing NTT on domain size 2^{}", lg_domain_size);

            let mut v = vec![];
            for _ in 0..domain_size {
                v.push(T::rand(rng));
            }

            for _ in 0..10 {
                ntt_cuda::NTT(DEFAULT_GPU, &mut v, NTTInputOutputOrder::NN);
            }
        }
    }

    let rng = &mut test_rng();

    test_ntt::<Fr, Fr, _, GeneralEvaluationDomain<Fr>>(rng);
}
