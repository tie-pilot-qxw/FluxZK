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
    #[cfg(feature = "mnt4753")]
    use ark_mnt4_753::Fr;
    #[cfg(feature = "pallas")]
    use ark_pallas::Fr;
    #[cfg(feature = "vesta")]
    use ark_vesta::Fr;

    use ark_ff::{PrimeField, UniformRand};
    use ark_poly::{domain::DomainCoeff, EvaluationDomain, GeneralEvaluationDomain};
    use ark_std::test_rng;

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .map(|value| value.parse::<usize>().expect("invalid integer"))
            .unwrap_or(default)
    }

    fn benchmark_sizes() -> Vec<usize> {
        match std::env::var("SPPARK_KS") {
            Ok(value) => value
                .split(',')
                .filter(|item| !item.trim().is_empty())
                .map(|item| item.trim().parse::<usize>().expect("invalid SPPARK_KS"))
                .collect(),
            Err(_) => (20..=30).step_by(2).collect(),
        }
    }

    fn test_ntt<
        F: PrimeField,
        T: DomainCoeff<F> + UniformRand + core::fmt::Debug + Eq,
        R: ark_std::rand::Rng,
        D: EvaluationDomain<F>,
    >(
        rng: &mut R,
    ) {
        let warmups = env_usize("SPPARK_WARMUPS", 1);
        let samples = env_usize("SPPARK_SAMPLES", 10);
        for lg_domain_size in benchmark_sizes() {
            let domain_size = 1usize << lg_domain_size;

            println!(
                "Testing NTT on domain size 2^{} (warmups={}, samples={})",
                lg_domain_size, warmups, samples
            );

            let mut v = vec![];
            for _ in 0..domain_size {
                v.push(T::rand(rng));
            }

            for _ in 0..warmups {
                ntt_cuda::NTT(DEFAULT_GPU, &mut v, NTTInputOutputOrder::NN);
            }
            println!("Measured samples for 2^{}", lg_domain_size);
            for _ in 0..samples {
                ntt_cuda::NTT(DEFAULT_GPU, &mut v, NTTInputOutputOrder::NN);
            }
        }
    }

    let rng = &mut test_rng();

    test_ntt::<Fr, Fr, _, GeneralEvaluationDomain<Fr>>(rng);
}
