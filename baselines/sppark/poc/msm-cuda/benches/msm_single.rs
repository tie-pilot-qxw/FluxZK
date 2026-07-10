// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "bls12_377")]
use ark_bls12_377::{G1Affine, G2Affine};
#[cfg(feature = "bls12_381")]
use ark_bls12_381::{G1Affine, G2Affine};
#[cfg(feature = "bn254")]
use ark_bn254::G1Affine;
use ark_ff::BigInteger256;

use std::str::FromStr;

use msm_cuda::*;

fn main() {
    let bench_npow = std::env::var("BENCH_NPOW").unwrap_or("23".to_string());
    let bench_nbatch = std::env::var("BENCH_NBATCH").unwrap_or("1".to_string());

    let npoints_npow = i32::from_str(&bench_npow).unwrap();
    let nbatch = i32::from_str(&bench_nbatch).unwrap();

    let (points, scalars) =
        util::generate_points_scalars::<G1Affine>(1usize << npoints_npow);

    format!("2**{}", npoints_npow);

    let mut msm = if nbatch == 1 { None } else { Some(Msm::new(points.as_slice())) };

    let start = std::time::Instant::now();

    if nbatch == 1 {
        let _ = multi_scalar_mult_arkworks(&points.as_slice(), unsafe {
            std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
        });
    }  else {
        for _ in 0..nbatch {
            let _ = msm.as_mut().unwrap().invoke::<G1Affine>(unsafe {
                std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
            });
        }
    }

    let duration = start.elapsed().as_secs() * 1000 + start.elapsed().subsec_millis() as u64;

    println!("repeated time : {}", duration);
}
