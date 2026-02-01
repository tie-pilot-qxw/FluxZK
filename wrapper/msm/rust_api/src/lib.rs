include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::any::type_name;

use group::ff::PrimeField;

pub use halo2curves::{CurveAffine, CurveExt};

pub fn gpu_msm<C: CurveAffine>(
    coeffs: &[C::Scalar], 
    bases: &[C],
    acc: &mut C::Curve
) -> Result<(), String> {
    if type_name::<C>() != "halo2curves::bn256::curve::G1Affine" {
        return Err("Only support bn256::G1Affine".to_string());
    }
    let len = coeffs.len();
    // let mut res = vec![0u32; 16]; // Assume PointAffine has 16 u32 values (8 for x and 8 for y)

    // Convert scalars and points to pointers
    let coeffs1: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();
    let scalers_ptr = coeffs1.as_ptr() as *const u32;
    let points_ptr = bases.as_ptr() as *const u32;

    let acc_ptr = acc as *mut C::Curve as *mut u32;

    // Call the CUDA function
    let success = unsafe {
        cuda_msm(len as u32, scalers_ptr, points_ptr, acc_ptr)
    };

    if !success {
        return Err("Failed to execute cuda_msm".to_string());
    }

    Ok(())
}
