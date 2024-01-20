use crate::{common::SIMDBase, intrinsics::*, macros::vec_impl_binary_op, vec4f::Vec4f};
use derive_more::{From, Into};
use std::ops::{Add, Div, Mul, Neg, Sub};

use super::Vec8fBase;

#[repr(transparent)]
#[derive(Clone, Copy, From, Into)]
pub struct Vec8f(__m256);

impl super::Vec8fBase for Vec8f {
    #[inline]
    fn new(v0: f32, v1: f32, v2: f32, v3: f32, v4: f32, v5: f32, v6: f32, v7: f32) -> Self {
        // SAFETY: the `cfg_if!` in `vec8f/mod.rs` guarantees the intrinsic is available.
        unsafe { _mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7) }.into()
    }

    #[inline]
    fn join(low: Vec4f, high: Vec4f) -> Self {
        // SAFETY: the `cfg_if!` in `vec8f/mod.rs` guarantees the intrinsic is available.
        unsafe { _mm256_set_m128(high.into(), low.into()) }.into()
    }

    #[inline]
    fn low(self) -> Vec4f {
        // SAFETY: the `cfg_if!` in `vec8f/mod.rs` guarantees the intrinsic is available.
        unsafe { _mm256_castps256_ps128(self.0) }.into()
    }

    #[inline]
    fn high(self) -> Vec4f {
        // SAFETY: the `cfg_if!` in `vec8f/mod.rs` guarantees the intrinsic is available.
        unsafe { _mm256_extractf128_ps(self.0, 1) }.into()
    }

    #[inline]
    unsafe fn load_ptr_aligned(addr: *const f32) -> Self {
        _mm256_load_ps(addr).into()
    }

    #[inline]
    unsafe fn store_ptr_aligned(self, addr: *mut f32) {
        _mm256_store_ps(addr, self.0);
    }

    #[inline]
    unsafe fn store_ptr_non_temporal(self, addr: *mut f32) {
        _mm256_stream_ps(addr, self.0)
    }
}

impl SIMDBase<8> for Vec8f {
    type Underlying = __m256;
    type Element = f32;

    #[inline]
    fn broadcast(value: f32) -> Self {
        // SAFETY: the `cfg_if!` in `vec8f/mod.rs` guarantees the intrinsic is available.
        unsafe { _mm256_set1_ps(value) }.into()
    }

    #[inline]
    unsafe fn load_ptr(addr: *const f32) -> Self {
        _mm256_loadu_ps(addr).into()
    }

    #[inline]
    unsafe fn store_ptr(self, addr: *mut Self::Element) {
        _mm256_storeu_ps(addr, self.0);
    }

    #[inline]
    fn sum(self) -> Self::Element {
        (self.low() + self.high()).sum()
    }
}

impl Default for Vec8f {
    #[inline]
    fn default() -> Self {
        // SAFETY: the `cfg_if!` in `vec8f/mod.rs` guarantees the intrinsic is available.
        unsafe { _mm256_setzero_ps() }.into()
    }
}

impl Neg for Vec8f {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        // SAFETY: the `cfg_if!` in `vec8f/mod.rs` guarantees the intrinsic is available.
        unsafe { _mm256_xor_ps(self.0, _mm256_set1_ps(-0.0)) }.into()
    }
}

impl PartialEq for Vec8f {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // SAFETY: the `cfg_if!` in `vec8f/mod.rs` guarantees the intrinsic is available.
        unsafe {
            let cmp_result = _mm256_cmp_ps::<0>(self.0, other.0);
            _mm256_testz_ps(cmp_result, cmp_result) == 0
        }
    }
}

vec_impl_binary_op!(Vec8f, Add, add, _mm256_add_ps);
vec_impl_binary_op!(Vec8f, Sub, sub, _mm256_sub_ps);
vec_impl_binary_op!(Vec8f, Mul, mul, _mm256_mul_ps);
vec_impl_binary_op!(Vec8f, Div, div, _mm256_div_ps);
