use crate::{intrinsics::*, macros::vecbase_impl_binary_op, Vec4f};
use std::ops::{Add, Div, Mul, Neg, Sub};

pub type Underlying = __m256;

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Vec8fBase(Underlying);

impl Vec8fBase {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(v0: f32, v1: f32, v2: f32, v3: f32, v4: f32, v5: f32, v6: f32, v7: f32) -> Self {
        unsafe { _mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7) }.into()
    }

    #[inline(always)]
    pub fn join(low: Vec4f, high: Vec4f) -> Self {
        unsafe { _mm256_set_m128(high.into(), low.into()) }.into()
    }

    #[inline(always)]
    pub fn broadcast(value: f32) -> Self {
        unsafe { _mm256_set1_ps(value) }.into()
    }

    #[inline(always)]
    pub fn low(self) -> Vec4f {
        unsafe { _mm256_castps256_ps128(self.0) }.into()
    }

    #[inline(always)]
    pub fn high(self) -> Vec4f {
        unsafe { _mm256_extractf128_ps(self.0, 1) }.into()
    }

    #[inline(always)]
    pub unsafe fn load_ptr(addr: *const f32) -> Self {
        _mm256_loadu_ps(addr).into()
    }

    #[inline(always)]
    pub unsafe fn load_ptr_aligned(addr: *const f32) -> Self {
        _mm256_load_ps(addr).into()
    }

    #[inline(always)]
    pub unsafe fn store_ptr(self, addr: *mut f32) {
        _mm256_storeu_ps(addr, self.0);
    }

    #[inline(always)]
    pub unsafe fn store_ptr_aligned(self, addr: *mut f32) {
        _mm256_store_ps(addr, self.0);
    }

    #[inline(always)]
    pub unsafe fn store_ptr_non_temporal(self, addr: *mut f32) {
        _mm256_stream_ps(addr, self.0)
    }
}

impl From<Underlying> for Vec8fBase {
    /// Wraps given `value` into [`Vec8f`].
    #[inline(always)]
    fn from(value: Underlying) -> Self {
        Self(value)
    }
}

impl From<Vec8fBase> for Underlying {
    /// Unwraps given vector into raw [`__m256`] value.
    #[inline(always)]
    fn from(value: Vec8fBase) -> Self {
        value.0
    }
}

impl Default for Vec8fBase {
    #[inline(always)]
    fn default() -> Self {
        unsafe { _mm256_setzero_ps() }.into()
    }
}

impl Neg for Vec8fBase {
    type Output = Self;

    fn neg(self) -> Self::Output {
        unsafe { _mm256_xor_ps(self.0, _mm256_set1_ps(-0.0)) }.into()
    }
}

impl PartialEq for Vec8fBase {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp_result = _mm256_cmp_ps::<0>(self.0, other.0);
            _mm256_testz_ps(cmp_result, cmp_result) == 0
        }
    }
}

vecbase_impl_binary_op!(Vec8fBase, Add, add, _mm256_add_ps);
vecbase_impl_binary_op!(Vec8fBase, Sub, sub, _mm256_sub_ps);
vecbase_impl_binary_op!(Vec8fBase, Mul, mul, _mm256_mul_ps);
vecbase_impl_binary_op!(Vec8fBase, Div, div, _mm256_div_ps);

// TODO: this should be in mod.rs
use super::Vec8f;

impl From<Underlying> for Vec8f {
    fn from(value: Underlying) -> Self {
        Self(Vec8fBase::from(value))
    }
}

impl From<Vec8f> for Underlying {
    fn from(value: Vec8f) -> Self {
        value.0.into()
    }
}
