use crate::{intrinsics::*, macros::vecbase_impl_binary_op};
use std::ops::{Add, Div, Mul, Neg, Sub};

pub type Underlying = __m128;

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Vec4fBase(__m128);

impl Vec4fBase {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(v0: f32, v1: f32, v2: f32, v3: f32) -> Self {
        unsafe { _mm_setr_ps(v0, v1, v2, v3) }.into()
    }

    #[inline(always)]
    pub fn broadcast(value: f32) -> Self {
        unsafe { _mm_set1_ps(value) }.into()
    }

    #[inline(always)]
    pub unsafe fn load_ptr(addr: *const f32) -> Self {
        _mm_loadu_ps(addr).into()
    }

    #[inline(always)]
    pub unsafe fn load_ptr_aligned(addr: *const f32) -> Self {
        _mm_load_ps(addr).into()
    }

    #[inline(always)]
    pub unsafe fn store_ptr(self, addr: *mut f32) {
        _mm_storeu_ps(addr, self.0);
    }

    #[inline(always)]
    pub unsafe fn store_ptr_aligned(self, addr: *mut f32) {
        _mm_store_ps(addr, self.0);
    }

    #[inline(always)]
    pub unsafe fn store_ptr_non_temporal(self, addr: *mut f32) {
        _mm_stream_ps(addr, self.0)
    }

    #[inline(always)]
    pub fn sum(self) -> f32 {
        // According to Agner Fog, using `hadd` is inefficient.
        // src: https://github.com/vectorclass/version2/blob/master/vectorf128.h#L1043
        // TODO: benchmark this implementation and `hadd`-based one
        unsafe {
            let t1 = _mm_movehl_ps(self.0, self.0);
            let t2 = _mm_add_ps(self.0, t1);
            let t3 = _mm_shuffle_ps(t2, t2, 1);
            let t4 = _mm_add_ss(t2, t3);
            _mm_cvtss_f32(t4)
        }
    }
}

impl From<Underlying> for Vec4fBase {
    /// Wraps given `value` into [`Vec8f`].
    #[inline(always)]
    fn from(value: Underlying) -> Self {
        Self(value)
    }
}

impl From<Vec4fBase> for Underlying {
    /// Unwraps given vector into raw [`__m256`] value.
    #[inline(always)]
    fn from(value: Vec4fBase) -> Self {
        value.0
    }
}

impl Default for Vec4fBase {
    #[inline(always)]
    fn default() -> Self {
        unsafe { _mm_setzero_ps() }.into()
    }
}

impl Neg for Vec4fBase {
    type Output = Self;

    fn neg(self) -> Self::Output {
        unsafe { _mm_xor_ps(self.0, _mm_set1_ps(-0.0)) }.into()
    }
}

impl PartialEq for Vec4fBase {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp_result = _mm_cmpeq_ps(self.0, other.0);
            _mm_movemask_ps(cmp_result) == 0x0F
        }
    }
}

vecbase_impl_binary_op!(Vec4fBase, Add, add, _mm_add_ps);
vecbase_impl_binary_op!(Vec4fBase, Sub, sub, _mm_sub_ps);
vecbase_impl_binary_op!(Vec4fBase, Mul, mul, _mm_mul_ps);
vecbase_impl_binary_op!(Vec4fBase, Div, div, _mm_div_ps);

// TODO: this should be in `mod.rs`
use super::Vec4f;

impl From<Underlying> for Vec4f {
    fn from(value: Underlying) -> Self {
        Self(Vec4fBase::from(value))
    }
}

impl From<Vec4f> for Underlying {
    fn from(value: Vec4f) -> Self {
        value.0.into()
    }
}
