use std::ops::{Add, Div, Mul, Neg, Sub};

use super::{SIMDRound, Vec4fBase};
use crate::{
    intrinsics::*,
    macros::{vec_impl_binary_op, vec_impl_broadcast_default, vec_impl_unary_op},
    prelude::{SIMDBase, SIMDFusedCalc},
};
use derive_more::{From, Into};

#[repr(transparent)]
#[derive(Copy, Clone, From, Into)]
pub struct Vec4f(float32x4_t);

impl SIMDBase<4> for Vec4f {
    type Underlying = float32x4_t;
    type Element = f32;

    #[inline]
    fn broadcast(value: Self::Element) -> Self {
        // SAFETY: the `cfg_if!` in `vec4f/mod.rs` guarantees the intrinsic is available.
        unsafe { vdupq_n_f32(value) }.into()
    }

    #[inline]
    unsafe fn load_ptr(addr: *const Self::Element) -> Self {
        vld1q_f32(addr).into()
    }

    #[inline]
    unsafe fn store_ptr(self, addr: *mut Self::Element) {
        vst1q_f32(addr, self.0)
    }

    #[inline]
    fn sum(self) -> Self::Element {
        // SAFETY: the `cfg_if!` in `vec4f/mod.rs` guarantees the intrinsic is available.
        unsafe { vaddvq_f32(self.0) }
    }
}

impl Vec4fBase for Vec4f {
    #[inline]
    fn new(v0: f32, v1: f32, v2: f32, v3: f32) -> Self {
        // SAFETY: the `cfg_if!` in `vec4f/mod.rs` guarantees the intrinsic is available.
        unsafe { Self::load_ptr([v0, v1, v2, v3].as_ptr()) }
    }
}

vec_impl_broadcast_default!(Vec4f, 0.0);
vec_impl_unary_op!(Vec4f, Neg, neg, vnegq_f32);
vec_impl_binary_op!(Vec4f, Add, add, vaddq_f32);
vec_impl_binary_op!(Vec4f, Sub, sub, vsubq_f32);
vec_impl_binary_op!(Vec4f, Mul, mul, vmulq_f32);
vec_impl_binary_op!(Vec4f, Div, div, vdivq_f32);

impl PartialEq for Vec4f {
    #[inline]
    fn eq(&self, other: &Vec4f) -> bool {
        // SAFETY: the `cfg_if!` in `vec4f/mod.rs` guarantees the intrinsic is available.
        unsafe { vminvq_u32(vceqq_f32(self.0, other.0)) != 0 }
    }
}

impl SIMDFusedCalc for Vec4f {
    #[inline]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // SAFETY: the `cfg_if!` in `vec4f/mod.rs` guarantees the intrinsic is available.
        unsafe { vfmaq_f32(c.0, b.0, self.0) }.into()
    }

    #[inline]
    fn mul_sub(self, b: Self, c: Self) -> Self {
        -self.nmul_add(b, c)
    }

    #[inline]
    fn nmul_add(self, b: Self, c: Self) -> Self {
        // SAFETY: the `cfg_if!` in `vec4f/mod.rs` guarantees the intrinsic is available.
        unsafe { vfmsq_f32(c.0, b.0, self.0) }.into()
    }

    #[inline]
    fn nmul_sub(self, b: Self, c: Self) -> Self {
        -self.mul_add(b, c)
    }
}

impl SIMDRound for Vec4f {
    fn round(self) -> Self {
        // SAFETY: the `cfg_if!` in `vec4f/mod.rs` guarantees the intrinsic is available.
        unsafe { vcvtq_f32_s32(vcvtaq_s32_f32(self.0)) }.into()
    }
}
