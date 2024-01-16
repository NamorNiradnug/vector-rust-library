use std::ops::{Add, Div, Mul, Neg, Sub};

use super::Vec8fBase;
use crate::{
    intrinsics::*,
    macros::vec_impl_broadcast_default,
    prelude::{SIMDBase, SIMDFusedCalc, Vec4f},
};
use derive_more::{From, Into};

#[derive(Copy, Clone, From, Into)]
pub struct Vec8f(float32x4x2_t);

impl SIMDBase<8> for Vec8f {
    type Underlying = float32x4x2_t;
    type Element = f32;

    #[inline]
    fn broadcast(value: Self::Element) -> Self {
        let half = Vec4f::broadcast(value);
        (half, half).into()
    }

    #[inline]
    unsafe fn load_ptr(addr: *const Self::Element) -> Self {
        vld1q_f32_x2(addr).into()
    }

    #[inline]
    unsafe fn store_ptr(self, addr: *mut Self::Element) {
        vst1q_f32_x2(addr, self.0)
    }

    #[inline]
    fn sum(self) -> Self::Element {
        (self.low() + self.high()).sum()
    }
}

impl Vec8fBase for Vec8f {
    #[inline]
    fn new(v0: f32, v1: f32, v2: f32, v3: f32, v4: f32, v5: f32, v6: f32, v7: f32) -> Self {
        [v0, v1, v2, v3, v4, v5, v6, v7].into()
    }

    #[inline]
    fn join(a: Vec4f, b: Vec4f) -> Self {
        float32x4x2_t(a.into(), b.into()).into()
    }

    #[inline]
    fn low(self) -> Vec4f {
        self.0 .0.into()
    }

    #[inline]
    fn high(self) -> Vec4f {
        self.0 .1.into()
    }
}

vec_impl_broadcast_default!(Vec8f, 0.0);

impl Neg for Vec8f {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        unsafe { float32x4x2_t(vnegq_f32(self.0 .0), vnegq_f32(self.0 .1)) }.into()
    }
}

macro_rules! impl_binary_op {
    ($trait: tt, $op: tt, $intrinsic: tt) => {
        impl $trait for Vec8f {
            type Output = Self;
            #[inline]
            fn $op(self, other: Self) -> Self::Output {
                unsafe {
                    float32x4x2_t(
                        $intrinsic(self.0 .0, other.0 .0),
                        $intrinsic(self.0 .1, other.0 .1),
                    )
                    .into()
                }
            }
        }
    };
}

impl_binary_op!(Add, add, vaddq_f32);
impl_binary_op!(Sub, sub, vsubq_f32);
impl_binary_op!(Mul, mul, vmulq_f32);
impl_binary_op!(Div, div, vdivq_f32);

impl PartialEq for Vec8f {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.split() == other.split()
    }
}

vec_impl_fused_low_high!(Vec8f);
