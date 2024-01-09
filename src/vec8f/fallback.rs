use derive_more::{Add, Div, Mul, Neg, Sub};

use crate::{
    prelude::SIMDBase,
    vec4f::{Vec4f, Vec4fBase},
};

use super::Vec8fBase;

#[repr(C)]
#[derive(Copy, Clone, Add, Sub, Mul, Div, Neg, Default, PartialEq)]
#[mul(forward)]
#[div(forward)]
pub struct Vec8f(pub Vec4f, pub Vec4f);

impl Vec8fBase for Vec8f {
    #[inline]
    fn new(v0: f32, v1: f32, v2: f32, v3: f32, v4: f32, v5: f32, v6: f32, v7: f32) -> Self {
        (Vec4f::new(v0, v1, v2, v3), Vec4f::new(v4, v5, v6, v7)).into()
    }

    #[inline]
    fn join(a: Vec4f, b: Vec4f) -> Self {
        Self(a, b)
    }

    #[inline]
    unsafe fn load_ptr_aligned(addr: *const f32) -> Self {
        (
            Vec4f::load_ptr_aligned(addr),
            Vec4f::load_ptr_aligned(addr.add(4)),
        )
            .into()
    }

    #[inline]
    unsafe fn store_ptr_aligned(self, addr: *mut f32) {
        self.0.store_ptr_aligned(addr);
        self.1.store_ptr_aligned(addr.add(4));
    }

    #[inline]
    unsafe fn store_ptr_non_temporal(self, addr: *mut f32) {
        self.0.store_ptr_non_temporal(addr);
        self.1.store_ptr_non_temporal(addr.add(4));
    }

    #[inline]
    fn low(self) -> Vec4f {
        self.0
    }

    #[inline]
    fn high(self) -> Vec4f {
        self.1
    }
}

impl SIMDBase<8> for Vec8f {
    type Underlying = (Vec4f, Vec4f);
    type Element = f32;

    #[inline]
    fn broadcast(value: Self::Element) -> Self {
        let half = Vec4f::broadcast(value);
        (half, half).into()
    }

    #[inline]
    unsafe fn load_ptr(addr: *const Self::Element) -> Self {
        (Vec4f::load_ptr(addr), Vec4f::load_ptr(addr.add(4))).into()
    }

    #[inline]
    unsafe fn store_ptr(self, addr: *mut Self::Element) {
        self.0.store_ptr(addr);
        self.1.store_ptr(addr.add(4));
    }

    #[inline]
    fn sum(self) -> Self::Element {
        (self.low() + self.high()).sum()
    }
}
