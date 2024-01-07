use crate::Vec4f;
use derive_more::{Add, Div, Mul, Neg, Sub};

pub type Underlying = (Vec4f, Vec4f);

#[repr(C)]
#[derive(Copy, Clone, Add, Sub, Mul, Div, Neg, Default, PartialEq)]
#[mul(forward)]
#[div(forward)]
pub struct Vec8fBase(pub Vec4f, pub Vec4f);

impl Vec8fBase {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(v0: f32, v1: f32, v2: f32, v3: f32, v4: f32, v5: f32, v6: f32, v7: f32) -> Self {
        Self(Vec4f::new(v0, v1, v2, v3), Vec4f::new(v4, v5, v6, v7))
    }

    #[inline(always)]
    pub fn join(low: Vec4f, high: Vec4f) -> Self {
        Self(low, high)
    }

    #[inline(always)]
    pub fn broadcast(value: f32) -> Self {
        let half = Vec4f::broadcast(value);
        Self(half, half)
    }

    #[inline(always)]
    pub fn low(self) -> Vec4f {
        self.0
    }

    #[inline(always)]
    pub fn high(self) -> Vec4f {
        self.1
    }

    #[inline(always)]
    pub unsafe fn load_ptr(addr: *const f32) -> Self {
        Self(Vec4f::load_ptr(addr), Vec4f::load_ptr(addr.add(4)))
    }

    #[inline(always)]
    pub unsafe fn load_ptr_aligned(addr: *const f32) -> Self {
        Self(
            Vec4f::load_ptr_aligned(addr),
            Vec4f::load_ptr_aligned(addr.add(4)),
        )
    }

    #[inline(always)]
    pub unsafe fn store_ptr(self, addr: *mut f32) {
        self.0.store_ptr(addr);
        self.1.store_ptr(addr.add(4));
    }

    #[inline(always)]
    pub unsafe fn store_ptr_aligned(self, addr: *mut f32) {
        self.0.store_ptr_aligned(addr);
        self.1.store_ptr_aligned(addr.add(4));
    }

    #[inline(always)]
    pub unsafe fn store_ptr_non_temporal(self, addr: *mut f32) {
        self.0.store_ptr_non_temporal(addr);
        self.1.store_ptr_non_temporal(addr.add(4));
    }
}
