use crate::{common::SIMDBase, intrinsics::*, macros::vec_impl_binary_op};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Vec4f(__m128);

impl Vec4f {
    /// Loads vector from an aligned array pointed by `addr`.
    ///
    /// # Safety
    /// Like [`load`], requires `addr` to be valid.
    /// Unlike [`load`], requires `addr` to be divisible by `16`, i.e. to be a `16`-bytes aligned address.
    ///
    /// [`load`]: Self::load
    ///
    /// # Examples
    /// ```
    /// # use vrl::prelude::*;
    /// #[repr(align(16))]
    /// struct AlignedArray([f32; 4]);
    ///
    /// let array = AlignedArray([42.0; 4]);
    /// let vec = unsafe { Vec4f::load_ptr_aligned(array.0.as_ptr()) };
    /// assert_eq!(vec, Vec4f::broadcast(42.0));
    /// ```
    /// In the following example `zeros` is aligned 2-bytes aligned. Therefore
    /// `zeros.as_ptr().byte_add(1)` is an odd address and hence not divisible by `16`.
    /// ```should_panic
    /// # use vrl::prelude::*;
    /// let zeros = unsafe { std::mem::zeroed::<[u16; 10]>() };
    /// unsafe { Vec4f::load_ptr_aligned(zeros.as_ptr().byte_add(1) as *const f32) };
    /// ```
    #[inline]
    pub unsafe fn load_ptr_aligned(addr: *const f32) -> Self {
        _mm_load_ps(addr).into()
    }

    /// Stores vector into aligned array at given address.
    ///
    /// # Safety
    /// Like [`store_ptr`], requires `addr` to be valid.
    /// Unlike [`store_ptr`], requires `addr` to be divisible by `16`, i.e. to be a 16-bytes aligned address.
    ///
    /// [`store_ptr`]: Self::store_ptr
    #[inline]
    pub unsafe fn store_ptr_aligned(self, addr: *mut f32) {
        _mm_store_ps(addr, self.0);
    }

    /// Stores vector into aligned array at given address in uncached memory (non-temporal store).
    /// This may be more efficient than [`store_ptr_aligned`] if it is unlikely that stored data will
    /// stay in cache until it is read again, for instance, when storing large blocks of memory.
    ///
    /// # Safety
    /// Has same requirements as [`store_ptr_aligned`]: `addr` must be valid and
    /// divisible by `16`, i.e. to be a 16-bytes aligned address.
    ///
    /// [`store_ptr_aligned`]: Self::store_ptr_aligned
    #[inline]
    pub unsafe fn store_ptr_non_temporal(self, addr: *mut f32) {
        _mm_stream_ps(addr, self.0)
    }
}

impl super::Vec4fBase for Vec4f {
    #[inline]
    fn new(v0: f32, v1: f32, v2: f32, v3: f32) -> Self {
        unsafe { _mm_setr_ps(v0, v1, v2, v3) }.into()
    }
}

impl SIMDBase<4> for Vec4f {
    type Underlying = __m128;
    type Element = f32;

    #[inline]
    fn broadcast(value: f32) -> Self {
        unsafe { _mm_set1_ps(value) }.into()
    }

    #[inline]
    unsafe fn load_ptr(addr: *const f32) -> Self {
        _mm_loadu_ps(addr).into()
    }

    #[inline]
    unsafe fn store_ptr(self, addr: *mut f32) {
        _mm_storeu_ps(addr, self.0);
    }

    #[inline]
    fn sum(self) -> f32 {
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

impl From<__m128> for Vec4f {
    #[inline]
    fn from(value: __m128) -> Self {
        Self(value)
    }
}

impl From<Vec4f> for __m128 {
    #[inline]
    fn from(value: Vec4f) -> Self {
        value.0
    }
}

impl Default for Vec4f {
    #[inline]
    fn default() -> Self {
        unsafe { _mm_setzero_ps() }.into()
    }
}

impl Neg for Vec4f {
    type Output = Self;

    fn neg(self) -> Self::Output {
        unsafe { _mm_xor_ps(self.0, _mm_set1_ps(-0.0)) }.into()
    }
}

impl PartialEq for Vec4f {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp_result = _mm_cmpeq_ps(self.0, other.0);
            _mm_movemask_ps(cmp_result) == 0x0F
        }
    }
}

vec_impl_binary_op!(Vec4f, Add, add, _mm_add_ps);
vec_impl_binary_op!(Vec4f, Sub, sub, _mm_sub_ps);
vec_impl_binary_op!(Vec4f, Mul, mul, _mm_mul_ps);
vec_impl_binary_op!(Vec4f, Div, div, _mm_div_ps);
