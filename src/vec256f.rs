use core::arch::x86_64::*;
use std::{
    fmt::Debug,
    mem::MaybeUninit,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{common::SIMDVector, macros::vec_overload_operator};

/// Represents a packed vector of 8 single-precision floating-point values.
/// [`__m256`] wrapper.
#[derive(Clone, Copy)]
pub struct Vec256f {
    ymm: __m256,
}

impl Vec256f {
    /// Initializes elements of returned vector with given values.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(v0: f32, v1: f32, v2: f32, v3: f32, v4: f32, v5: f32, v6: f32, v7: f32) -> Self {
        Self {
            ymm: unsafe { _mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7) },
        }
    }

    /// Loads vector from array pointer by `addr`.
    /// `addr` is not required to be aligned.
    ///
    /// # Safety
    /// `addr` must be a valid pointer.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec256f;
    /// let array = [42.0; 8];
    /// let vec = unsafe { Vec256f::load(&array) };
    /// ```
    #[inline(always)]
    pub unsafe fn load(addr: *const [f32; 8]) -> Self {
        Self {
            ymm: _mm256_loadu_ps(addr as *const f32),
        }
    }

    /// Loads vector from aligned array pointed by `addr`.
    ///
    /// # Safety
    /// Like [`load`], requires `addr` to be valid.
    /// Unlike [`load`], requires `addr` to be divisible by `32`, i.e. to be a `32`-bytes aligned address.
    ///
    /// [`load`]: Self::load
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec256f;
    /// #[repr(align(32))]
    /// struct AlignedArray([f32; 8]);
    ///
    /// let array = AlignedArray([42.0; 8]);
    /// let vec = unsafe { Vec256f::load_aligned(&array.0) };
    /// assert_eq!(vec, 42.0.into());
    /// ```
    #[inline(always)]
    pub unsafe fn load_aligned(addr: *const [f32; 8]) -> Self {
        Self {
            ymm: _mm256_loadu_ps(addr as *const f32),
        }
    }

    /// Returns vector with all its elements initialized with a given `value`, i.e. broadcasts
    /// `value` to all elements of returned vector.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec256f;
    /// assert_eq!(
    ///     Vec256f::broadcast(42.0),
    ///     [42.0; 8].into()
    /// );
    /// ```
    #[inline(always)]
    pub fn broadcast(value: f32) -> Self {
        Self {
            ymm: unsafe { _mm256_set1_ps(value) },
        }
    }

    /// Stores vector into array at given address.
    ///
    /// # Safety
    /// `addr` must be a valid pointer.
    #[inline(always)]
    pub unsafe fn store(&self, addr: *mut [f32; 8]) {
        _mm256_storeu_ps(addr as *mut f32, self.ymm)
    }

    /// Stores vector into aligned array at given address.
    ///
    /// # Safety
    /// Like [`store`], requires `addr` to be valid.
    /// Unlike [`store`], requires `addr` to be divisible by `32`, i.e. to be a 32-bytes aligned address.
    ///
    /// [`store`]: Self::store
    #[inline(always)]
    pub unsafe fn store_aligned(&self, addr: *mut [f32; 8]) {
        _mm256_store_ps(addr as *mut f32, self.ymm)
    }

    /// Stores vector into given `array`.
    #[inline(always)]
    pub fn extract(&self, array: &mut [f32; 8]) {
        unsafe { self.store(array) }
    }
}

impl SIMDVector for Vec256f {
    type Underlying = __m256;
    type Element = f32;
    const ELEMENTS: usize = 8;
}

impl Default for Vec256f {
    /// Initializes all elements of returned vector with zero.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec256f;
    /// assert_eq!(Vec256f::default(), 0.0.into());
    /// ```
    #[inline(always)]
    fn default() -> Self {
        Self {
            ymm: unsafe { _mm256_setzero_ps() },
        }
    }
}

impl Neg for Vec256f {
    type Output = Self;

    /// Flips sign bit of each element including non-finite ones.
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self {
            ymm: unsafe { _mm256_xor_ps(self.ymm, _mm256_set1_ps(-0f32)) },
        }
    }
}

vec_overload_operator!(Vec256f, Add, add, _mm256_add_ps);
vec_overload_operator!(Vec256f, Sub, sub, _mm256_sub_ps);
vec_overload_operator!(Vec256f, Mul, mul, _mm256_mul_ps);
vec_overload_operator!(Vec256f, Div, div, _mm256_div_ps);

impl From<__m256> for Vec256f {
    /// Wraps given `value` into [`Vec256f`].
    #[inline(always)]
    fn from(value: __m256) -> Self {
        Self { ymm: value }
    }
}

impl From<Vec256f> for __m256 {
    /// Unwraps given vector into raw [`__m256`] value.
    #[inline(always)]
    fn from(value: Vec256f) -> Self {
        value.ymm
    }
}

impl From<&[f32; 8]> for Vec256f {
    /// Does same as [`load`](Self::load).
    #[inline(always)]
    fn from(value: &[f32; 8]) -> Self {
        unsafe { Self::load(value) }
    }
}

impl From<[f32; 8]> for Vec256f {
    #[inline(always)]
    fn from(value: [f32; 8]) -> Self {
        (&value).into()
    }
}

impl From<&Vec256f> for [f32; 8] {
    #[inline(always)]
    fn from(value: &Vec256f) -> Self {
        let mut result = MaybeUninit::<Self>::uninit();
        unsafe {
            value.store(result.as_mut_ptr());
            result.assume_init()
        }
    }
}

impl From<Vec256f> for [f32; 8] {
    fn from(value: Vec256f) -> Self {
        (&value).into()
    }
}

impl From<f32> for Vec256f {
    /// Does same as [`broadcast`](Self::broadcast).
    #[inline(always)]
    fn from(value: f32) -> Self {
        Self::broadcast(value)
    }
}

impl PartialEq for Vec256f {
    /// Checks whether all elements of vectors are equal.
    /// Note that comparing with [`NaN`](`f32::NAN`) always evaluates `false`.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec256f;
    /// let a = Vec256f::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    /// assert_eq!(a, a);
    /// ```
    ///
    /// ```
    /// # use vrl::Vec256f;
    /// let a = Vec256f::broadcast(f32::NAN);
    /// assert_ne!(a, a);
    /// ```
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp_result = _mm256_cmp_ps::<0>(self.ymm, other.ymm);
            _mm256_testz_ps(cmp_result, cmp_result) == 0
        }
    }
}

impl Debug for Vec256f {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_tuple = f.debug_tuple("Vec256f");
        for value in Into::<[f32; 8]>::into(self) {
            debug_tuple.field(&value);
        }
        debug_tuple.finish()
    }
}

#[test]
#[inline(never)] // in order to find the function in disassembled binary
fn it_works() {
    let a: Vec256f = 1.0.into();
    assert_eq!(Into::<[f32; 8]>::into(a), [1.0; 8]);
    assert_eq!(a, [1.0; 8].into());

    let b = 2.0 * a;
    assert_ne!(a, b);

    let mut c = b / 2.0;
    assert_eq!(a, c);

    c += Vec256f::from(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]);
    let d = -c;

    const EXPECTED_D: [f32; 8] = [-2.0, -1.0, -3.0, -1.0, -4.0, -1.0, -5.0, -1.0];
    assert_eq!(d, EXPECTED_D.into());
    assert_eq!(Into::<[f32; 8]>::into(d), EXPECTED_D);
}
