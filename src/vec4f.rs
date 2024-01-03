use std::{
    fmt::Debug,
    mem::MaybeUninit,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{common::SIMDVector, intrinsics::*, macros::vec_overload_operator};

/// Represents a packed vector of 4 single-precision floating-point values. [`__m128`] wrapper.
#[derive(Clone, Copy)]
pub struct Vec4f {
    xmm: __m128,
}

impl Vec4f {
    /// Initializes elements of returned vector with given values.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec4f;
    /// assert_eq!(
    ///     Vec4f::new(1.0, 2.0, 3.0, 4.0),
    ///     [1.0, 2.0, 3.0, 4.0].into()
    /// );
    /// ```
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(v0: f32, v1: f32, v2: f32, v3: f32) -> Self {
        unsafe { _mm_setr_ps(v0, v1, v2, v3) }.into()
    }

    /// Loads vector from array pointer by `addr`.
    /// `addr` is not required to be aligned.
    ///
    /// # Safety
    /// `addr` must be a valid pointer.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec4f;
    /// let array = [42.0; 4];
    /// let vec = unsafe { Vec4f::load(&array) };
    /// ```
    #[inline(always)]
    pub unsafe fn load(addr: *const [f32; 4]) -> Self {
        _mm_loadu_ps(addr as *const f32).into()
    }

    /// Loads vector from aligned array pointed by `addr`.
    ///
    /// # Safety
    /// Like [`load`], requires `addr` to be valid.
    /// Unlike [`load`], requires `addr` to be divisible by `16`, i.e. to be a `16`-bytes aligned address.
    ///
    /// [`load`]: Self::load
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec4f;
    /// #[repr(align(16))]
    /// struct AlignedArray([f32; 4]);
    ///
    /// let array = AlignedArray([42.0; 4]);
    /// let vec = unsafe { Vec4f::load_aligned(&array.0) };
    /// assert_eq!(vec, 42.0.into());
    /// ```
    ///
    /// ```should_panic
    /// # use vrl::Vec4f;
    /// let zeros = unsafe { std::mem::zeroed::<[u8; 20]>() };
    /// unsafe { Vec4f::load_aligned(zeros.as_ptr().offset(1) as *const [f32; 4]) };
    /// ```
    #[inline(always)]
    pub unsafe fn load_aligned(addr: *const [f32; 4]) -> Self {
        _mm_load_ps(addr as *const f32).into()
    }

    /// Returns vector with all its elements initialized with a given `value`, i.e. broadcasts
    /// `value` to all elements of returned vector.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec4f;
    /// assert_eq!(
    ///     Vec4f::broadcast(42.0),
    ///     [42.0; 4].into()
    /// );
    /// ```
    #[inline(always)]
    pub fn broadcast(value: f32) -> Self {
        unsafe { _mm_set1_ps(value) }.into()
    }

    /// Stores vector into array at given address.
    ///
    /// # Safety
    /// `addr` must be a valid pointer.
    #[inline(always)]
    pub unsafe fn store(&self, addr: *mut [f32; 4]) {
        _mm_storeu_ps(addr as *mut f32, self.xmm)
    }

    /// Stores vector into aligned array at given address.
    ///
    /// # Safety
    /// Like [`store`], requires `addr` to be valid.
    /// Unlike [`store`], requires `addr` to be divisible by `16`, i.e. to be a 16-bytes aligned address.
    ///
    /// [`store`]: Self::store
    #[inline(always)]
    pub unsafe fn store_aligned(&self, addr: *mut [f32; 4]) {
        _mm_store_ps(addr as *mut f32, self.xmm)
    }

    /// Stores vector into aligned array at given address in uncached memory (non-temporal store).
    /// This may be more efficient than [`store_aligned`] if it is unlikely that stored data will
    /// stay in cache until it is read again, for instance, when storing large blocks of memory.
    ///
    /// # Safety
    /// Has same requirements as [`store_aligned`]: `addr` must be valid and
    /// divisible by `16`, i.e. to be a 16-bytes aligned address.
    ///
    /// [`store_aligned`]: Self::store_aligned
    #[inline(always)]
    pub unsafe fn store_non_temporal(&self, addr: *mut [f32; 4]) {
        _mm_stream_ps(addr as *mut f32, self.xmm)
    }

    /// Stores vector into given `array`.
    #[inline(always)]
    pub fn extract(&self, array: &mut [f32; 4]) {
        unsafe { self.store(array) }
    }

    /// Calculates the sum of all elements of vector.
    #[inline(always)]
    pub fn horizontal_add(self) -> f32 {
        todo!()
    }
}

impl SIMDVector for Vec4f {
    type Underlying = __m128;
    type Element = f32;
    const ELEMENTS: usize = 4;
}

impl Default for Vec4f {
    /// Initializes all elements of returned vector with zero.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec4f;
    /// assert_eq!(Vec4f::default(), 0.0.into());
    /// ```
    #[inline(always)]
    fn default() -> Self {
        unsafe { _mm_setzero_ps() }.into()
    }
}

impl Neg for Vec4f {
    type Output = Self;

    /// Flips sign bit of each element including non-finite ones.
    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { _mm_xor_ps(self.xmm, _mm_set1_ps(-0f32)) }.into()
    }
}

vec_overload_operator!(Vec4f, Add, add, _mm_add_ps);
vec_overload_operator!(Vec4f, Sub, sub, _mm_sub_ps);
vec_overload_operator!(Vec4f, Mul, mul, _mm_mul_ps);
vec_overload_operator!(Vec4f, Div, div, _mm_div_ps);

impl From<__m128> for Vec4f {
    /// Wraps given `value` into [`Vec4f`].
    #[inline(always)]
    fn from(value: __m128) -> Self {
        Self { xmm: value }
    }
}

impl From<Vec4f> for __m128 {
    /// Unwraps given vector into raw [`__m128`] value.
    #[inline(always)]
    fn from(value: Vec4f) -> Self {
        value.xmm
    }
}

impl From<&[f32; 4]> for Vec4f {
    /// Does same as [`load`](Self::load).
    #[inline(always)]
    fn from(value: &[f32; 4]) -> Self {
        unsafe { Self::load(value) }
    }
}

impl From<[f32; 4]> for Vec4f {
    #[inline(always)]
    fn from(value: [f32; 4]) -> Self {
        (&value).into()
    }
}

impl From<&Vec4f> for [f32; 4] {
    #[inline(always)]
    fn from(value: &Vec4f) -> Self {
        let mut result = MaybeUninit::<Self>::uninit();
        unsafe {
            value.store(result.as_mut_ptr());
            result.assume_init()
        }
    }
}

impl From<Vec4f> for [f32; 4] {
    fn from(value: Vec4f) -> Self {
        (&value).into()
    }
}

impl From<f32> for Vec4f {
    /// Does same as [`broadcast`](Self::broadcast).
    #[inline(always)]
    fn from(value: f32) -> Self {
        Self::broadcast(value)
    }
}

impl PartialEq for Vec4f {
    /// Checks whether all elements of vectors are equal.
    ///
    /// __Note__: when [`NaN`](`f32::NAN`) is an element of one of the operands the result is always `false`.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec4f;
    /// let a = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(a, a);
    /// ```
    ///
    /// ```
    /// # use vrl::Vec4f;
    /// let a = Vec4f::broadcast(f32::NAN);
    /// assert_ne!(a, a);
    /// ```
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp_result = _mm_cmp_ps::<0>(self.xmm, other.xmm);
            _mm_testz_ps(cmp_result, cmp_result) == 0
        }
    }
}

impl Debug for Vec4f {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_tuple = f.debug_tuple("Vec4f");
        for value in Into::<[f32; 4]>::into(self) {
            debug_tuple.field(&value);
        }
        debug_tuple.finish()
    }
}

#[test]
#[inline(never)] // in order to find the function in disassembled binary
fn it_works() {
    let a: Vec4f = 1.0.into();
    assert_eq!(Into::<[f32; 4]>::into(a), [1.0; 4]);
    assert_eq!(a, [1.0; 4].into());

    let b = 2.0 * a;
    assert_ne!(a, b);

    let mut c = b / 2.0;
    assert_eq!(a, c);

    c += Vec4f::from(&[1.0, 0.0, 2.0, 0.0]);
    let d = -c;

    const EXPECTED_D: [f32; 4] = [-2.0, -1.0, -3.0, -1.0];
    assert_eq!(d, EXPECTED_D.into());
    assert_eq!(Into::<[f32; 4]>::into(d), EXPECTED_D);
}
