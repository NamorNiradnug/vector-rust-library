use std::{
    fmt::Debug,
    mem::MaybeUninit,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{common::SIMDVector, intrinsics::*, macros::vec_overload_operator, Vec4f};

/// Represents a packed vector of 8 single-precision floating-point values.
/// [`__m256`] wrapper.
#[derive(Clone, Copy)]
pub struct Vec8f {
    ymm: __m256,
}

impl Vec8f {
    /// Initializes elements of returned vector with given values.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec8f;
    /// assert_eq!(
    ///     Vec8f::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
    ///     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].into()
    /// );
    /// ```
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(v0: f32, v1: f32, v2: f32, v3: f32, v4: f32, v5: f32, v6: f32, v7: f32) -> Self {
        unsafe { _mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7) }.into()
    }

    /// Joins two [`Vec4f`] into a single [`Vec8f`]. The first four elements of returned vector are
    /// elements of `a` and the last four elements are elements of `b`.
    ///
    /// See also [`split`](Self::split).
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::{Vec4f, Vec8f};
    /// let a = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// let b = Vec4f::new(5.0, 6.0, 7.0, 8.0);
    /// let joined = Vec8f::join(a, b);
    /// assert_eq!(a, joined.low());
    /// assert_eq!(b, joined.high());
    /// assert_eq!(joined.split(), (a, b));
    /// ```
    #[inline(always)]
    pub fn join(a: Vec4f, b: Vec4f) -> Self {
        unsafe { _mm256_set_m128(b.into(), a.into()) }.into()
    }

    /// Loads vector from array pointer by `addr`.
    /// `addr` is not required to be aligned.
    ///
    /// # Safety
    /// `addr` must be a valid pointer.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec8f;
    /// let array = [42.0; 8];
    /// let vec = unsafe { Vec8f::load(&array) };
    /// ```
    #[inline(always)]
    pub unsafe fn load(addr: *const [f32; 8]) -> Self {
        _mm256_loadu_ps(addr as *const f32).into()
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
    /// # use vrl::Vec8f;
    /// #[repr(align(32))]
    /// struct AlignedArray([f32; 8]);
    ///
    /// let array = AlignedArray([42.0; 8]);
    /// let vec = unsafe { Vec8f::load_aligned(&array.0) };
    /// assert_eq!(vec, 42.0.into());
    /// ```
    ///
    /// ```should_panic
    /// # use vrl::Vec8f;
    /// let zeros = unsafe { std::mem::zeroed::<[u8; 40]>() };
    /// unsafe { Vec8f::load_aligned(zeros.as_ptr().offset(1) as *const [f32; 8]) };
    /// ```
    #[inline(always)]
    pub unsafe fn load_aligned(addr: *const [f32; 8]) -> Self {
        _mm256_load_ps(addr as *const f32).into()
    }

    /// Returns vector with all its elements initialized with a given `value`, i.e. broadcasts
    /// `value` to all elements of returned vector.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec8f;
    /// assert_eq!(
    ///     Vec8f::broadcast(42.0),
    ///     [42.0; 8].into()
    /// );
    /// ```
    #[inline(always)]
    pub fn broadcast(value: f32) -> Self {
        unsafe { _mm256_set1_ps(value) }.into()
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

    /// Stores vector into aligned array at given address in uncached memory (non-temporal store).
    /// This may be more efficient than [`store_aligned`] if it is unlikely that stored data will
    /// stay in cache until it is read again, for instance, when storing large blocks of memory.
    ///
    /// # Safety
    /// Has same requirements as [`store_aligned`]: `addr` must be valid and
    /// divisible by `32`, i.e. to be a 32-bytes aligned address.
    ///
    /// [`store_aligned`]: Self::store_aligned
    #[inline(always)]
    pub unsafe fn store_non_temporal(&self, addr: *mut [f32; 8]) {
        _mm256_stream_ps(addr as *mut f32, self.ymm)
    }

    /// Stores vector into given `array`.
    #[inline(always)]
    pub fn extract(&self, array: &mut [f32; 8]) {
        unsafe { self.store(array) }
    }

    /// Calculates the sum of all elements of vector.
    #[inline(always)]
    pub fn horizontal_add(self) -> f32 {
        todo!()
    }

    /// Returns the first four elements of vector.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::{Vec4f, Vec8f};
    /// let vec8 = Vec8f::new(1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0);
    /// assert_eq!(vec8.low(), Vec4f::broadcast(1.0));
    /// ```
    #[inline(always)]
    pub fn low(self) -> Vec4f {
        unsafe { _mm256_castps256_ps128(self.ymm) }.into()
    }

    /// Returns the last four element of vector.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::{Vec4f, Vec8f};
    /// let vec8 = Vec8f::new(1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0);
    /// assert_eq!(vec8.high(), Vec4f::broadcast(2.0));
    /// ```
    #[inline(always)]
    pub fn high(self) -> Vec4f {
        unsafe { _mm256_extractf128_ps(self.ymm, 1) }.into()
    }

    /// Splits vector into low and high halfs.
    ///
    /// See also [`join`](Self::join).
    ///
    /// # Examples
    /// ```
    /// # use::vrl::{Vec4f, Vec8f};
    /// let vec = Vec8f::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    /// let (low, high) = vec.split();
    /// assert_eq!(low, vec.low());
    /// assert_eq!(high, vec.high());
    /// assert_eq!(Vec8f::join(low, high), vec);
    /// ```
    #[inline(always)]
    pub fn split(self) -> (Vec4f, Vec4f) {
        (self.low(), self.high())
    }
}

impl SIMDVector for Vec8f {
    type Underlying = __m256;
    type Element = f32;
    const ELEMENTS: usize = 8;
}

impl Default for Vec8f {
    /// Initializes all elements of returned vector with zero.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec8f;
    /// assert_eq!(Vec8f::default(), 0.0.into());
    /// ```
    #[inline(always)]
    fn default() -> Self {
        unsafe { _mm256_setzero_ps() }.into()
    }
}

impl Neg for Vec8f {
    type Output = Self;

    /// Flips sign bit of each element including non-finite ones.
    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { _mm256_xor_ps(self.ymm, _mm256_set1_ps(-0f32)) }.into()
    }
}

vec_overload_operator!(Vec8f, Add, add, _mm256_add_ps);
vec_overload_operator!(Vec8f, Sub, sub, _mm256_sub_ps);
vec_overload_operator!(Vec8f, Mul, mul, _mm256_mul_ps);
vec_overload_operator!(Vec8f, Div, div, _mm256_div_ps);

impl From<__m256> for Vec8f {
    /// Wraps given `value` into [`Vec8f`].
    #[inline(always)]
    fn from(value: __m256) -> Self {
        Self { ymm: value }
    }
}

impl From<Vec8f> for __m256 {
    /// Unwraps given vector into raw [`__m256`] value.
    #[inline(always)]
    fn from(value: Vec8f) -> Self {
        value.ymm
    }
}

impl From<&[f32; 8]> for Vec8f {
    /// Does same as [`load`](Self::load).
    #[inline(always)]
    fn from(value: &[f32; 8]) -> Self {
        unsafe { Self::load(value) }
    }
}

impl From<[f32; 8]> for Vec8f {
    #[inline(always)]
    fn from(value: [f32; 8]) -> Self {
        (&value).into()
    }
}

impl From<&Vec8f> for [f32; 8] {
    #[inline(always)]
    fn from(value: &Vec8f) -> Self {
        let mut result = MaybeUninit::<Self>::uninit();
        unsafe {
            value.store(result.as_mut_ptr());
            result.assume_init()
        }
    }
}

impl From<Vec8f> for [f32; 8] {
    fn from(value: Vec8f) -> Self {
        (&value).into()
    }
}

impl From<f32> for Vec8f {
    /// Does same as [`broadcast`](Self::broadcast).
    #[inline(always)]
    fn from(value: f32) -> Self {
        Self::broadcast(value)
    }
}

impl PartialEq for Vec8f {
    /// Checks whether all elements of vectors are equal.
    ///
    /// __Note__: when [`NaN`](`f32::NAN`) is an element of one of the operands the result is always `false`.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec8f;
    /// let a = Vec8f::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    /// assert_eq!(a, a);
    /// ```
    ///
    /// ```
    /// # use vrl::Vec8f;
    /// let a = Vec8f::broadcast(f32::NAN);
    /// assert_ne!(a, a);
    /// ```
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp_result = _mm256_cmp_ps::<0>(self.ymm, other.ymm);
            _mm256_testz_ps(cmp_result, cmp_result) == 0
        }
    }
}

impl Debug for Vec8f {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_tuple = f.debug_tuple("Vec8f");
        for value in Into::<[f32; 8]>::into(self) {
            debug_tuple.field(&value);
        }
        debug_tuple.finish()
    }
}

#[test]
#[inline(never)] // in order to find the function in disassembled binary
fn it_works() {
    let a: Vec8f = 1.0.into();
    assert_eq!(Into::<[f32; 8]>::into(a), [1.0; 8]);
    assert_eq!(a, [1.0; 8].into());

    let b = 2.0 * a;
    assert_ne!(a, b);

    let mut c = b / 2.0;
    assert_eq!(a, c);

    c += Vec8f::from(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]);
    let d = -c;

    const EXPECTED_D: [f32; 8] = [-2.0, -1.0, -3.0, -1.0, -4.0, -1.0, -5.0, -1.0];
    assert_eq!(d, EXPECTED_D.into());
    assert_eq!(Into::<[f32; 8]>::into(d), EXPECTED_D);
}
