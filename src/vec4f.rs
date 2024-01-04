use std::{
    fmt::Debug,
    mem::MaybeUninit,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{common::SIMDVector, intrinsics::*, macros::vec_overload_operator};

/// Represents a packed vector of 4 single-precision floating-point values. [`__m128`] wrapper.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vec4f {
    xmm: __m128,
}

impl Vec4f {
    /// Initializes elements of returned vector with given values.
    ///
    /// # Example
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
    /// # Example
    /// ```
    /// # use vrl::Vec4f;
    /// let array = [42.0; 4];
    /// let vec = unsafe { Vec4f::load_ptr(&array) };
    /// ```
    #[inline(always)]
    pub unsafe fn load_ptr(addr: *const [f32; 4]) -> Self {
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
    /// let vec = unsafe { Vec4f::load_ptr_aligned(&array.0) };
    /// assert_eq!(vec, Vec4f::broadcast(42.0));
    /// ```
    /// In the following example `zeros` is aligned 2-bytes aligned. Therefore
    /// `zeros.as_ptr().byte_add(1)` is an odd address and hence not divisible by `16`.
    /// ```should_panic
    /// # use vrl::Vec4f;
    /// let zeros = unsafe { std::mem::zeroed::<[u16; 10]>() };
    /// unsafe { Vec4f::load_ptr_aligned(zeros.as_ptr().byte_add(1) as *const [f32; 4]) };
    /// ```
    #[inline(always)]
    pub unsafe fn load_ptr_aligned(addr: *const [f32; 4]) -> Self {
        _mm_load_ps(addr as *const f32).into()
    }

    /// Loads values of returned vector from given data.
    ///
    /// # Exmaple
    /// ```
    /// # use vrl::Vec4f;
    /// assert_eq!(
    ///     Vec4f::new(1.0, 2.0, 3.0, 4.0),
    ///     Vec4f::load(&[1.0, 2.0, 3.0, 4.0])
    /// );
    /// ```
    #[inline(always)]
    pub fn load(data: &[f32; 4]) -> Self {
        unsafe { Self::load_ptr(data) }
    }

    /// Checks that data contains exactly four elements and loads them into vector.
    ///
    /// # Panics
    /// Panics if `data.len()` isn't `4`.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec4f;
    /// assert_eq!(
    ///     Vec4f::load_checked(&[1.0, 2.0, 3.0, 4.0]),
    ///     Vec4f::new(1.0, 2.0, 3.0, 4.0)
    /// );
    /// ```
    /// ```should_panic
    /// # use vrl::Vec4f;
    /// Vec4f::load_checked(&[1.0, 2.0, 3.0]);
    /// ```
    /// ```should_panic
    /// # use vrl::Vec4f;
    /// Vec4f::load_checked(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// ```
    #[inline(always)]
    pub fn load_checked(data: &[f32]) -> Self {
        Self::load(
            data.try_into()
                .expect("data must contain exactly 4 elements"),
        )
    }

    /// Loads the first four elements of `data` into vector.
    ///
    /// # Panics
    /// Panics if `data` contains less than four elements.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::Vec4f;
    /// assert_eq!(
    ///     Vec4f::load_prefix(&[1.0, 2.0, 3.0, 4.0, 5.0]),
    ///     Vec4f::new(1.0, 2.0, 3.0, 4.0)
    /// );
    /// ```
    ///
    /// ```should_panic
    /// # use vrl::Vec4f;
    /// Vec4f::load_prefix(&[1.0, 2.0, 3.0]);
    /// ```
    #[inline(always)]
    pub fn load_prefix(data: &[f32]) -> Self {
        if data.len() < 4 {
            panic!("data must contain at least 4 elements");
        }
        unsafe { Self::load_ptr(data.as_ptr() as *const [f32; 4]) }
    }

    /// Loads first 4 elements of `data` if available otherwise initializes first elements of
    /// returned vector with values of `data` and rest elements with zeros.
    ///
    /// # Example
    /// ```
    /// # use vrl::Vec4f;
    /// let values = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// assert_eq!(
    ///     Vec4f::load_partial(&values),
    ///     Vec4f::from(&values[..4].try_into().unwrap())
    /// );
    /// assert_eq!(
    ///     Vec4f::load_partial(&values[..2]),
    ///     Vec4f::new(1.0, 2.0, 0.0, 0.0)  // note zeros here
    /// );
    /// ```
    #[inline]
    pub fn load_partial(data: &[f32]) -> Self {
        match data.len() {
            4.. => unsafe { Self::load_ptr(data.as_ptr() as *const [f32; 4]) },
            3 => Self::new(data[0], data[1], data[2], 0.0),
            2 => Self::new(data[0], data[1], 0.0, 0.0),
            1 => Self::new(data[0], 0.0, 0.0, 0.0),
            0 => Self::default(),
        }
    }

    /// Returns vector with all its elements initialized with a given `value`, i.e. broadcasts
    /// `value` to all elements of returned vector.
    ///
    /// # Example
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
    pub unsafe fn store_ptr(&self, addr: *mut [f32; 4]) {
        _mm_storeu_ps(addr as *mut f32, self.xmm)
    }

    /// Stores vector into aligned array at given address.
    ///
    /// # Safety
    /// Like [`store_ptr`], requires `addr` to be valid.
    /// Unlike [`store_ptr`], requires `addr` to be divisible by `16`, i.e. to be a 16-bytes aligned address.
    ///
    /// [`store_ptr`]: Self::store_ptr
    #[inline(always)]
    pub unsafe fn store_ptr_aligned(&self, addr: *mut [f32; 4]) {
        _mm_store_ps(addr as *mut f32, self.xmm)
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
    #[inline(always)]
    pub unsafe fn store_ptr_non_temporal(&self, addr: *mut [f32; 4]) {
        _mm_stream_ps(addr as *mut f32, self.xmm)
    }

    /// Stores vector into given `array`.
    #[inline(always)]
    pub fn store(&self, array: &mut [f32; 4]) {
        unsafe { self.store_ptr(array) }
    }

    /// Checkes that `slice` contains exactly four elements and store elements of vector there.
    ///
    /// # Panics
    /// Panics if `slice.len()` isn't `4`.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec4f;
    /// let mut data = [-1.0; 4];
    /// Vec4f::default().store_checked(&mut data);
    /// assert_eq!(data, [0.0; 4]);
    /// ```
    /// ```should_panic
    /// # use vrl::Vec4f;
    /// let mut data = [-1.0; 3];
    /// Vec4f::default().store_checked(&mut data);
    /// ```
    /// ```should_panic
    /// # use vrl::Vec4f;
    /// let mut data = [-1.0; 5];
    /// Vec4f::default().store_checked(&mut data);
    /// ```
    pub fn store_checked(&self, slice: &mut [f32]) {
        self.store(
            slice
                .try_into()
                .expect("slice must contain at least 4 elements"),
        )
    }

    /// Stores elements of vector into the first four elements of `slice`.
    ///
    /// # Panics
    /// Panics if `slice` contains less then four elements.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::Vec4f;
    /// let mut data = [-1.0; 5];
    /// Vec4f::broadcast(2.0).store_prefix(&mut data);
    /// assert_eq!(data, [2.0, 2.0, 2.0, 2.0, -1.0]);
    /// ```
    /// ```should_panic
    /// # use vrl::Vec4f;
    /// let mut data = [-1.0; 3];
    /// Vec4f::default().store_prefix(&mut data);
    /// ```
    #[inline(always)]
    pub fn store_prefix(&self, slice: &mut [f32]) {
        if slice.len() < 4 {
            panic!("slice.len() must at least 4");
        }
        unsafe { self.store_ptr(slice.as_ptr() as *mut [f32; 4]) };
    }
    /// Stores `min(4, slice.len())` elements of vector into prefix of `slice`.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::Vec4f;
    /// let mut data = [0.0; 3];
    /// Vec4f::broadcast(1.0).store_partial(&mut data);
    /// assert_eq!(data, [1.0; 3]);
    /// ```
    /// ```
    /// # use vrl::Vec4f;
    /// let mut data = [0.0; 5];
    /// Vec4f::broadcast(1.0).store_partial(&mut data);
    /// assert_eq!(data, [1.0, 1.0, 1.0, 1.0, 0.0]);  // note last zero
    /// ```
    pub fn store_partial(&self, slice: &mut [f32]) {
        match slice.len() {
            4.. => unsafe { self.store_ptr(slice.as_mut_ptr() as *mut [f32; 4]) },
            _ => slice.copy_from_slice(&<[f32; 4]>::from(self)[..slice.len()]),
        }
    }

    /// Calculates the sum of all elements of vector.
    ///
    /// # Exmaple
    /// ```
    /// # use vrl::Vec4f;
    /// assert_eq!(Vec4f::new(1.0, 2.0, 3.0, 4.0).sum(), 10.0);
    /// ```
    #[inline(always)]
    pub fn sum(self) -> f32 {
        // Acoording to Agner Fog, using `hadd` is inefficient.
        // src: https://github.com/vectorclass/version2/blob/master/vectorf128.h#L1043
        // TODO: benchmark this implementation and `hadd`-based one
        unsafe {
            let t1 = _mm_movehl_ps(self.xmm, self.xmm);
            let t2 = _mm_add_ps(self.xmm, t1);
            let t3 = _mm_shuffle_ps(t2, t2, 1);
            let t4 = _mm_add_ss(t2, t3);
            _mm_cvtss_f32(t4)
        }
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
    /// # Example
    /// ```
    /// # use vrl::Vec4f;
    /// assert_eq!(Vec4f::default(), Vec4f::broadcast(0.0));
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

vec_overload_operator!(Vec4f, Add, add, _mm_add_ps, sse);
vec_overload_operator!(Vec4f, Sub, sub, _mm_sub_ps, sse);
vec_overload_operator!(Vec4f, Mul, mul, _mm_mul_ps, sse);
vec_overload_operator!(Vec4f, Div, div, _mm_div_ps, sse);

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
        Self::load(value)
    }
}

impl From<[f32; 4]> for Vec4f {
    #[inline(always)]
    fn from(value: [f32; 4]) -> Self {
        (&value).into()
    }
}

impl From<Vec4f> for [f32; 4] {
    #[inline(always)]
    fn from(value: Vec4f) -> Self {
        let mut result = MaybeUninit::<Self>::uninit();
        unsafe {
            value.store_ptr(result.as_mut_ptr());
            result.assume_init()
        }
    }
}

impl From<&Vec4f> for [f32; 4] {
    #[inline(always)]
    fn from(value: &Vec4f) -> Self {
        unsafe { *(value as *const Vec4f as *const [f32; 4]) }
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
    /// assert_ne!(a, Vec4f::default());
    /// ```
    ///
    /// ```
    /// # use vrl::Vec4f;
    /// let a = Vec4f::broadcast(f32::NAN);
    /// assert_ne!(a, a);
    /// ```
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp_result = _mm_cmpeq_ps(self.xmm, other.xmm);
            _mm_movemask_ps(cmp_result) == 0x0F
        }
    }
}

impl Debug for Vec4f {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_tuple = f.debug_tuple("Vec4f");
        for value in <[f32; 4]>::from(self) {
            debug_tuple.field(&value);
        }
        debug_tuple.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::Vec4f;

    #[test]
    #[inline(never)] // in order to find the function in disassembled binary
    fn it_works() {
        let a = Vec4f::broadcast(1.0);
        assert_eq!(<[f32; 4]>::from(a), [1.0; 4]);
        assert_eq!(a, [1.0; 4].into());

        let b = 2.0 * a;
        assert_ne!(a, b);

        let mut c = b / 2.0;
        assert_eq!(a, c);

        c += Vec4f::from(&[1.0, 0.0, 2.0, 0.0]);
        let d = -c;

        const EXPECTED_D: [f32; 4] = [-2.0, -1.0, -3.0, -1.0];
        assert_eq!(d, EXPECTED_D.into());
        assert_eq!(<[f32; 4]>::from(d), EXPECTED_D);
    }
}
