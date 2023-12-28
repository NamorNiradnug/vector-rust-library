use core::arch::x86_64::*;
use std::{
    fmt::Debug,
    mem::MaybeUninit,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Represents a packed vector of 8 single-precision floating-point values.
/// [`__m256`] wrapper.
#[derive(Clone, Copy, Debug)]
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
    /// `addr` must not be null.
    #[inline(always)]
    pub unsafe fn load(addr: *const f32) -> Self {
        Self {
            ymm: _mm256_loadu_ps(addr),
        }
    }

    /// Returns vector with all its elements initialized with a given `value`, i.e. broadcasts
    /// `value` to all elements of returned vector.
    /// ```non_run
    /// use vector_rust_library::vec256::Vec256f;
    /// assert_eq!(
    ///     Vec256f::broadcast(42.0),
    ///     Vec256f::new(42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0)
    /// );
    /// ```
    #[inline(always)]
    pub fn broadcast(value: f32) -> Self {
        Self {
            ymm: unsafe { _mm256_set1_ps(value) },
        }
    }

    /// Loads vector from aligned array pointed by `addr`.
    ///
    /// # Safety
    /// Like [`load`], requires `addr` to be not null.
    /// Unlike [`load`], requires `addr` to be divisible by `32`, i.e. to be a `32`-byte aligned address.
    ///
    /// [`load`]: Self::load
    #[inline(always)]
    pub unsafe fn load_aligned(addr: *const [f32; 8]) -> Self {
        Self {
            ymm: _mm256_loadu_ps(addr as *const f32),
        }
    }

    /// Stores vector into array at given address.
    ///
    /// # Safety
    /// `addr` must not be null pointer.
    #[inline(always)]
    pub unsafe fn store(&self, addr: *mut [f32; 8]) {
        _mm256_storeu_ps(addr as *mut f32, self.ymm)
    }

    /// Stores vector into aligned array at given address.
    ///
    /// # Safety
    /// Like [`store`], requires `addr` to be not null.
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

impl Default for Vec256f {
    /// Initializes all elements of returned vector with zero.
    #[inline(always)]
    fn default() -> Self {
        Self {
            ymm: unsafe { _mm256_setzero_ps() },
        }
    }
}

impl Neg for Vec256f {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self {
            ymm: unsafe { _mm256_xor_ps(self.ymm, _mm256_set1_ps(-0f32)) },
        }
    }
}

impl<T: Into<Self>> Add<T> for Vec256f {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: T) -> Self::Output {
        Self {
            ymm: unsafe { _mm256_add_ps(self.ymm, rhs.into().ymm) },
        }
    }
}

impl<T> AddAssign<T> for Vec256f
where
    Self: Add<T, Output = Self>,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}

impl<T: Into<Self>> Sub<T> for Vec256f {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: T) -> Self::Output {
        Self {
            ymm: unsafe { _mm256_sub_ps(self.ymm, rhs.into().ymm) },
        }
    }
}

impl<T> SubAssign<T> for Vec256f
where
    Self: Sub<T, Output = Self>,
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs
    }
}

impl<T: Into<Vec256f>> Mul<T> for Vec256f {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: T) -> Self::Output {
        Self {
            ymm: unsafe { _mm256_mul_ps(self.ymm, rhs.into().ymm) },
        }
    }
}

impl<T> MulAssign<T> for Vec256f
where
    Self: Mul<T, Output = Self>,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T: Into<Vec256f>> Div<T> for Vec256f {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: T) -> Self::Output {
        Self {
            ymm: unsafe { _mm256_div_ps(self.ymm, rhs.into().ymm) },
        }
    }
}

impl<T> DivAssign<T> for Vec256f
where
    Self: Div<T, Output = Self>,
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

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
    #[inline(always)]
    fn from(value: &[f32; 8]) -> Self {
        unsafe { Self::load(value.as_ptr()) }
    }
}

impl From<Vec256f> for [f32; 8] {
    #[inline(always)]
    fn from(value: Vec256f) -> Self {
        let mut result = MaybeUninit::<Self>::uninit();
        unsafe {
            value.store(result.as_mut_ptr());
            result.assume_init()
        }
    }
}

impl From<f32> for Vec256f {
    #[inline(always)]
    fn from(value: f32) -> Self {
        Self::broadcast(value)
    }
}

// TODO: Debug, Display

#[test]
fn is_compiles() {
    let a: Vec256f = 1.0.into();
    let b = a * 2.0;
    let mut c = b / 2.0;
    c += Vec256f::from(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]);
    let d = -c;
    println!("{d:?}");
}
