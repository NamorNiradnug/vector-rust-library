use std::{
    mem::MaybeUninit,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use crate::{
    common::{SIMDBase, SIMDRound, SIMDVector},
    macros::{
        vec_impl_generic_traits, vec_impl_partial_load, vec_impl_partial_store, vec_impl_sum_prod,
        vec_overload_operator,
    },
    vec4f::Vec4f,
};

cfg_if::cfg_if! {
    if #[cfg(avx)] {
        mod avx;
        pub use avx::Vec8f;
    } else if #[cfg(neon)] {
        mod neon;
        pub use neon::Vec8f;
    } else {
        mod fallback;
        pub use fallback::Vec8f;
    }
}

/// Base trait for [`Vec8f`].
pub trait Vec8fBase: SIMDBase<8> + SIMDRound + Copy + Clone {
    /// Initializes elements of returned vector with given values.
    ///
    /// # Example
    /// ```
    /// # use vrl::prelude::*;
    /// assert_eq!(
    ///     Vec8f::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
    ///     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].into()
    /// );
    /// ```
    #[allow(clippy::too_many_arguments)]
    fn new(v0: f32, v1: f32, v2: f32, v3: f32, v4: f32, v5: f32, v6: f32, v7: f32) -> Self;

    /// Joins two [`Vec4f`] into a single [`Vec8f`]. The first four elements of returned vector are
    /// elements of `a` and the last four elements are elements of `b`.
    ///
    /// See also [`split`](Self::split).
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::prelude::*;
    /// let a = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// let b = Vec4f::new(5.0, 6.0, 7.0, 8.0);
    /// let joined = Vec8f::join(a, b);
    /// assert_eq!(a, joined.low());
    /// assert_eq!(b, joined.high());
    /// assert_eq!(joined.split(), (a, b));
    /// ```
    fn join(a: Vec4f, b: Vec4f) -> Self;

    /// Loads vector from aligned array pointed by `addr`.
    ///
    /// # Safety
    /// Like [`load_ptr`], requires `addr` to be valid.
    /// Unlike [`load_ptr`], requires `addr` to be divisible by `32`, i.e. to be a `32`-bytes aligned address.
    ///
    /// [`load_ptr`]: SIMDBase::load_ptr
    ///
    /// # Examples
    /// ```
    /// # use vrl::prelude::*;
    /// #[repr(align(32))]
    /// struct AlignedArray([f32; 8]);
    ///
    /// let array = AlignedArray([42.0; 8]);
    /// let vec = unsafe { Vec8f::load_ptr_aligned(array.0.as_ptr()) };
    /// assert_eq!(vec, Vec8f::broadcast(42.0));
    /// ```
    /// In the following example `zeros` is aligned as `u16`, i.e. 2-bytes aligned.
    /// Therefore `zeros.as_ptr().byte_add(1)` is an odd address and hence not divisible by `32`.
    /// ```should_panic
    /// # use vrl::prelude::*;
    /// let zeros = unsafe { std::mem::zeroed::<[u16; 20]>() };
    /// unsafe { Vec8f::load_ptr_aligned(zeros.as_ptr().byte_add(1) as *const f32) };
    /// ```
    #[cfg(target_feature = "sse")]
    unsafe fn load_ptr_aligned(addr: *const f32) -> Self;

    /// Stores vector into aligned array at given address.
    ///
    /// # Safety
    /// Like [`store_ptr`], requires `addr` to be valid.
    /// Unlike [`store_ptr`], requires `addr` to be divisible by `32`, i.e. to be a 32-bytes aligned address.
    ///
    /// [`store_ptr`]: SIMDBase::store_ptr
    #[cfg(target_feature = "sse")]
    unsafe fn store_ptr_aligned(self, addr: *mut f32);

    /// Stores vector into aligned array at given address in uncached memory (non-temporal store).
    /// This may be more efficient than [`store_ptr_aligned`] if it is unlikely that stored data will
    /// stay in cache until it is read again, for instance, when storing large blocks of memory.
    ///
    /// # Safety
    /// Has same requirements as [`store_ptr_aligned`]: `addr` must be valid and
    /// divisible by `32`, i.e. to be a 32-bytes aligned address.
    ///
    /// [`store_ptr_aligned`]: Self::store_ptr_aligned
    #[cfg(target_feature = "sse")]
    unsafe fn store_ptr_non_temporal(self, addr: *mut f32);

    /// Returns the first four elements of vector.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::prelude::*;
    /// let vec8 = Vec8f::new(1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0);
    /// assert_eq!(vec8.low(), Vec4f::broadcast(1.0));
    /// ```
    fn low(self) -> Vec4f;

    /// Returns the last four elements of vector.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::prelude::*;
    /// let vec8 = Vec8f::new(1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0);
    /// assert_eq!(vec8.high(), Vec4f::broadcast(2.0));
    /// ```
    fn high(self) -> Vec4f;

    /// Splits vector into low and high halfs.
    ///
    /// See also [`join`](Self::join).
    ///
    /// # Example
    /// ```
    /// # use vrl::prelude::*;
    /// let vec = Vec8f::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    /// let (low, high) = vec.split();
    /// assert_eq!(low, vec.low());
    /// assert_eq!(high, vec.high());
    /// assert_eq!(Vec8f::join(low, high), vec);
    /// ```
    #[inline]
    fn split(self) -> (Vec4f, Vec4f) {
        (self.low(), self.high())
    }
}

impl From<(Vec4f, Vec4f)> for Vec8f {
    /// Does same as [`join`](Self::join).
    #[inline]
    fn from((low, high): (Vec4f, Vec4f)) -> Self {
        Self::join(low, high)
    }
}

impl From<Vec8f> for (Vec4f, Vec4f) {
    /// Does same as [`split`](Vec8f::split).
    #[inline]
    fn from(vec: Vec8f) -> (Vec4f, Vec4f) {
        vec.split()
    }
}

vec_impl_generic_traits!(Vec8f, f32, 8);
vec_impl_partial_load!(Vec8f, f32, Vec4f, 8);
vec_impl_partial_store!(Vec8f, f32, 8);

impl SIMDVector<8> for Vec8f {}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    #[inline(never)] // in order to find the function in disassembled binary
    fn it_works() {
        let a = Vec8f::broadcast(1.0);
        assert_eq!(<[f32; 8]>::from(a), [1.0; 8]);
        assert_eq!(a, [1.0; 8].into());

        let b = 2.0 * a;
        assert_ne!(a, b);

        let mut c = b / 2.0;
        assert_eq!(a, c);

        c += Vec8f::from(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]);
        let d = -c;

        const EXPECTED_D: [f32; 8] = [-2.0, -1.0, -3.0, -1.0, -4.0, -1.0, -5.0, -1.0];
        assert_eq!(d, EXPECTED_D.into());
        assert_eq!(<[f32; 8]>::from(d), EXPECTED_D);
    }

    #[test]
    fn test_load_partial() {
        const VALUES: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        for i in 0..8 {
            let vec_values = <[f32; 8]>::from(Vec8f::load_partial(&VALUES[..i]));
            assert_eq!(vec_values[..i], VALUES[..i]);
            assert!(vec_values[i..].iter().all(|x| *x == 0.0));
        }
        assert_eq!(
            Vec8f::load_partial(VALUES),
            Vec8f::from(&VALUES[..8].try_into().unwrap())
        );
    }

    #[cfg(avx)]
    #[test]
    fn test_m256_conv() {
        use crate::{intrinsics::__m256, prelude::*};
        let vec = Vec8f::join(Vec4f::broadcast(1.0), Vec4f::broadcast(2.0));
        assert_eq!(vec, __m256::from(vec).into());
    }
}
