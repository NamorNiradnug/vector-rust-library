use std::{
    fmt::Debug,
    mem::MaybeUninit,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use crate::{
    common::SIMDVector,
    intrinsics::_mm_extract_ps,
    macros::{vec_impl_sum_prod, vec_overload_operator},
};

use derive_more::{Add, Div, Mul, Neg, Sub};

cfg_if::cfg_if! {
    if #[cfg(sse)] {
        mod sse;
        use sse::{Vec4fBase, Underlying};
    } else {
        compile_error!("Currently SSE is required for Vec4f");
        mod fallback;
        use fallback::{Vec4fBase, Underlying};
    }
}
/// Represents a packed vector of 4 single-precision floating-point values. [`__m128`] wrapper.
///
/// [`__m128`]: crate::intrinsics::__m128
#[derive(Clone, Copy, Add, Sub, Mul, Div, Neg, PartialEq)]
#[mul(forward)]
#[div(forward)]
#[repr(transparent)]
pub struct Vec4f(Vec4fBase);

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
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn new(v0: f32, v1: f32, v2: f32, v3: f32) -> Self {
        Self(Vec4fBase::new(v0, v1, v2, v3))
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
    /// let vec = unsafe { Vec4f::load_ptr(array.as_ptr()) };
    /// ```
    #[inline]
    pub unsafe fn load_ptr(addr: *const f32) -> Self {
        Self(Vec4fBase::load_ptr(addr))
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
    /// let vec = unsafe { Vec4f::load_ptr_aligned(array.0.as_ptr()) };
    /// assert_eq!(vec, Vec4f::broadcast(42.0));
    /// ```
    /// In the following example `zeros` is aligned 2-bytes aligned. Therefore
    /// `zeros.as_ptr().byte_add(1)` is an odd address and hence not divisible by `16`.
    /// ```should_panic
    /// # use vrl::Vec4f;
    /// let zeros = unsafe { std::mem::zeroed::<[u16; 10]>() };
    /// unsafe { Vec4f::load_ptr_aligned(zeros.as_ptr().byte_add(1) as *const f32) };
    /// ```
    #[inline]
    pub unsafe fn load_ptr_aligned(addr: *const f32) -> Self {
        Self(Vec4fBase::load_ptr_aligned(addr))
    }

    /// Loads values of returned vector from given data.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::Vec4f;
    /// assert_eq!(
    ///     Vec4f::new(1.0, 2.0, 3.0, 4.0),
    ///     Vec4f::load(&[1.0, 2.0, 3.0, 4.0])
    /// );
    /// ```
    #[inline]
    pub fn load(data: &[f32; 4]) -> Self {
        unsafe { Self::load_ptr(data.as_ptr()) }
    }

    /// Checks that `data` contains exactly four elements and loads them into vector.
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
    #[inline]
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
    #[inline]
    pub fn load_prefix(data: &[f32]) -> Self {
        if data.len() < 4 {
            panic!("data must contain at least 4 elements");
        }
        unsafe { Self::load_ptr(data.as_ptr()) }
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
            4.. => unsafe { Self::load_ptr(data.as_ptr()) },
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
    #[inline]
    pub fn broadcast(value: f32) -> Self {
        Self(Vec4fBase::broadcast(value))
    }

    /// Stores vector into array at given address.
    ///
    /// # Safety
    /// `addr` must be a valid pointer.
    #[inline]
    pub unsafe fn store_ptr(&self, addr: *mut f32) {
        self.0.store_ptr(addr);
    }

    /// Stores vector into aligned array at given address.
    ///
    /// # Safety
    /// Like [`store_ptr`], requires `addr` to be valid.
    /// Unlike [`store_ptr`], requires `addr` to be divisible by `16`, i.e. to be a 16-bytes aligned address.
    ///
    /// [`store_ptr`]: Self::store_ptr
    #[inline]
    pub unsafe fn store_ptr_aligned(&self, addr: *mut f32) {
        self.0.store_ptr_aligned(addr);
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
    pub unsafe fn store_ptr_non_temporal(&self, addr: *mut f32) {
        self.0.store_ptr_non_temporal(addr);
    }

    /// Stores vector into given `array`.
    #[inline]
    pub fn store(&self, array: &mut [f32; 4]) {
        unsafe { self.store_ptr(array.as_mut_ptr()) }
    }

    /// Checks that `slice` contains exactly four elements and stores elements of vector there.
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
    #[inline]
    pub fn store_checked(&self, slice: &mut [f32]) {
        self.store(
            slice
                .try_into()
                .expect("slice must contain exactly 4 elements"),
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
    #[inline]
    pub fn store_prefix(&self, slice: &mut [f32]) {
        if slice.len() < 4 {
            panic!("slice must contain at least 4 elements");
        }
        unsafe { self.store_ptr(slice.as_mut_ptr()) };
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
    #[inline]
    pub fn store_partial(&self, slice: &mut [f32]) {
        match slice.len() {
            4.. => unsafe { self.store_ptr(slice.as_mut_ptr()) },
            _ => slice.copy_from_slice(&<[f32; 4]>::from(self)[..slice.len()]),
        }
    }

    /// Calculates the sum of all elements of vector.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::Vec4f;
    /// assert_eq!(Vec4f::new(1.0, 2.0, 3.0, 4.0).sum(), 10.0);
    /// ```
    #[inline]
    pub fn sum(self) -> f32 {
        self.0.sum()
    }

    /// Extracts `index`-th element of the vector. Index `0` corresponds to the "most left"
    /// element.
    ///
    /// # Panic
    /// Panics if `index` is invalid, i.e. greater than 3.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec4f;
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(vec.extract(2), 3.0);
    /// ```
    /// ```should_panic
    /// # use vrl::Vec4f;
    /// Vec4f::default().extract(5);
    ///
    /// ```
    ///
    /// # Note
    /// If `index` is known at compile time consider using [`extract_const`](Self::extract_const).
    #[inline]
    pub fn extract(self, index: usize) -> f32 {
        // NOTE: Agner Fog uses `int` as `index` type. Should we make it `isize` or `i32` too?
        if index >= Self::ELEMENTS {
            panic!("invalid index");
        }

        // NOTE: Agner Fog doesn't use aligned store here.
        // TODO: benchmark if this vertion actually faster then alignless one.
        #[repr(C)]
        #[repr(align(16))]
        struct AlignedArray([f32; 4]);

        let mut stored = std::mem::MaybeUninit::<AlignedArray>::uninit();
        unsafe {
            self.store_ptr_aligned(stored.as_mut_ptr() as *mut f32);
            stored.assume_init().0[index]
        }
    }

    /// Extracts `index % 4`-th element of the vector. This corresponds to the original [`extract`]
    /// function from VCL.
    ///
    /// ```
    /// # use vrl::Vec4f;
    /// assert_eq!(Vec4f::new(1.0, 2.0, 3.0, 4.0).extract_wrapping(6), 3.0);
    /// ```
    ///
    /// [`extract`]: https://github.com/vectorclass/version2/blob/f4617df57e17efcd754f5bbe0ec87883e0ed9ce6/vectorf128.h#L616
    pub fn extract_wrapping(self, index: usize) -> f32 {
        self.extract(index & 3)
    }

    /// Extracts `INDEX`-th element of the vector. Does same as [`extract`](Self::extract) with compile-time known
    /// index, but works faster on platforms with `sse4.1` support.
    ///
    /// # Examples
    /// ```
    /// # use vrl::Vec4f;
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(vec.extract_const::<2>(), 3.0);
    /// ```
    /// ```compile_fail
    /// # use vrl::Vec4f;
    /// Vec4f::default().extract_const::<5>();
    /// # #[cfg(miri)] { compile_error!() }
    /// ```
    #[inline]
    pub fn extract_const<const INDEX: i32>(self) -> f32 {
        if cfg!(sse41) {
            // TODO: benchmark this code to make sure it's actually faster than fallback version
            f32::from_bits(unsafe { _mm_extract_ps(self.into(), INDEX) } as u32)
        } else {
            self.extract(INDEX as usize)
        }
    }
}

impl SIMDVector for Vec4f {
    type Underlying = Underlying;
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
    #[inline]
    fn default() -> Self {
        Self(Vec4fBase::default())
    }
}

vec_overload_operator!(Vec4f, Add, add);
vec_overload_operator!(Vec4f, Sub, sub);
vec_overload_operator!(Vec4f, Mul, mul);
vec_overload_operator!(Vec4f, Div, div);
vec_impl_sum_prod!(Vec4f);

impl From<&[f32; 4]> for Vec4f {
    /// Does same as [`load`](Self::load).
    #[inline]
    fn from(value: &[f32; 4]) -> Self {
        Self::load(value)
    }
}

impl From<[f32; 4]> for Vec4f {
    #[inline]
    fn from(value: [f32; 4]) -> Self {
        (&value).into()
    }
}

impl From<Vec4f> for [f32; 4] {
    #[inline]
    fn from(value: Vec4f) -> Self {
        let mut result = MaybeUninit::<Self>::uninit();
        unsafe {
            value.store_ptr(result.as_mut_ptr() as *mut f32);
            result.assume_init()
        }
    }
}

impl From<&Vec4f> for [f32; 4] {
    #[inline]
    fn from(value: &Vec4f) -> Self {
        unsafe { std::mem::transmute_copy(value) }
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

impl Index<usize> for Vec4f {
    type Output = f32;

    /// Extracts `index`-th element of the vector.  If value of the vector is expected to be
    /// in a register consider using [`extract`](`Vec4f::extract`). Use this function if
    /// only the vector is probably stored in memory.
    ///
    /// # Panics
    /// Panics if `index` is invalid, i.e. greater than `3`.
    ///
    /// # Examples
    /// In the following example the vector is stored in the heap. Using `[]`-indexing in
    /// the case is as efficient as dereferencing the corresponding pointer.
    /// ```
    /// # use vrl::Vec4f;
    /// # use std::ops::Index;
    /// let many_vectors = vec![Vec4f::new(1.0, 2.0, 3.0, 4.0); 128];
    /// assert_eq!(many_vectors.index(64)[2], 3.0);
    /// ```
    /// ```should_panic
    /// # use vrl::Vec4f;
    /// Vec4f::default()[4];
    /// ```
    /// Here is an example if inefficient usage of the function. The vector wouldn't even reach memory
    /// and would stay in a register without that `[1]`. [`extract`](Vec4f::extract) should be used instead.
    /// ```
    /// # use vrl::Vec4f;
    /// let mut vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// vec *= 3.0;
    /// vec -= 2.0;
    /// let second_value = vec[1];
    /// assert_eq!(second_value, 4.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        if index >= Self::ELEMENTS {
            panic!("invalid index");
        }
        unsafe { &*(self as *const Self as *const f32).add(index) }
    }
}

impl IndexMut<usize> for Vec4f {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= Self::ELEMENTS {
            panic!("invalid index");
        }
        unsafe { &mut *(self as *mut Self as *mut f32).add(index) }
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
