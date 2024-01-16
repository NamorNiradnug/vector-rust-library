use std::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Base trait for fixed-size vector types.
pub trait SIMDBase<const N: usize>
where
    Self: Into<Self::Underlying> + From<Self::Underlying>,
    Self::Element: Copy,
{
    /// Underlying intrinsic type or tuple of types.
    type Underlying;

    /// Type of a single element of packed vector.
    type Element;

    /// Number of elements in vector.
    const N: usize = N;

    /// Initializes all values of returned vector with a given value.
    ///
    /// # Examples
    /// ```
    /// # use vrl::prelude::*;
    /// assert_eq!(
    ///     Vec4f::broadcast(42.0),
    ///     [42.0; 4].into()
    /// );
    /// ```
    ///
    /// # Note
    /// Prefer using [`default`](Default::default) instead of [`broadcast`]-ing zero.
    ///
    /// [`broadcast`]: SIMDBase::broadcast
    fn broadcast(value: Self::Element) -> Self;

    /// Loads vector from an array pointed by `addr`.
    /// `addr` is not required to be aligned.
    ///
    /// # Safety
    /// `addr` must be a valid pointer to an [`N`](Self::N)-sized array.
    ///
    /// # Example
    /// ```
    /// # use vrl::prelude::*;
    /// let array = [42.0; 4];
    /// let vec = unsafe { Vec4f::load_ptr(array.as_ptr()) };
    /// ```
    unsafe fn load_ptr(addr: *const Self::Element) -> Self;

    /// Loads vector from a given array.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::prelude::*;
    /// assert_eq!(
    ///     Vec4f::new(1.0, 2.0, 3.0, 4.0),
    ///     Vec4f::load(&[1.0, 2.0, 3.0, 4.0])
    /// );
    /// ```
    #[inline]
    fn load(data: &[Self::Element; N]) -> Self {
        unsafe { Self::load_ptr(data.as_ptr()) }
    }

    /// Checks that `data` contains exactly [`N`] elements and loads them into vector.
    ///
    /// # Panics
    /// Panics if `data.len()` doesn't equal [`N`].
    ///
    /// # Examples
    /// ```
    /// # use vrl::prelude::*;
    /// assert_eq!(
    ///     Vec4f::load_checked(&[1.0, 2.0, 3.0, 4.0]),
    ///     Vec4f::new(1.0, 2.0, 3.0, 4.0)
    /// );
    /// ```
    /// ```should_panic
    /// # use vrl::prelude::*;
    /// Vec4f::load_checked(&[1.0, 2.0, 3.0]);
    /// ```
    /// ```should_panic
    /// # use vrl::prelude::*;
    /// Vec4f::load_checked(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// ```
    ///
    /// [`N`]: Self::N
    #[inline]
    fn load_checked(data: &[Self::Element]) -> Self {
        if data.len() != Self::N {
            panic!("data must contain exactly {} elements", Self::N);
        }
        unsafe { Self::load_ptr(data.as_ptr()) }
    }

    /// Loads first [`N`] elements of `data` into vector.
    ///
    /// # Panics
    /// Panics if `data` contains less than [`N`] elements.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::prelude::*;
    /// assert_eq!(
    ///     Vec4f::load_prefix(&[1.0, 2.0, 3.0, 4.0, 5.0]),
    ///     Vec4f::new(1.0, 2.0, 3.0, 4.0)
    /// );
    /// ```
    ///
    /// ```should_panic
    /// # use vrl::prelude::*;
    /// Vec4f::load_prefix(&[1.0, 2.0, 3.0]);
    /// ```
    ///
    /// [`N`]: Self::N
    #[inline]
    fn load_prefix(data: &[Self::Element]) -> Self {
        if data.len() < Self::N {
            panic!("data must contain at least {} elements", Self::N);
        }
        unsafe { Self::load_ptr(data.as_ptr()) }
    }

    /// Stores vector into array at given address.
    ///
    /// # Safety
    /// `addr` must be a valid pointer.
    unsafe fn store_ptr(self, addr: *mut Self::Element);

    /// Stores vector into given `array`.
    #[inline]
    fn store(self, array: &mut [Self::Element; N]) {
        unsafe { self.store_ptr(array.as_mut_ptr()) }
    }

    /// Checks that `slice` contains exactly [`N`] elements and stores elements of vector there.
    ///
    /// # Panics
    /// Panics if `slice.len()` doesn't equal [`N`].
    ///
    /// # Examples
    /// ```
    /// # use vrl::prelude::*;
    /// let mut data = [-1.0; 4];
    /// Vec4f::default().store_checked(&mut data);
    /// assert_eq!(data, [0.0; 4]);
    /// ```
    /// ```should_panic
    /// # use vrl::prelude::*;
    /// let mut data = [-1.0; 3];
    /// Vec4f::default().store_checked(&mut data);
    /// ```
    /// ```should_panic
    /// # use vrl::prelude::*;
    /// let mut data = [-1.0; 5];
    /// Vec4f::default().store_checked(&mut data);
    /// ```
    ///
    /// [`N`]: Self::N
    #[inline]
    fn store_checked(self, slice: &mut [Self::Element]) {
        if slice.len() != Self::N {
            panic!("slice must contain exactly {} elements", Self::N);
        }
        unsafe { self.store_ptr(slice.as_mut_ptr()) };
    }

    /// Stores elements of the vector into prefix of `slice`.
    ///
    /// # Panics
    /// Panics if `slice.len()` is less than [`N`](Self::N).
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::prelude::*;
    /// let mut data = [-1.0; 5];
    /// Vec4f::broadcast(2.0).store_prefix(&mut data);
    /// assert_eq!(data, [2.0, 2.0, 2.0, 2.0, -1.0]);
    /// ```
    /// ```should_panic
    /// # use vrl::prelude::*;
    /// let mut data = [-1.0; 3];
    /// Vec4f::default().store_prefix(&mut data);
    /// ```
    #[inline]
    fn store_prefix(self, slice: &mut [Self::Element]) {
        if slice.len() < Self::N {
            panic!("slice must contain at least {} elements", Self::N);
        }
        unsafe { self.store_ptr(slice.as_mut_ptr()) };
    }

    /// Extracts `index`-th element of the vector. Index `0` corresponds to the "most left"
    /// element.
    ///
    /// # Panic
    /// Panics if `index` is invalid, i.e. isn't less than [`N`](Self::N).
    ///
    /// # Examples
    /// ```
    /// # use vrl::prelude::*;
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(vec.extract(2), 3.0);
    /// ```
    /// ```should_panic
    /// # use vrl::prelude::*;
    /// Vec4f::default().extract(5);
    ///
    /// ```
    ///
    /// # Note
    /// If `index` is known at compile time consider using [`extract_const`](Self::extract_const).
    #[inline]
    fn extract(self, index: usize) -> Self::Element {
        if index >= Self::N {
            panic!("invalid index");
        }
        let mut stored = std::mem::MaybeUninit::<[Self::Element; N]>::uninit();
        unsafe {
            self.store_ptr(stored.as_mut_ptr() as *mut Self::Element);
            stored.assume_init()[index]
        }
    }

    /// Extracts `index % N`-th element of the vector. This corresponds to the original [`extract`]
    /// function from VCL.
    ///
    /// # Examples
    /// ```
    /// # use vrl::prelude::*;
    /// assert_eq!(Vec4f::new(1.0, 2.0, 3.0, 4.0).extract_wrapping(6), 3.0);
    /// ```
    ///
    /// [`extract`]: https://github.com/vectorclass/version2/blob/f4617df57e17efcd754f5bbe0ec87883e0ed9ce6/vectorf128.h#L616
    #[inline]
    fn extract_wrapping(self, index: usize) -> Self::Element {
        self.extract(index % Self::N)
    }

    /// Extracts `INDEX`-th element of the vector. Does same as [`extract`](Self::extract) with compile-time
    /// known index.
    ///
    /// # Examples
    /// ```
    /// # use vrl::prelude::*;
    /// let vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(vec.extract_const::<2>(), 3.0);
    /// ```
    /// ```compile_fail
    /// # use vrl::prelude::*;
    /// Vec4f::default().extract_const::<5>();
    /// # compile_error!("out-of-range index for extract_const")
    /// ```
    ///
    /// # Note
    /// Currently not all implementations assures that `INDEX` is valid at compile time.
    #[inline]
    fn extract_const<const INDEX: i32>(self) -> Self::Element {
        self.extract(INDEX as usize)
    }

    /// Calculates the sum of all elements of vector.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::prelude::*;
    /// assert_eq!(Vec4f::new(1.0, 2.0, 3.0, 4.0).sum(), 10.0);
    /// ```
    fn sum(self) -> Self::Element;
}

pub trait SIMDPartialLoad<T> {
    /// Loads first `min(N, data.len())` elements of `data` into vector.
    ///
    /// # Example
    /// ```
    /// # use vrl::prelude::*;
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
    fn load_partial(data: &[T]) -> Self;
}

pub trait SIMDPartialStore<T> {
    /// Stores `min(N, slice.len())` elements of vector into prefix of `slice`.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::prelude::*;
    /// let mut data = [0.0; 3];
    /// Vec4f::broadcast(1.0).store_partial(&mut data);
    /// assert_eq!(data, [1.0; 3]);
    /// ```
    /// ```
    /// # use vrl::prelude::*;
    /// let mut data = [0.0; 5];
    /// Vec4f::broadcast(1.0).store_partial(&mut data);
    /// assert_eq!(data, [1.0, 1.0, 1.0, 1.0, 0.0]);  // note last zero
    /// ```
    fn store_partial(&self, slice: &mut [T]);
}

pub trait SIMDFusedCalc {
    /// Multiplies vector by `b` and adds `c` to the product.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::prelude::*;
    /// let a = Vec4f::new(1.0, 2.0, 0.5, 2.0);
    /// let b = Vec4f::new(1.0, 0.5, 2.0, 3.0);
    /// let c = Vec4f::new(4.0, 2.0, 3.0, 1.0);
    /// assert_eq!(a.mul_add(b, c), a * b + c);
    /// ```
    fn mul_add(self, b: Self, c: Self) -> Self;

    /// Multiplies vector by `b` ans substracts `c` from the procuct.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::prelude::*;
    /// let a = Vec4f::new(1.0, 2.0, 0.5, 2.0);
    /// let b = Vec4f::new(1.0, 0.5, 2.0, 3.0);
    /// let c = Vec4f::new(4.0, 2.0, 3.0, 1.0);
    /// assert_eq!(a.mul_sub(b, c), a * b - c);
    /// ```
    fn mul_sub(self, b: Self, c: Self) -> Self;

    /// Multiplies vector by `b` and substracts the product from `c`.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::prelude::*;
    /// let a = Vec4f::new(1.0, 2.0, 0.5, 2.0);
    /// let b = Vec4f::new(1.0, 0.5, 2.0, 3.0);
    /// let c = Vec4f::new(4.0, 2.0, 3.0, 1.0);
    /// assert_eq!(a.nmul_add(b, c), c - a * b);
    /// ```
    fn nmul_add(self, b: Self, c: Self) -> Self;

    /// Multiplies vector by `b` and substracts the product from `-c`.
    ///
    /// # Exmaples
    /// ```
    /// # use vrl::prelude::*;
    /// let a = Vec4f::new(1.0, 2.0, 0.5, 2.0);
    /// let b = Vec4f::new(1.0, 0.5, 2.0, 3.0);
    /// let c = Vec4f::new(4.0, 2.0, 3.0, 1.0);
    /// assert_eq!(a.nmul_sub(b, c), -(a * b + c));
    /// ```
    fn nmul_sub(self, b: Self, c: Self) -> Self;
}

pub(crate) trait SIMDFusedCalcFallback {}

impl<T: SIMDFusedCalcFallback + Arithmetic + Neg<Output = Self>> SIMDFusedCalc for T {
    #[inline]
    fn mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }

    #[inline]
    fn mul_sub(self, b: Self, c: Self) -> Self {
        self * b - c
    }

    #[inline]
    fn nmul_add(self, b: Self, c: Self) -> Self {
        c - self * b
    }

    #[inline]
    fn nmul_sub(self, b: Self, c: Self) -> Self {
        -(self * b + c)
    }
}

pub trait Arithmetic<Rhs = Self, Output = Self>:
    Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output>
    + Mul<Rhs, Output = Output>
    + Div<Rhs, Output = Output>
{
}
impl<T, Rhs, Output> Arithmetic<Rhs, Output> for T where
    T: Add<Rhs, Output = Output>
        + Sub<Rhs, Output = Output>
        + Mul<Rhs, Output = Output>
        + Div<Rhs, Output = Output>
{
}

pub trait ArithmeticAssign<Rhs = Self>:
    AddAssign<Rhs> + SubAssign<Rhs> + MulAssign<Rhs> + DivAssign<Rhs>
{
}
impl<T, Rhs> ArithmeticAssign<Rhs> for T where
    T: AddAssign<Rhs> + SubAssign<Rhs> + MulAssign<Rhs> + DivAssign<Rhs>
{
}

/// Represents a packed vector containing [`N`] values of type [`Element`].
///
/// [`Default::default`] initializes all elements of vector with zero.
///
/// All arithmetic operations ([`Neg`], [`Add`], etc) are applied vertically, i.e. "elementwise".
///
/// # [`extract`] vs [`index`]
///
/// [`index`] (aka operator `[]`) extracts `index`-th element of the vector. If value of the vector is expected to be
/// in a register consider using [`extract`]. Use this function if only the vector is probably stored in memory.
///
/// In the following example the vector is stored in the heap. Using `[]`-indexing in
/// the case is as efficient as dereferencing the corresponding pointer.
/// ```
/// # use vrl::prelude::*;
/// # use std::ops::Index;
/// let many_vectors = vec![Vec4f::new(1.0, 2.0, 3.0, 4.0); 128];
/// assert_eq!(many_vectors.index(64)[2], 3.0);
/// ```
/// Here is an example of inefficient usage of [`index`]. The vector wouldn't even reach memory
/// and would stay in a register without that `[1]`. [`extract`] should be used instead.
/// ```
/// # use vrl::prelude::*;
/// let mut vec = Vec4f::new(1.0, 2.0, 3.0, 4.0);
/// vec *= 3.0;
/// vec -= 2.0;
/// let second_value = vec[1];
/// assert_eq!(second_value, 4.0);
/// ```
///
/// [`N`]: SIMDBase::N
/// [`Element`]: SIMDBase::Element
/// [`extract`]: SIMDBase::extract
/// [`index`]: Index::index
pub trait SIMDVector<const N: usize>:
    SIMDBase<N>
    + Neg<Output = Self>
    + Arithmetic
    + ArithmeticAssign<Self>
    + Arithmetic<Self::Element>
    + ArithmeticAssign<Self::Element>
    + PartialEq
    + Into<Self::Underlying>
    + Into<[Self::Element; N]>
    + for<'a> From<&'a [Self::Element; N]>
    + SIMDPartialLoad<Self::Element>
    + SIMDPartialStore<Self::Element>
    + SIMDFusedCalc
    + Default
    + Copy
    + Clone
    + Debug
    + Index<usize>
    + IndexMut<usize>
    + Sum
    + Product
where
    [Self::Element; N]: Into<Self>,
    Self::Element: Arithmetic<Self, Self>,
    Self::Underlying: Into<Self>,
{
}
