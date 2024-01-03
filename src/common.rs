use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Represents a packed vector containing [`ELEMENTS`](SIMDVector::ELEMENTS)
/// values of type [`Element`].
///
/// Converting [`Element`] to [`SIMDVector`] works as `broadcast`, i.e. assign
/// the converting value to all elements of the vector.
///
/// [`Default::default`] initializes all elements of vector with zero.
///
/// All arithmetic operations ([`Neg`], [`Add`], etc) are applied vertically, i.e. "elementwise".
///
/// [`Element`]: Self::Element
pub trait SIMDVector
where
    Self: From<Self::Underlying>
        + From<Self::Element>
        + Default
        + Neg<Output = Self>
        + Add<Self>
        + Add<Self::Element>
        + Sub<Self>
        + Sub<Self::Element>
        + Mul<Self>
        + Mul<Self::Element>
        + Div<Self>
        + Div<Self::Element>
        + AddAssign<Self>
        + AddAssign<Self::Element>
        + SubAssign<Self>
        + SubAssign<Self::Element>
        + MulAssign<Self>
        + MulAssign<Self::Element>
        + DivAssign<Self>
        + DivAssign<Self::Element>, // + [Self::Element; Self::ELEMENTS]
    Self::Underlying: From<Self>,
    Self::Element: Add<Self> + Sub<Self> + Mul<Self> + Div<Self>,
{
    /// Underlying intrinsic type.
    type Underlying;

    /// Type of a single element of [`SIMDVector`].
    type Element;

    /// Number of elements in [`SIMDVector`].
    const ELEMENTS: usize;
}
