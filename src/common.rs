/// Represents a packed vector containing [`ELEMENTS`](SIMDVector::ELEMENTS)
/// values of type [`Element`].
///
/// Converting [`Element`] to [`SIMDVector`] should work as `broadcast`, i.e. assign
/// the converting value to all elements of the vector.
///
/// [`Element`]: Self::Element
pub trait SIMDVector
where
    Self: From<Self::Underlying> + From<Self::Element>, // + [Self::Element; Self::ELEMENTS]
    Self::Underlying: From<Self>,
{
    /// Underlying intrinsic type.
    type Underlying;

    /// Type of a single element of [`SIMDVector`].
    type Element;

    /// Number of elements in [`SIMDVector`].
    const ELEMENTS: usize;
}
