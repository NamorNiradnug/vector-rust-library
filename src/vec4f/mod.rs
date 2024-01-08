use std::{
    mem::MaybeUninit,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use crate::{common::*, macros::*};

cfg_if::cfg_if! {
    if #[cfg(sse)] {
        mod sse;
        pub use sse::*;
    } else {
        compile_error!("Currently SSE is required for Vec4f");
        mod fallback;
        pub use fallback::*;
    }
}

pub trait Vec4fBase {
    /// Initializes elements of returned vector with given values.
    ///
    /// # Example
    /// ```
    /// # use vrl::prelude::*;
    /// assert_eq!(
    ///     Vec4f::new(1.0, 2.0, 3.0, 4.0),
    ///     [1.0, 2.0, 3.0, 4.0].into()
    /// );
    /// ```
    fn new(v0: f32, v1: f32, v2: f32, v3: f32) -> Self;
}

vec_impl_generic_traits!(Vec4f, f32, 4);
vec_impl_partial_load!(Vec4f, f32, 4);
vec_impl_partial_store!(Vec4f, f32, 4);

impl SIMDVector<4> for Vec4f {}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

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
