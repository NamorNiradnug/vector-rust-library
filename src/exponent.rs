use crate::{common::Arithmetic, prelude::*};

/// Calculates `pow(2, vec)` where elements of `vec` must be integers.
/// Doesn't check for over-/underflow.
fn pow2n<const N: usize, VecT: SIMDBase<N, Element = f32>>(_vec: VecT) -> VecT {
    todo!()
}

#[allow(unused_assignments)]
#[allow(clippy::excessive_precision)]
fn exp_f32<
    const N: usize,
    VecT: SIMDBase<N, Element = f32> + Arithmetic + Arithmetic<f32> + SIMDFusedCalc + SIMDRound + Copy,
>(
    vec: VecT,
) -> VecT {
    // Taylor coefficients
    const P0: f32 = 1.0 / 2.0;
    const P1: f32 = 1.0 / 6.0;
    const P2: f32 = 1.0 / 24.0;
    const P3: f32 = 1.0 / 120.0;
    const P4: f32 = 1.0 / 720.0;
    const P5: f32 = 1.0 / 5040.0;

    const LN2_HI: f32 = 0.693359375;
    const LN2_LO: f32 = -2.12194440e-4;
    const _MAX_X: f32 = 87.3;

    let r = (vec * std::f32::consts::LOG2_E).round();
    let mut x = vec;
    x = VecT::mul_add(r, VecT::broadcast(LN2_HI), x); //  x -= r * ln2_hi;
    x = VecT::nmul_add(r, VecT::broadcast(LN2_LO), x); //  x -= r * ln2_lo;

    let x2 = x * x;
    let x4 = x2 * x2;
    // NOTE: not using a separate function like `polynom_5` here to reuse the `x2` value.
    // result = sum_(i=0..5) P_i * x^i =
    // = (x * P5 + P4) * x^4 + ((x * P3 + P2) * x^2 + x * P1 + P0)
    let mut result = x.mul_add(VecT::broadcast(P5), VecT::broadcast(P4)).mul_add(
        x4,
        x.mul_add(VecT::broadcast(P3), VecT::broadcast(P2))
            .mul_add(x2, x.mul_add(VecT::broadcast(P1), VecT::broadcast(P0))),
    );
    result = result.mul_add(x2, x);

    result = (result + 1.0) * pow2n(r);

    todo!("Overflow check");

    #[allow(unreachable_code)]
    result
}

pub trait SIMDTaylorExponent {
    /// Calculates an exponent of each element in vector.
    ///
    /// ```no_run
    /// # use vrl::prelude::*;
    /// use std::f32::consts::E;
    /// let vec = Vec4f::broadcast(2.0);
    /// let exp = vec.exp();
    /// approx::assert_abs_diff_eq!(exp[0], E * E);
    /// ```
    fn exp(self) -> Self;
}

macro_rules! vec_impl_exp {
    ($vectype: ty, $N: literal, f32) => {
        impl SIMDTaylorExponent for $vectype {
            fn exp(self) -> Self {
                exp_f32(self)
            }
        }
    };
}

vec_impl_exp!(Vec4f, 4, f32);
vec_impl_exp!(Vec8f, 4, f32);
