#![allow(non_snake_case)]

use paste::paste;
use std::{fmt::Debug, ops::Neg, usize};
use vrl::prelude::*;

fn iota_array<T: From<i16> + Copy, const N: usize>(first: usize) -> [T; N] {
    let mut array = [0.into(); N];
    for i in first..first + N {
        array[i - first] = (i as i16).into();
    }
    array
}

fn test_store_load_impl<VecT, const N: usize>()
where
    VecT: SIMDBase<N>
        + From<[VecT::Element; N]>
        + Into<[VecT::Element; N]>
        + Copy
        + Debug
        + PartialEq
        + SIMDPartialStore<VecT::Element>
        + SIMDPartialLoad<VecT::Element>,
    VecT::Element: From<i16> + Debug + PartialEq,
{
    const MAGIC_VALUE: i16 = 42;

    let values = iota_array(0);
    let mut result = [MAGIC_VALUE.into(); N];
    let loaded = VecT::load(&values);
    loaded.store(&mut result);

    // test simple load/store
    assert_eq!(values, result);
    assert_eq!(values, Into::<[VecT::Element; N]>::into(loaded));
    assert_eq!(Into::<VecT>::into(values), loaded);

    // test load_partial
    for len in 0..N {
        let prefix = VecT::load_partial(&values[..len]);
        let prefix_arr: [VecT::Element; N] = prefix.into();
        assert_eq!(&prefix_arr[..len], &values[..len]);
    }

    // test store_partial
    for len in 0..N {
        let mut buffer = vec![MAGIC_VALUE.into(); N];
        loaded.store_partial(buffer.split_at_mut(len).0);
        assert_eq!(&buffer[..len], &values[..len]);
        assert_eq!(&buffer[len..], vec![MAGIC_VALUE.into(); N - len])
    }
    {
        let mut buffer = vec![MAGIC_VALUE.into(); N + 1];
        loaded.store_partial(buffer.as_mut_slice());
        assert_eq!(&buffer[..N], values);
        assert_eq!(buffer[N], MAGIC_VALUE.into());
    }
}

macro_rules! test_checked_load_store {
    ($vectype: ty) => {
        paste! {
            #[test]
            #[should_panic(expected = "data must contain exactly ")]
            fn [<test_load_checked_too_small_ $vectype>]() {
                $vectype::load_checked(&[0i8.into(); 0]);
            }

            #[test]
            #[should_panic(expected = "data must contain exactly ")]
            fn [<test_load_checked_too_big_ $vectype>]() {
                $vectype::load_checked(&[0i8.into(); 100]);
            }

            #[test]
            #[should_panic(expected = "slice must contain exactly ")]
            fn [<test_store_checked_too_small_ $vectype>]() {
                $vectype::default().store_checked(&mut [0i8.into(); 0]);
            }

            #[test]
            #[should_panic(expected = "slice must contain exactly ")]
            fn [<test_store_checked_too_big_ $vectype>]() {
                $vectype::default().store_checked(&mut [0i8.into(); 100]);
            }
        }
    };
}

macro_rules! test_store_load {
    ($vectype: ty, $N: literal) => {
        paste! {
            test_checked_load_store!($vectype);

            #[test]
            fn [<test_store_load_ $vectype>]() {
                test_store_load_impl::<$vectype, $N>();
            }
        }
    };
}

test_store_load!(Vec4f, 4);
test_store_load!(Vec8f, 8);

fn test_fused_impl<
    const N: usize,
    VecT: SIMDFusedCalc
        + SIMDBase<N>
        + From<[VecT::Element; N]>
        + PartialEq
        + Arithmetic
        + Neg<Output = VecT>
        + Debug
        + Copy,
>()
where
    VecT::Element: From<i16> + Copy,
{
    let a = VecT::from(iota_array::<VecT::Element, N>(0));
    let b = VecT::from(iota_array::<VecT::Element, N>(2));
    let c = VecT::from(iota_array::<VecT::Element, N>(4));
    assert_eq!(a.mul_add(b, c), a * b + c);
    assert_eq!(a.nmul_add(b, c), -(a * b) + c);
    assert_eq!(a.mul_sub(b, c), a * b - c);
    assert_eq!(a.nmul_sub(b, c), -(a * b + c));
}

macro_rules! test_fused {
    ($vectype: ty, $N: literal) => {
        paste! {
            #[test]
            fn [<test_fused_ops_ $vectype>]() {
                test_fused_impl::<$N, $vectype>();
            }
        }
    };
}

test_fused!(Vec4f, 4);
test_fused!(Vec8f, 8);

fn test_round_impl<const N: usize, VecT: SIMDBase<N> + SIMDRound>()
where
    VecT::Element: From<i16> + From<f32> + Copy + PartialEq,
    VecT: From<[VecT::Element; N]>
        + PartialEq
        + Debug
        + Copy
        + Arithmetic<VecT::Element>
        + Into<[VecT::Element; N]>,
{
    let int_vec = VecT::from(iota_array::<VecT::Element, N>(0));
    assert_eq!(int_vec.round(), int_vec);
    assert_eq!((int_vec + 0.4.into()).round(), int_vec);
    assert_eq!((int_vec + 0.6.into()).round(), int_vec + 1.into());
    assert_eq!((int_vec - 2.4.into()).round(), int_vec - 2.into());
    assert_eq!((int_vec - 2.6.into()).round(), int_vec - 3.into());
    assert_eq!(
        ((int_vec - 0.1.into()) * 0.5.into()).round(),
        VecT::load_prefix(
            &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0]
                .map(|x| x.into())
        )
    );
}

macro_rules! test_round {
    ($vectype: ty, $N: literal) => {
        paste! {
            #[test]
            fn [<test_round_ $vectype>]() {
                test_round_impl::<$N, $vectype>();
            }
        }
    };
}

test_round!(Vec4f, 4);
test_round!(Vec8f, 8);
