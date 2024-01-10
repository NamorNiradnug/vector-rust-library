#![allow(non_snake_case)]

use std::fmt::Debug;
use paste::paste;
use vrl::prelude::*;

fn iota_array<T: From<i16> + Copy, const N: usize>() -> [T; N] {
    let mut array = [0.into(); N];
    (0..N).for_each(|i| {
        array[i] = (i as i16).into();
    });
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

    let values = iota_array();
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
            #[should_panic]
            fn [<test_load_checked_too_small_ $vectype>]() {
                $vectype::load_checked(&[0i8.into(); 0]);
            }

            #[test]
            #[should_panic]
            fn [<test_load_checked_too_big_ $vectype>]() {
                $vectype::load_checked(&[0i8.into(); 100]);
            }

            #[test]
            #[should_panic]
            fn [<test_store_checked_too_small_ $vectype>]() {
                $vectype::default().store_checked(&mut [0i8.into(); 0]);
            }

            #[test]
            #[should_panic]
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
            #[allow(non_snake_case)]
            fn [<test_store_load_ $vectype>]() {
                test_store_load_impl::<$vectype, $N>();
            }
        }
    };
}

test_store_load!(Vec4f, 4);
test_store_load!(Vec8f, 8);
