#[allow(unused_macros)]
macro_rules! vec_impl_unary_op {
    ($vectype: ty, $op_trait: tt, $op_name: tt, $intrinsic: tt) => {
        impl $op_trait for $vectype {
            type Output = Self;
            #[inline]
            fn $op_name(self) -> Self::Output {
                unsafe { $intrinsic(self.into()).into() }
            }
        }
    };
}

macro_rules! vec_impl_binary_op {
    ($vectype: ty, $op_trait: tt, $op_name: tt, $intrinsic: tt) => {
        impl $op_trait for $vectype {
            type Output = Self;
            #[inline]
            fn $op_name(self, rhs: Self) -> Self::Output {
                // SAFETY: the intrinsic won't compile on a platform it isn't available;
                // cfg_if! should do its job
                unsafe { $intrinsic(self.into(), rhs.into()).into() }
            }
        }
    };
}

macro_rules! vec_overload_operator {
    ($vectype: ty, $eltype: ty, $op_trait: ident, $op_name: ident) => {
        impl $op_trait<$eltype> for $vectype {
            type Output = Self;
            #[inline]
            fn $op_name(self, rhs: $eltype) -> Self::Output {
                self.$op_name(<$vectype>::broadcast(rhs))
            }
        }

        impl $op_trait<$vectype> for $eltype {
            type Output = $vectype;
            #[inline]
            fn $op_name(self, rhs: $vectype) -> Self::Output {
                rhs.$op_name(self)
            }
        }

        paste::paste! {
        impl<T> [<$op_trait Assign>]<T> for $vectype
        where
            Self: $op_trait<T, Output = Self>,
        {
            #[inline]
            fn [<$op_name _assign>](&mut self, rhs: T) {
                *self = self.$op_name(rhs);
            }
        }
        }
    };
}

macro_rules! vec_impl_sum_prod {
    ($vectype: ty) => {
        impl std::iter::Sum for $vectype {
            #[inline]
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(<$vectype>::default(), |a, b| a + b)
            }
        }

        impl std::iter::Product for $vectype {
            #[inline]
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(<$vectype>::default(), |a, b| a * b)
            }
        }
    };
}

macro_rules! vec_impl_generic_traits {
    ($vectype: tt, $eltype: tt, $N: tt) => {
        vec_overload_operator!($vectype, $eltype, Add, add);
        vec_overload_operator!($vectype, $eltype, Sub, sub);
        vec_overload_operator!($vectype, $eltype, Mul, mul);
        vec_overload_operator!($vectype, $eltype, Div, div);
        vec_impl_sum_prod!($vectype);

        impl From<&[$eltype; $N]> for $vectype {
            /// Does same as [`load`](SIMDBase::load).
            #[inline]
            fn from(value: &[$eltype; $N]) -> Self {
                Self::load(value)
            }
        }

        impl From<[$eltype; $N]> for $vectype {
            #[inline]
            fn from(value: [$eltype; $N]) -> Self {
                (&value).into()
            }
        }

        impl From<$vectype> for [$eltype; $N] {
            #[inline]
            fn from(value: $vectype) -> Self {
                let mut result = MaybeUninit::<Self>::uninit();
                // SAFETY: currently all the supported platforms (x86 and NEON) store
                // SIMD types as a packed array of `f32`/`f64`/etc
                // Also the underlying intrinsic types are aligned enough
                // Hence the pointer casting is valid
                unsafe {
                    value.store_ptr(result.as_mut_ptr() as *mut $eltype);
                    result.assume_init()
                }
            }
        }

        impl From<&$vectype> for [$eltype; $N] {
            #[inline]
            fn from(value: &$vectype) -> Self {
                // SAFETY: see safety comment for `impl From<vectype> for [eltype; N]` above
                unsafe { std::mem::transmute_copy(value) }
            }
        }

        impl std::fmt::Debug for $vectype {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut debug_tuple = f.debug_tuple("$vectype");
                for value in <[$eltype; Self::N]>::from(self) {
                    debug_tuple.field(&value);
                }
                debug_tuple.finish()
            }
        }

        impl std::ops::Index<usize> for $vectype {
            type Output = $eltype;
            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                if index >= Self::N {
                    panic!("invalid index");
                }
                // SAFETY: on both x86 and NEON intrinsic types are stored as an array of several values
                unsafe { &*(self as *const Self as *const $eltype).add(index) }
            }
        }

        impl std::ops::IndexMut<usize> for $vectype {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                if index >= Self::N {
                    panic!("invalid index");
                }
                // SAFETY: on both x86 and NEON intrinsic types are stored as an array of several values
                unsafe { &mut *(self as *mut Self as *mut $eltype).add(index) }
            }
        }
    };
}

macro_rules! vec_impl_partial_load {
    ($vectype: ty, $eltype: ty, 4) => {
        impl crate::common::SIMDPartialLoad<$eltype> for $vectype {
            #[inline]
            fn load_partial(data: &[$eltype]) -> Self {
                match data.len() {
                    // SAFETY: if data.len() is at least 4 hence it's safe to simply load the prefix
                    4.. => unsafe { Self::load_ptr(data.as_ptr()) },
                    3 => Self::new(data[0], data[1], data[2], 0 as $eltype),
                    2 => Self::new(data[0], data[1], 0 as $eltype, 0 as $eltype),
                    1 => Self::new(data[0], 0 as $eltype, 0 as $eltype, 0 as $eltype),
                    0 => Self::default(),
                }
            }
        }
    };

    ($vectype: ty, $eltype: ty, $halfvectype: ty, 8) => {
        impl crate::common::SIMDPartialLoad<$eltype> for $vectype {
            #[inline]
            fn load_partial(data: &[$eltype]) -> Self {
                match data.len() {
                    // SAFETY: if data.len() is at least 8 hence it's safe to simply load the prefix
                    8.. => unsafe { Self::load_ptr(data.as_ptr()) },
                    4.. => Self::join(
                        // SAFETY: data.len() is at least 4 hence it's safe to load the first 4
                        // element into vector type of size 4.
                        unsafe { <$halfvectype>::load_ptr(data.as_ptr()) },
                        <$halfvectype>::load_partial(data.split_at(4).1),
                    ),
                    0.. => Self::join(
                        <$halfvectype>::load_partial(data),
                        <$halfvectype>::default(),
                    ),
                }
            }
        }
    };
}

macro_rules! vec_impl_partial_store {
    ($vectype: ty, $eltype: ty, 4) => {
        impl crate::common::SIMDPartialStore<$eltype> for $vectype {
            #[inline]
            fn store_partial(&self, slice: &mut [$eltype]) {
                match slice.len() {
                    // SAFETY: slice.len() is at least 4 hence it's valid to write the vector into the prefix
                    4.. => unsafe { self.store_ptr(slice.as_mut_ptr()) },
                    _ => slice.copy_from_slice(&<[$eltype; 4]>::from(self)[..slice.len()]),
                }
            }
        }
    };
    ($vectype: ty, $eltype: ty, 8) => {
        impl crate::common::SIMDPartialStore<$eltype> for $vectype {
            #[inline]
            fn store_partial(&self, slice: &mut [$eltype]) {
                match slice.len() {
                    // SAFETY: slice.len() is at least 8 hence it's valid to write the vector into the prefix
                    8.. => unsafe { self.store_ptr(slice.as_mut_ptr()) },
                    4.. => {
                        // SAFETY: slice.len() is at least 4 hence it's valid to write the first 4
                        // elements of the vector into the slice's prefix
                        unsafe { self.low().store_ptr(slice.as_mut_ptr()) };
                        self.high().store_partial(slice.split_at_mut(4).1)
                    }
                    0.. => self.low().store_partial(slice),
                }
            }
        }
    };
}

#[allow(unused_macros)]
macro_rules! vec_impl_broadcast_default {
    ($vectype: ty, $zero: literal) => {
        impl Default for $vectype {
            #[inline]
            fn default() -> Self {
                Self::broadcast($zero)
            }
        }
    };
}

#[allow(unused_imports)]
pub(crate) use {
    vec_impl_binary_op, vec_impl_broadcast_default, vec_impl_generic_traits, vec_impl_partial_load,
    vec_impl_partial_store, vec_impl_sum_prod, vec_impl_unary_op, vec_overload_operator,
};
