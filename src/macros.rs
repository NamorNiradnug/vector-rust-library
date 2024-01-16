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
                unsafe {
                    value.store_ptr(result.as_mut_ptr() as *mut $eltype);
                    result.assume_init()
                }
            }
        }

        impl From<&$vectype> for [$eltype; $N] {
            #[inline]
            fn from(value: &$vectype) -> Self {
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
                unsafe { &*(self as *const Self as *const $eltype).add(index) }
            }
        }

        impl std::ops::IndexMut<usize> for $vectype {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                if index >= Self::N {
                    panic!("invalid index");
                }
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
                    8.. => unsafe { Self::load_ptr(data.as_ptr()) },
                    4.. => Self::join(
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
                    8.. => unsafe { self.store_ptr(slice.as_mut_ptr()) },
                    4.. => {
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
macro_rules! vec_impl_fused_low_high {
    ($vectype: ty) => {
        impl SIMDFusedCalc for $vectype {
            #[inline]
            fn mul_add(self, b: Self, c: Self) -> Self {
                (
                    self.low().mul_add(b.low(), c.low()),
                    self.high().mul_add(b.high(), c.high()),
                )
                    .into()
            }

            #[inline]
            fn mul_sub(self, b: Self, c: Self) -> Self {
                (
                    self.low().mul_sub(b.low(), c.low()),
                    self.high().mul_sub(b.high(), c.high()),
                )
                    .into()
            }

            #[inline]
            fn nmul_add(self, b: Self, c: Self) -> Self {
                (
                    self.low().nmul_add(b.low(), c.low()),
                    self.high().nmul_add(b.high(), c.high()),
                )
                    .into()
            }

            #[inline]
            fn nmul_sub(self, b: Self, c: Self) -> Self {
                (
                    self.low().nmul_sub(b.low(), c.low()),
                    self.high().nmul_sub(b.high(), c.high()),
                )
                    .into()
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
    vec_impl_binary_op, vec_impl_broadcast_default, vec_impl_fused_low_high,
    vec_impl_generic_traits, vec_impl_partial_load, vec_impl_partial_store, vec_impl_sum_prod,
    vec_impl_unary_op, vec_overload_operator,
};
