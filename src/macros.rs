macro_rules! vec_overload_operator {
    ($vectype: ty, $op_trait: ident, $op_name: ident, $intrinsic: ident) => {
        impl $op_trait for $vectype {
            type Output = Self;
            #[inline(always)]
            fn $op_name(self, rhs: Self) -> Self::Output {
                unsafe { $intrinsic(self.into(), rhs.into()).into() }
            }
        }

        impl $op_trait<<$vectype as SIMDVector>::Element> for $vectype {
            type Output = Self;
            #[inline(always)]
            fn $op_name(self, rhs: <$vectype as SIMDVector>::Element) -> Self::Output {
                self.$op_name(Into::<Self>::into(rhs))
            }
        }

        impl $op_trait<$vectype> for <$vectype as SIMDVector>::Element {
            type Output = $vectype;
            #[inline(always)]
            fn $op_name(self, rhs: $vectype) -> Self::Output {
                rhs.$op_name(self)
            }
        }

        paste::paste! {
        impl<T> [<$op_trait Assign>]<T> for $vectype
        where
            Self: $op_trait<T, Output = Self>,
        {
            #[inline(always)]
            fn [<$op_name _assign>](&mut self, rhs: T) {
                *self = self.$op_name(rhs);
            }
        }
        }
    };
}

pub(crate) use vec_overload_operator;
