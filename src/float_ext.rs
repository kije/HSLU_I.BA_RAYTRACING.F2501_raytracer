use simba::simd::{SimdBool, WideF32x4, WideF32x8, WideF64x4};

pub trait AbsDiffEq<Rhs = Self>: PartialEq<Rhs>
where
    Rhs: ?Sized,
{
    /// Used for specifying relative comparisons.
    type Epsilon;
    type Output: SimdBool;

    /// The default tolerance to use when testing values that are close together.
    ///
    /// This is used when no `epsilon` value is supplied to the [`abs_diff_eq!`], [`relative_eq!`],
    /// or [`ulps_eq!`] macros.
    fn default_epsilon() -> Self::Epsilon;

    /// A test for equality that uses the absolute difference to compute the approximate
    /// equality of two numbers.
    fn abs_diff_eq(&self, other: &Rhs, epsilon: Self::Epsilon) -> Self::Output;

    #[inline(always)]
    fn abs_diff_ne(&self, other: &Rhs, epsilon: Self::Epsilon) -> Self::Output {
        !Self::abs_diff_eq(self, other, epsilon)
    }

    #[inline(always)]
    fn abs_diff_eq_default(&self, other: &Rhs) -> Self::Output {
        Self::abs_diff_eq(self, other, Self::default_epsilon())
    }

    #[inline(always)]
    fn abs_diff_ne_default(&self, other: &Rhs) -> Self::Output {
        !Self::abs_diff_eq_default(self, other)
    }
}

macro_rules! impl_abs_diff_eq_simba_wide {
    ($WideF32xX:ty, $f32:ty) => {
        impl AbsDiffEq for $WideF32xX {
            type Epsilon = $WideF32xX;
            type Output = <$WideF32xX as simba::simd::SimdValue>::SimdBool;

            #[inline(always)]
            fn default_epsilon() -> Self::Epsilon {
                <$f32>::EPSILON.into()
            }

            #[inline(always)]
            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> Self::Output {
                use simba::simd::{SimdComplexField, SimdPartialOrd};
                (*self - *other).simd_abs().simd_le(epsilon)
            }
        }
    };
}

macro_rules! impl_abs_diff_eq_approx_forward {
    ($($t:ty),*) => {
        $(
            impl AbsDiffEq for $t {
                type Epsilon = <$t as approx::AbsDiffEq>::Epsilon;
                type Output = bool;

                #[inline(always)]
                fn default_epsilon() -> Self::Epsilon {
                    <$t as approx::AbsDiffEq>::default_epsilon()
                }

                #[inline(always)]
                fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> Self::Output {
                    <$t as approx::AbsDiffEq>::abs_diff_eq(self, other, epsilon)
                }
            }
        )*
    };
}

impl_abs_diff_eq_simba_wide!(WideF32x4, f32);
impl_abs_diff_eq_simba_wide!(WideF32x8, f32);
impl_abs_diff_eq_simba_wide!(WideF64x4, f64);

impl_abs_diff_eq_approx_forward!(f32, f64, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
