use crate::helpers::{ColorType, Splatable};
use crate::simd_compat::SimdValueRealSimplified;
use crate::vector::{Vector, Vector3DAccessor, VectorFixedDimensions};
use palette::num::{One, Zero};
use palette::{Hsv, IntoColor, Srgb};
use simba::scalar::SubsetOf;
use simba::simd::SimdValue;

// fixme: this might be a trait thai is not only useful to have for colors, e.g. geometry and basically anything VectorAware (or SimdSclaraAware) could use this?
pub trait ColorSimdExt<Scalar>
where
    Scalar: crate::scalar::Scalar + SimdValue,
{
    fn blend(mask: <Scalar as SimdValue>::SimdBool, t: &Self, f: &Self) -> Self;

    fn splat(v: &ColorType<Scalar::Element>) -> Self
    where
        Self: Splatable<ColorType<Scalar::Element>>;

    fn abs_diff_eq_default_all(&self, v: &Self) -> Scalar::SimdBool
    where
        Scalar: crate::float_ext::AbsDiffEq<Scalar, Output = Scalar::SimdBool>;

    fn abs_diff_eq_default_any(&self, v: &Self) -> Scalar::SimdBool
    where
        Scalar: crate::float_ext::AbsDiffEq<Scalar, Output = Scalar::SimdBool>;

    fn zero() -> Self
    where
        Scalar: Zero;
    fn one() -> Self
    where
        Scalar: One;

    fn to_vec3<V: Vector<Scalar = Scalar> + VectorFixedDimensions<3>>(&self) -> V;
    fn from_vec3<V: Vector<Scalar = Scalar> + VectorFixedDimensions<3> + Vector3DAccessor>(
        vec: &V,
    ) -> Self;
}

impl<SourceScalar, Scalar: Splatable<SourceScalar>> Splatable<ColorType<SourceScalar>>
    for ColorType<Scalar>
{
    #[inline(always)]
    fn splat(
        ColorType {
            red, green, blue, ..
        }: &ColorType<SourceScalar>,
    ) -> Self {
        ColorType::new(
            Splatable::splat(red),
            Splatable::splat(green),
            Splatable::splat(blue),
        )
    }
}

impl<Scalar> ColorSimdExt<Scalar> for ColorType<Scalar>
where
    Scalar: crate::scalar::Scalar + SimdValue,
    <Scalar as SimdValue>::Element: SubsetOf<Scalar>,
{
    #[inline(always)]
    fn blend(mask: <Scalar as SimdValue>::SimdBool, t: &Self, f: &Self) -> Self {
        ColorType::<Scalar>::new(
            t.red.clone().select(mask, f.red.clone()),
            t.green.clone().select(mask, f.green.clone()),
            t.blue.clone().select(mask, f.blue.clone()),
        )
    }

    #[inline(always)]
    fn splat(v: &ColorType<Scalar::Element>) -> Self
    where
        Self: Splatable<ColorType<Scalar::Element>>,
    {
        <Self as Splatable<_>>::splat(v)
    }

    fn abs_diff_eq_default_all(&self, v: &Self) -> Scalar::SimdBool
    where
        Scalar: crate::float_ext::AbsDiffEq<Scalar, Output = Scalar::SimdBool>,
    {
        use crate::float_ext::AbsDiffEq;
        self.blue.abs_diff_eq_default(&v.blue)
            & self.green.abs_diff_eq_default(&v.green)
            & self.red.abs_diff_eq_default(&v.red)
    }

    fn abs_diff_eq_default_any(&self, v: &Self) -> Scalar::SimdBool
    where
        Scalar: crate::float_ext::AbsDiffEq<Scalar, Output = Scalar::SimdBool>,
    {
        use crate::float_ext::AbsDiffEq;
        self.blue.abs_diff_eq_default(&v.blue)
            | self.green.abs_diff_eq_default(&v.green)
            | self.red.abs_diff_eq_default(&v.red)
    }

    fn zero() -> Self
    where
        Scalar: Zero,
    {
        Self::new(Scalar::zero(), Scalar::zero(), Scalar::zero())
    }
    fn one() -> Self
    where
        Scalar: One,
    {
        Self::new(Scalar::one(), Scalar::one(), Scalar::one())
    }

    fn to_vec3<V: Vector<Scalar = Scalar> + VectorFixedDimensions<3>>(&self) -> V {
        V::from_components([self.red.clone(), self.green.clone(), self.blue.clone()])
    }

    fn from_vec3<V: Vector<Scalar = Scalar> + VectorFixedDimensions<3> + Vector3DAccessor>(
        vec: &V,
    ) -> Self {
        ColorType::<Scalar>::new(vec.x().clone(), vec.y().clone(), vec.z().clone())
    }
}

#[inline(always)]
pub fn maximize_value<S: SimdValueRealSimplified>(color: ColorType<S>) -> ColorType<S> {
    let x: Srgb<S> = color.into_encoding();
    let mut y: Hsv<_, S> = x.into_color();
    y.value = <S as palette::num::One>::one();

    IntoColor::<Srgb<S>>::into_color(y).into_linear()
}
