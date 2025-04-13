use crate::helpers::Splatable;
use crate::scalar::Scalar;
use num_traits::{Float, FromPrimitive, Num, NumAssignOps, NumOps, One, Zero};
use palette::bool_mask::{BoolMask, HasBoolMask, LazySelect, Select};
use simba::scalar::{SubsetOf, SupersetOf};
use simba::simd::{SimdBool, SimdPartialOrd, SimdRealField, SimdSigned, SimdValue};
use std::fmt::Debug;
use std::ops::Neg;

pub(crate) trait SimdValueBoolExt: SimdValue<SimdBool: SimdValue<Element = bool>> {
    fn create_mask(mask: <Self::SimdBool as SimdValue>::Element) -> Self::SimdBool;
}

impl<T> SimdValueBoolExt for T
where
    T: SimdValue<SimdBool: SimdValue<Element = bool>>,
{
    fn create_mask(mask: <Self::SimdBool as SimdValue>::Element) -> Self::SimdBool {
        <Self::SimdBool as SimdValue>::splat(mask)
    }
}

pub(crate) trait SimdValueSimplified:
    Scalar
    + SubsetOf<Self>
    + SupersetOf<<Self as SimdValue>::Element>
    + Default
    + HasBoolMask<Mask = <Self as SimdValue>::SimdBool>
    + FromPrimitive
    + SimdValue<
        Element: SubsetOf<Self> + Copy + Send + Sync,
        SimdBool: SimdValue<Element = bool, SimdBool = <Self as SimdValue>::SimdBool>
                      + BoolMask
                      + Debug
                      + PartialEq
                      + Default
                      + SimdBool
                      + Send
                      + Sync
                      + Select<Self>
                      + LazySelect<Self>,
    > + NumAssignOps<Self>
    + NumOps<Self>
    + Copy
    + Zero
    + One
    + SimdValueBoolExt
    + Num
    + Send
    + Sync
    + SimdPartialOrd
    + PartialEq
    + Splatable<Self::Element>
{
}

impl<V> SimdValueSimplified for V where
    V: Scalar
        + SubsetOf<Self>
        + SupersetOf<<Self as SimdValue>::Element>
        + Default
        + HasBoolMask<Mask = <Self as SimdValue>::SimdBool>
        + FromPrimitive
        + SimdValue<
            Element: SubsetOf<Self> + Copy + Send + Sync,
            SimdBool: SimdValue<Element = bool, SimdBool = <Self as SimdValue>::SimdBool>
                          + BoolMask
                          + Debug
                          + PartialEq
                          + Default
                          + SimdBool
                          + Send
                          + Sync
                          + Select<Self>
                          + LazySelect<Self>,
        > + NumAssignOps<Self>
        + NumOps<Self>
        + Copy
        + Zero
        + One
        + SimdValueBoolExt
        + Send
        + Sync
        + Num
        + SimdPartialOrd
        + PartialEq
        + Splatable<Self::Element>
{
}

pub(crate) trait SimdValueSignedSimplified:
    SimdValueSimplified + Neg<Output = Self> + SimdSigned
{
}

impl<V> SimdValueSignedSimplified for V where
    V: SimdValueSimplified + Neg<Output = Self> + SimdSigned
{
}

pub(crate) trait SimdValueRealSimplified:
    SimdValueSignedSimplified<Element: Float>
    + SimdRealField
    + palette::num::Real
    + palette::num::Sqrt
    + palette::num::Zero
    + palette::num::One
    + palette::num::Arithmetics
    + palette::num::Clamp
    + palette::num::Abs
    + palette::num::PartialCmp
    + palette::num::MinMax
{
}

impl<V> SimdValueRealSimplified for V where
    V: SimdValueSignedSimplified<Element: Float>
        + SimdRealField
        + palette::num::Real
        + palette::num::Sqrt
        + palette::num::Zero
        + palette::num::One
        + palette::num::Arithmetics
        + palette::num::Clamp
        + palette::num::Abs
        + palette::num::PartialCmp
        + palette::num::MinMax
{
}
