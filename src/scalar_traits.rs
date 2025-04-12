use crate::scalar::Scalar;
use palette::bool_mask::HasBoolMask;
use simba::simd::{SimdRealField, SimdValue};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A consolidated trait for scalar types used in lighting calculations
///
/// This combines all the common requirements for scalars used in light calculations
/// to simplify trait bounds throughout the codebase.
pub trait LightScalar:
    Scalar
    + SimdValue
    + SimdRealField
    + num_traits::One
    + num_traits::Zero
    + palette::num::Real
    + palette::num::Zero
    + palette::num::One
    + palette::num::Arithmetics
    + palette::num::Clamp
    + palette::num::Sqrt
    + palette::num::Abs
    + palette::num::PartialCmp
    + HasBoolMask
    + palette::num::MinMax
    + Sub<Self, Output = Self>
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Neg<Output = Self>
    + Copy
{
}

// Blanket implementation for any type that satisfies the requirements
impl<T> LightScalar for T where
    T: Scalar
        + SimdValue
        + SimdRealField
        + num_traits::One
        + num_traits::Zero
        + palette::num::Real
        + palette::num::Zero
        + palette::num::One
        + palette::num::Arithmetics
        + palette::num::Clamp
        + palette::num::Sqrt
        + palette::num::Abs
        + palette::num::PartialCmp
        + HasBoolMask
        + palette::num::MinMax
        + Copy
{
}
