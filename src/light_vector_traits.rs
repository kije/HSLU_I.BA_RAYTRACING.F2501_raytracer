use crate::helpers::ColorType;
use crate::vector::Vector;
use simba::simd::SimdValue;

/// A simple trait that provides just the scalar operations needed for light calculations
/// without complex trait bounds
pub trait LightVectorOps: Sized {
    type Vector: Vector;
    type Scalar: SimdValue;

    // Essential scalar operations
    fn zero() -> Self::Scalar;
    fn one() -> Self::Scalar;
    fn from_f32(val: f32) -> Self::Scalar;

    // Vector operations
    fn dot(v1: Self::Vector, v2: Self::Vector) -> Self::Scalar;
    fn mag(v: Self::Vector) -> Self::Scalar;
    fn subtract(v1: Self::Vector, v2: Self::Vector) -> Self::Vector;

    // SIMD operations
    fn is_greater_than(
        scalar: Self::Scalar,
        other: Self::Scalar,
    ) -> <Self::Scalar as SimdValue>::SimdBool;
    fn select(
        condition: <Self::Scalar as SimdValue>::SimdBool,
        if_true: Self::Scalar,
        if_false: Self::Scalar,
    ) -> Self::Scalar;
    fn clamp(value: Self::Scalar, min: Self::Scalar, max: Self::Scalar) -> Self::Scalar;

    // Color operations
    fn blend_colors(
        mask: <Self::Scalar as SimdValue>::SimdBool,
        true_color: ColorType<Self::Scalar>,
        false_color: ColorType<Self::Scalar>,
    ) -> ColorType<Self::Scalar>;

    // Math operations
    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
    fn div(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
    fn negate(a: Self::Scalar) -> Self::Scalar;
}

// Concrete implementations for the types we need
pub struct Vec3Helper;
pub struct Vec3x4Helper;
pub struct Vec3x8Helper;

// Implementation for the standard Vec3
impl LightVectorOps for Vec3Helper {
    type Vector = ultraviolet::Vec3;
    type Scalar = f32;

    fn zero() -> Self::Scalar {
        0.0
    }
    fn one() -> Self::Scalar {
        1.0
    }
    fn from_f32(val: f32) -> Self::Scalar {
        val
    }

    fn dot(v1: Self::Vector, v2: Self::Vector) -> Self::Scalar {
        v1.dot(v2)
    }

    fn mag(v: Self::Vector) -> Self::Scalar {
        v.mag()
    }

    fn subtract(v1: Self::Vector, v2: Self::Vector) -> Self::Vector {
        v1 - v2
    }

    fn is_greater_than(
        scalar: Self::Scalar,
        other: Self::Scalar,
    ) -> <Self::Scalar as SimdValue>::SimdBool {
        scalar > other
    }

    fn select(
        condition: <Self::Scalar as SimdValue>::SimdBool,
        if_true: Self::Scalar,
        if_false: Self::Scalar,
    ) -> Self::Scalar {
        if condition { if_true } else { if_false }
    }

    fn clamp(value: Self::Scalar, min: Self::Scalar, max: Self::Scalar) -> Self::Scalar {
        value.clamp(min, max)
    }

    fn blend_colors(
        mask: <Self::Scalar as SimdValue>::SimdBool,
        true_color: ColorType<Self::Scalar>,
        false_color: ColorType<Self::Scalar>,
    ) -> ColorType<Self::Scalar> {
        if mask { true_color } else { false_color }
    }

    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a + b
    }
    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a * b
    }
    fn div(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a / b
    }
    fn negate(a: Self::Scalar) -> Self::Scalar {
        -a
    }
}

// Similar implementations could be added for Vec3x4Helper and Vec3x8Helper
// with SIMD-specific operations
