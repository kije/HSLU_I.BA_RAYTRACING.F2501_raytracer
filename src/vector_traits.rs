use crate::scalar_traits::LightScalar;
use crate::vector::{
    CommonVecOperations, CommonVecOperationsFloat, CommonVecOperationsReflectable,
    CommonVecOperationsSimdOperations, Vector,
};
use palette::bool_mask::{HasBoolMask, LazySelect};
use simba::scalar::SubsetOf;
use simba::simd::SimdValue;
use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

/// A basic vector trait combining common vector operations
pub(crate) trait VectorBasic:
    Vector
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Copy
    + CommonVecOperations
{
}

// Blanket implementation
impl<V> VectorBasic for V where
    V: Vector
        + Add<Self, Output = Self>
        + Sub<Self, Output = Self>
        + Mul<Self, Output = Self>
        + Copy
        + CommonVecOperations
{
}

/// A trait for 3D vectors with common operations for rendering
pub(crate) trait Vector3D:
    VectorBasic + CommonVecOperationsFloat + CommonVecOperationsReflectable
where
    Self::Scalar: LightScalar,
{
}

// Blanket implementation
impl<V> Vector3D for V
where
    V: VectorBasic + CommonVecOperationsFloat + CommonVecOperationsReflectable,
    V::Scalar: LightScalar,
{
}

/// A trait for SIMD-compatible vectors with enhanced features needed for rendering
pub(crate) trait SimdVector: Vector3D + CommonVecOperationsSimdOperations
where
    Self::Scalar: LightScalar,
    // Basic SIMD vector requirements
    <<Self as Vector>::Scalar as SimdValue>::Element: SubsetOf<<Self as Vector>::Scalar>,
    <<Self as Vector>::Scalar as HasBoolMask>::Mask: LazySelect<Self::Scalar>,
    <<Self as Vector>::Scalar as SimdValue>::SimdBool: Debug + SimdValue<Element = bool>,
    // Make sure single value vector is Vector3D
    <Self as CommonVecOperationsSimdOperations>::SingleValueVector: Vector3D,
    // Make sure scalar types are compatible
    <<Self as CommonVecOperationsSimdOperations>::SingleValueVector as Vector>::Scalar:
        LightScalar + SubsetOf<Self::Scalar>,
{
}

// Blanket implementation
impl<V> SimdVector for V
where
    V: Vector3D + CommonVecOperationsSimdOperations,
    V::Scalar: LightScalar,
    <<V as Vector>::Scalar as SimdValue>::Element: SubsetOf<<V as Vector>::Scalar>,
    <<V as Vector>::Scalar as HasBoolMask>::Mask: LazySelect<V::Scalar>,
    <<V as Vector>::Scalar as SimdValue>::SimdBool: Debug + SimdValue<Element = bool>,
    <V as CommonVecOperationsSimdOperations>::SingleValueVector: Vector3D,
    <<V as CommonVecOperationsSimdOperations>::SingleValueVector as Vector>::Scalar:
        LightScalar + SubsetOf<V::Scalar>,
{
}
