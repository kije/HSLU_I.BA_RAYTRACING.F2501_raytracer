use crate::scalar_traits::LightScalar;
use crate::vector::{
    NormalizableVector, ReflectableVector, SimdCapableVector, Vector, VectorOperations,
};
use palette::bool_mask::{HasBoolMask, LazySelect};
use simba::scalar::SubsetOf;
use simba::simd::SimdValue;
use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

/// A basic vector trait combining common vector operations
pub(crate) trait BaseVector:
    Vector
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Copy
    + VectorOperations
    + Sync
{
}

// Blanket implementation
impl<V> BaseVector for V where
    V: Vector
        + Add<Self, Output = Self>
        + Sub<Self, Output = Self>
        + Mul<Self, Output = Self>
        + Copy
        + VectorOperations
        + Sync
{
}

/// A trait for vectors with common operations for rendering
pub(crate) trait RenderingVector:
    BaseVector + NormalizableVector + ReflectableVector
where
    Self::Scalar: LightScalar,
{
}

// Blanket implementation
impl<V> RenderingVector for V
where
    V: BaseVector + NormalizableVector + ReflectableVector,
    V::Scalar: LightScalar,
{
}

/// A trait for SIMD-compatible vectors with enhanced features needed for rendering,
/// including mask operations that support lazy selection
pub(crate) trait SimdRenderingVector: RenderingVector + SimdCapableVector
where
    Self::Scalar: LightScalar,
    // Basic SIMD vector requirements
    <<Self as Vector>::Scalar as SimdValue>::Element: SubsetOf<<Self as Vector>::Scalar>,
    <<Self as Vector>::Scalar as HasBoolMask>::Mask: LazySelect<Self::Scalar>,
    <<Self as Vector>::Scalar as SimdValue>::SimdBool: Debug + SimdValue<Element = bool>,
    // Make sure single value vector is RenderingVector
    <Self as SimdCapableVector>::SingleValueVector: RenderingVector,
    // Make sure scalar types are compatible
    <<Self as SimdCapableVector>::SingleValueVector as Vector>::Scalar:
        LightScalar + SubsetOf<Self::Scalar>,
{
}

// Blanket implementation
impl<V> SimdRenderingVector for V
where
    V: RenderingVector + SimdCapableVector,
    V::Scalar: LightScalar,
    <<V as Vector>::Scalar as SimdValue>::Element: SubsetOf<<V as Vector>::Scalar>,
    <<V as Vector>::Scalar as HasBoolMask>::Mask: LazySelect<V::Scalar>,
    <<V as Vector>::Scalar as SimdValue>::SimdBool: Debug + SimdValue<Element = bool>,
    <V as SimdCapableVector>::SingleValueVector: RenderingVector,
    <<V as SimdCapableVector>::SingleValueVector as Vector>::Scalar:
        LightScalar + SubsetOf<V::Scalar>,
{
}
