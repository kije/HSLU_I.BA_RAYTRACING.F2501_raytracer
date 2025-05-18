use crate::helpers::Splatable;
use crate::matrix::{MatrixFixedDimensions, MatrixOperations};
use crate::simd_compat::SimdValueRealSimplified;
use crate::vector::{
    NormalizableVector, ReflectableVector, RefractableVector, RotatableVector, SimdCapableVector,
    Vector, Vector3DAccessor, Vector3DOperations, VectorAssociations, VectorFixedDimensions,
    VectorLerp, VectorOperations,
};
use simba::scalar::{SubsetOf, SupersetOf};
use simba::simd::SimdValue;
use std::ops::{Add, Div, Mul, Neg, Sub};
use ultraviolet::Lerp;

/// A basic vector trait combining common vector operations
pub trait BaseVector:
    Vector
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
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
        + Div<Self, Output = Self>
        + Copy
        + VectorOperations
        + Sync
{
}

/// A trait for vectors with common operations for rendering
pub trait RenderingVector:
    BaseVector<Scalar: SimdValueRealSimplified>
    + NormalizableVector
    + ReflectableVector
    + RefractableVector
    + RotatableVector
    + crate::float_ext::AbsDiffEq<Epsilon = Self, Output = <Self::Scalar as SimdValue>::SimdBool>
    + Vector3DOperations
    + Neg<Output = Self>
    + Vector3DAccessor
    + VectorFixedDimensions<3>
    + VectorLerp
    + VectorAssociations<Matrix: MatrixFixedDimensions<3> + MatrixOperations>
{
}

// Blanket implementation
impl<V> RenderingVector for V where
    V: BaseVector<Scalar: SimdValueRealSimplified>
        + NormalizableVector
        + ReflectableVector
        + RefractableVector
        + RotatableVector
        + crate::float_ext::AbsDiffEq<Epsilon = Self, Output = <Self::Scalar as SimdValue>::SimdBool>
        + Vector3DOperations
        + Neg<Output = V>
        + Vector3DAccessor
        + VectorFixedDimensions<3>
        + VectorLerp
        + VectorAssociations<Matrix: MatrixFixedDimensions<3> + MatrixOperations>
{
}

/// A trait for SIMD-compatible vectors with enhanced features needed for rendering,
/// including mask operations that support lazy selection
pub trait SimdRenderingVector:
RenderingVector + SimdCapableVector<Scalar: SimdValueRealSimplified<Element: SubsetOf<<<Self as SimdCapableVector>::SingleValueVector as Vector>::Scalar >, SimdBool: Splatable<<<<Self as SimdCapableVector>::SingleValueVector as Vector>::Scalar as SimdValue>::SimdBool>>, SingleValueVector: RenderingVector>
{
}

// Blanket implementation
impl<V> SimdRenderingVector for V where
    V: RenderingVector + SimdCapableVector<Scalar: SimdValueRealSimplified<Element: SubsetOf<<<Self as SimdCapableVector>::SingleValueVector as Vector>::Scalar >, SimdBool: Splatable<<<<Self as SimdCapableVector>::SingleValueVector as Vector>::Scalar as SimdValue>::SimdBool>>, SingleValueVector: RenderingVector>
{
}
