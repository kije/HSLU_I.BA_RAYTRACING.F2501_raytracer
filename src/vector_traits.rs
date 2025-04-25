use crate::helpers::Splatable;
use crate::matrix::{MatrixFixedDimensions, MatrixOperations};
use crate::simd_compat::SimdValueRealSimplified;
use crate::vector::{
    NormalizableVector, ReflectableVector, RefractableVector, SimdCapableVector, Vector,
    Vector3DAccessor, Vector3DOperations, VectorAssociations, VectorOperations,
};
use simba::simd::SimdValue;
use std::ops::{Add, Mul, Neg, Sub};

/// A basic vector trait combining common vector operations
pub trait BaseVector:
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
pub trait RenderingVector:
    BaseVector<Scalar: SimdValueRealSimplified>
    + NormalizableVector
    + ReflectableVector
    + RefractableVector
    + crate::float_ext::AbsDiffEq<Output = <Self::Scalar as SimdValue>::SimdBool>
    + Vector3DOperations
    + Neg<Output = Self>
    + Vector3DAccessor
    + VectorAssociations<Matrix: MatrixFixedDimensions<3> + MatrixOperations>
{
}

// Blanket implementation
impl<V> RenderingVector for V where
    V: BaseVector<Scalar: SimdValueRealSimplified>
        + NormalizableVector
        + ReflectableVector
        + RefractableVector
        + crate::float_ext::AbsDiffEq<Output = <Self::Scalar as SimdValue>::SimdBool>
        + Vector3DOperations
        + Neg<Output = V>
        + Vector3DAccessor
        + VectorAssociations<Matrix: MatrixFixedDimensions<3> + MatrixOperations>
{
}

/// A trait for SIMD-compatible vectors with enhanced features needed for rendering,
/// including mask operations that support lazy selection
pub trait SimdRenderingVector:
    RenderingVector + SimdCapableVector<Scalar:  SimdValueRealSimplified<SimdBool: Splatable<<<<Self as SimdCapableVector>::SingleValueVector as Vector>::Scalar as SimdValue>::SimdBool>>, SingleValueVector: RenderingVector>
{
}

// Blanket implementation
impl<V> SimdRenderingVector for V where
    V: RenderingVector + SimdCapableVector<Scalar:  SimdValueRealSimplified<SimdBool: Splatable<<<<Self as SimdCapableVector>::SingleValueVector as Vector>::Scalar as SimdValue>::SimdBool>>,SingleValueVector: RenderingVector>
{
}
