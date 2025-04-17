use crate::matrix::{MatrixFixedDimensions, MatrixOperations};
use crate::scalar_traits::LightScalar;
use crate::vector::{
    NormalizableVector, ReflectableVector, SimdCapableVector, Vector, Vector3DAccessor,
    Vector3DOperations, VectorAssociations, VectorOperations,
};
use std::ops::{Add, Mul, Neg, Sub};

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
    BaseVector<Scalar: LightScalar>
    + NormalizableVector
    + ReflectableVector
    + Vector3DOperations
    + Neg<Output = Self>
    + Vector3DAccessor
    + VectorAssociations<Matrix: MatrixFixedDimensions<3> + MatrixOperations>
{
}

// Blanket implementation
impl<V> RenderingVector for V where
    V: BaseVector<Scalar: LightScalar>
        + NormalizableVector
        + ReflectableVector
        + Vector3DOperations
        + Neg<Output = V>
        + Vector3DAccessor
        + VectorAssociations<Matrix: MatrixFixedDimensions<3> + MatrixOperations>
{
}

/// A trait for SIMD-compatible vectors with enhanced features needed for rendering,
/// including mask operations that support lazy selection
pub(crate) trait SimdRenderingVector:
    RenderingVector + SimdCapableVector<SingleValueVector: RenderingVector>
{
}

// Blanket implementation
impl<V> SimdRenderingVector for V where
    V: RenderingVector + SimdCapableVector<SingleValueVector: RenderingVector>
{
}
