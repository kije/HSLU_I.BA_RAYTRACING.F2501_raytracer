use crate::raytracing::Intersectable;
use crate::simd_compat::SimdValueRealSimplified;
use crate::vector::Vector;

//pub mod plane;
pub mod sphere;
pub mod triangle;

pub trait BasicGeometry<V: Vector<Scalar: SimdValueRealSimplified>>: Intersectable<V> {}
