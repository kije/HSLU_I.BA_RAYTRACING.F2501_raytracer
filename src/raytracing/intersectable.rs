use crate::geometry::Ray;
use crate::raytracing::surface_interaction::SurfaceInteraction;
use crate::simd_compat::SimdValueRealSimplified;
use crate::vector::Vector;
/// Trait for objects that can be intersected by rays
pub trait Intersectable<V: Vector<Scalar: SimdValueRealSimplified>> {
    /// Check if a ray intersects this object and return surface interaction
    fn intersect(&self, ray: &Ray<V>) -> Option<SurfaceInteraction<V>>;
}
