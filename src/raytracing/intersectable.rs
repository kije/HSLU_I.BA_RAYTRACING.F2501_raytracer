use crate::geometry::Ray;
use crate::raytracing::surface_interaction::SurfaceInteraction;
use crate::vector::Vector;
/// Trait for objects that can be intersected by rays
pub trait Intersectable<V: Vector> {
    /// Check if a ray intersects this object and return surface interaction
    fn intersect(&self, ray: &Ray<V>) -> Option<SurfaceInteraction<V>>;
}
