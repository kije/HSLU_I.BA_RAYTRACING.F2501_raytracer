use crate::helpers::ColorType;
use crate::vector::Vector;

/// Trait for objects that can be lit
pub trait Lightable<V>
where
    V: Vector,
{
    /// Get the material color at the given point
    fn get_material_color_at(&self, point: V) -> ColorType<V::Scalar>;

    /// Get the surface normal at the given point
    fn get_surface_normal_at(&self, point: V) -> V;
}
