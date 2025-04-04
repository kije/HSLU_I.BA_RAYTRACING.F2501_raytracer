use crate::helpers::ColorType;
use crate::vector::Vector;
use simba::simd::SimdValue;

/// Trait for objects that can be lit
pub(crate) trait Lightable<V>
where
    V: Vector,
    V::Scalar: SimdValue,
{
    /// Get the material color at the given point
    fn get_material_color_at(&self, point: V) -> ColorType<V::Scalar>;

    /// Get the surface normal at the given point
    fn get_surface_normal_at(&self, point: V) -> V;
}
