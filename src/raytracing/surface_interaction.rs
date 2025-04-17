use crate::color::ColorSimdExt;
use crate::helpers::ColorType;
use crate::raytracing::material::Material;
use crate::scene::Lightable;
use crate::vector::{SimdCapableVector, Vector};
use crate::vector_traits::RenderingVector;
use simba::simd::SimdValue;

/// Represents all the data needed for shading at an intersection point
/// This decouples the geometry type from the shading calculations
#[derive(Clone, Debug)]
pub struct SurfaceInteraction<V: Vector> {
    /// The point of intersection in 3D space
    pub point: V,

    /// Surface normal at the intersection point
    pub normal: V,

    // fixme for reflexion/refraction, we likely need an incident_vector
    /// Distance from ray origin to intersection point
    pub distance: V::Scalar,

    /// Surface material properties
    pub material: Material<V::Scalar>,

    /// Valid mask for SIMD operations
    pub valid_mask: <<V as Vector>::Scalar as SimdValue>::SimdBool,
}

impl<V: RenderingVector + SimdCapableVector> SurfaceInteraction<V> {
    /// Create a new surface interaction
    pub fn new(
        point: V,
        normal: V,
        distance: V::Scalar,
        material: Material<V::Scalar>,
        valid_mask: <<V as Vector>::Scalar as SimdValue>::SimdBool,
    ) -> Self {
        Self {
            point,
            normal,
            distance,
            material,
            valid_mask,
        }
    }

    /// Blend two surface interactions based on a mask
    pub fn blend(mask: <<V as Vector>::Scalar as SimdValue>::SimdBool, a: &Self, b: &Self) -> Self {
        Self {
            point: V::blend(mask.clone(), a.point, b.point),
            normal: V::blend(mask.clone(), a.normal, b.normal),
            distance: a.distance.select(mask.clone(), b.distance),
            material: Material::blend(mask.clone(), &a.material, &b.material),
            valid_mask: (a.valid_mask & mask) | (b.valid_mask & !mask),
        }
    }
}

impl<V: Vector> Lightable<V> for SurfaceInteraction<V> {
    #[inline(always)]
    fn get_material_color_at(&self, point: V) -> ColorType<V::Scalar> {
        debug_assert!(point == self.point);
        self.material.color
    }

    #[inline(always)]
    fn get_surface_normal_at(&self, point: V) -> V {
        debug_assert!(point == self.point);
        self.normal.clone()
    }
}
