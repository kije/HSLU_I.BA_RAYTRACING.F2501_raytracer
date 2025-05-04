use crate::geometry::{GeometryCollection, Ray, RenderGeometry};
use crate::helpers::Splatable;
use crate::raytracing::{Intersectable, SurfaceInteraction, TransmissionProperties};
use crate::vector::{NormalizableVector, Vector};
use crate::vector_traits::{RenderingVector, SimdRenderingVector};
use num_traits::One;
use num_traits::Zero;
use simba::simd::SimdPartialOrd;
use simba::simd::{SimdBool, SimdValue};

pub struct Raytracer;

impl Raytracer {
    pub fn has_any_intersection<V>(
        from: V,
        direction: V,
        check_objects: &GeometryCollection<V::SingleValueVector>,
        max_distance: V::Scalar,
    ) -> <V::Scalar as SimdValue>::SimdBool
    where
        V: SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector>,
    {
        let ray = Ray::<V>::new_with_mask(
            from,
            direction,
            TransmissionProperties::default(),
            <<<V as Vector>::Scalar as SimdValue>::SimdBool as SimdValue>::splat(true),
        );

        let mut has_intersection =
            <<<V as Vector>::Scalar as SimdValue>::SimdBool as SimdValue>::splat(false);

        for object in check_objects.get_all() {
            let object = RenderGeometry::<V>::splat(object);

            // Check intersection
            if let Some(interaction) = object.intersect(&ray) {
                has_intersection = has_intersection
                    | (interaction.valid_mask
                        & interaction.distance.simd_le(max_distance)
                        & interaction.distance.simd_ge(<V as Vector>::Scalar::zero()));

                if has_intersection.all() {
                    break;
                }
            }
        }

        has_intersection
    }

    /// Cast a ray in the scene and returns its intersection with the closest object.
    ///
    /// `IS_ANTIALIASING_RAY` param may be set to true if the ray is used for antialiasing and might be used to skip complex calculations that are not relevant for antialiasing
    ///
    pub fn cast_ray<const IS_ANTIALIASING_RAY: bool, V>(
        from: V,
        direction: V,
        start_refraction_index: V::Scalar,
        check_objects: &GeometryCollection<V::SingleValueVector>,
    ) -> Option<(Ray<V>, SurfaceInteraction<V>)>
    where
        V: SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector>,
    {
        let ray = Ray::<V>::new_with_mask(
            from,
            direction,
            TransmissionProperties::new(<V as Vector>::Scalar::one(), start_refraction_index),
            <<<V as Vector>::Scalar as SimdValue>::SimdBool as SimdValue>::splat(true),
        );

        let mut nearest_interaction: Option<SurfaceInteraction<V>> = None;

        for object in check_objects.get_all() {
            let object = RenderGeometry::<V>::splat(object);

            // Check intersection
            if let Some(interaction) = object.intersect(&ray) {
                if interaction.valid_mask.none() {
                    continue;
                }

                // Update nearest interaction
                nearest_interaction = match nearest_interaction {
                    None => Some(interaction),
                    Some(ref current) => {
                        let closer =
                            interaction.distance.simd_le(current.distance) & interaction.valid_mask;

                        if closer.none() {
                            Some(SurfaceInteraction::blend(
                                current.valid_mask,
                                current,
                                &interaction,
                            ))
                        } else if closer.all() {
                            Some(interaction)
                        } else {
                            let pick_old_mask = (interaction.valid_mask & !current.valid_mask)
                                | (interaction.valid_mask & current.valid_mask & closer);

                            Some(SurfaceInteraction::blend(
                                pick_old_mask,
                                &interaction,
                                current,
                            ))
                        }
                    }
                };
            }
        }

        nearest_interaction.map(|i| (ray, i))
    }
}
