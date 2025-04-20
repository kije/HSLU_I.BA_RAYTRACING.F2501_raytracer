use crate::geometry::{GeometryCollection, Ray, RenderGeometry};
use crate::helpers::Splatable;
use crate::raytracing::{Intersectable, SurfaceInteraction};
use crate::vector::{NormalizableVector, SimdCapableVector, Vector};
use crate::vector_traits::{RenderingVector, SimdRenderingVector};
use num_traits::Zero;
use simba::simd::SimdPartialOrd;
use simba::simd::{SimdBool, SimdValue};

pub type SingleValueVectorScalar<V> =
    <<V as SimdCapableVector>::SingleValueVector as Vector>::Scalar;

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
            <<V as Vector>::Scalar as SimdValue>::SimdBool::splat(true),
        );

        let mut has_intersection = <<V as Vector>::Scalar as SimdValue>::SimdBool::splat(false);

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

    pub fn cast_ray<V>(
        from: V,
        direction: V,
        check_objects: &GeometryCollection<V::SingleValueVector>,
    ) -> Option<(Ray<V>, SurfaceInteraction<V>)>
    where
        V: SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector>,
    {
        let ray = Ray::<V>::new_with_mask(
            from,
            direction,
            <<V as Vector>::Scalar as SimdValue>::SimdBool::splat(false),
        );

        // fixme why?
        // For SIMD rays, we process each geometry type separately and find the nearest
        let mut nearest_interaction: Option<SurfaceInteraction<V>> = None;

        let grouped_by_kind = check_objects
            .keys()
            .map(|&kind| check_objects.get_by_kind(kind));

        for objects in grouped_by_kind {
            for object in objects {
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
                            let closer = interaction.distance.simd_le(current.distance)
                                & interaction.valid_mask;

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
        }

        nearest_interaction.map(|i| (ray, i))
    }
}
