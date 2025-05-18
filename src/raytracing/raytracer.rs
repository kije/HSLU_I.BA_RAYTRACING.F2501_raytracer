use crate::color::ColorSimdExt;
use crate::color_traits::LightCompatibleColor;
use crate::float_ext::AbsDiffEq;
use crate::geometry::{GeometryCollection, Ray, RenderGeometry};
use crate::helpers::{ColorType, Splatable};
use crate::raytracing::{Intersectable, SurfaceInteraction, TransmissionProperties};
use crate::vector::{NormalizableVector, Vector};
use crate::vector_traits::{RenderingVector, SimdRenderingVector};
use num_traits::Zero;
use num_traits::{One, one};
use palette::blend::Premultiply;
use simba::simd::SimdPartialOrd;
use simba::simd::{SimdBool, SimdValue};

pub struct Raytracer;

pub struct IntersectionTest<S: SimdValue> {
    pub has_intersection: S::SimdBool,
    pub completely_occluded: S::SimdBool,
    pub combined_opacity: S,
    pub color_filter: ColorType<S>,
}
impl Raytracer {
    pub fn has_any_intersection<V>(
        from: V,
        direction: V,
        check_objects: &GeometryCollection<V::SingleValueVector>,
        max_distance: V::Scalar,
    ) -> IntersectionTest<V::Scalar>
    where
        V: SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector>,
        ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
    {
        let one = <<V as Vector>::Scalar as One>::one();
        let zero = <<V as Vector>::Scalar as Zero>::zero();
        let ray = Ray::<V>::new_with_mask(
            from,
            direction,
            TransmissionProperties::default(),
            <<<V as Vector>::Scalar as SimdValue>::SimdBool as SimdValue>::splat(true),
        );

        let mut has_intersection: <<V as Vector>::Scalar as SimdValue>::SimdBool = false.into(); //<<<V as Vector>::Scalar as SimdValue>::SimdBool as SimdValue>::splat(false);
        let mut completely_occluded: <<V as Vector>::Scalar as SimdValue>::SimdBool = false.into(); //<<<V as Vector>::Scalar as SimdValue>::SimdBool as SimdValue>::splat(false);
        let mut opacity = one;
        let mut filter = ColorSimdExt::one();

        for object in check_objects.get_all() {
            let object = RenderGeometry::<V>::splat(object);

            // Check intersection
            if let Some(interaction) = object.intersect(&ray) {
                let interaction_has_intersection =
                    interaction.valid_mask & interaction.distance.simd_le(max_distance);
                &interaction.distance.simd_ge(zero);

                let (_, transmittance) = interaction.material.compute_fresnel(
                    interaction.normal,
                    -ray.direction,
                    ray.transmission.refraction_index().simd_unwrap_or(|| one),
                );

                let transmittance = ColorSimdExt::blend(
                    interaction.material.transmission.mask(),
                    &transmittance,
                    &ColorSimdExt::zero(),
                );

                let interaction_opacity = interaction
                    .material
                    .transmission
                    .opacity()
                    .simd_unwrap_or(|| zero)
                    * transmittance.red; // transmittance.red = transmittance.green = transmittance.blue
                opacity = (opacity - (one - interaction_opacity))
                    .simd_clamp(zero, one)
                    .select(interaction_has_intersection, opacity);
                let interaction_completely_occluded = (interaction_has_intersection
                    & !interaction.material.transmission.opacity().mask())
                    & opacity.abs_diff_eq_default(&zero);
                has_intersection = has_intersection | interaction_has_intersection;
                completely_occluded = (completely_occluded | interaction_completely_occluded)
                    .select(interaction_has_intersection, completely_occluded);

                let material_absorbition = interaction.material.absorption();

                // fixme use interaction.material.absorption()
                filter = ColorSimdExt::blend(
                    interaction_has_intersection,
                    &(filter - material_absorbition),
                    &filter,
                );

                if completely_occluded.all() && has_intersection.all() {
                    break;
                }
            }
        }

        IntersectionTest {
            has_intersection,
            completely_occluded,
            combined_opacity: opacity,
            color_filter: filter,
        }
    }
    // pub fn has_any_intersection<V>(
    //     from: V,
    //     direction: V,
    //     check_objects: &GeometryCollection<V::SingleValueVector>,
    //     max_distance: V::Scalar,
    // ) -> IntersectionTest<V::Scalar>
    // where
    //     V: SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector>,
    //     ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
    // {
    //     let one = <<V as Vector>::Scalar as One>::one();
    //     let zero = <<V as Vector>::Scalar as Zero>::zero();
    //     let ray = Ray::<V>::new_with_mask(
    //         from,
    //         direction,
    //         TransmissionProperties::default(),
    //         <<<V as Vector>::Scalar as SimdValue>::SimdBool as SimdValue>::splat(true),
    //     );
    //
    //     let mut has_intersection: <<V as Vector>::Scalar as SimdValue>::SimdBool = false.into(); //<<<V as Vector>::Scalar as SimdValue>::SimdBool as SimdValue>::splat(false);
    //     let mut completely_occluded: <<V as Vector>::Scalar as SimdValue>::SimdBool = false.into(); //<<<V as Vector>::Scalar as SimdValue>::SimdBool as SimdValue>::splat(false);
    //     let mut opacity = one;
    //     let mut filter = ColorSimdExt::one();
    //
    //     for object in check_objects.get_all() {
    //         let object = RenderGeometry::<V>::splat(object);
    //
    //         // Check intersection
    //         if let Some(interaction) = object.intersect(&ray) {
    //             has_intersection = has_intersection
    //                 | (interaction.valid_mask
    //                     & interaction.distance.simd_le(max_distance)
    //                     & interaction.distance.simd_gt(<V as Vector>::Scalar::zero()));
    //
    //             // fixme replace with implementation above
    //             opacity = zero.select(has_intersection, opacity);
    //
    //             if has_intersection.all() {
    //                 break;
    //             }
    //         }
    //     }
    //
    //     IntersectionTest {
    //         has_intersection,
    //         completely_occluded,
    //         combined_opacity: opacity,
    //         color_filter: filter,
    //     }
    // }

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
