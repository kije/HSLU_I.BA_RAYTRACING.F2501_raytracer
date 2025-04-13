use crate::geometry::{Ray, SphereData};
use crate::helpers::Splatable;
use crate::raytracing::RayIntersectionCandidate;
use crate::scalar_traits::LightScalar;
use crate::scene::{Scene, SceneObject};
use crate::vector::{NormalizableVector, SimdCapableVector, Vector};
use crate::vector_traits::{RenderingVector, SimdRenderingVector};
use num_traits::Float;
use palette::bool_mask::{BoolMask, HasBoolMask, LazySelect};
use simba::scalar::SubsetOf;
use simba::simd::{SimdBool, SimdValue};

pub(crate) type SingleValueVectorScalar<V> =
    <<V as SimdCapableVector>::SingleValueVector as Vector>::Scalar;

struct Raytracer;

impl Raytracer {
    fn find_nearest_intersection<'a, V>(
        ray: &Ray<V>,
        scene: &'a Scene,
    ) -> Option<RayIntersectionCandidate<V::Scalar, SphereData<V>>>
    where
        V: 'a + SimdRenderingVector,
        <V::Scalar as SimdValue>::Element: SubsetOf<V::Scalar>,
        V::Scalar: LightScalar<SimdBool: SimdBool + BoolMask> + Splatable<V::Scalar>,
        <V::Scalar as SimdValue>::Element: Float + Copy,
        <V::Scalar as HasBoolMask>::Mask: LazySelect<V::Scalar>,
        <V::Scalar as SimdValue>::SimdBool: SimdValue<Element = bool>,
        SingleValueVectorScalar<V>: LightScalar + SubsetOf<V::Scalar>,
        V::SingleValueVector: RenderingVector + NormalizableVector,
    {
        spheres
            .clone()
            .into_iter()
            .fold(None, |previous_intersection, sphere| {
                let sphere = SphereData::<V>::splat(sphere);
                let new_intersection = sphere.check_intersection(ray, sphere);

                // If no intersection with current object, keep previous result
                if new_intersection.valid_mask.none() {
                    return previous_intersection;
                }

                // If no previous intersection, use current
                let Some(previous_intersection) = previous_intersection else {
                    return Some(new_intersection);
                };

                // If previous has no valid intersections, use current
                if previous_intersection.valid_mask.none() {
                    return Some(new_intersection);
                }

                // Compare distances
                let new_is_nearer = previous_intersection.t.simd_ge(new_intersection.t);

                // Simple cases: one is entirely closer than the other
                if new_is_nearer.none() {
                    return Some(previous_intersection);
                } else if new_is_nearer.all() {
                    return Some(new_intersection);
                }

                // Complex case: merge intersections based on which is closer for each lane
                Self::merge_intersections(&previous_intersection, &new_intersection, new_is_nearer)
            })
    }

    // Merge two intersections based on which one is closer for each SIMD lane
    fn merge_intersections<V>(
        previous: &RayIntersectionCandidate<V::Scalar, SphereData<V>>,
        new: &RayIntersectionCandidate<V::Scalar, SphereData<V>>,
        new_is_nearer: <crate::vector::Scalar as SimdValue>::SimdBool,
    ) -> Option<RayIntersectionCandidate<V::Scalar, SphereData<V>>>
    where
        V: SimdRenderingVector,
        <crate::vector::Scalar as SimdValue>::Element: SubsetOf<crate::vector::Scalar>,
        V::Scalar: LightScalar,
        <crate::vector::Scalar as HasBoolMask>::Mask: LazySelect<crate::vector::Scalar>,
        <crate::vector::Scalar as SimdValue>::SimdBool: SimdValue<Element = bool>,
        crate::renderer::raytracer_renderer::SingleValueVectorScalar<V>:
            LightScalar + SubsetOf<V::Scalar>,
        crate::vector::SingleValueVector: RenderingVector + NormalizableVector,
    {
        let previous_valid = previous.valid_mask;
        let new_valid = new.valid_mask;

        // Compute the mask for picking previous intersection values
        let pick_old_mask =
            (previous_valid & !new_valid) | (previous_valid & new_valid & !new_is_nearer);

        // Blend fields using the mask
        let merged_t = previous.t.select(pick_old_mask.clone(), new.t);
        let merged_payload = SphereData::<V>::blend(pick_old_mask, &previous.payload, &new.payload);
        let merged_valid = previous_valid | new_valid;

        Some(RayIntersectionCandidate::new(
            merged_t,
            merged_payload,
            merged_valid,
        ))
    }
}
