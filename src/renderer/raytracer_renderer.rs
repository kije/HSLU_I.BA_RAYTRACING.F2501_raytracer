use crate::extensions::SrgbColorConvertExt;
use crate::float_ext::AbsDiffEq;
use crate::helpers::{ColorType, Pixel, Splatable};
use crate::image_buffer::ImageBuffer;
use crate::output::OutputColorEncoder;
use crate::renderer::{RenderCoordinates, RenderCoordinatesVectorized, Renderer};
use crate::{
    AVERAGE_SCENE_FACTOR, DEFAULT_REFRACTION_INDEX, RENDER_RAY_FOCUS, SCENE_DEPTH, SCENE_HEIGHT,
    SCENE_WIDTH, WINDOW_HEIGHT, WINDOW_TO_SCENE_HEIGHT_FACTOR, WINDOW_TO_SCENE_WIDTH_FACTOR,
    WINDOW_WIDTH,
};
use itertools::{Itertools, izip};
use rayon::ThreadPool;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use simba::simd::SimdRealField;
use simba::simd::{SimdComplexField, SimdOption, WideF32x4, WideF32x8};
use std::cell::LazyCell;

// Import the consolidated traits
use crate::color_traits::LightCompatibleColor;
use crate::vector_traits::{RenderingVector, SimdRenderingVector};

use crate::vector::{
    NormalizableVector, SimdCapableVector, Vector, VectorFixedDimensions, VectorOperations,
};

use crate::color::ColorSimdExt;
use crate::geometry::{
    BoundedPlane, CompositeGeometry, GeometryCollection, HasRenderObjectId, Ray, SphereData,
    TriangleData,
};
use crate::raytracing::Raytracer;
use crate::raytracing::SurfaceInteraction;
use crate::raytracing::{Material, TransmissionProperties};
use crate::scene::{AmbientLight, Light, PointLight, Scene, SceneLightSource};
use fast_poisson::Poisson2D;
use num_traits::{One, Zero};
use palette::Mix;
use palette::blend::{Blend, Premultiply};
use palette::bool_mask::BoolMask;
use palette::cast::AsArrays;
use rand::distributions::{Distribution, Standard};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::ParallelBridge;
use simba::scalar::SupersetOf;
use simba::simd::{SimdBool, SimdPartialOrd, SimdValue};
use std::fmt::Debug;
use std::hint::{likely, unlikely};
use std::marker::PhantomData;
use std::sync::{Arc, LazyLock, RwLock};
use ultraviolet::{Rotor3, Vec3, Vec3x4, Vec3x8, f32x8};
use wide::f32x4;

const RAYTRACE_REFLECTION_MAX_DEPTH: usize = if cfg!(feature = "high_quality") {
    if cfg!(feature = "extreme_quality") {
        28
    } else {
        18
    }
} else {
    9
};

const RAYTRACE_REFRACTION_MAX_DEPTH: usize = if cfg!(feature = "high_quality") {
    if cfg!(feature = "extreme_quality") {
        25
    } else {
        17
    }
} else {
    8
};

const POINT_LIGHT_MULTIPLICATOR: usize = if cfg!(feature = "soft_shadows") {
    if cfg!(feature = "high_quality") {
        if cfg!(feature = "extreme_quality") {
            28
        } else {
            18
        }
    } else {
        8
    }
} else {
    1
};

const ANTIALIASING_SAMPLES_PER_PIXEL: usize = if cfg!(feature = "high_quality") {
    if cfg!(feature = "extreme_quality") {
        45
    } else {
        27
    }
} else {
    8
};

// Thread pool for ray processing
static RAY_PROCESSING_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    rayon::ThreadPoolBuilder::default()
        .num_threads(8) // Adjust based on available cores
        .thread_name(|i| format!("ray-thread-{}", i))
        .build()
        .expect("Failed to build ray processing thread pool")
});

// Pre-calculated antialiasing samples for faster access
static ANTIALIASING_SAMPLES: LazyLock<Vec<[f32; 2]>> = LazyLock::new(|| {
    let total_rays = ANTIALIASING_SAMPLES_PER_PIXEL.next_multiple_of(8);
    if cfg!(feature = "anti_aliasing_randomness") {
        let mut samples = vec![[0.0, 0.0]];
        samples.extend(
            Poisson2D::new()
                .with_dimensions([1.2, 1.2], (3.0 / total_rays as f32))
                .with_samples(total_rays as u32)
                .into_iter()
                .take(total_rays - 1),
        );
        samples
    } else {
        let mut samples = vec![[0.0, 0.0]];
        samples.extend(vec![[1.0, 1.0]; total_rays - 1]);
        samples
    }
});

trait SceneLightIterator<'a, V: 'a + SimdRenderingVector>:
    IntoIterator<Item: AsRef<SceneLightSource<V::SingleValueVector>>> + Clone + Sync
{
}

impl<'a, T, V: 'a + SimdRenderingVector> SceneLightIterator<'a, V> for T where
    T: IntoIterator<Item: AsRef<SceneLightSource<V::SingleValueVector>>> + Clone + Sync
{
}

#[derive(Debug, Copy, Clone, Default)]
pub struct RaytracerRenderer<C: OutputColorEncoder>(PhantomData<C>);

impl<C: OutputColorEncoder> RaytracerRenderer<C> {
    /// Does a single raytracing pass.
    ///
    /// `IS_ANTIALIASING_RAY` param may be set to true if the ray is used for antialiasing and might be used to skip complex calculations that are not relevant for antialiasing
    ///
    fn single_raytrace<
        'a,
        const IS_ANTIALIASING_RAY: bool,
        const IS_REFLECTION_RAY: bool,
        const IS_REFRACTION_RAY: bool,
        V,
    >(
        coords: V,
        direction: V,
        start_refraction_index: V::Scalar,
        check_objects: &GeometryCollection<V::SingleValueVector>,
        lights: impl SceneLightIterator<'a, V> + Send + 'static,
        recursion_depth: Option<usize>,
    ) -> Option<(
        ColorType<V::Scalar>,
        <<V as Vector>::Scalar as SimdValue>::SimdBool,
        SurfaceInteraction<V>,
    )>
    where
        V: 'a
            + SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector + Send>
            + Send
            + 'static,
        ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
        Standard: Distribution<<V as Vector>::Scalar>,
    {
        // Early recursion depth check
        if let Some(recursion_depth) = recursion_depth {
            if recursion_depth == 0 {
                return None;
            }
        }

        let zero = V::Scalar::zero();
        let zero_color = ColorSimdExt::zero();

        // Cast the ray and get the closest intersection
        let (ray, nearest_interaction) = Raytracer::cast_ray::<IS_ANTIALIASING_RAY, _>(
            coords,
            direction,
            start_refraction_index,
            check_objects,
        )?;

        if nearest_interaction.valid_mask.none() {
            return None;
        }

        // Calculate direct lighting and specular highlights
        let (direct_light, specular_color): (ColorType<V::Scalar>, ColorType<V::Scalar>) =
            Self::calculate_lighting::<IS_ANTIALIASING_RAY, IS_REFLECTION_RAY, _>(
                &nearest_interaction,
                ray.direction,
                start_refraction_index,
                check_objects,
                lights.clone(),
            );

        // Apply distance-based attenuation
        let distance_factor =
            Self::attenuation_factor_based_on_distance::<V>(nearest_interaction.distance);
        let direct_light = direct_light * distance_factor;
        let specular_color = specular_color * distance_factor;

        // Calculate reflection and refraction only when needed
        let is_transmissive = nearest_interaction.material.transmission.mask();
        let is_reflective = nearest_interaction.material.metallic.simd_gt(zero) | is_transmissive;

        // Skip expensive calculations if material properties don't require them
        let reflection_color: ColorType<V::Scalar> = if cfg!(feature = "reflections")
            && is_reflective.any()
        {
            Self::calculate_reflection::<IS_ANTIALIASING_RAY, IS_REFLECTION_RAY, IS_REFRACTION_RAY, _>(
                &nearest_interaction,
                ray.direction,
                start_refraction_index,
                check_objects,
                lights.clone(),
                recursion_depth,
            )
        } else {
            zero_color
        };

        let refraction_color: ColorType<V::Scalar> =
            if cfg!(feature = "refractions") && is_transmissive.any() {
                Self::calculate_refractions::<
                    IS_ANTIALIASING_RAY,
                    IS_REFLECTION_RAY,
                    IS_REFRACTION_RAY,
                    _,
                >(
                    &nearest_interaction,
                    ray.direction,
                    start_refraction_index,
                    check_objects,
                    lights,
                    recursion_depth,
                )
            } else {
                zero_color
            };

        // Combine all lighting components
        let blend_color: ColorType<V::Scalar> = ColorSimdExt::blend(
            is_transmissive,
            // For transparent materials, combine reflection, refraction and specular
            &(reflection_color + refraction_color + specular_color),
            // For opaque surfaces, direct lighting plus reflection and specular
            &(direct_light + reflection_color + specular_color),
        );

        Some((
            blend_color,
            nearest_interaction.valid_mask,
            nearest_interaction,
        ))
    }

    fn attenuation_factor_based_on_distance<V: SimdRenderingVector>(
        distance: V::Scalar,
    ) -> V::Scalar {
        let distance = distance.simd_abs();
        let zero = V::Scalar::zero();
        let one = V::Scalar::one();

        let scale = V::Scalar::from_subset(&0.1);
        let attenuation = one / (one + distance + scale * distance * distance);

        attenuation.simd_clamp(zero, one)
    }

    fn calculate_refractions<
        'a,
        const IS_ANTIALIASING_RAY: bool,
        const IS_REFLECTION_RAY: bool,
        const IS_REFRACTION_RAY: bool,
        V,
    >(
        interaction: &SurfaceInteraction<V>,
        view_dir: V,
        start_refraction_index: V::Scalar,
        check_objects: &GeometryCollection<V::SingleValueVector>,
        lights: impl SceneLightIterator<'a, V> + Send + 'static,
        recursion_depth: Option<usize>,
    ) -> ColorType<V::Scalar>
    where
        V: 'a
            + SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector + Send>
            + Send
            + 'static,
        ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
        Standard: Distribution<<V as Vector>::Scalar>,
    {
        let zero = V::Scalar::zero();
        let one = V::Scalar::one();
        let bool_false =
            <<<V as Vector>::Scalar as SimdValue>::SimdBool as BoolMask>::from_bool(false);

        let material_is_refractive = interaction.material.transmission.mask();

        if interaction.valid_mask.none() || material_is_refractive.none() {
            return ColorSimdExt::zero();
        }

        macro_rules! impl_refraction_raytrace {
            (
                let $refraction_contribution:ident = $single_raytrace:ident(
                    $interaction_point:expr,
                    $light_dir:expr,
                    $start_refraction_index:expr,
                    $check_objects:expr,
                    $lights:expr,
                    $recursion_depth:expr,
                ) $(if $condition:expr)?
            ) => {
                let $refraction_contribution = if cfg!(feature = "refractions") $( && $condition)?  {
                    if IS_REFRACTION_RAY {
                        impl_refraction_raytrace!(
                            $single_raytrace,
                            $interaction_point,
                            $light_dir,
                            $start_refraction_index,
                            $check_objects,
                            $lights,
                            $recursion_depth
                        )
                    } else {
                        let lights = ($lights).clone();
                        let interaction_point = ($interaction_point).clone();
                        let check_objects = ($check_objects).clone();
                        let light_dir = ($light_dir).clone();
                        let start_refraction_index = ($start_refraction_index).clone();
                        std::thread::spawn(move || {
                            impl_refraction_raytrace!(
                                $single_raytrace,
                                interaction_point,
                                light_dir,
                                start_refraction_index,
                                &check_objects,
                                lights,
                                $recursion_depth
                            )
                        })
                        .join()
                        .unwrap()
                    }
                } else {
                    None
                };
            };
            ($single_raytrace:ident, $interaction_point:expr, $light_dir:expr, $start_refraction_index:expr, $check_objects:expr, $lights:expr, $recursion_depth:expr) => {
                Self::$single_raytrace::<IS_ANTIALIASING_RAY, IS_REFLECTION_RAY, true, _>(
                    $interaction_point,
                    $light_dir,
                    $start_refraction_index,
                    $check_objects,
                    $lights,
                    $recursion_depth
                        .map(|d| d - 1)
                        .or(Some(RAYTRACE_REFRACTION_MAX_DEPTH)),
                )
            };
        }

        let cos_theta = view_dir.dot(interaction.normal);
        let is_inside_object = cos_theta.simd_le(zero);

        let inormal = V::blend(is_inside_object, -interaction.normal, interaction.normal);

        let interaction_refraction_index =
            *interaction.material.transmission.refraction_index().value();

        let new_medium_refraction_index = interaction_refraction_index.select(
            is_inside_object,
            V::Scalar::from_subset(&DEFAULT_REFRACTION_INDEX),
        );

        let eta = (new_medium_refraction_index / start_refraction_index).select(
            is_inside_object,
            start_refraction_index / new_medium_refraction_index,
        );

        let (_, transmittance) =
            interaction
                .material
                .compute_fresnel(inormal, view_dir, eta.simd_recip());

        let has_transmittance = transmittance.abs_diff_eq_default_any(&ColorSimdExt::zero());

        // if has_transmittance.none() {
        //     return ColorSimdExt::zero();
        // }

        //
        // // Check for total internal reflection
        // let cos_theta_i = cos_theta.simd_abs();
        // let sin_theta_t_squared = eta * eta * (one - cos_theta_i * cos_theta_i);
        // let total_internal_reflection = sin_theta_t_squared.simd_ge(one);
        //
        // // Return early if we have total internal reflection everywhere
        // if total_internal_reflection.all() {
        //     return ColorSimdExt::zero();
        // }

        let refraction_direction = view_dir.refracted(-inormal, eta.simd_recip()).normalized();

        // // Calculate the next recursion depth
        // let next_depth = recursion_depth
        //     .map(|d| d - 1)
        //     .or(Some(RAYTRACE_REFRACTION_MAX_DEPTH));
        //
        // // Reusing thread pool instead of spawning new threads
        // let refraction_color = if cfg!(feature = "refractions") && IS_REFRACTION_RAY {
        //     Self::single_raytrace::<IS_ANTIALIASING_RAY, IS_REFLECTION_RAY, true, _>(
        //         interaction.point + (refraction_direction * V::default_epsilon_distance()),
        //         refraction_direction,
        //         new_medium_refraction_index,
        //         check_objects,
        //         lights.clone(),
        //         next_depth,
        //     )
        // } else if cfg!(feature = "refractions") {
        //     let interaction_point =
        //         interaction.point + (refraction_direction * V::default_epsilon_distance());
        //     let check_objects = check_objects.clone();
        //     let lights = lights.clone();
        //
        //     match RAY_PROCESSING_POOL.install(|| {
        //         Self::single_raytrace::<IS_ANTIALIASING_RAY, IS_REFLECTION_RAY, true, _>(
        //             interaction_point,
        //             refraction_direction,
        //             new_medium_refraction_index,
        //             &check_objects,
        //             lights,
        //             next_depth,
        //         )
        //     }) {
        //         Some(result) => Some(result),
        //         None => None,
        //     }
        // } else {
        //     None
        // };

        impl_refraction_raytrace! {
            let refraction_color = single_raytrace(
                (interaction.point + (refraction_direction * V::default_epsilon_distance())),
                refraction_direction,
                new_medium_refraction_index,
                check_objects,
                lights.clone(),
                recursion_depth,
            )
        }

        let (refraction, refraction_valid, intersection_distance) = refraction_color
            .map(|(refraction, refraction_valid, intersection)| {
                (refraction, refraction_valid, intersection.distance)
            })
            .unwrap_or((ColorSimdExt::zero(), bool_false, zero));

        let refraction_valid = refraction_valid & interaction.valid_mask & material_is_refractive;

        let refraction_boost = interaction
            .material
            .transmission
            .boost()
            .simd_unwrap_or(|| zero)
            + one;

        let refracted_light = refraction * refraction_boost * transmittance;

        ColorType::<V::Scalar>::blend(refraction_valid, &refracted_light, &ColorSimdExt::zero())
    }

    fn calculate_reflection<
        'a,
        const IS_ANTIALIASING_RAY: bool,
        const IS_REFLECTION_RAY: bool,
        const IS_REFRACTION_RAY: bool,
        V,
    >(
        interaction: &SurfaceInteraction<V>,
        view_dir: V,
        start_refraction_index: V::Scalar,
        check_objects: &GeometryCollection<V::SingleValueVector>,
        lights: impl SceneLightIterator<'a, V> + Send + 'static,
        recursion_depth: Option<usize>,
    ) -> ColorType<V::Scalar>
    where
        V: 'a
            + SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector + Send>
            + Send
            + 'static,
        ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
        Standard: Distribution<<V as Vector>::Scalar>,
    {
        let zero = V::Scalar::zero();
        let one = V::Scalar::one();
        let bool_false =
            <<<V as Vector>::Scalar as SimdValue>::SimdBool as BoolMask>::from_bool(false);

        if interaction.valid_mask.none() {
            return ColorSimdExt::zero();
        }

        // For transmissive materials, check for total internal reflection
        let cos_theta = view_dir.dot(interaction.normal);
        let is_inside_object = cos_theta.simd_lt(zero);

        let inormal = V::blend(is_inside_object, -interaction.normal, interaction.normal);

        let interaction_refraction_index =
            *interaction.material.transmission.refraction_index().value();

        let new_medium_refraction_index = interaction_refraction_index.select(
            is_inside_object,
            V::Scalar::from_subset(&DEFAULT_REFRACTION_INDEX),
        );

        let eta = (new_medium_refraction_index / start_refraction_index).select(
            is_inside_object,
            start_refraction_index / new_medium_refraction_index,
        );

        // Check for total internal reflection
        let cos_theta_i = cos_theta.simd_abs();
        let sin_theta_t_squared = eta * eta * (one - cos_theta_i * cos_theta_i);
        let total_internal_reflection = sin_theta_t_squared.simd_ge(one);

        let material_is_reflective = interaction.material.metallic.simd_gt(zero)
            | (interaction.material.transmission.mask() & total_internal_reflection);

        if material_is_reflective.none() {
            return ColorSimdExt::zero();
        }

        let reflection_direction = view_dir.reflected(interaction.normal).normalized();
        let reflection_is_none = reflection_direction.abs_diff_eq_default(&V::broadcast(zero));

        if reflection_is_none.all() {
            return ColorSimdExt::zero();
        }

        // Calculate Fresnel reflectance
        let (reflectance, _) =
            interaction
                .material
                .compute_fresnel(inormal, -view_dir, start_refraction_index);

        // let reflection_color = if cfg!(feature = "reflections") && IS_REFLECTION_RAY {
        //     Self::single_raytrace::<IS_ANTIALIASING_RAY, true, IS_REFRACTION_RAY, _>(
        //         interaction.point + (reflection_direction * V::default_epsilon_distance()),
        //         reflection_direction,
        //         start_refraction_index,
        //         check_objects,
        //         lights.clone(),
        //         next_depth,
        //     )
        // } else if cfg!(feature = "reflections") {
        //     let interaction_point =
        //         interaction.point + (reflection_direction * V::default_epsilon_distance());
        //     let check_objects = check_objects.clone();
        //     let lights = lights.clone();
        //
        //     match RAY_PROCESSING_POOL.install(|| {
        //         Self::single_raytrace::<IS_ANTIALIASING_RAY, true, IS_REFRACTION_RAY, _>(
        //             interaction_point,
        //             reflection_direction,
        //             start_refraction_index,
        //             &check_objects,
        //             lights,
        //             next_depth,
        //         )
        //     }) {
        //         Some(result) => Some(result),
        //         None => None,
        //     }
        // } else {
        //     None
        // };

        macro_rules! impl_reflection_raytrace {
            (
                let $reflection_contribution:ident = $single_raytrace:ident(
                    $interaction_point:expr,
                    $light_dir:expr,
                    $start_refraction_index:expr,
                    $check_objects:expr,
                    $lights:expr,
                    $recursion_depth:expr,
                ) $(if $condition:expr)?
            ) => {
                let $reflection_contribution = if cfg!(feature = "reflections") $( && $condition)?  {
                    if IS_REFLECTION_RAY {
                        impl_reflection_raytrace!(
                            $single_raytrace,
                            $interaction_point,
                            $light_dir,
                            $start_refraction_index,
                            $check_objects,
                            $lights,
                            $recursion_depth
                        )
                    } else {
                        let lights = ($lights).clone();
                        let interaction_point = ($interaction_point).clone();
                        let check_objects = ($check_objects).clone();
                        let light_dir = ($light_dir).clone();
                        let start_refraction_index = ($start_refraction_index).clone();
                        std::thread::spawn(move || {
                            impl_reflection_raytrace!(
                                $single_raytrace,
                                interaction_point,
                                light_dir,
                                start_refraction_index,
                                &check_objects,
                                lights,
                                $recursion_depth
                            )
                        })
                        .join()
                        .unwrap()
                    }
                } else {
                    None
                };
            };
            ($single_raytrace:ident, $interaction_point:expr, $light_dir:expr, $start_refraction_index:expr, $check_objects:expr, $lights:expr, $recursion_depth:expr) => {
                Self::$single_raytrace::<IS_ANTIALIASING_RAY, true, IS_REFRACTION_RAY, _>(
                    $interaction_point,
                    $light_dir,
                    $start_refraction_index,
                    $check_objects,
                    $lights,
                    $recursion_depth
                        .map(|d| d - 1)
                        .or(Some(RAYTRACE_REFLECTION_MAX_DEPTH)),
                )
            };
        }

        impl_reflection_raytrace! {
            let reflection_color = single_raytrace(
                (interaction.point + (reflection_direction * V::default_epsilon_distance())),
                reflection_direction,
                start_refraction_index,
                check_objects,
                lights.clone(),
                recursion_depth,
            ) if (reflection_is_none.none())
        }

        let (reflection, reflection_valid, interaction_distance) = reflection_color
            .map(|(refraction, refraction_valid, interaction)| {
                (refraction, refraction_valid, interaction.distance)
            })
            .unwrap_or((ColorSimdExt::zero(), bool_false, zero));

        let reflection_valid = reflection_valid
            & interaction.valid_mask
            & material_is_reflective
            & !reflection_is_none;

        let distance_factor = Self::attenuation_factor_based_on_distance::<V>(interaction_distance);

        ColorSimdExt::blend(
            reflection_valid,
            &(reflection * distance_factor * reflectance),
            &ColorSimdExt::zero(),
        )
    }

    fn calculate_lighting<'a, const IS_ANTIALIASING_RAY: bool, const IS_REFLECTION_RAY: bool, V>(
        interaction: &SurfaceInteraction<V>,
        view_dir: V,
        start_refraction_index: V::Scalar,
        check_objects: &GeometryCollection<V::SingleValueVector>,
        lights: impl SceneLightIterator<'a, V> + Send + 'static,
    ) -> (ColorType<V::Scalar>, ColorType<V::Scalar>)
    where
        V: 'a
            + SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector + Send>
            + Send
            + 'static,
        ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
        Standard: Distribution<<V as Vector>::Scalar>,
    {
        let zero = V::Scalar::zero();
        let one = V::Scalar::one();
        if interaction.valid_mask.none() {
            return (ColorSimdExt::zero(), ColorSimdExt::zero());
        }

        // Create ambient light
        let ambient_light =
            AmbientLight::<V>::new(ColorSimdExt::one(), V::Scalar::from_subset(&0.08));

        // Get material color
        let material_color = interaction.material.color.clone();

        // Calculate ambient lighting
        let ambient_contribution = <ColorType<V::Scalar> as ColorSimdExt<V::Scalar>>::blend(
            interaction.valid_mask,
            &(material_color.clone() * ambient_light.color),
            &ColorSimdExt::zero(),
        ) * ambient_light.intensity;

        // Calculate direct lighting from all point lights
        let mut light_color = <ColorType<V::Scalar> as ColorSimdExt<V::Scalar>>::zero();
        let mut specular_color = <ColorType<V::Scalar> as ColorSimdExt<V::Scalar>>::zero();

        // Only calculate specular if the material has specular properties
        let has_specular = interaction.material.shininess().simd_gt(V::Scalar::zero());

        for light in lights.clone() {
            // Create SIMD light
            let light = match light.as_ref() {
                SceneLightSource::PointLight(light) => PointLight::<V>::splat(&light),
            };

            let light_position = light.position;

            // Calculate light direction (from intersection point to light)
            let light_to_point = light_position - interaction.point;
            let light_dir = light_to_point.normalized();

            // Move point a bit away from the surface of the object
            let check_for_light_blocking_point =
                interaction.point + (light_dir * V::default_epsilon_distance());
            let check_for_light_blocking_light_to_point =
                light_position - check_for_light_blocking_point;

            let obstacles_in_path_to_light = Raytracer::has_any_intersection(
                check_for_light_blocking_point,
                light_dir,
                check_objects,
                check_for_light_blocking_light_to_point.mag(),
            );
            let light_can_reach_point =
                !obstacles_in_path_to_light.completely_occluded & interaction.valid_mask;

            if light_can_reach_point.none() {
                continue;
            }

            let contribution =
                light.calculate_contribution_at(interaction, interaction.point, view_dir);

            let light_color_simd = ColorSimdExt::blend(
                light_can_reach_point,
                &(contribution.color / obstacles_in_path_to_light.color_filter),
                &contribution.color,
            );
            let light_intensity = contribution.intensity;

            // Calculate diffuse factor
            let diffuse_factor = interaction.normal.dot(light_dir).simd_max(zero);

            // Only calculate specular for materials with specular properties
            let specular_factor = if !has_specular.none() {
                // Calculate specular reflection
                let reflection = light_dir.reflected(interaction.normal);
                let specular = reflection
                    .normalized()
                    .dot(view_dir)
                    .simd_max(zero)
                    .simd_powf(
                        (interaction.material.shininess() * V::Scalar::from_subset(&(512.0)))
                            .simd_max(V::Scalar::one()),
                    );

                specular.select(has_specular, V::Scalar::zero())
            } else {
                V::Scalar::zero()
            };

            // Combined lighting for this light
            let light_factor = diffuse_factor
                * light_intensity
                * obstacles_in_path_to_light
                    .combined_opacity
                    .select(light_can_reach_point, one);

            let specular_factor = light_intensity
                * obstacles_in_path_to_light
                    .combined_opacity
                    .select(light_can_reach_point, one)
                * specular_factor;

            // Only apply light where the surface faces the light & is not blocked by the light
            let light_valid = diffuse_factor.simd_gt(zero) & light_can_reach_point;

            let diffuse_contribution = material_color * light_color_simd * light_factor;
            let specular_contribution = light.color * specular_factor;

            // Add light contribution
            light_color = light_color
                + ColorType::blend(
                    light_valid & interaction.valid_mask,
                    &diffuse_contribution,
                    &ColorSimdExt::zero(),
                );

            if !has_specular.none() {
                specular_color = specular_color
                    + ColorType::blend(
                        light_valid & interaction.valid_mask & has_specular,
                        &specular_contribution,
                        &ColorSimdExt::zero(),
                    );
            }
        }

        // Combine ambient and direct lighting
        (ambient_contribution + light_color, specular_color)
    }

    fn get_antialiasing_simpling_directions<V>() -> (V, V, V, V, V, V, V, V)
    where
        V: SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector>,
    {
        let (x_r, y_r) = if cfg!(feature = "anti_aliasing_rotation_scale") {
            let rotate_angle = (0.5f32).atan();

            let sin = V::broadcast(V::Scalar::from_subset(&(rotate_angle.sin())));
            let cos = V::broadcast(V::Scalar::from_subset(&(rotate_angle.cos())));

            let x = V::unit_x();
            let y = V::unit_y();

            let x_r = x.mul_add(cos, y * sin);
            let y_r = x.mul_add(-sin, y * cos);

            (x_r, y_r)
        } else {
            (V::unit_x(), V::unit_y())
        };

        let t = -y_r;
        let l = -x_r;
        let r = x_r;
        let b = y_r;
        let tl = t + l;
        let tr = t + r;
        let bl = b + l;
        let br = b + r;

        (
            t.normalized(),
            l.normalized(),
            b.normalized(),
            r.normalized(),
            tl.normalized(),
            tr.normalized(),
            bl.normalized(),
            br.normalized(),
        )
    }

    fn antialiased_raytrace<'a, V>(
        sample_coordinates: Vec<V>,
        direction: V,
        start_refraction_index: <V as Vector>::Scalar,
        check_objects: &GeometryCollection<<V as SimdCapableVector>::SingleValueVector>,
        lights: impl SceneLightIterator<'a, V> + Send + 'static,
    ) -> Option<(
        ColorType<<<V as SimdCapableVector>::SingleValueVector as Vector>::Scalar>,
        <<<V as SimdCapableVector>::SingleValueVector as Vector>::Scalar as SimdValue>::SimdBool,
    )>
    where
        V: 'a
            + SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector + Send>
            + Send
            + 'static,
        ColorType<<V as Vector>::Scalar>: LightCompatibleColor<<V as Vector>::Scalar>,
        Standard: Distribution<<V as Vector>::Scalar>,
    {
        let mut scale =
            V::Scalar::from_subset(&(1.0 / (sample_coordinates.len() * V::LANES) as f32));

        let zero = V::Scalar::zero();
        let bool_false =
            <<<V as Vector>::Scalar as SimdValue>::SimdBool as BoolMask>::from_bool(false);

        let (initial_coords, mut rest_coordinates) =
            (sample_coordinates[0], &sample_coordinates[1..]);
        let default_color = ColorSimdExt::zero();
        let default_mask = bool_false;
        let (initial_color, initial_mask, interaction) =
            Self::single_raytrace::<false, false, false, V>(
                initial_coords,
                direction,
                start_refraction_index,
                check_objects,
                lights.clone(),
                None,
            )
            .map(|(c, m, interaction)| (c, m, Some(interaction)))
            .unwrap_or((default_color, default_mask, None));

        // let all_same_object = interaction
        //     .map(|i| {
        //         i.get_render_object_id()
        //             .id()
        //             .simd_horizontal_min()
        //             .abs_diff_eq_default(&i.get_render_object_id().id().simd_horizontal_max())
        //     })
        //     .unwrap_or(false);

        // if all_same_object && V::LANES > 2 {
        //     rest_coordinates = &rest_coordinates[..(rest_coordinates.len() / 2)];
        //     scale =
        //         V::Scalar::from_subset(&(1.0 / ((1 + rest_coordinates.len()) * V::LANES) as f32))
        // }

        let initial_color = initial_color * scale;

        // fixme: maybe reordering the rays here so that 1 simd ray represents the different samples of the antialiasing for the same "real" pixel, because they cohere better, instead of the rays of independent "real" pixels.
        // Sample multiple rays and average the results
        let (res_color, valid_mask) = rest_coordinates
            .chunks(2)
            .par_bridge()
            .flat_map(|vecs| {
                vecs.iter()
                    .filter_map(|&coords| {
                        let (c, m, _) = Self::single_raytrace::<true, false, false, V>(
                            coords,
                            direction,
                            start_refraction_index,
                            check_objects,
                            lights.clone(),
                            None,
                        )?;
                        Some((c * scale, m))
                    })
                    .collect::<Vec<_>>()
            })
            .reduce(
                || (default_color, default_mask),
                |(res_color, res_mask), (color, mask)| (color + res_color, res_mask | mask),
            );

        let result_final_color = res_color + initial_color;
        Some((
            ColorType::<<<V as SimdCapableVector>::SingleValueVector as Vector>::Scalar>::new(
                <<V as SimdCapableVector>::SingleValueVector as Vector>::Scalar::from_subset(
                    &result_final_color.red.simd_horizontal_sum(),
                ),
                <<V as SimdCapableVector>::SingleValueVector as Vector>::Scalar::from_subset(
                    &result_final_color.green.simd_horizontal_sum(),
                ),
                <<V as SimdCapableVector>::SingleValueVector as Vector>::Scalar::from_subset(
                    &result_final_color.blue.simd_horizontal_sum(),
                ),
            ),
            (valid_mask | initial_mask).any().into(),
        ))
    }

    const fn get_total_rays<V: Vector>() -> usize {
        ANTIALIASING_SAMPLES_PER_PIXEL.next_multiple_of(<V as Vector>::LANES)
    }
    fn bundle_rays_for_simd_antialiased_raytracing<'a, V>(coords: Vec3) -> Vec<V>
    where
        Vec3: 'a + SimdRenderingVector<SingleValueVector: NormalizableVector>,
        V: 'a
            + SimdRenderingVector<
                SingleValueVector: NormalizableVector,
                Scalar: SimdValue<Element = f32>,
            >,
        Standard: Distribution<<Vec3 as Vector>::Scalar>,
        [(); { Self::get_total_rays::<V>() }]:,
        [(); Self::get_total_rays::<V>() - 1]:,
        [(); { <<V as Vector>::Scalar as SimdValue>::LANES }]:,
        [(); <V as Vector>::DIMENSIONS]:,
    {
        let total_rays: usize = Self::get_total_rays::<V>();

        // be generic over output vector
        // bundle rays in bundles of ANTIALIASING_SAMPLES_PER_PIXEL.next_multiple_of(V::LANES) -> has the nice effect that if ANTIALIASING_SAMPLES_PER_PIXEL does not fit into lanes size, we get higher antialiasing supersampling "for free"
        // In non-simd code path, the caller just specifies that it wants a non-simd vector returned, and due to our trait magic, this logic will not bundle rays since lane-size is 1
        // fixme how to prevent non-simd path to simd this?
        let direction_bias = Self::get_antialiasing_simpling_directions::<Vec3>();

        let directions = [
            direction_bias.0,
            direction_bias.1,
            direction_bias.2,
            direction_bias.3,
            direction_bias.4,
            direction_bias.5,
            direction_bias.6,
            direction_bias.7,
        ];

        let scale_factor = if cfg!(feature = "anti_aliasing_rotation_scale") {
            ((5.0f32).sqrt() / 2.05)
        } else {
            0.85
        };

        let bundle_iter = (0..total_rays)
            .collect_array::<{ Self::get_total_rays::<V>() }>()
            .expect("Has enough rays");
        let bundle_iter = bundle_iter.chunks_exact(<V as Vector>::LANES);

        // let random_iter = if cfg!(feature = "anti_aliasing_randomness") {
        //     [[0.0; 2]]
        //         .into_iter()
        //         .chain(
        //             Poisson2D::new()
        //                 .with_dimensions([1.2, 1.2], (3.0 / total_rays as f32))
        //                 .with_samples(total_rays as u32)
        //                 .into_iter()
        //                 .take(total_rays - 1), // only return TOTAL_RAYS -1 because we defined the 1st point in the iterator to be 0 to sample at the exact position of the ray
        //         )
        //         .pad_using(total_rays, |_| -> fast_poisson::Point<2> {
        //             let random_vec = Vec3::sample_random() * 1.15;
        //             [random_vec.x, random_vec.y]
        //         })
        //         .map(|p| -> fast_poisson::Point<2> {
        //             [
        //                 p[0] * WINDOW_TO_SCENE_WIDTH_FACTOR * scale_factor,
        //                 p[1] * WINDOW_TO_SCENE_HEIGHT_FACTOR * scale_factor,
        //             ]
        //         })
        //         .collect::<Vec<_>>()
        // } else {
        //     [[0.0; 2]]
        //         .into_iter()
        //         .chain([[1.0; 2]; Self::get_total_rays::<V>() - 1])
        //         .map(|p| -> fast_poisson::Point<2> {
        //             [
        //                 p[0] * WINDOW_TO_SCENE_WIDTH_FACTOR * scale_factor,
        //                 p[1] * WINDOW_TO_SCENE_HEIGHT_FACTOR * scale_factor,
        //             ]
        //         })
        //         .collect::<Vec<_>>()
        // };
        let random_iter = ANTIALIASING_SAMPLES
            .iter()
            .map(|p| -> fast_poisson::Point<2> {
                [
                    p[0] * WINDOW_TO_SCENE_WIDTH_FACTOR * scale_factor,
                    p[1] * WINDOW_TO_SCENE_HEIGHT_FACTOR * scale_factor,
                ]
            })
            .take(total_rays)
            .collect::<Vec<_>>();

        let random_iter = random_iter.chunks(<V as Vector>::LANES);

        izip!(bundle_iter, random_iter)
            .map(|(is, random_vec)| {
                //    println!("{random_vec:?}");
                let (xs, ys, zs): (Vec<_>, Vec<_>, Vec<_>) =
                    izip!(is, random_vec, directions.iter().cycle())
                        .map(|(i, random_vec, sampling_direction_bias)| {
                            (
                                coords.x + random_vec[0] * sampling_direction_bias.x,
                                coords.y + random_vec[1] * sampling_direction_bias.y,
                                coords.z,
                            )
                        })
                        .collect();

                <V as VectorFixedDimensions<3>>::from_scalar_components([
                    xs.into_iter()
                        .collect_array::<{ <<V as Vector>::Scalar as SimdValue>::LANES }>()
                        .expect("Should have enough values"),
                    ys.into_iter()
                        .collect_array::<{ <<V as Vector>::Scalar as SimdValue>::LANES }>()
                        .expect("Should have enough values"),
                    zs.into_iter()
                        .collect_array::<{ <<V as Vector>::Scalar as SimdValue>::LANES }>()
                        .expect("Should have enough values"),
                ])
            })
            .collect()
    }

    fn get_pixel_color(
        RenderCoordinates { x, y }: RenderCoordinates,
        scene: &Scene<Vec3>,
    ) -> Option<Pixel>
    where
        [(); { Self::get_total_rays::<Vec3>() }]:,
        [(); { Self::get_total_rays::<Vec3>() - 1 }]:,
    {
        let coords = Vec3::new(x as f32, y as f32, 0.0);
        let render_ray_direction = coords - RENDER_RAY_FOCUS;
        let start_refraction_index = DEFAULT_REFRACTION_INDEX; // air

        let (colors, valid) = if cfg!(feature = "anti_aliasing") {
            let bundle_rays_coords =
                Self::bundle_rays_for_simd_antialiased_raytracing::<Vec3>(coords);
            Self::antialiased_raytrace(
                bundle_rays_coords,
                render_ray_direction,
                start_refraction_index,
                &scene.scene_objects,
                (scene
                    .scene_lights
                    .iter()
                    .flat_map(|light| light.preprocess::<{ POINT_LIGHT_MULTIPLICATOR }>())
                    .collect::<Vec<_>>()),
            )?
        } else {
            let (color, mask, _) = Self::single_raytrace::<false, false, false, _>(
                coords,
                render_ray_direction,
                start_refraction_index,
                &scene.scene_objects,
                scene
                    .scene_lights
                    .iter()
                    .flat_map(|light| light.preprocess::<{ POINT_LIGHT_MULTIPLICATOR }>())
                    .collect::<Vec<_>>(),
                None,
            )?;

            (color, mask)
        };

        if unlikely(!valid) {
            return None;
        }

        Some(Pixel(colors))
    }

    fn render_pixel_colors<'a>(
        coords: RenderCoordinatesVectorized<'a>,
        scene: &Scene<Vec3>,
        set_pixel: &dyn Fn(usize, Pixel),
    ) where
        [(); { Self::get_total_rays::<Vec3x8>() }]:,
        [(); { Self::get_total_rays::<Vec3x8>() - 1 }]:,
    {
        let start_refraction_index = DEFAULT_REFRACTION_INDEX; // air
        if cfg!(feature = "anti_aliasing") {
            izip!(coords.i, coords.x, coords.y, coords.z,).for_each(|(&i, &x, &y, &z)| {
                let coords = Vec3::new(x, y, z);
                let bundle_rays_coords =
                    Self::bundle_rays_for_simd_antialiased_raytracing::<Vec3x8>(coords);
                let render_ray_direction = Vec3x8::splat(coords - RENDER_RAY_FOCUS);

                // println!("{bundle_rays_coords:?}");

                if let Some((color, true)) = Self::antialiased_raytrace(
                    bundle_rays_coords,
                    render_ray_direction,
                    WideF32x8::from_subset(&start_refraction_index),
                    &scene.scene_objects,
                    scene
                        .scene_lights
                        .iter()
                        .flat_map(|light| light.preprocess::<{ POINT_LIGHT_MULTIPLICATOR }>())
                        .collect::<Vec<_>>(),
                ) {
                    set_pixel(i, Pixel(color));
                }
            });
        } else {
            const CHUNK_SIZE: usize = 8;

            // fn render_simd<V>(
            //     coords: V,
            //     start_refraction_index: f32,
            //     scene: &Scene<Vec3>,
            // ) -> Option<Vec<(usize, Pixel)>>
            // where
            //     V: SimdRenderingVector<SingleValueVector = Vec3> + Send + 'static,
            //     ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
            //     Standard: Distribution<<V as Vector>::Scalar>,
            // {
            //     let render_ray_direction = coords - V::splat(RENDER_RAY_FOCUS);
            //     let (colors, valid_mask, _) = Self::single_raytrace::<false, false, false, V>(
            //         coords,
            //         render_ray_direction,
            //         WideF32x8::from_subset(&start_refraction_index),
            //         &scene.scene_objects,
            //         LIGHTS.iter(),
            //         None,
            //     )?;
            //
            //     let x = colors.extract_values(Some(valid_mask));
            //
            //     for (pixel_index, &c) in x.iter().enumerate().filter_map(|(i, v)| {
            //         let Some(v) = v else {
            //             return None;
            //         };
            //         Some((i, v))
            //     }) {
            //         set_pixel(idxs[pixel_index], Pixel(c));
            //     }
            // }
            izip!(
                coords.i.chunks(CHUNK_SIZE),
                coords.x.chunks(CHUNK_SIZE),
                coords.y.chunks(CHUNK_SIZE),
                coords.z.chunks(CHUNK_SIZE),
            )
            .for_each(|(idxs, xs, ys, zs)| {
                let len = idxs.len();

                if likely(len == 8) {
                    let coords = Vec3x8::new(f32x8::from(xs), f32x8::from(ys), f32x8::from(zs));
                    let render_ray_direction = coords - Vec3x8::splat(RENDER_RAY_FOCUS);
                    let Some((colors, valid_mask, _)) =
                        Self::single_raytrace::<false, false, false, _>(
                            coords,
                            render_ray_direction,
                            WideF32x8::from_subset(&start_refraction_index),
                            &scene.scene_objects,
                            scene
                                .scene_lights
                                .iter()
                                .flat_map(|light| {
                                    light.preprocess::<{ POINT_LIGHT_MULTIPLICATOR }>()
                                })
                                .collect::<Vec<_>>(),
                            None,
                        )
                    else {
                        return;
                    };

                    let x = colors.extract_values(Some(valid_mask));

                    for (pixel_index, &c) in x.iter().enumerate().filter_map(|(i, v)| {
                        let Some(v) = v else {
                            return None;
                        };
                        Some((i, v))
                    }) {
                        set_pixel(idxs[pixel_index], Pixel(c));
                    }
                    return;
                } else if len == 4 {
                    let coords = Vec3x4::new(f32x4::from(xs), f32x4::from(ys), f32x4::from(zs));
                    let render_ray_direction = coords - Vec3x4::splat(RENDER_RAY_FOCUS);
                    let Some((colors, valid_mask, _)) =
                        Self::single_raytrace::<false, false, false, _>(
                            coords,
                            render_ray_direction,
                            WideF32x4::from_subset(&start_refraction_index),
                            &scene.scene_objects,
                            scene
                                .scene_lights
                                .iter()
                                .flat_map(|light| {
                                    light.preprocess::<{ POINT_LIGHT_MULTIPLICATOR }>()
                                })
                                .collect::<Vec<_>>(),
                            None,
                        )
                    else {
                        return;
                    };

                    let x = colors.extract_values(Some(valid_mask));

                    for (pixel_index, &c) in x.iter().enumerate().filter_map(|(i, v)| {
                        let Some(v) = v else {
                            return None;
                        };
                        Some((i, v))
                    }) {
                        set_pixel(idxs[pixel_index], Pixel(c));
                    }
                    return;
                } else {
                    for (&x, (&y, &i)) in xs.iter().zip(ys.iter().zip(idxs.iter())) {
                        let coords = Vec3::new(x, y, 0.0);
                        let render_ray_direction = coords - RENDER_RAY_FOCUS;
                        if let Some((color, true, _)) =
                            Self::single_raytrace::<false, false, false, _>(
                                coords,
                                render_ray_direction,
                                start_refraction_index,
                                &scene.scene_objects,
                                scene
                                    .scene_lights
                                    .iter()
                                    .flat_map(|light| {
                                        light.preprocess::<{ POINT_LIGHT_MULTIPLICATOR }>()
                                    })
                                    .collect::<Vec<_>>(),
                                None,
                            )
                        {
                            set_pixel(i, Pixel(color));
                        }
                    }
                }
            });
        };
    }
}

impl<const W: usize, const H: usize, C: OutputColorEncoder> Renderer<W, H, C>
    for RaytracerRenderer<C>
where
    [(); W * H]:,
    [(); { Self::get_total_rays::<Vec3x8>() }]:,
    [(); { Self::get_total_rays::<Vec3x8>() - 1 }]:,
    [(); { Self::get_total_rays::<Vec3>() }]:,
    [(); { Self::get_total_rays::<Vec3>() - 1 }]:,
{
    fn render(&self, buffer: &ImageBuffer<W, H>, scene: &Scene<Vec3>)
    where
        [(); W * H]:,
    {
        if cfg!(feature = "simd_render") {
            Self::render_to_buffer_chunked_inplace(buffer, scene, Self::render_pixel_colors)
        } else {
            Self::render_to_buffer(buffer, scene, Self::get_pixel_color)
        }
    }
}
