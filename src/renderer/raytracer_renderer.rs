use crate::extensions::SrgbColorConvertExt;
use crate::helpers::{ColorType, Pixel, Splatable};
use crate::image_buffer::ImageBuffer;
use crate::output::OutputColorEncoder;
use crate::renderer::{RenderCoordinates, RenderCoordinatesVectorized, Renderer};
use crate::{WINDOW_HEIGHT, WINDOW_WIDTH};
use itertools::izip;
use rayon::iter::ParallelIterator;

// Import the consolidated traits
use crate::color_traits::LightCompatibleColor;
use crate::vector_traits::{RenderingVector, SimdRenderingVector};

use crate::vector::{NormalizableVector, Vector};

use crate::color::ColorSimdExt;
use crate::geometry::{Ray, SphereData};
use crate::raytracing::{Intersectable, RayIntersection, RayIntersectionCandidate};
use crate::scene::{AmbientLight, Light, PointLight};
use num_traits::{One, Zero};
use palette::Srgb;
use palette::bool_mask::BoolMask;
use rand::distributions::{Distribution, Standard};
use rayon::iter::IntoParallelIterator;
use simba::scalar::SupersetOf;
use simba::simd::{SimdBool, SimdPartialOrd, SimdValue};
use std::fmt::Debug;
use std::hint::{likely, unlikely};
use std::marker::PhantomData;
use std::sync::LazyLock;
use ultraviolet::{Vec3, Vec3x8, f32x8};

// Helper type alias to make the bounds more readable

static SPHERES: LazyLock<[SphereData<Vec3>; 8]> = LazyLock::new(|| {
    [
        SphereData::new(
            Vec3::new(WINDOW_WIDTH as f32 / 2.0, WINDOW_HEIGHT as f32 / 2.0, 150.0),
            70.0,
            ColorType::new(255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0),
        ),
        SphereData::new(
            Vec3::new(WINDOW_WIDTH as f32 / 2.5, WINDOW_HEIGHT as f32 / 2.5, 150.0),
            90.0,
            ColorType::new(0.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0),
        ),
        SphereData::new(
            Vec3::new(
                2.0 * (WINDOW_WIDTH as f32 / 2.5),
                WINDOW_HEIGHT as f32 / 2.5,
                150.0,
            ),
            90.0,
            ColorType::new(111.0 / 255.0, 255.0 / 255.0, 222.0 / 255.0),
        ),
        SphereData::new(
            Vec3::new(
                2.0 * (WINDOW_WIDTH as f32 / 2.5),
                2.0 * (WINDOW_HEIGHT as f32 / 2.5),
                250.0,
            ),
            120.0,
            ColorType::new(158.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0),
        ),
        SphereData::new(
            Vec3::new(
                1.25 * (WINDOW_WIDTH as f32 / 2.5),
                0.5 * (WINDOW_HEIGHT as f32 / 2.5),
                90.0,
            ),
            30.0,
            ColorType::new(128.0 / 255.0, 210.0 / 255.0, 255.0 / 255.0),
        ),
        SphereData::new(
            Vec3::new(
                WINDOW_WIDTH as f32 / 2.5,
                2.25 * (WINDOW_HEIGHT as f32 / 2.5),
                500.0,
            ),
            250.0,
            ColorType::new(254.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0),
        ),
        SphereData::new(
            Vec3::new(
                WINDOW_WIDTH as f32 / 4.0,
                3.0 * (WINDOW_HEIGHT as f32 / 4.0),
                20.0,
            ),
            10.0,
            ColorType::new(255.0 / 255.0, 55.0 / 255.0, 77.0 / 255.0),
        ),
        SphereData::new(
            Vec3::new(
                WINDOW_WIDTH as f32 / 3.0,
                3.0 * (WINDOW_HEIGHT as f32 / 6.0),
                30.0,
            ),
            25.0,
            ColorType::new(55.0 / 255.0, 230.0 / 255.0, 180.0 / 255.0),
        ),
    ]
});

static RENDER_RAY_DIRECTION: Vec3 = Vec3::new(0.0, 0.0, 1.0);

static LIGHTS: LazyLock<[PointLight<Vec3>; 4]> = LazyLock::new(|| {
    [
        PointLight::new(
            Vec3::new(-100.0, 1000.0, -10.0),
            ColorType::new(0.822, 0.675, 0.45),
            1.0,
        ),
        PointLight::new(
            Vec3::new(1.0, -1.0, 100.0),
            ColorType::new(0.0, 0.675, 0.9),
            1.0,
        ),
        PointLight::new(
            Vec3::new(0.0, 0.0, -100.0),
            ColorType::new(0., 0., 0.),
            0.95,
        ),
        PointLight::new(
            Vec3::new((WINDOW_WIDTH / 2) as f32, (WINDOW_HEIGHT / 2) as f32, 120.0),
            ColorType::new(0.7, 0.56, 0.5),
            1.0,
        ),
    ]
});

trait SceneObjectIterator<'a, V: 'a + SimdRenderingVector>:
    IntoIterator<Item = &'a SphereData<V::SingleValueVector>> + Clone + Sync
{
}

impl<'a, T, V: 'a + SimdRenderingVector> SceneObjectIterator<'a, V> for T where
    T: IntoIterator<Item = &'a SphereData<V::SingleValueVector>> + Clone + Sync
{
}

trait SceneLightIterator<'a, V: 'a + SimdRenderingVector>:
    IntoIterator<Item = &'a PointLight<V::SingleValueVector>> + Clone + Sync
{
}

impl<'a, T, V: 'a + SimdRenderingVector> SceneLightIterator<'a, V> for T where
    T: IntoIterator<Item = &'a PointLight<V::SingleValueVector>> + Clone + Sync
{
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct RaytracerRenderer<C: OutputColorEncoder>(PhantomData<C>);

impl<C: OutputColorEncoder> RaytracerRenderer<C> {
    fn get_pixel_color(RenderCoordinates { x, y }: RenderCoordinates) -> Option<Pixel> {
        let (colors, valid) = Self::get_pixel_color_vectorized(
            Vec3::new(x as f32, y as f32, 0.0),
            RENDER_RAY_DIRECTION,
            SPHERES.iter(),
            LIGHTS.iter(),
        )?;

        if unlikely(!valid) {
            return None;
        }

        Some(Pixel(colors))
    }

    // Significantly simplified trait bounds using consolidated traits
    fn get_pixel_color_vectorized<'a, V>(
        coords: V,
        unit_z: V,
        spheres: impl SceneObjectIterator<'a, V>,
        lights: impl SceneLightIterator<'a, V>,
    ) -> Option<(
        ColorType<V::Scalar>,
        <<V as Vector>::Scalar as SimdValue>::SimdBool,
    )>
    where
        V: 'a + SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector>,
        ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
        Standard: Distribution<<V as Vector>::Scalar>,
        [(); <V as Vector>::LANES]:,
    {
        // Determine if we should use antialiasing
        let antialiasing = cfg!(feature = "anti_aliasing");

        if antialiasing {
            Self::antialiased_raytrace(coords, unit_z, spheres, lights)
        } else {
            Self::single_raytrace(coords, unit_z, spheres, lights)
        }
    }

    // Separate function for a single raytrace calculation
    fn single_raytrace<'a, V>(
        coords: V,
        unit_z: V,
        spheres: impl SceneObjectIterator<'a, V>,
        lights: impl SceneLightIterator<'a, V>,
    ) -> Option<(
        ColorType<V::Scalar>,
        <<V as Vector>::Scalar as SimdValue>::SimdBool,
    )>
    where
        V: 'a + SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector>,
        ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
        Standard: Distribution<<V as Vector>::Scalar>,
    {
        let ray = Ray::<V>::new_with_mask(
            coords,
            unit_z,
            <V::Scalar as SimdValue>::SimdBool::splat(false),
        );

        // Find nearest intersection
        let nearest_intersection = Self::find_nearest_intersection(&ray, spheres)?;

        if nearest_intersection.valid_mask.none() {
            return None;
        }

        // Calculate intersection details
        let intersected_object = nearest_intersection.payload;
        let RayIntersection {
            intersection: intersection_point,
            valid_mask,
            normal: _intersection_normal,
            ..
        } = intersected_object.intersect(
            &ray,
            &RayIntersectionCandidate::new(
                nearest_intersection.t,
                &intersected_object,
                nearest_intersection.valid_mask,
            ),
        );

        // Calculate lighting
        let color = Self::calculate_lighting(
            &intersected_object,
            intersection_point,
            ray.direction.normalized(),
            valid_mask,
            lights,
        );

        Some((color, valid_mask))
    }

    // Find the nearest intersection among all objects
    fn find_nearest_intersection<'a, V>(
        ray: &Ray<V>,
        spheres: impl SceneObjectIterator<'a, V>,
    ) -> Option<RayIntersectionCandidate<V::Scalar, SphereData<V>>>
    where
        V: 'a + SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector>,
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
        new_is_nearer: <<V as Vector>::Scalar as SimdValue>::SimdBool,
    ) -> Option<RayIntersectionCandidate<V::Scalar, SphereData<V>>>
    where
        V: SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector>,
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

    // Calculate lighting for an intersection point
    fn calculate_lighting<'a, V>(
        object: &SphereData<V>,
        intersection_point: V,
        view_dir: V,
        valid_mask: <<V as Vector>::Scalar as SimdValue>::SimdBool,
        lights: impl SceneLightIterator<'a, V>,
    ) -> ColorType<V::Scalar>
    where
        V: 'a + SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector>,
        ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
    {
        let zero = V::Scalar::zero();
        let one = V::Scalar::one();

        // Create ambient light
        let ambient_light = AmbientLight::new(ColorType::new(one, one, one), zero);

        // Calculate direct lighting from all point lights
        let mut light_color = Srgb::<V::Scalar>::new(zero, zero, zero);
        for light in lights.into_iter() {
            let light = PointLight::<V>::splat(light);
            let contribution =
                light.calculate_contribution_at(object, intersection_point, view_dir);

            light_color = light_color
                + ColorType::blend(
                    contribution.valid_mask,
                    &(contribution.color * contribution.intensity),
                    &Srgb::<V::Scalar>::new(zero, zero, zero),
                );
        }

        // Calculate ambient lighting
        let ambient_contribution =
            ambient_light.calculate_contribution_at(object, intersection_point, view_dir);

        // Combine lighting and apply valid mask
        ColorType::blend(
            valid_mask,
            &((ambient_contribution.color * ambient_contribution.intensity) + light_color),
            &Srgb::<V::Scalar>::new(zero, zero, zero),
        )
    }

    #[inline(always)]
    // Separate function for antialiased raytracing
    fn antialiased_raytrace<'a, V>(
        coords: V,
        unit_z: V,
        spheres: impl SceneObjectIterator<'a, V>,
        lights: impl SceneLightIterator<'a, V>,
    ) -> Option<(
        ColorType<V::Scalar>,
        <<V as Vector>::Scalar as SimdValue>::SimdBool,
    )>
    where
        V: 'a + SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector>,
        ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
        Standard: Distribution<<V as Vector>::Scalar>,
    {
        let samples_per_pixel = 32;
        let scale = V::Scalar::from_subset(&(1.0 / samples_per_pixel as f32));

        let zero = V::Scalar::zero();
        let bool_false =
            <<<V as Vector>::Scalar as SimdValue>::SimdBool as BoolMask>::from_bool(false);

        let initial_color = ColorType::new(zero, zero, zero);

        // Sample multiple rays and average the results
        let (res_color, valid_mask) = (0..samples_per_pixel)
            .into_par_iter()
            .filter_map(|_| {
                Self::single_raytrace(
                    coords + V::sample_random(),
                    unit_z,
                    spheres.clone(),
                    lights.clone(),
                )
                .map(|(c, m)| (c * scale, m))
            })
            .reduce(
                || (initial_color, bool_false),
                |(res_color, res_mask), (color, mask)| (color + res_color, res_mask | mask),
            );

        Some((res_color, valid_mask))
    }

    fn render_pixel_colors<'a>(
        coords: RenderCoordinatesVectorized<'a>,
        set_pixel: &dyn Fn(usize, Pixel),
    ) {
        const CHUNK_SIZE: usize = 8;

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
                let Some((colors, valid_mask)) = Self::get_pixel_color_vectorized(
                    coords,
                    Vec3x8::unit_z(),
                    SPHERES.iter(),
                    LIGHTS.iter(),
                ) else {
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
                    let coords = RenderCoordinates {
                        x: x.floor() as usize,
                        y: y.floor() as usize,
                    };
                    if let Some(pixel) = Self::get_pixel_color(coords) {
                        set_pixel(i, pixel);
                    }
                }
            }
        });
    }
}

impl<const W: usize, const H: usize, C: OutputColorEncoder> Renderer<W, H, C>
    for RaytracerRenderer<C>
{
    fn render(&self, buffer: &ImageBuffer<W, H>)
    where
        [(); W * H]:,
    {
        if cfg!(feature = "simd_render") {
            Self::render_to_buffer_chunked_inplace(buffer, Self::render_pixel_colors)
        } else {
            Self::render_to_buffer(buffer, Self::get_pixel_color)
        }
    }
}
