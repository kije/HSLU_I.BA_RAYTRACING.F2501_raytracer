use crate::extensions::SrgbColorConvertExt;
use crate::helpers::{ColorType, Pixel, Splatable};
use crate::image_buffer::ImageBuffer;
use crate::output::OutputColorEncoder;
use crate::renderer::{RenderCoordinates, RenderCoordinatesVectorized, Renderer};
use crate::{WINDOW_HEIGHT, WINDOW_WIDTH};
use itertools::izip;
use rayon::iter::ParallelIterator;
use simba::simd::SimdComplexField;

// Import the consolidated traits
use crate::color_traits::LightCompatibleColor;
use crate::vector_traits::{RenderingVector, SimdRenderingVector};

use crate::vector::{NormalizableVector, Vector};

use crate::color::ColorSimdExt;
use crate::geometry::{Ray, SphereData, TriangleData};
use crate::raytracing::Intersectable;
use crate::raytracing::Material;
use crate::raytracing::SurfaceInteraction;
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
use std::rc::Rc;
use std::sync::Arc;
use std::sync::LazyLock;
use ultraviolet::{Vec3, Vec3x8, f32x8};

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

// For SIMD, we need to split objects by type for efficient processing
// For a real raytracer, you'd want to implement spatial partitioning (BVH, KD-tree, etc.)

static SIMD_SPHERES: LazyLock<[SphereData<Vec3>; 8]> = LazyLock::new(|| {
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
        SphereData::with_material(
            Vec3::new(
                2.0 * (WINDOW_WIDTH as f32 / 2.5),
                2.0 * (WINDOW_HEIGHT as f32 / 2.5),
                250.0,
            ),
            120.0,
            Material::new(
                ColorType::new(158.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0),
                0.85,
                0.25,
            ),
        ),
        SphereData::with_material(
            Vec3::new(
                1.25 * (WINDOW_WIDTH as f32 / 2.5),
                0.5 * (WINDOW_HEIGHT as f32 / 2.5),
                90.0,
            ),
            30.0,
            Material::new(
                ColorType::new(128.0 / 255.0, 210.0 / 255.0, 255.0 / 255.0),
                1.0,
                0.5,
            ),
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

static SIMD_TRIANGLES: LazyLock<Vec<TriangleData<Vec3>>> = LazyLock::new(|| {
    vec![TriangleData::with_material(
        Vec3::new(WINDOW_WIDTH as f32 * 0.2, WINDOW_HEIGHT as f32 * 0.2, 240.0),
        Vec3::new(WINDOW_WIDTH as f32 * 0.5, WINDOW_HEIGHT as f32 * 0.8, 200.0),
        Vec3::new(WINDOW_WIDTH as f32 * 0.7, WINDOW_HEIGHT as f32 * 0.2, 160.0),
        Material::new(ColorType::new(0.5, 0.7, 0.8), 0.5, 0.5),
    )]
});

trait SceneSphereIterator<'a, V: 'a + SimdRenderingVector>:
    IntoIterator<Item = &'a SphereData<V::SingleValueVector>> + Clone + Sync
{
}

impl<'a, T, V: 'a + SimdRenderingVector> SceneSphereIterator<'a, V> for T where
    T: IntoIterator<Item = &'a SphereData<V::SingleValueVector>> + Clone + Sync
{
}

trait SceneTriangleIterator<'a, V: 'a + SimdRenderingVector>:
    IntoIterator<Item = &'a TriangleData<V::SingleValueVector>> + Clone + Sync
{
}

impl<'a, T, V: 'a + SimdRenderingVector> SceneTriangleIterator<'a, V> for T where
    T: IntoIterator<Item = &'a TriangleData<V::SingleValueVector>> + Clone + Sync
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
            SIMD_SPHERES.iter(),
            SIMD_TRIANGLES.iter(),
            LIGHTS.iter(),
        )?;

        if unlikely(!valid) {
            return None;
        }

        Some(Pixel(colors))
    }

    fn get_pixel_color_vectorized<'a, V>(
        coords: V,
        unit_z: V,
        spheres: impl SceneSphereIterator<'a, V>,
        triangles: impl SceneTriangleIterator<'a, V>,
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
            Self::antialiased_raytrace(coords, unit_z, spheres, triangles, lights)
        } else {
            Self::single_raytrace(coords, unit_z, spheres, triangles, lights)
        }
    }

    fn single_raytrace<'a, V>(
        coords: V,
        unit_z: V,
        spheres: impl SceneSphereIterator<'a, V>,
        triangles: impl SceneTriangleIterator<'a, V>,
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

        // For SIMD rays, we process each geometry type separately and find the nearest
        let mut nearest_interaction: Option<SurfaceInteraction<V>> = None;

        // Process spheres
        for sphere in spheres {
            // Create SIMD sphere
            let sphere_simd = SphereData::<V>::splat(sphere);

            // Check intersection
            if let Some(interaction) = sphere_simd.intersect(&ray) {
                if interaction.valid_mask.none() {
                    continue;
                }

                // Update nearest interaction
                nearest_interaction = match nearest_interaction {
                    None => Some(interaction),
                    Some(ref current) => {
                        let closer =
                            interaction.distance.simd_lt(current.distance) & interaction.valid_mask;

                        if closer.none() {
                            nearest_interaction
                        } else if closer.all() {
                            Some(interaction)
                        } else {
                            Some(SurfaceInteraction::blend(closer, &interaction, current))
                        }
                    }
                };
            }
        }

        // Process triangles
        for triangle in triangles {
            // Create SIMD triangle
            let triangle_simd = TriangleData::<V>::splat(triangle);

            // Check intersection
            if let Some(interaction) = triangle_simd.intersect(&ray) {
                if interaction.valid_mask.none() {
                    continue;
                }

                // Update nearest interaction
                nearest_interaction = match nearest_interaction {
                    None => Some(interaction),
                    Some(ref current) => {
                        let closer =
                            interaction.distance.simd_lt(current.distance) & interaction.valid_mask;

                        if closer.none() {
                            nearest_interaction
                        } else if closer.all() {
                            Some(interaction)
                        } else {
                            Some(SurfaceInteraction::blend(closer, &interaction, current))
                        }
                    }
                };
            }
        }

        // If no intersection found, return None
        let interaction = nearest_interaction?;

        // Calculate lighting
        let color = Self::calculate_lighting(&interaction, ray.direction.normalized(), lights);

        Some((color, interaction.valid_mask))
    }

    fn calculate_lighting<'a, V>(
        interaction: &SurfaceInteraction<V>,
        view_dir: V,
        lights: impl SceneLightIterator<'a, V>,
    ) -> ColorType<V::Scalar>
    where
        V: 'a + SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector>,
        ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
    {
        let zero = V::Scalar::zero();
        let one = V::Scalar::one();

        // Create ambient light
        let ambient_light =
            AmbientLight::<V>::new(ColorType::new(one, one, one), V::Scalar::from_subset(&0.1));

        // Get material color
        let material_color = interaction.material.color.clone();

        // Calculate ambient lighting
        let ambient_contribution = ColorType::blend(
            interaction.valid_mask,
            &(material_color.clone() * ambient_light.color),
            &Srgb::<V::Scalar>::new(zero, zero, zero),
        ) * ambient_light.intensity;

        // Calculate direct lighting from all point lights
        let mut light_color = Srgb::<V::Scalar>::new(zero, zero, zero);

        for light in lights {
            // Create SIMD light
            let light = PointLight::<V>::splat(&light);
            let contribution =
                light.calculate_contribution_at(interaction, interaction.point, view_dir);

            let light_position = light.position;
            let light_color_simd = contribution.color;
            let light_intensity = contribution.intensity;

            // Calculate light direction (from intersection point to light)
            let light_dir = (light_position - interaction.point).normalized();

            // Calculate diffuse factor
            let diffuse_factor = interaction.normal.dot(light_dir).simd_max(zero);

            // Calculate specular reflection
            let reflection = light_dir.reflected(interaction.normal);
            let specular_factor = reflection
                .normalized()
                .dot(view_dir)
                .simd_max(zero)
                .simd_powi(32);

            // Adjust specular based on material roughness
            let roughness_factor = V::Scalar::one() - interaction.material.roughness;
            let specular = specular_factor * roughness_factor;

            // Combined lighting for this light
            let light_factor = (diffuse_factor + specular) * light_intensity;

            // Only apply light where the surface faces the light
            let light_valid = diffuse_factor.simd_gt(zero);

            // Add light contribution
            light_color = light_color
                + ColorType::blend(
                    light_valid & interaction.valid_mask,
                    &(light_color_simd * light_factor),
                    &Srgb::<V::Scalar>::new(zero, zero, zero),
                );
        }

        // Combine ambient and direct lighting
        ambient_contribution + light_color
    }

    #[inline(always)]
    fn antialiased_raytrace<'a, V>(
        coords: V,
        unit_z: V,
        spheres: impl SceneSphereIterator<'a, V>,
        triangles: impl SceneTriangleIterator<'a, V>,
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
                    triangles.clone(),
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
                    SIMD_SPHERES.iter(),
                    SIMD_TRIANGLES.iter(),
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
