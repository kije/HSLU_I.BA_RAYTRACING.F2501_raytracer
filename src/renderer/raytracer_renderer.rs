use crate::extensions::SrgbColorConvertExt;
use crate::helpers::{ColorType, Pixel, Splatable};
use crate::image_buffer::ImageBuffer;
use crate::output::OutputColorEncoder;
use crate::renderer::{RenderCoordinates, RenderCoordinatesVectorized, Renderer};
use crate::{WINDOW_HEIGHT, WINDOW_WIDTH};
use itertools::{Itertools, izip};
use rayon::iter::ParallelIterator;
use simba::simd::SimdComplexField;

// Import the consolidated traits
use crate::color_traits::LightCompatibleColor;
use crate::vector_traits::{RenderingVector, SimdRenderingVector};

use crate::vector::{NormalizableVector, Vector};

use crate::color::ColorSimdExt;
use crate::geometry::{
    BoundedPlane, CompositeGeometry, GeometryCollection, Ray, RenderGeometry, RenderGeometryKind,
    SphereData, TriangleData,
};
use crate::raytracing::Material;
use crate::raytracing::SurfaceInteraction;
use crate::raytracing::{Intersectable, Raytracer};
use crate::scene::{AmbientLight, Light, PointLight, Scene};
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
use ultraviolet::{Rotor3, Vec3, Vec3x8, f32x8};

const SCENE_DEPTH: f32 = 1000.0;
static RENDER_RAY_DIRECTION: Vec3 = Vec3::new(0.0, 0.0, 1.0);
static RENDER_RAY_FOCUS: Vec3 = Vec3::new(
    WINDOW_WIDTH as f32 / 2.0,
    WINDOW_HEIGHT as f32 / 2.0,
    -2.0 * SCENE_DEPTH,
);

const RAYTRACE_REFLECTION_MAX_DEPTH: usize = if cfg!(feature = "high_quality") {
    18
} else {
    9
};

const POINT_LIGHT_MULTIPLICATOR: usize = if cfg!(feature = "soft_shadows") {
    if cfg!(feature = "high_quality") {
        14
    } else {
        6
    }
} else {
    1
};

static LIGHTS: LazyLock<[PointLight<Vec3>; { POINT_LIGHT_MULTIPLICATOR * 7 }]> =
    LazyLock::new(|| {
        [
            PointLight::new(
                Vec3::new(WINDOW_WIDTH as f32 / 2.0, WINDOW_HEIGHT as f32 / 2.1, 20.0),
                ColorType::new(0.822, 0.675, 0.45),
                0.25,
            ),
            PointLight::new(
                Vec3::new(WINDOW_WIDTH as f32 / 3.5, WINDOW_HEIGHT as f32 / 3.5, 55.0),
                ColorType::new(0.822, 0.675, 0.45),
                1.0,
            ),
            PointLight::new(
                Vec3::new(WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32 / 2.5, 150.0),
                ColorType::new(1.0, 1.0, 1.0),
                0.85,
            ),
            PointLight::new(
                Vec3::new((WINDOW_WIDTH / 2) as f32, (WINDOW_HEIGHT / 6) as f32, 60.0),
                ColorType::new(0.75, 0.56, 0.5),
                0.5,
            ),
            PointLight::new(
                Vec3::new((WINDOW_WIDTH / 4) as f32, (WINDOW_HEIGHT / 6) as f32, 10.0),
                ColorType::new(0.0, 0.5, 0.4),
                0.3,
            ),
            PointLight::new(
                Vec3::new(
                    (WINDOW_WIDTH as f32) / 1.25,
                    (WINDOW_HEIGHT / 3) as f32,
                    80.0,
                ),
                ColorType::new(0.6, 0.2, 0.3),
                0.35,
            ),
            PointLight::new(
                Vec3::new(
                    (WINDOW_WIDTH as f32) / 2.0,
                    WINDOW_HEIGHT as f32 / 1.1,
                    140.0,
                ),
                ColorType::new(0.5, 0.5, 0.5),
                0.6,
            ),
        ]
        .map(|light| light.to_point_light_cloud::<{ POINT_LIGHT_MULTIPLICATOR }>())
        .into_iter()
        .flatten()
        .collect_array::<{ POINT_LIGHT_MULTIPLICATOR * 7 }>()
        .unwrap()
    });

// Create a lazy-initialized GeometryCollection
static SCENE: LazyLock<Scene<Vec3>> = LazyLock::new(|| {
    let mut scene = Scene::<Vec3>::with_capacities(20);

    // Add spheres
    scene.add_sphere(SphereData::new(
        Vec3::new(
            WINDOW_WIDTH as f32 / 2.5,
            WINDOW_HEIGHT as f32 / 2.75,
            170.0,
        ),
        70.0,
        ColorType::new(255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0),
    ));

    scene.add_sphere(SphereData::with_material(
        Vec3::new(WINDOW_WIDTH as f32 / 2.5, WINDOW_HEIGHT as f32 / 1.5, 170.0),
        70.0,
        Material::new(
            ColorType::new(255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0),
            0.8,
            0.0,
        ),
    ));

    // scene.add_sphere(SphereData::new(
    //     Vec3::new(WINDOW_WIDTH as f32 / 2.5, WINDOW_HEIGHT as f32 / 2.5, 150.0),
    //     90.0,
    //     ColorType::new(0.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0),
    // ));

    scene.add_sphere(SphereData::with_material(
        Vec3::new(
            1.9 * (WINDOW_WIDTH as f32 / 2.5),
            WINDOW_HEIGHT as f32 / 2.5,
            160.0,
        ),
        88.0,
        Material::new(
            ColorType::new(111.0 / 255.0, 255.0 / 255.0, 222.0 / 255.0),
            1.0,
            0.2,
        ),
    ));
    //
    // scene.add_sphere(SphereData::with_material(
    //     Vec3::new(
    //         2.0 * (WINDOW_WIDTH as f32 / 2.5),
    //         2.0 * (WINDOW_HEIGHT as f32 / 2.5),
    //         250.0,
    //     ),
    //     120.0,
    //     Material::new(
    //         ColorType::new(158.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0),
    //         0.85,
    //         0.25,
    //     ),
    // ));
    //
    // scene.add_sphere(SphereData::with_material(
    //     Vec3::new(
    //         1.25 * (WINDOW_WIDTH as f32 / 2.5),
    //         0.5 * (WINDOW_HEIGHT as f32 / 2.5),
    //         90.0,
    //     ),
    //     30.0,
    //     Material::new(
    //         ColorType::new(128.0 / 255.0, 210.0 / 255.0, 255.0 / 255.0),
    //         1.0,
    //         0.5,
    //     ),
    // ));
    //
    scene.add_sphere(SphereData::with_material(
        Vec3::new(
            WINDOW_WIDTH as f32 / 2.5,
            2.25 * (WINDOW_HEIGHT as f32 / 2.5),
            500.0,
        ),
        250.0,
        Material::new(
            ColorType::new(254.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0),
            0.5,
            0.05,
        ),
    ));
    //
    // scene.add_sphere(SphereData::new(
    //     Vec3::new(
    //         WINDOW_WIDTH as f32 / 4.0,
    //         3.0 * (WINDOW_HEIGHT as f32 / 4.0),
    //         20.0,
    //     ),
    //     10.0,
    //     ColorType::new(255.0 / 255.0, 55.0 / 255.0, 77.0 / 255.0),
    // ));
    //
    // scene.add_sphere(SphereData::new(
    //     Vec3::new(
    //         WINDOW_WIDTH as f32 / 3.0,
    //         3.0 * (WINDOW_HEIGHT as f32 / 6.0),
    //         30.0,
    //     ),
    //     25.0,
    //     ColorType::new(55.0 / 255.0, 230.0 / 255.0, 180.0 / 255.0),
    // ));

    // Add triangles
    let mut plane_up = Vec3::unit_y();
    let mut plane_normal = -Vec3::unit_z();
    plane_normal.rotate_by(Rotor3::from_rotation_yz(-0.325));
    plane_up.rotate_by(Rotor3::from_rotation_yz(-0.325));

    scene.add_triangle(TriangleData::with_material(
        Vec3::new(
            WINDOW_WIDTH as f32 * 0.05,
            WINDOW_HEIGHT as f32 * 0.2,
            200.0,
        ),
        Vec3::new(WINDOW_WIDTH as f32 * 0.3, WINDOW_HEIGHT as f32 * 0.5, 200.0),
        Vec3::new(
            WINDOW_WIDTH as f32 * 0.25,
            WINDOW_HEIGHT as f32 * 0.15,
            150.0,
        ),
        Material::new(ColorType::new(0.5, 0.7, 0.8), 0.5, 0.5),
    ));

    scene.add_triangle(TriangleData::with_material(
        Vec3::new(
            WINDOW_WIDTH as f32 * 0.55,
            WINDOW_HEIGHT as f32 * 0.45,
            200.0,
        ),
        Vec3::new(
            WINDOW_WIDTH as f32 * 0.7,
            WINDOW_HEIGHT as f32 * 0.72,
            200.0,
        ),
        Vec3::new(
            WINDOW_WIDTH as f32 * 0.65,
            WINDOW_HEIGHT as f32 * 0.35,
            140.0,
        ),
        Material::new(ColorType::new(0.7, 0.7, 0.8), 0.1, 0.3),
    ));

    // Convert BoundedPlane to basic geometries and add them
    let plane_triangles = BoundedPlane::with_material(
        plane_normal,
        Vec3::new(WINDOW_WIDTH as f32 * 0.5, WINDOW_HEIGHT as f32 * 0.5, 250.0),
        plane_up,
        WINDOW_WIDTH as f32 * 0.55,
        WINDOW_HEIGHT as f32 * 0.55,
        Material::new(ColorType::new(0.6, 0.7, 0.5), 0.95, 0.05),
    )
    .to_basic_geometries();

    for triangle in plane_triangles {
        scene.add_triangle(triangle);
    }

    let back_plane_triangle = BoundedPlane::with_material(
        -Vec3::unit_z(),
        Vec3::new(
            WINDOW_WIDTH as f32 * 0.5,
            WINDOW_HEIGHT as f32 * 0.5,
            10000.0,
        ),
        Vec3::unit_y(),
        WINDOW_WIDTH as f32 * 10.,
        WINDOW_HEIGHT as f32 * 10.,
        Material::new(ColorType::new(0.2, 0.2, 0.2), 0.0, 0.8),
    )
    .to_basic_geometries();

    for triangle in back_plane_triangle {
        scene.add_triangle(triangle);
    }

    scene
});

trait RenderGeometryIterator<'a, V: 'a + SimdRenderingVector>:
    IntoIterator<Item = &'a RenderGeometry<V::SingleValueVector>> + Clone + Sync
{
}

impl<'a, T, V: 'a + SimdRenderingVector> RenderGeometryIterator<'a, V> for T where
    T: IntoIterator<Item = &'a RenderGeometry<V::SingleValueVector>> + Clone + Sync
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
pub struct RaytracerRenderer<C: OutputColorEncoder>(PhantomData<C>);

impl<C: OutputColorEncoder> RaytracerRenderer<C> {
    fn get_pixel_color(RenderCoordinates { x, y }: RenderCoordinates) -> Option<Pixel> {
        let coords = Vec3::new(x as f32, y as f32, 0.0);
        let render_ray_direction = coords - RENDER_RAY_FOCUS;

        let (colors, valid) = Self::get_pixel_color_vectorized(
            coords,
            render_ray_direction,
            &SCENE.scene_objects,
            LIGHTS.iter(),
        )?;

        if unlikely(!valid) {
            return None;
        }

        Some(Pixel(colors))
    }

    fn get_pixel_color_vectorized<'a, V>(
        coords: V,
        direction: V,
        check_objects: &GeometryCollection<V::SingleValueVector>,
        lights: impl SceneLightIterator<'a, V> + Send + 'static,
    ) -> Option<(
        ColorType<V::Scalar>,
        <<V as Vector>::Scalar as SimdValue>::SimdBool,
    )>
    where
        V: 'a
            + SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector + Send>
            + Send
            + 'static,
        ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
        Standard: Distribution<<V as Vector>::Scalar>,
        [(); <V as Vector>::LANES]:,
    {
        if cfg!(feature = "anti_aliasing") {
            Self::antialiased_raytrace(coords, direction, check_objects, lights)
        } else {
            Self::single_raytrace::<false, false, _>(coords, direction, check_objects, lights, None)
        }
    }

    /// Does a single raytracing pass.
    ///
    /// `IS_ANTIALIASING_RAY` param may be set to true if the ray is used for antialiasing and might be used to skip complex calculations that are not relevant for antialiasing
    ///
    fn single_raytrace<'a, const IS_ANTIALIASING_RAY: bool, const IS_REFLECTION_RAY: bool, V>(
        coords: V,
        direction: V,
        check_objects: &GeometryCollection<V::SingleValueVector>,
        lights: impl SceneLightIterator<'a, V> + Send + 'static,
        recursion_depth: Option<usize>,
    ) -> Option<(
        ColorType<V::Scalar>,
        <<V as Vector>::Scalar as SimdValue>::SimdBool,
    )>
    where
        V: 'a
            + SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector + Send>
            + Send
            + 'static,
        ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
        Standard: Distribution<<V as Vector>::Scalar>,
    {
        if let Some(recursion_depth) = recursion_depth {
            if recursion_depth == 0 {
                return None;
            }
        }
        let zero = V::Scalar::zero();

        let (ray, nearest_interaction) =
            Raytracer::cast_ray::<IS_ANTIALIASING_RAY, _>(coords, direction, check_objects)?;

        let color = Self::calculate_lighting::<IS_ANTIALIASING_RAY, IS_REFLECTION_RAY, _>(
            &nearest_interaction,
            ray.direction,
            check_objects,
            lights.clone(),
        );

        let reflection_color = if cfg!(feature = "reflections") {
            Self::calculate_reflection::<IS_ANTIALIASING_RAY, IS_REFLECTION_RAY, _>(
                &nearest_interaction,
                ray.direction,
                check_objects,
                lights,
                recursion_depth,
            )
        } else {
            ColorType::new(zero, zero, zero)
        };

        Some((color + reflection_color, nearest_interaction.valid_mask))
    }

    fn calculate_reflection<'a, const IS_ANTIALIASING_RAY: bool, const IS_REFLECTION_RAY: bool, V>(
        interaction: &SurfaceInteraction<V>,
        view_dir: V,
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
        let bool_false =
            <<<V as Vector>::Scalar as SimdValue>::SimdBool as BoolMask>::from_bool(false);

        if interaction.valid_mask.none() {
            return ColorType::new(zero, zero, zero);
        }

        let material_is_reflective = interaction.material.reflectivity.simd_gt(zero);

        if material_is_reflective.none() {
            return ColorType::new(zero, zero, zero);
        }

        macro_rules! impl_reflection_raytrace {
            (
                let $reflection_contribution:ident = $single_raytrace:ident(
                    $interaction_point:expr,
                    $light_dir:expr,
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
                            $check_objects,
                            $lights,
                            $recursion_depth
                        )
                    } else {
                        let lights = $lights.clone();
                        let interaction_point = $interaction_point.clone();
                        let check_objects = $check_objects.clone();
                        let light_dir = $light_dir.clone();
                        std::thread::spawn(move || {
                            impl_reflection_raytrace!(
                                $single_raytrace,
                                interaction_point,
                                light_dir,
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
            ($single_raytrace:ident, $interaction_point:expr, $light_dir:expr, $check_objects:expr, $lights:expr, $recursion_depth:expr) => {
                Self::$single_raytrace::<IS_ANTIALIASING_RAY, true, _>(
                    $interaction_point,
                    $light_dir,
                    $check_objects,
                    $lights,
                    $recursion_depth
                        .map(|d| d - 1)
                        .or(Some(RAYTRACE_REFLECTION_MAX_DEPTH)),
                )
            };
        }

        let reflection_direction = view_dir.reflected(interaction.normal).normalized();

        impl_reflection_raytrace! {
            let reflection_color = single_raytrace(
                (interaction.point + (reflection_direction * V::broadcast(<V as Vector>::Scalar::from_subset(&(1.5e-02_f32))))),
                reflection_direction,
                check_objects,
                lights.clone(),
                recursion_depth,
            )
        }

        let (reflection, reflection_valid) =
            reflection_color.unwrap_or((ColorType::<V::Scalar>::new(zero, zero, zero), bool_false));

        let reflection_valid = reflection_valid & interaction.valid_mask & material_is_reflective;

        ColorType::<V::Scalar>::blend(
            reflection_valid,
            &(reflection * interaction.material.reflectivity),
            &ColorType::<V::Scalar>::new(zero, zero, zero),
        )
    }

    fn calculate_lighting<'a, const IS_ANTIALIASING_RAY: bool, const IS_REFLECTION_RAY: bool, V>(
        interaction: &SurfaceInteraction<V>,
        view_dir: V,
        check_objects: &GeometryCollection<V::SingleValueVector>,
        lights: impl SceneLightIterator<'a, V> + Send + 'static,
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

        if interaction.valid_mask.none() {
            return ColorType::new(zero, zero, zero);
        }

        // Create ambient light
        let ambient_light =
            AmbientLight::<V>::new(ColorType::new(one, one, one), V::Scalar::from_subset(&0.1));

        // Get material color
        let material_color = interaction.material.color.clone();

        // Calculate ambient lighting
        let ambient_contribution = ColorType::blend(
            interaction.valid_mask,
            &(material_color.clone() * ambient_light.color),
            &ColorType::<V::Scalar>::new(zero, zero, zero),
        ) * ambient_light.intensity;

        // Calculate direct lighting from all point lights
        let mut light_color = ColorType::<V::Scalar>::new(zero, zero, zero);

        for light in lights.clone() {
            // Create SIMD light
            let light = PointLight::<V>::splat(&light);

            let light_position = light.position;

            // Calculate light direction (from intersection point to light)
            let light_to_point = (light_position - interaction.point);
            let light_dir = light_to_point.normalized();

            // fixme translucent objects?
            let check_for_light_blocking_point = interaction.point
                + (light_dir * V::broadcast(<V as Vector>::Scalar::from_subset(&1.0e-03_f32))); // move point a bit away from the surface of the object
            let check_for_light_blocking_light_to_point =
                light_position - check_for_light_blocking_point;
            let light_can_reach_point = (!Raytracer::has_any_intersection(
                check_for_light_blocking_point,
                light_dir,
                check_objects,
                check_for_light_blocking_light_to_point.mag(),
            )) & interaction.valid_mask;

            let contribution =
                light.calculate_contribution_at(interaction, interaction.point, view_dir);

            let light_color_simd = contribution.color;
            let light_intensity = contribution.intensity;

            // Calculate diffuse factor
            let diffuse_factor = interaction.normal.dot(light_dir).simd_max(zero);

            // Calculate specular reflection
            let reflection = light_dir.reflected(interaction.normal);

            let specular_factor = reflection
                .normalized()
                .dot(view_dir)
                .simd_max(zero)
                .simd_powf(
                    (interaction.material.shininess * V::Scalar::from_subset(&(512.0)))
                        .simd_max(V::Scalar::one()),
                );

            let specular = specular_factor.select(
                interaction.material.shininess.simd_gt(V::Scalar::zero()),
                V::Scalar::zero(),
            );
            // Combined lighting for this light
            let light_factor = (diffuse_factor) * light_intensity;

            let specular_light = light.color * specular * light_intensity;

            // Only apply light where the surface faces the light & is not blocked by the light
            let light_valid = diffuse_factor.simd_gt(zero) & light_can_reach_point;

            // Add light contribution
            light_color = light_color
                + ColorType::blend(
                    light_valid & interaction.valid_mask,
                    &((light_color_simd * light_factor) + specular_light),
                    &ColorType::<V::Scalar>::new(zero, zero, zero),
                );
        }

        // Combine ambient and direct lighting
        ambient_contribution + light_color
    }

    #[inline(always)]
    fn antialiased_raytrace<'a, V>(
        coords: V,
        direction: V,
        check_objects: &GeometryCollection<V::SingleValueVector>,
        lights: impl SceneLightIterator<'a, V> + Send + 'static,
    ) -> Option<(
        ColorType<V::Scalar>,
        <<V as Vector>::Scalar as SimdValue>::SimdBool,
    )>
    where
        V: 'a
            + SimdRenderingVector<SingleValueVector: RenderingVector + NormalizableVector + Send>
            + Send
            + 'static,
        ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
        Standard: Distribution<<V as Vector>::Scalar>,
    {
        let x = V::unit_x();
        let y = V::unit_y();

        let tl = -x - y;
        let tr = x - y;
        let bl = -x + y;
        let br = x + y;

        let directions = [
            tl.normalized(),
            tr.normalized(),
            bl.normalized(),
            br.normalized(),
        ];

        let samples_per_pixel = if cfg!(feature = "high_quality") {
            36usize
        } else {
            9usize
        };
        let scale = V::Scalar::from_subset(&(1.0 / (samples_per_pixel) as f32));
        let sample_radius = V::broadcast(V::Scalar::from_subset(
            &(0.85 + (samples_per_pixel as f32 / 100.0)),
        ));

        let zero = V::Scalar::zero();
        let bool_false =
            <<<V as Vector>::Scalar as SimdValue>::SimdBool as BoolMask>::from_bool(false);

        let default_color = ColorType::new(zero, zero, zero);
        let default_mask = bool_false;
        let (initial_color, initial_mask) = Self::single_raytrace::<false, false, _>(
            coords,
            direction,
            check_objects,
            lights.clone(),
            None,
        )
        .map(|(c, m)| (c * scale, m))
        .unwrap_or((default_color, default_mask));

        // fixme: maybe reordering the rays here so that 1 simd ray represents the different samples of the antialiasing for the same "real" pixel, because they cohere better, instead of the rays of independent "real" pixels.
        // Sample multiple rays and average the results
        let (res_color, valid_mask) = (0..(samples_per_pixel - 1))
            .into_par_iter()
            .filter_map(|i| {
                let sampling_direction_bias = directions[i % directions.len()];
                let random_vec = V::sample_random() * sampling_direction_bias;
                let (c, m) = Self::single_raytrace::<true, false, _>(
                    coords + (random_vec * sample_radius),
                    direction,
                    check_objects,
                    lights.clone(),
                    None,
                )?;
                Some((c * scale, m))
            })
            .reduce(
                || (default_color, default_mask),
                |(res_color, res_mask), (color, mask)| (color + res_color, res_mask | mask),
            );

        Some((res_color + initial_color, valid_mask | initial_mask))
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
                let render_ray_direction = coords - Vec3x8::splat(RENDER_RAY_FOCUS);
                let Some((colors, valid_mask)) = Self::get_pixel_color_vectorized(
                    coords,
                    render_ray_direction,
                    &SCENE.scene_objects,
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
