use crate::extensions::SrgbColorConvertExt;
use crate::helpers::{ColorType, Pixel, Splatable};
use crate::image_buffer::ImageBuffer;
use crate::output::OutputColorEncoder;
use crate::renderer::{RenderCoordinates, RenderCoordinatesVectorized, Renderer};
use crate::{WINDOW_HEIGHT, WINDOW_WIDTH};
use itertools::{Itertools, izip};

use crate::vector::{
    CommonVecOperations, CommonVecOperationsFloat, CommonVecOperationsSimdOperations,
};

use crate::color::ColorSimdExt;
use crate::geometry::{Ray, SphereData};
use crate::raytracing::{Intersectable, RayIntersection, RayIntersectionCandidate};
use crate::scene::{AmbientLight, Light, PointLight};
use num_traits::{Float, NumOps, One, Zero};
use palette::blend::{Blend, Premultiply};
use palette::bool_mask::{HasBoolMask, LazySelect};
use palette::cast::ArrayCast;
use palette::stimulus::StimulusColor;
use palette::{Darken, Mix, Srgb};
use rand::distributions::{Distribution, Standard};
use rand::{Rng, random};
use simba::scalar::{SubsetOf, SupersetOf};
use simba::simd::{SimdBool, SimdComplexField, SimdPartialOrd, SimdRealField, SimdValue};
use std::borrow::Cow;
use std::fmt::Debug;
use std::intrinsics::{likely, unlikely};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};
use std::sync::LazyLock;
use std::thread;
use ultraviolet::{Vec3, Vec3x8, f32x8};
// Todo optimization-idea: ensure that ray direction & normals (on intersection) are unit vectors in constructors. This way we can simplify the maths in certain cases

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

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct TestRenderer3DSW03CommonCode<C: OutputColorEncoder>(PhantomData<C>);

impl<C: OutputColorEncoder> TestRenderer3DSW03CommonCode<C> {
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

    fn get_pixel_color_vectorized<'a, Vector>(coords: Vector, unit_z: Vector, spheres: impl IntoIterator<Item=&'a SphereData<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector>>+Clone, lights: impl IntoIterator<Item=&'a PointLight<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector>>+Clone) -> Option<(ColorType<Vector::Scalar>,  <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool)>
    where
        Vector: 'a + crate::vector::Vector + Copy + CommonVecOperations + CommonVecOperationsFloat + CommonVecOperationsSimdOperations
        + Add<Vector, Output = Vector>
        + Sub<Vector, Output = Vector>
        + Mul<Vector, Output = Vector>,
        Vector::Scalar:
        Zero  + One + Copy + NumOps<Vector::Scalar, Vector::Scalar> + SimdRealField + SimdPartialOrd + SubsetOf<<Vector as crate::vector::Vector>::Scalar>
        + Splatable<<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar>
    + palette::num::Real
    + palette::num::Zero
    + palette::num::One
    + palette::num::Arithmetics
    + palette::num::Clamp
    + palette::num::Sqrt
    + palette::num::Abs
    + palette::num::PartialCmp
    + HasBoolMask
    + palette::num::MinMax,
     Standard: Distribution<Vector::Scalar>,
    ColorType<<Vector as crate::vector::Vector>::Scalar>: Premultiply<Scalar = Vector::Scalar> + StimulusColor + ArrayCast<Array = [Vector::Scalar; <Vector as crate::vector::Vector>::DIMENSIONS]>,
        <<Vector as crate::vector::Vector>::Scalar as HasBoolMask>::Mask: LazySelect<<Vector as crate::vector::Vector>::Scalar>,
        <Vector::Scalar as SimdValue>::Element: Float + Copy,
        <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool: Debug + SimdValue<Element = bool>,
        [(); <Vector as crate::vector::Vector>::LANES]:,
        <<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar: SubsetOf<<Vector as crate::vector::Vector>::Scalar>, <<Vector as crate::vector::Vector>::Scalar as SimdValue>::Element: SubsetOf<<Vector as crate::vector::Vector>::Scalar>
    {
        let raytrace = move |coords: Vector| {
            let ray = Ray::<Vector>::new_with_mask(
                coords,
                unit_z,
                <Vector::Scalar as SimdValue>::SimdBool::splat(false), // todo check if it matters if we use false or tre here: logic-wise and performance-wise
            );

            let nearest_intersection =
                spheres
                    .clone()
                    .into_iter()
                    .fold(None, |previous_intersection, sphere| {
                        let sphere = SphereData::<Vector>::splat(sphere);
                        let new_intersection = sphere.check_intersection(&ray, Cow::Owned(sphere));

                        // if we have no intersection, return previous nearest_intersection
                        if new_intersection.valid_mask.none() {
                            return previous_intersection;
                        }

                        // if nearest_intersection is none, return current intersection
                        let Some(previous_intersection) = previous_intersection else {
                            return Some(new_intersection);
                        };

                        // if nearest_intersection has no intersections, return current intersection (as this is by now guaranteed to have at least one intersection)
                        if previous_intersection.valid_mask.none() {
                            return Some(new_intersection);
                        }

                        let new_is_nearer = previous_intersection.t.simd_ge(new_intersection.t);

                        if new_is_nearer.none() {
                            return Some(previous_intersection);
                        } else if new_is_nearer.all() {
                            return Some(new_intersection);
                        }

                        // now the complex case:
                        // we need to merge the two intersections
                        // compare nearest_intersection's with intersection (take the minimum of the two)
                        // but take care that we only consider valid values (the ones where both valid masks are ture)
                        // for the values where only one of the valid maks are true (xor both valiud masks), take the one
                        // from the intersection where it is true

                        // Suppose previous_intersection is the best intersection found so far.
                        // new_intersect is from the current sphere (using the final_t & final_valid_mask above).
                        let previous_valid = previous_intersection.valid_mask;
                        let new_valid = new_intersection.valid_mask;

                        // 1) Compute the “pick old” mask in a single step.
                        let pick_old_mask =
                            // If old is valid but new is not:
                            (previous_valid & !new_valid)
                                // OR if both are valid but old is nearer:
                                | (previous_valid & new_valid & !new_is_nearer);

                        // 2) Blend each field once using the same mask.
                        let merged_t = previous_intersection
                            .t
                            .select(pick_old_mask.clone(), new_intersection.t);

                        let merged_payload = SphereData::<Vector>::blend(
                            pick_old_mask,
                            &previous_intersection.payload,
                            &new_intersection.payload,
                        );

                        // 3) The new valid lanes are old OR new
                        let merged_valid = previous_valid | new_valid;

                        let merged_candidate = RayIntersectionCandidate::new(
                            merged_t,
                            Cow::Owned(merged_payload),
                            merged_valid,
                        );

                        Some(merged_candidate)
                    });

            let Some(nearest_intersection) = nearest_intersection else {
                return None;
            };

            if nearest_intersection.valid_mask.none() {
                return None;
            }

            let intersected_object = nearest_intersection.payload;
            let RayIntersection {
                distance,
                intersection: intersection_point,
                valid_mask,
                normal: intersection_normal,
                ..
            } = intersected_object.intersect(
                &ray,
                &RayIntersectionCandidate::new(
                    nearest_intersection.t,
                    intersected_object.as_ref(),
                    nearest_intersection.valid_mask,
                ),
            );

            //////////////////////

            let zero = Vector::Scalar::zero();

            let ambient_light = AmbientLight::new(
                ColorType::new(
                    Vector::Scalar::from_subset(&1.0),
                    Vector::Scalar::from_subset(&1.0),
                    Vector::Scalar::from_subset(&1.0),
                ),
                Vector::Scalar::from_subset(&0.08),
            );

            // Fixme replace manual ambient lighting amount/color with AmbientIllumination struct
            let mut light_color = Srgb::<Vector::Scalar>::new(zero, zero, zero);

            for light in lights.clone().into_iter() {
                let light = PointLight::<Vector>::splat(light);
                let direct_light_contribution = light.calculate_contribution_at(
                    intersected_object.as_ref(),
                    intersection_point,
                    ray.direction.normalized(),
                );
                light_color = light_color
                    + ColorType::blend(
                        direct_light_contribution.valid_mask,
                        &(direct_light_contribution.color * direct_light_contribution.intensity),
                        &Srgb::<Vector::Scalar>::new(zero, zero, zero),
                    );
            }

            let ambient_color = ambient_light.calculate_contribution_at(
                intersected_object.as_ref(),
                intersection_point,
                ray.direction.normalized(),
            );

            Some((
                ColorType::blend(
                    valid_mask,
                    &((ambient_color.color * ambient_color.intensity) + light_color),
                    &Srgb::<Vector::Scalar>::new(zero, zero, zero),
                ),
                valid_mask,
            ))
        };

        // fixme move this antialiasing somewhere else
        // probably better if we would do it in such a way as to fill a entire simd x8 lane with variations to raytrace that map to 1 pixel
        // this way we would
        let antialiasing = false;

        if antialiasing {
            // FIXME: Antialiasing does not need to do the lighting step (?), just the intersection and object color step
            let mut valid_mask = None;
            let mut res_color = ColorType::new(
                Vector::Scalar::zero(),
                Vector::Scalar::zero(),
                Vector::Scalar::zero(),
            );
            let samples_per_pixel = 32;
            let scale = Vector::Scalar::from_subset(&(1.0 / samples_per_pixel as f32));
            for _ in 0..samples_per_pixel {
                let vec = Vector::sample_random();
                let (color, mask) = raytrace(coords + vec).unwrap_or_else(|| {
                    (
                        res_color,
                        <Vector::Scalar as SimdValue>::SimdBool::splat(false),
                    )
                });

                if valid_mask.is_none() {
                    valid_mask = Some(mask);
                }

                valid_mask = valid_mask.map(|valid_mask| valid_mask | mask);

                res_color +=
                    ColorType::new(color.red * scale, color.green * scale, color.blue * scale);
            }

            valid_mask.map(|valid_mask| (res_color, valid_mask))
        } else {
            raytrace(coords)
        }
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
    for TestRenderer3DSW03CommonCode<C>
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
