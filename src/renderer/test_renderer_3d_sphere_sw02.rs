use crate::helpers::Pixel;
use crate::image_buffer::ImageBuffer;
use crate::output::OutputColorEncoder;
use crate::renderer::{RenderCoordinates, RenderCoordinatesVectorized, Renderer};
use crate::{WINDOW_HEIGHT, WINDOW_WIDTH};
use color::{OpaqueColor, Srgb};
use itertools::{Chunk, Itertools, izip};
use std::intrinsics::{likely, unlikely};
use std::marker::PhantomData;
use ultraviolet::{Vec2x4, Vec3, Vec3x4, Vec3x8, f32x4, f32x8};
use wide::CmpGt;

struct Ray {
    origin: Vec3,
    direction: Vec3,
    direction_mag_squared: f32,
}

impl Ray {
    const INVALID_VALUE: f32 = f32::INFINITY;

    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
            direction_mag_squared: direction.mag_sq(),
        }
    }

    #[inline(always)]
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + t * self.direction
    }
}

struct Rayx4 {
    origin: Vec3x4,
    direction: Vec3x4,
    direction_mag_squared: f32x4,
}

impl Rayx4 {
    const INVALID_VALUE: f32 = f32::INFINITY;

    const INVALID_VALUE_SPLATTED: f32x4 = f32x4::new([Self::INVALID_VALUE; 4]);

    pub fn new(origin: Vec3x4, direction: Vec3x4) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
            direction_mag_squared: direction.mag_sq(),
        }
    }

    #[inline(always)]
    pub fn at(&self, t: f32x4) -> Vec3x4 {
        self.origin + t * self.direction
    }
}

struct Rayx8 {
    origin: Vec3x8,
    direction: Vec3x8,
    direction_mag_squared: f32x8,
}

impl Rayx8 {
    const INVALID_VALUE: f32 = f32::INFINITY;
    const INVALID_VALUE_SPLATTED: f32x8 = f32x8::new([Self::INVALID_VALUE; 8]);

    pub fn new(origin: Vec3x8, direction: Vec3x8) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
            direction_mag_squared: direction.mag_sq(),
        }
    }

    #[inline(always)]
    pub fn at(&self, t: f32x8) -> Vec3x8 {
        self.origin + t * self.direction
    }
}

trait Intersectable {
    fn intersect(&self, ray: &Ray) -> Option<(Vec3, f32)>;
    fn intersect_x4(&self, ray: &Rayx4) -> (Vec3x4, f32x4);
    fn intersect_x8(&self, ray: &Rayx8) -> (Vec3x8, f32x8);
}

struct SphereData {
    c: Vec3,
    r_sq: f32,
    r_inv: f32,
    color: OpaqueColor<Srgb>,
}

impl SphereData {
    const fn new(c: Vec3, r: f32, color: OpaqueColor<Srgb>) -> Self {
        Self {
            c,
            r_sq: r * r,
            r_inv: 1.0 / r,
            color,
        }
    }
}

macro_rules! intersect_sphere_simd_impl {
    ($fnname: ident,$x: ident) => {
        fn $fnname(
            &self,
            ray: &concat_idents!(Ray, $x),
        ) -> (concat_idents!(Vec3, $x), concat_idents!(f32, $x)) {
            type VecType = concat_idents!(Vec3, $x);
            type F32Type = concat_idents!(f32, $x);
            type RayType = concat_idents!(Ray, $x);

            let u = ray.direction;
            let v = ray.origin - VecType::splat(self.c);

            let a = 2.0 * ray.direction_mag_squared; // u dot u
            let b = 2.0 * u.dot(v);
            let c = v.dot(v) - F32Type::splat(self.r_sq);

            let discriminant = b * b - 2.0 * a * c;

            let discriminant_pos = discriminant.cmp_gt(F32Type::ZERO);
            let discriminant_sqrt = discriminant.sqrt();

            let t1 = (-b - discriminant_sqrt) / a;
            let t1_valid = t1.cmp_gt(F32Type::ZERO) & discriminant_pos;

            // let t2 = (-b + discriminant_sqrt) / a;
            // let t2_valid = t2.cmp_gt(F32Type::ZERO) & discriminant_pos;
            //
            // let t = t2_valid.blend(t2, RayType::INVALID_VALUE_SPLATTED);
            // let t = t1_valid.blend(t1, t);

            let t = t1_valid.blend(t1, RayType::INVALID_VALUE_SPLATTED);

            (ray.at(t), t)
        }
    };
}

impl Intersectable for SphereData {
    fn intersect(&self, ray: &Ray) -> Option<(Vec3, f32)> {
        let u = ray.direction;
        let v = ray.origin - self.c;

        let a = 2.0 * ray.direction_mag_squared;
        let b = 2.0 * u.dot(v);
        let c = v.dot(v) - self.r_sq;

        let discriminant = b * b - 2.0 * a * c;

        if discriminant < 0.0 {
            return None;
        }

        let t = (-b - discriminant.sqrt()) / a;

        // if t <= 0.0 {
        //     let t2 = (-b + discriminant.sqrt()) / a;
        //     if t2 > 0.0 {
        //         t = t2;
        //     } else {
        //         return None;
        //     }
        // }

        Some((ray.at(t), t))
    }

    intersect_sphere_simd_impl!(intersect_x4, x4);
    intersect_sphere_simd_impl!(intersect_x8, x8);
}

static SPHERE_1: SphereData = SphereData::new(
    Vec3::new(WINDOW_WIDTH as f32 / 2.0, WINDOW_HEIGHT as f32 / 2.0, 150.0),
    70.0,
    OpaqueColor::from_rgb8(255, 0, 0),
);

static SPHERE_2: SphereData = SphereData::new(
    Vec3::new(WINDOW_WIDTH as f32 / 2.5, WINDOW_HEIGHT as f32 / 2.5, 150.0),
    90.0,
    OpaqueColor::from_rgb8(0, 255, 0),
);

static SPHERE_3: SphereData = SphereData::new(
    Vec3::new(
        2.0 * (WINDOW_WIDTH as f32 / 2.5),
        WINDOW_HEIGHT as f32 / 2.5,
        150.0,
    ),
    90.0,
    OpaqueColor::from_rgb8(0, 0, 255),
);

static SPHERE_4: SphereData = SphereData::new(
    Vec3::new(
        2.0 * (WINDOW_WIDTH as f32 / 2.5),
        2.0 * (WINDOW_HEIGHT as f32 / 2.5),
        250.0,
    ),
    120.0,
    OpaqueColor::from_rgb8(158, 0, 255),
);

static SPHERE_5: SphereData = SphereData::new(
    Vec3::new(
        1.25 * (WINDOW_WIDTH as f32 / 2.5),
        0.5 * (WINDOW_HEIGHT as f32 / 2.5),
        90.0,
    ),
    30.0,
    OpaqueColor::from_rgb8(128, 210, 255),
);

static SPHERE_6: SphereData = SphereData::new(
    Vec3::new(
        (WINDOW_WIDTH as f32 / 2.5),
        2.25 * (WINDOW_HEIGHT as f32 / 2.5),
        500.0,
    ),
    250.0,
    OpaqueColor::from_rgb8(254, 255, 255),
);

static SPHERE_7: SphereData = SphereData::new(
    Vec3::new(
        WINDOW_WIDTH as f32 / 4.0,
        3.0 * (WINDOW_HEIGHT as f32 / 4.0),
        20.0,
    ),
    10.0,
    OpaqueColor::from_rgb8(255, 55, 77),
);

static SPHERE_8: SphereData = SphereData::new(
    Vec3::new(
        WINDOW_WIDTH as f32 / 3.0,
        3.0 * (WINDOW_HEIGHT as f32 / 6.0),
        30.0,
    ),
    25.0,
    OpaqueColor::from_rgb8(55, 230, 180),
);

static SPHERES: [&'static SphereData; 8] = [
    &SPHERE_1, &SPHERE_2, &SPHERE_3, &SPHERE_4, &SPHERE_5, &SPHERE_6, &SPHERE_7, &SPHERE_8,
];

static RENDER_RAY_DIRECTION: Vec3 = Vec3::new(0.0, 0.0, 1.0);

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct TestRenderer3DSphereSW02<C: OutputColorEncoder>(PhantomData<C>);

impl<C: OutputColorEncoder> TestRenderer3DSphereSW02<C> {
    fn get_pixel_color(RenderCoordinates { x, y }: RenderCoordinates) -> Option<Pixel> {
        let ray = Ray::new(Vec3::new(x as f32, y as f32, 0.0), RENDER_RAY_DIRECTION);

        let mut pixel_color: Option<(f32, OpaqueColor<Srgb>)> = None;

        for sphere in SPHERES {
            if let Some((_, distance)) = sphere.intersect(&ray) {
                let d = distance / sphere.c.mag();
                let color = sphere.color.map_lightness(|l| l - 2.0 * d);
                if let Some((prev_dist, _)) = pixel_color {
                    if prev_dist > distance {
                        pixel_color = Some((distance, color));
                    }
                } else {
                    pixel_color = Some((distance, color));
                }
            }
        }

        pixel_color.map(|c| Pixel(c.1))
    }

    fn render_pixel_colors<'a>(
        coords: RenderCoordinatesVectorized<'a>,
        set_pixel: &dyn Fn(usize, Pixel),
    ) {
        izip!(
            coords.i.chunks(8),
            coords.x.chunks(8),
            coords.y.chunks(8),
            coords.z.chunks(8),
        ).for_each(|(idxs, xs, ys, zs)| {
            let len = idxs.len();

            macro_rules! impl_render_pixel_colors_simd {
                ($fnname: ident, $xn: ident, $xs: ident, $ys: ident, $zs: ident, $idxs: ident, $set_pixel: ident) => {
                    type VecType = concat_idents!(Vec3, $xn);
                    type F32Type = concat_idents!(f32, $xn);
                    type RayType = concat_idents!(Ray, $xn);

                    let ray = RayType::new(
                        VecType::new(F32Type::from($xs), F32Type::from($ys), F32Type::from($zs)),
                        VecType::unit_z()
                    );


                    let mut min_distances = vec![f32::INFINITY; $idxs.len()];

                    for sphere in SPHERES {
                        let (_, distances) = sphere.$fnname(&ray);

                        let d = distances / sphere.c.mag();

                        for (i, &v) in d.as_array_ref().iter().enumerate() {
                            if  min_distances[i] > v {
                                min_distances[i] = v;
                                if v.is_finite() && !v.is_nan()  {
                                    $set_pixel($idxs[i], Pixel(sphere.color.map_lightness(|l| l - 2.0 * v)));
                                }
                            }
                        }
                    }
                };
            }



            if likely(len == 8) {
                impl_render_pixel_colors_simd!(intersect_x8, x8, xs, ys, zs, idxs, set_pixel);
            } else if unlikely(len == 4) {
                impl_render_pixel_colors_simd!(intersect_x4, x4, xs, ys, zs, idxs, set_pixel);
            }  else {
                izip!(
                    xs.iter(),
                    ys.iter(),
                    idxs.iter()
                )
                    .map(|(x, y, i)| (*i, RenderCoordinates { x: x.floor() as usize, y: y.floor() as usize }))
                    .map(|(i, coords)| (i, Self::get_pixel_color(coords)))
                    .for_each(|(i, color)| {
                        if let Some(pixel_color) = color {
                            set_pixel(i, pixel_color);
                        }
                    });
            }
        });
    }
}

impl<const W: usize, const H: usize, C: OutputColorEncoder> Renderer<W, H, C>
    for TestRenderer3DSphereSW02<C>
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
