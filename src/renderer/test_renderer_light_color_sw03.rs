use crate::helpers::Pixel;
use crate::image_buffer::ImageBuffer;
use crate::output::OutputColorEncoder;
use crate::renderer::{RenderCoordinates, RenderCoordinatesVectorized, Renderer};
use crate::{WINDOW_HEIGHT, WINDOW_WIDTH};
use color::{OpaqueColor, Srgb};
use itertools::{Chunk, Itertools, concat, izip};
use std::borrow::Cow;
use std::intrinsics::{cold_path, likely, unlikely};
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::Not;
use std::ops::{BitAnd, BitXor};
use std::sync::LazyLock;
use ultraviolet::{Vec2x4, Vec3, Vec3x4, Vec3x8, f32x4, f32x8, m32x4, m32x8};
use wide::{CmpEq, CmpGe, CmpGt, i32x8, u32x8};
#[derive(Clone, Debug, Copy)]
struct RayIntersectionCandidate<Scalar, Payload, ValidMask = ()>
where
    Scalar: Sized + Copy,
    ValidMask: Copy,
    Payload: ?Sized,
{
    /// Distance from ray origin
    t: Scalar,
    valid_mask: ValidMask,
    payload: Payload,
}

impl<Scalar, Payload, ValidMask> RayIntersectionCandidate<Scalar, Payload, ValidMask>
where
    Scalar: Sized + Copy,
    ValidMask: Copy,
    Payload: Sized,
{
    const fn new(t: Scalar, payload: Payload, valid_mask: ValidMask) -> Self {
        Self {
            t,
            valid_mask,
            payload,
        }
    }
}

#[derive(Clone, Debug)]
struct RayIntersection<Vector, Scalar, ValidMask = ()>
where
    Vector: Sized,
    Scalar: Sized + Copy,
    ValidMask: Copy,
{
    intersection: Vector,
    normal: Vector,
    distance: Scalar,
    incident_angle_cos: Scalar,
    valid_mask: ValidMask,
}

impl<Vector, Scalar, ValidMask> RayIntersection<Vector, Scalar, ValidMask>
where
    Vector: Sized,
    Scalar: Sized + Copy,
    ValidMask: Copy,
{
    pub const fn new(
        intersection: Vector,
        normal: Vector,
        distance: Scalar,
        incident_angle_cos: Scalar,
        valid_mask: ValidMask,
    ) -> Self {
        Self {
            intersection,
            normal,
            distance,
            incident_angle_cos,
            valid_mask,
        }
    }
}

#[derive(Clone, Debug, Copy)]
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

    //pub fn reflect_at_intersection(&mut self, intersection: &RayIntersection<Vec3, f32, ()>) {}
}

#[derive(Clone, Debug, Copy)]
struct Rayx4 {
    origin: Vec3x4,
    direction: Vec3x4,
    direction_mag_squared: f32x4,
    valid_mask: Option<m32x4>,
}

impl Rayx4 {
    const INVALID_VALUE: f32 = f32::INFINITY;
    const INVALID_VALUE_SPLATTED: f32x4 = f32x4::new([Self::INVALID_VALUE; 4]);

    const INVALID_VECTOR: Vec3x4 = Vec3x4::new(
        Self::INVALID_VALUE_SPLATTED,
        Self::INVALID_VALUE_SPLATTED,
        Self::INVALID_VALUE_SPLATTED,
    );

    pub fn new(origin: Vec3x4, direction: Vec3x4) -> Self {
        Self::new_with_mask(origin, direction, None)
    }

    pub fn new_with_mask(origin: Vec3x4, direction: Vec3x4, valid_mask: Option<m32x4>) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
            direction_mag_squared: direction.mag_sq(),
            valid_mask,
        }
    }

    #[inline(always)]
    pub fn at(&self, t: f32x4) -> Vec3x4 {
        self.origin + t * self.direction
    }

    #[inline(always)]
    pub fn at_masked(&self, t: f32x4, additional_valid_mask: Option<m32x4>) -> Vec3x4 {
        let mask = self
            .valid_mask
            .map_or(
                additional_valid_mask.unwrap_or(m32x4::ONE),
                |s| match additional_valid_mask {
                    Some(m) => s.bitand(m),
                    _ => s,
                },
            );

        if mask.none() {
            return Self::INVALID_VECTOR;
        }

        Vec3x4::blend(mask, self.origin + t * self.direction, Self::INVALID_VECTOR)
    }
}

#[derive(Clone, Debug, Copy)]
struct Rayx8 {
    origin: Vec3x8,
    direction: Vec3x8,
    direction_mag_squared: f32x8,
    valid_mask: Option<m32x8>,
}

impl Rayx8 {
    const INVALID_VALUE: f32 = f32::INFINITY;
    const INVALID_VALUE_SPLATTED: f32x8 = f32x8::new([Self::INVALID_VALUE; 8]);

    const INVALID_VECTOR: Vec3x8 = Vec3x8::new(
        Self::INVALID_VALUE_SPLATTED,
        Self::INVALID_VALUE_SPLATTED,
        Self::INVALID_VALUE_SPLATTED,
    );

    pub fn new(origin: Vec3x8, direction: Vec3x8) -> Self {
        Self::new_with_mask(origin, direction, None)
    }

    pub fn new_with_mask(origin: Vec3x8, direction: Vec3x8, valid_mask: Option<m32x8>) -> Self {
        Self {
            origin,
            direction: direction.normalized(),
            direction_mag_squared: direction.mag_sq(),
            valid_mask,
        }
    }

    #[inline(always)]
    pub fn at(&self, t: f32x8) -> Vec3x8 {
        self.origin + t * self.direction
    }

    #[inline(always)]
    pub fn at_masked(&self, t: f32x8, additional_valid_mask: Option<m32x8>) -> Vec3x8 {
        let mask = self
            .valid_mask
            .map_or(
                additional_valid_mask.unwrap_or(m32x8::ONE),
                |s| match additional_valid_mask {
                    Some(m) => s.bitand(m),
                    _ => s,
                },
            );

        if mask.none() {
            return Self::INVALID_VECTOR;
        }

        Vec3x8::blend(mask, self.origin + t * self.direction, Self::INVALID_VECTOR)
    }
}

trait Intersectable
where
    Self::ScalarType: Sized + Copy,
    Self::MaskType: Sized + Copy,
{
    type RayType;
    type ScalarType;
    type VectorType;
    type MaskType;

    type ReturnTypeWrapper<T>;

    fn check_intersection<'a, P>(
        &'a self,
        ray: &'_ Self::RayType,
        payload: P,
    ) -> Self::ReturnTypeWrapper<RayIntersectionCandidate<Self::ScalarType, P, Self::MaskType>>;

    // FIXME: we might want to move that to the RayIntersectionCandidate type, as you normaly would call this anyyway by candidate.payload.intersect(ray, candidate)?
    // or we want to have a shortcut from RayIntersectionCandidate
    fn intersect<'a>(
        &'a self,
        ray: &'_ Self::RayType,
        candidate: &'_ RayIntersectionCandidate<Self::ScalarType, &'a Self, Self::MaskType>,
    ) -> Self::ReturnTypeWrapper<RayIntersection<Self::VectorType, Self::ScalarType, Self::MaskType>>;
}

#[derive(Debug, Copy, Clone)]
struct SphereData<Vector, Scalar>
where
    Scalar: Sized + Copy,
    Vector: Sized,
{
    c: Vector,
    r_sq: Scalar,
    r_inv: Scalar,
    color: OpaqueColor<Srgb>, // fixme simd
}

impl<Vector, Scalar> SphereData<Vector, Scalar>
where
    Scalar:
        Sized + Copy + From<f32> + std::ops::Div<Output = Scalar> + std::ops::Mul<Output = Scalar>,
    Vector: Sized,
{
    fn new(c: Vector, r: Scalar, color: OpaqueColor<Srgb>) -> Self {
        Self {
            c,
            r_sq: r * r,
            r_inv: Scalar::from(1.0) / r,
            color,
        }
    }
}

macro_rules! intersect_sphere_simd_impl {
    ($x: ident) => {
        impl Intersectable for SphereData<concat_idents!(Vec3, $x), concat_idents!(f32, $x)> {
            type RayType = concat_idents!(Ray, $x);
            type MaskType = concat_idents!(m32, $x);
            type ScalarType = concat_idents!(f32, $x);
            type VectorType = concat_idents!(Vec3, $x);

            type ReturnTypeWrapper<T> = T;

            fn check_intersection<'a, P>(
                &'a self,
                ray: &'_ Self::RayType,
                payload: P,
            ) -> Self::ReturnTypeWrapper<
                RayIntersectionCandidate<Self::ScalarType, P, Self::MaskType>,
            > {
                type VecType = concat_idents!(Vec3, $x);
                type F32Type = concat_idents!(f32, $x);
                type RayType = concat_idents!(Ray, $x);

                let ray: &concat_idents!(Ray, $x) = ray;

                let u = ray.direction;
                let v = ray.origin - self.c;

                let a = 2.0 * ray.direction_mag_squared; // u dot u
                let b = 2.0 * u.dot(v);
                let c = v.dot(v) - self.r_sq;

                let discriminant = b * b - (2.0 * a * c);

                let discriminant_pos = discriminant.cmp_ge(F32Type::ZERO);
                let discriminant_sqrt = discriminant.sqrt();

                let t1 = (-b - discriminant_sqrt) / a;

                let t1_valid = t1.cmp_ge(F32Type::ZERO) & discriminant_pos;
                //println!("{discriminant:?}: \t {discriminant_pos:?} | {t1_valid:?}");

                // let t2 = (-b + discriminant_sqrt) / a;
                // let t2_valid = t2.cmp_gt(F32Type::ZERO) & di scriminant_pos;
                //
                // let t = t2_valid.blend(t2, RayType::INVALID_VALUE_SPLATTED);
                // let t = t1_valid.blend(t1, t);

                //let t = t1_valid.blend(t1, RayType::INVALID_VALUE_SPLATTED);

                // not needed, do belnding only at the end?
                // let t = discriminant_pos.blend(t1, RayType::INVALID_VALUE_SPLATTED);

                RayIntersectionCandidate::new(t1, payload, t1_valid)
            }

            fn intersect<'a>(
                &'a self,
                ray: &'_ Self::RayType,
                candidate: &'_ RayIntersectionCandidate<Self::ScalarType, &'a Self, Self::MaskType>,
            ) -> Self::ReturnTypeWrapper<
                RayIntersection<Self::VectorType, Self::ScalarType, Self::MaskType>,
            > {
                type VecType = concat_idents!(Vec3, $x);
                type F32Type = concat_idents!(f32, $x);
                type RayType = concat_idents!(Ray, $x);

                let ray: &concat_idents!(Ray, $x) = ray;

                let t = candidate.t;
                let valid_mask = candidate.valid_mask;

                let p = ray.at(t);
                let r = p - VecType::from(self.c);
                let n = r.normalized();

                // we can remove   n.mag() probably, as it is 1 anyways
                let incident_angle = p.dot(n) / (p.mag() * n.mag());

                RayIntersection::new(p, n, (p - ray.origin).mag(), incident_angle, valid_mask)
            }
        }

        impl SphereData<concat_idents!(Vec3, $x), concat_idents!(f32, $x)> {
            fn blend(mask: concat_idents!(m32, $x), t: &Self, f: &Self) -> Self {
                Self {
                    c: <concat_idents!(Vec3, $x)>::blend(mask, t.c, f.c),
                    r_inv: mask.blend(t.r_inv, f.r_inv),
                    r_sq: mask.blend(t.r_sq, f.r_sq),
                    color: t.color, // todo handle multiple colors
                }
            }

            fn splat(v: &SphereData<Vec3, f32>) -> Self {
                Self {
                    c: <concat_idents!(Vec3, $x)>::splat(v.c),
                    r_inv: <concat_idents!(f32, $x)>::splat(v.r_inv),
                    r_sq: <concat_idents!(f32, $x)>::splat(v.r_sq),
                    color: v.color, // todo handle multiple colors
                }
            }
        }
    };
}

intersect_sphere_simd_impl!(x4);
intersect_sphere_simd_impl!(x8);

impl Intersectable for SphereData<Vec3, f32> {
    type RayType = Ray;
    type MaskType = ();
    type ScalarType = f32;
    type VectorType = Vec3;

    type ReturnTypeWrapper<T> = Option<T>;

    fn check_intersection<'a, P>(
        &'a self,
        ray: &'_ Self::RayType,
        payload: P,
    ) -> Self::ReturnTypeWrapper<RayIntersectionCandidate<Self::ScalarType, P, Self::MaskType>>
    {
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

        Some(RayIntersectionCandidate::new(t, payload, ()))
    }
    fn intersect<'a>(
        &'a self,
        ray: &'_ Self::RayType,
        candidate: &'_ RayIntersectionCandidate<Self::ScalarType, &'a Self, Self::MaskType>,
    ) -> Self::ReturnTypeWrapper<RayIntersection<Self::VectorType, Self::ScalarType, Self::MaskType>>
    {
        let t = candidate.t;

        let p = ray.at(t);
        let r = p - self.c;
        let n = r.normalized();

        // we can remove   n.mag() probably, as it is 1 anyways
        let incident_angle = p.dot(n) / (p.mag() * n.mag());

        Some(RayIntersection::new(
            p,
            n,
            (p - ray.origin).mag(),
            incident_angle,
            (),
        ))
    }
}

static SPHERES: LazyLock<[SphereData<Vec3, f32>; 16]> = LazyLock::new(|| {
    [
        SphereData::new(
            Vec3::new(WINDOW_WIDTH as f32 / 2.0, WINDOW_HEIGHT as f32 / 2.0, 150.0),
            70.0,
            OpaqueColor::from_rgb8(255, 0, 0),
        ),
        SphereData::new(
            Vec3::new(WINDOW_WIDTH as f32 / 2.5, WINDOW_HEIGHT as f32 / 2.5, 150.0),
            90.0,
            OpaqueColor::from_rgb8(0, 255, 0),
        ),
        SphereData::new(
            Vec3::new(
                2.0 * (WINDOW_WIDTH as f32 / 2.5),
                WINDOW_HEIGHT as f32 / 2.5,
                150.0,
            ),
            90.0,
            OpaqueColor::from_rgb8(111, 255, 222),
        ),
        SphereData::new(
            Vec3::new(
                2.0 * (WINDOW_WIDTH as f32 / 2.5),
                2.0 * (WINDOW_HEIGHT as f32 / 2.5),
                250.0,
            ),
            120.0,
            OpaqueColor::from_rgb8(158, 0, 255),
        ),
        SphereData::new(
            Vec3::new(
                1.25 * (WINDOW_WIDTH as f32 / 2.5),
                0.5 * (WINDOW_HEIGHT as f32 / 2.5),
                90.0,
            ),
            30.0,
            OpaqueColor::from_rgb8(128, 210, 255),
        ),
        SphereData::new(
            Vec3::new(
                (WINDOW_WIDTH as f32 / 2.5),
                2.25 * (WINDOW_HEIGHT as f32 / 2.5),
                500.0,
            ),
            250.0,
            OpaqueColor::from_rgb8(254, 255, 255),
        ),
        SphereData::new(
            Vec3::new(
                WINDOW_WIDTH as f32 / 4.0,
                3.0 * (WINDOW_HEIGHT as f32 / 4.0),
                20.0,
            ),
            10.0,
            OpaqueColor::from_rgb8(255, 55, 77),
        ),
        SphereData::new(
            Vec3::new(
                WINDOW_WIDTH as f32 / 3.0,
                3.0 * (WINDOW_HEIGHT as f32 / 6.0),
                30.0,
            ),
            25.0,
            OpaqueColor::from_rgb8(55, 230, 180),
        ),
        // dupl
        SphereData::new(
            Vec3::new(WINDOW_WIDTH as f32 / 2.5, WINDOW_HEIGHT as f32 / 2.5, 150.0),
            90.0,
            OpaqueColor::from_rgb8(0, 255, 0),
        ),
        SphereData::new(
            Vec3::new(
                2.0 * (WINDOW_WIDTH as f32 / 2.5),
                WINDOW_HEIGHT as f32 / 2.5,
                150.0,
            ),
            90.0,
            OpaqueColor::from_rgb8(111, 255, 222),
        ),
        SphereData::new(
            Vec3::new(
                2.0 * (WINDOW_WIDTH as f32 / 2.5),
                2.0 * (WINDOW_HEIGHT as f32 / 2.5),
                250.0,
            ),
            120.0,
            OpaqueColor::from_rgb8(158, 0, 255),
        ),
        SphereData::new(
            Vec3::new(
                1.25 * (WINDOW_WIDTH as f32 / 2.5),
                0.5 * (WINDOW_HEIGHT as f32 / 2.5),
                90.0,
            ),
            30.0,
            OpaqueColor::from_rgb8(128, 210, 255),
        ),
        SphereData::new(
            Vec3::new(
                (WINDOW_WIDTH as f32 / 2.5),
                2.25 * (WINDOW_HEIGHT as f32 / 2.5),
                500.0,
            ),
            250.0,
            OpaqueColor::from_rgb8(254, 255, 255),
        ),
        SphereData::new(
            Vec3::new(
                WINDOW_WIDTH as f32 / 4.0,
                3.0 * (WINDOW_HEIGHT as f32 / 4.0),
                20.0,
            ),
            10.0,
            OpaqueColor::from_rgb8(255, 55, 77),
        ),
        SphereData::new(
            Vec3::new(
                WINDOW_WIDTH as f32 / 3.0,
                3.0 * (WINDOW_HEIGHT as f32 / 6.0),
                30.0,
            ),
            25.0,
            OpaqueColor::from_rgb8(55, 230, 180),
        ),
        SphereData::new(
            Vec3::new(WINDOW_WIDTH as f32 / 2.0, WINDOW_HEIGHT as f32 / 2.0, 150.0),
            70.0,
            OpaqueColor::from_rgb8(255, 0, 0),
        ),
    ]
});

static SPHERES_x4: LazyLock<[SphereData<Vec3x4, f32x4>; 2]> = LazyLock::new(|| {
    // fixme grouping like this likely needs to happen at runtime?
    SPHERES
        .chunks(4)
        .map(|c| {
            SphereData::new(
                Vec3x4::from(c.iter().map(|x| x.c).collect_array::<4>().unwrap()),
                f32x4::from(
                    c.iter()
                        .map(|x| 1.0 / x.r_inv)
                        .collect_array::<4>()
                        .unwrap(),
                ),
                c.get(0).unwrap().color,
            )
        })
        .collect_array::<2>()
        .unwrap()
});

static SPHERES_x8: LazyLock<[SphereData<Vec3x8, f32x8>; 2]> = LazyLock::new(|| {
    // fixme grouping like this likely needs to happen at runtime?
    SPHERES
        .chunks(8)
        .map(|c| {
            SphereData::new(
                Vec3x8::from(c.iter().map(|x| x.c).collect_array::<8>().unwrap()),
                f32x8::from(
                    c.iter()
                        .map(|x| 1.0 / x.r_inv)
                        .collect_array::<8>()
                        .unwrap(),
                ),
                c.get(0).unwrap().color,
            )
        })
        .collect_array::<2>()
        .unwrap()
});

static RENDER_RAY_DIRECTION: Vec3 = Vec3::new(0.0, 0.0, 1.0);

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct TestRenderer3DLightColorSW03<C: OutputColorEncoder>(PhantomData<C>);

impl<C: OutputColorEncoder> TestRenderer3DLightColorSW03<C> {
    fn get_pixel_color(RenderCoordinates { x, y }: RenderCoordinates) -> Option<Pixel> {
        let ray = Ray::new(Vec3::new(x as f32, y as f32, 0.0), RENDER_RAY_DIRECTION);

        let nearest_intersection = SPHERES.iter().fold(None, |nearest_intersection, sphere| {
            let intersection = sphere.check_intersection(&ray, sphere);

            match (nearest_intersection, intersection) {
                (None, intersection) => intersection,
                (nearest_intersection, None) => nearest_intersection,
                (Some(nearest_intersection), Some(intersection))
                    if intersection.t > nearest_intersection.t =>
                {
                    Some(nearest_intersection)
                }
                (Some(_), intersection) => intersection,
            }
        })?;

        let intersection_object = nearest_intersection.payload;

        let RayIntersection {
            distance,
            incident_angle_cos,
            ..
        } = intersection_object.intersect(&ray, &nearest_intersection)?;

        // println!("cos(theta) = {}", incident_angle_cos);
        let d = distance / intersection_object.c.mag();

        Some(Pixel(
            intersection_object.color.map_lightness(|l| l - 2.0 * d),
        ))
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
                        let RayIntersection { distance: distances, .. } = sphere.$fnname(&ray);

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
                let coords = Vec3x8::new(f32x8::from(xs), f32x8::from(ys), f32x8::from(zs));
                let coords_2d = coords.xy();
                let ray = Rayx8::new(
                    coords,
                    Vec3x8::unit_z(),
                );

                // FIXME the issue here is:
                // check_intersection should be called for each sphere individually -> e.g. all 8 rays shall be checked for the SAME sheper for intersection (solution: just simp,y splat the spere data 8x)
                // then blend based on distance (t) these intersections together, so we have for each ray it's intersecting sphere
                // then we can calculate at once the intersection angle etc.. for the entire ray bundle (x8) via spheres.intersection(ray), where spheres is the x8 variant of all intersections
                // we later will need to generalie this approach so it supports not only spheres -> enum ObjectType {Sphere(SphereData), Plana(PlaneData), ...} -> so we have a homogenious data structure to accumulate intersections on...

                let mut pixel_colors = [None; 8];
                for sphere in SPHERES.iter() {
                    let sphere_x8 = SphereData::<Vec3x8,f32x8>::splat(sphere);
                    let intersection: RayIntersectionCandidate<f32x8, _, m32x8> = sphere_x8.check_intersection(&ray, sphere_x8);

                    let d = intersection.t / sphere.c.mag();
                    let d = intersection.valid_mask.blend(d, Rayx8::INVALID_VALUE_SPLATTED);

                    for (pixel_index, &v) in d.as_array_ref().iter().enumerate().filter(|&(_, &v)| !v.is_nan() && v.is_finite() && v >= 0.0 && v != Rayx8::INVALID_VALUE) {
                        let p_idx = idxs[pixel_index];
                        println!("{pixel_index} / {p_idx}");
                        pixel_colors[pixel_index] = Some(Pixel(
                            sphere.color //.map_lightness(|l| l - 2.0 * v),
                        ));
                    }
                }
                for (pixel_index, pixel_color) in pixel_colors.into_iter().enumerate() {
                    if let Some(color) = pixel_color {
                        set_pixel(idxs[pixel_index], color);
                    }
                }
                return;

                let nearest_intersection = SPHERES_x8.iter().fold(None, |nearest_intersection, sphere| {
                    let intersection: RayIntersectionCandidate<f32x8, _, m32x8> = sphere.check_intersection(&ray, Cow::Borrowed(sphere));


                    let i_has_more_than_one_intersection = intersection.valid_mask.move_mask() & (intersection.valid_mask.move_mask() - 1) > 0;

                    if (i_has_more_than_one_intersection) {
                        //println!("{coords_2d:?}: \t Intersection {intersection:?} has more than one intersection");
                    }

                    //println!("{:?} / {:?}", nearest_intersection.clone().map(|x: RayIntersectionCandidate<f32x8, _, m32x8>| x.valid_mask), intersection.valid_mask);

                    // if we have no intersection, return previous nearest_intersection
                    if intersection.valid_mask.none() {
                        //println!("{coords_2d:?}: \t Skip because intersection.valid_mask.none()");
                        return nearest_intersection;
                    }


                    // if nearest_intersection is none, return current intersection
                    let Some(nearest_intersection) = nearest_intersection else {
                        //println!("{coords_2d:?}: \t Skip because nearest_intersection.is_none()");
                        return Some(intersection);
                    };

                    let ni_has_more_than_one_intersection = nearest_intersection.valid_mask.move_mask() & (nearest_intersection.valid_mask.move_mask() - 1) > 0;


                    // if nearest_intersection has no intersections, return current intersection (as this is by now guaranteed to have at least one intersection)
                    if nearest_intersection.valid_mask.none() {
                        //println!("{coords_2d:?}: \t Skip because nearest_intersection.valid_mask.none()");
                        return Some(intersection);
                    }

                    let intersection_has_lower_value = nearest_intersection.t.cmp_ge(intersection.t);

                    if intersection_has_lower_value.none() {
                        return Some(nearest_intersection);
                    } else if intersection_has_lower_value.all() {
                        return Some(intersection);
                    }

                    // now the complex case:
                    // we need to merge the two intersections
                    // compare nearest_intersection's with intersection (take the minimum of the two)
                    // but take care that we only consider valid values (the ones where both valid masks are ture)
                    // for the values where only one of the valid maks are true (xor both valiud masks), take the one
                    // from the intersection where it is true

                    // todo

                    let intersection_valid_mask = intersection.valid_mask;
                    let nearest_intersection_valid_mask = nearest_intersection.valid_mask;

                    let valid_both = nearest_intersection_valid_mask & intersection_valid_mask;
                    let valid_either = nearest_intersection_valid_mask | intersection_valid_mask;
                    let valid_exclusive = nearest_intersection_valid_mask ^ intersection_valid_mask;

                    let valid_only_intersection = intersection_valid_mask & valid_exclusive;
                    let valid_only_nearest_intersection = nearest_intersection_valid_mask & valid_exclusive;


                    let mut new_t = valid_both.blend(
                        intersection_has_lower_value.blend(intersection.t, nearest_intersection.t),
                        Rayx8::INVALID_VALUE_SPLATTED,
                    );
                    new_t = valid_only_intersection.blend(intersection.t, new_t);
                    new_t = valid_only_nearest_intersection.blend(nearest_intersection.t, new_t);

                    let intersection_has_lower_value = intersection_has_lower_value;

                    let new_payload = SphereData::<Vec3x8, f32x8>::blend(intersection_has_lower_value, &intersection.payload, &nearest_intersection.payload);

                    if (i_has_more_than_one_intersection || ni_has_more_than_one_intersection) && (intersection_valid_mask.move_mask() != nearest_intersection_valid_mask.move_mask()) {
                        println!("------\n\
i_valid: {intersection_valid_mask:?} \t / \t {nearest_intersection_valid_mask:?}\n\
both: {valid_both:?}\n\
eiter: {valid_either:?}\n\
exclusive: {valid_exclusive:?}\n\
only_i: {valid_only_intersection:?}\n\
only_ni: {valid_only_nearest_intersection:?}\n\
lower_val_i: {intersection_has_lower_value:?}\n
new_t: {new_t:?}\n\
new_payload: {new_payload:?}\n\
------
");
                    }
                    //println!("{nearest_intersection:?} + {intersection:?}: {new_t:?} / {new_payload:?} / {valid_either:?}");

                    Some(RayIntersectionCandidate::new(new_t, Cow::Owned(new_payload), valid_either))
                });

                let Some(nearest_intersection) = nearest_intersection else {
                    return;
                };

                if nearest_intersection.valid_mask.none() {
                    return;
                }


                // println!("{:?}", nearest_intersection);


                let RayIntersection {
                    distance,
                    incident_angle_cos,
                    valid_mask,
                    ..
                } = nearest_intersection.payload.intersect(&ray, &RayIntersectionCandidate::new(nearest_intersection.t, nearest_intersection.payload.as_ref(), nearest_intersection.valid_mask));
                //
                // // println!("cos(theta) = {}", incident_angle_cos);
                let d = distance / nearest_intersection.payload.c.mag();
                let d = valid_mask.blend(d, Rayx8::INVALID_VALUE_SPLATTED);

                //println!("{d:?}");

                for (pixel_index, &v) in d.as_array_ref().iter().enumerate().filter(|&(_, &v)| !v.is_nan() && v.is_finite() && v >= 0.0 && v != Rayx8::INVALID_VALUE) {
                    set_pixel(idxs[pixel_index], Pixel(
                        nearest_intersection.payload.color //.map_lightness(|l| l - 2.0 * v),
                    ));
                }


            // if likely(len == 8) {
            //     impl_render_pixel_colors_simd!(intersect_x8, x8, xs, ys, zs, idxs, set_pixel);
            // } else if unlikely(len == 4) {
            //     impl_render_pixel_colors_simd!(intersect_x4, x4, xs, ys, zs, idxs, set_pixel);
            } else {
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
    for TestRenderer3DLightColorSW03<C>
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
