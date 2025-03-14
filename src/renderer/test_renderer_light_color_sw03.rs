use crate::extensions::SrgbColorConvertExt;
use crate::helpers::{ColorType, Pixel};
use crate::image_buffer::ImageBuffer;
use crate::output::OutputColorEncoder;
use crate::renderer::{RenderCoordinates, RenderCoordinatesVectorized, Renderer};
use crate::{WINDOW_HEIGHT, WINDOW_WIDTH};
use itertools::{Itertools, izip};
use wide::CmpLt;

use palette::blend::Blend;
use palette::{Darken, Lighten, Mix, Srgb};
use std::borrow::Cow;
use std::intrinsics::{cold_path, likely, unlikely};
use std::marker::PhantomData;
use std::ops::{BitAnd, Mul};
use std::sync::LazyLock;
use ultraviolet::{Vec3, Vec3x4, Vec3x8, f32x4, f32x8, m32x4, m32x8};
use wide::CmpGe;

// Todo optimization-idea: ensure that ray direction & normals (on intersection) are unit vectors in constructors. This way we can simplify the maths in certain cases

#[derive(Clone, Debug, Copy)]
struct RayIntersectionCandidate<Scalar, Payload, ValidMask = ()>
where
    Scalar: Sized + Copy,
    ValidMask: Copy,
    Payload: ?Sized,
{
    /// Distance from ray origin
    t: Scalar,
    estimated_distance_sq: Scalar,
    valid_mask: ValidMask,
    payload: Payload,
}

impl<Scalar, Payload, ValidMask> RayIntersectionCandidate<Scalar, Payload, ValidMask>
where
    Scalar: Sized + Copy,
    ValidMask: Copy,
    Payload: Sized,
{
    const fn new(
        t: Scalar,
        estimated_distance_sq: Scalar,
        payload: Payload,
        valid_mask: ValidMask,
    ) -> Self {
        Self {
            t,
            estimated_distance_sq,
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
    intersection_direction: Vector,
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
        intersection_direction: Vector,
        normal: Vector,
        distance: Scalar,
        incident_angle_cos: Scalar,
        valid_mask: ValidMask,
    ) -> Self {
        Self {
            intersection,
            intersection_direction,
            normal,
            distance,
            incident_angle_cos,
            valid_mask,
        }
    }
}

impl RayIntersection<Vec3, f32, ()> {
    pub fn to_reflected_ray(&self) -> Ray {
        //let sin = (1.0 - self.incident_angle_cos).sqrt();

        Ray::new(
            self.intersection.clone(),
            self.intersection_direction.reflected(self.normal),
        )
    }
}

impl RayIntersection<Vec3x4, f32x4, m32x4> {
    pub fn to_reflected_ray(&self) -> Rayx4 {
        //let sin = (1.0 - self.incident_angle_cos).sqrt();

        Rayx4::new(
            self.intersection.clone(),
            self.intersection_direction.reflected(self.normal),
        )
    }
}

impl RayIntersection<Vec3x8, f32x8, m32x8> {
    pub fn to_reflected_ray(&self) -> Rayx8 {
        //let sin = (1.0 - self.incident_angle_cos).sqrt();

        Rayx8::new(
            self.intersection.clone(),
            self.intersection_direction.reflected(self.normal),
        )
    }
}

#[cfg(test)]
mod test_ray_intersection_struct {
    use super::*;

    #[test]
    fn test_ray_reflection_at_intersection() {
        let deg: f32 = 90.0;
        let intersection = RayIntersection::new(
            Vec3::new(1.0, 1.0, 0.),
            Vec3::new(1.0, 1., 0.),
            Vec3::new(0.0, 1., 0.),
            Vec3::new(1.0, 1.0, 0.).mag(),
            deg.cos(),
            (),
        );

        let reflection = intersection.to_reflected_ray();

        println!("reflection: {:?}", reflection);
    }
}

#[derive(Clone, Debug, Copy)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
    #[deprecated(note = "Is always 1, as direction is normalized")]
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
    #[deprecated(note = "Is always 1, as direction is normalized")]
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
    #[deprecated(note = "Is always 1, as direction is normalized")]
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
    color: ColorType<Scalar>,
}

impl<Vector, Scalar> SphereData<Vector, Scalar>
where
    Scalar:
        Sized + Copy + From<f32> + std::ops::Div<Output = Scalar> + std::ops::Mul<Output = Scalar>,
    Vector: Sized,
{
    fn new(c: Vector, r: Scalar, color: ColorType<Scalar>) -> Self {
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

                // FIXME if  we garantuee ray direction is normalized, we can avoid multiplying by direction_mag_squared here, as it will be 1 anyways => a = 2.0
                // CHATGPT:
                // Diskriminant Δ=(2(u⋅v))^2 − 4(v⋅v−r^2)
                // You can factor out common terms (like the constant 4) and simplify the square root and division. This may let you avoid some multiplications and divisions in the inner loop.
                // this then further down leads to the optimization that we can calculate the inverse of a (maybe use the fast inverse from minmath crate) and convert the division by a futher down t a multiplication by inv_a -> multiplication is generally faster than division
                //let a = 2.0 * ray.direction_mag_squared; // u dot u
                static a: f32 = 2.0; // 2 * (u dot u) => 2 * direction_mag_squared => 2 * 1 => 2
                static a_inv: f32 = a.recip(); // crate::helpers::fast_inverse(a);
                let b = 2.0 * u.dot(v);
                let c = v.dot(v) - self.r_sq;

                let discriminant = b * b - ((2.0 * a) * c);

                let discriminant_pos = discriminant.cmp_ge(F32Type::ZERO);
                let discriminant_sqrt = discriminant.sqrt();

                // FIXME optimize by replacing / a by * inv_a
                //let t1 = (-b - discriminant_sqrt) / a;
                let t0 = (-b - discriminant_sqrt) * a_inv;
                let t1 = (-b + discriminant_sqrt) * a_inv;

                //let t1_valid = t1.cmp_ge(F32Type::ZERO) & discriminant_pos;

                let t0_valid = t0.cmp_ge(F32Type::ZERO) & discriminant_pos;
                let t1_valid = t1.cmp_ge(F32Type::ZERO) & discriminant_pos;

                // Prefer t0 if it's valid, else t1 if that is valid.
                // If both are valid, t0 is nearer.
                let use_t0 = t0_valid & (!t1_valid | (t0.cmp_lt(t1)));
                let use_t1 = t1_valid & !use_t0;

                // Start with invalid for all lanes
                let mut final_t = RayType::INVALID_VALUE_SPLATTED;

                // Where t0 is chosen, blend in t0
                final_t = use_t0.blend(t0, final_t);

                // Where t1 is chosen, blend in t1
                final_t = use_t1.blend(t1, final_t);

                // final_t_valid is lanes where we picked something
                let final_t_valid = use_t0 | use_t1;

                let estimated_distance = final_t * final_t;

                RayIntersectionCandidate::new(final_t, estimated_distance, payload, final_t_valid)
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

                let p_mag = p.mag();

                let incident_angle = p.dot(n) / (p_mag);

                RayIntersection::new(
                    p,
                    ray.direction.clone(),
                    n,
                    (p - ray.origin).mag(),
                    incident_angle,
                    valid_mask,
                )
            }
        }

        impl SphereData<concat_idents!(Vec3, $x), concat_idents!(f32, $x)> {
            fn blend(mask: concat_idents!(m32, $x), t: &Self, f: &Self) -> Self {
                Self {
                    c: <concat_idents!(Vec3, $x)>::blend(mask, t.c, f.c),
                    r_inv: mask.blend(t.r_inv, f.r_inv),
                    r_sq: mask.blend(t.r_sq, f.r_sq),
                    color: ColorType::<concat_idents!(f32, $x)>::new(
                        mask.blend(t.color.red, f.color.red),
                        mask.blend(t.color.green, f.color.green),
                        mask.blend(t.color.blue, f.color.blue),
                    ),
                }
            }

            fn splat(v: &SphereData<Vec3, f32>) -> Self {
                Self {
                    c: <concat_idents!(Vec3, $x)>::splat(v.c),
                    r_inv: <concat_idents!(f32, $x)>::splat(v.r_inv),
                    r_sq: <concat_idents!(f32, $x)>::splat(v.r_sq),
                    color: ColorType::<concat_idents!(f32, $x)>::new(
                        <concat_idents!(f32, $x)>::splat(v.color.red),
                        <concat_idents!(f32, $x)>::splat(v.color.green),
                        <concat_idents!(f32, $x)>::splat(v.color.blue),
                    ),
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

        static a: f32 = 2.0;
        static a_inv: f32 = a.recip();
        let b = 2.0 * u.dot(v);
        let c = v.dot(v) - self.r_sq;

        let discriminant = b * b - 2.0 * a * c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrt_d = discriminant.sqrt();

        let mut t = (-b - sqrt_d) * a_inv;

        if t <= 0.0 {
            let t2 = (-b + sqrt_d) * a_inv;
            if t2 > 0.0 {
                t = t2;
            } else {
                return None;
            }
        }

        Some(RayIntersectionCandidate::new(
            t,
            (t * u).mag_sq(),
            payload,
            (),
        ))
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
            ray.direction.clone(),
            n,
            (p - ray.origin).mag(),
            incident_angle,
            (),
        ))
    }
}

#[derive(Debug, Copy, Clone)]
struct PointData<Vector>
where
    Vector: Sized,
{
    p: Vector,
}

impl<Vector> PointData<Vector>
where
    Vector: Sized,
{
    const fn new(p: Vector) -> Self {
        Self { p }
    }
}

macro_rules! intersect_point_simd_impl {
    ($x: ident) => {
        impl Intersectable for PointData<concat_idents!(Vec3, $x)> {
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

                let p = VecType::from(self.p);
                let v = p - ray.origin;
                let t = v.dot(ray.direction);

                let intersection_valid = (ray.at(t) - p).mag_sq().cmp_ge(F32Type::splat(0.01));

                let estimated_distance = t * t;

                RayIntersectionCandidate::new(t, estimated_distance, payload, intersection_valid)
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
                let n = ray.at(t * -1.0).normalized();

                RayIntersection::new(
                    p,
                    ray.direction.clone(),
                    n,
                    (p - ray.origin).mag(),
                    F32Type::splat(1.0),
                    valid_mask,
                )
            }
        }

        impl PointData<concat_idents!(Vec3, $x)> {
            fn blend(mask: concat_idents!(m32, $x), t: &Self, f: &Self) -> Self {
                Self {
                    p: <concat_idents!(Vec3, $x)>::blend(mask, t.p, f.p),
                }
            }

            fn splat(v: &PointData<Vec3>) -> Self {
                Self {
                    p: <concat_idents!(Vec3, $x)>::splat(v.p),
                }
            }
        }
    };
}

intersect_point_simd_impl!(x4);
intersect_point_simd_impl!(x8);

impl Intersectable for PointData<Vec3> {
    type RayType = Ray;
    type ScalarType = f32;
    type VectorType = Vec3;
    type MaskType = ();
    type ReturnTypeWrapper<T> = Option<T>;

    fn check_intersection<'a, P>(
        &'a self,
        ray: &'_ Self::RayType,
        payload: P,
    ) -> Self::ReturnTypeWrapper<RayIntersectionCandidate<Self::ScalarType, P, Self::MaskType>>
    {
        let v = self.p - ray.origin;

        let t = v.dot(ray.direction);
        let p_t = ray.at(t);

        if (p_t - self.p).mag_sq() > 0.01 {
            return None;
        }

        Some(RayIntersectionCandidate::new(t, v.mag_sq(), payload, ()))
    }

    fn intersect<'a>(
        &'a self,
        ray: &'_ Self::RayType,
        candidate: &'_ RayIntersectionCandidate<Self::ScalarType, &'a Self, Self::MaskType>,
    ) -> Self::ReturnTypeWrapper<RayIntersection<Self::VectorType, Self::ScalarType, Self::MaskType>>
    {
        let t = candidate.t;

        let p = ray.at(t);
        let n = ray.at(t * -1.0).normalized();

        Some(RayIntersection::new(
            p,
            ray.direction.clone(),
            n,
            (p - ray.origin).mag(),
            1.0,
            (),
        ))
    }
}

#[cfg(test)]
mod test_point_intersection {
    use super::*;

    #[test]
    fn test_check_intersection_point() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        let point = PointData::new(2.0 * v);

        let ray = Ray::new(Vec3::zero(), v);

        let i = point.check_intersection(&ray, ());

        assert!(i.is_some());
        assert_eq!(i.unwrap().estimated_distance_sq, (2.0 * v).mag_sq());
    }

    #[test]
    fn test_check_intersection_point_not() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        let point = PointData::new(2.0 * Vec3::new(0.0, 1.0, 0.0));

        let ray = Ray::new(Vec3::zero(), v);

        let i = point.check_intersection(&ray, ());

        assert!(i.is_none());
    }
}

static SPHERES: LazyLock<[SphereData<Vec3, f32>; 8]> = LazyLock::new(|| {
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
                (WINDOW_WIDTH as f32 / 2.5),
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

const N_RANDOM_SPHERES: usize = 2_000;
static SPHERES_RANDOM: LazyLock<[SphereData<Vec3, f32>; N_RANDOM_SPHERES]> = LazyLock::new(|| {
    (0..N_RANDOM_SPHERES)
        .map(|i| {
            let rnd =
                f32::from(((i % 255) + ((i / 45) % 255) + ((i / 4) % 255)) as u16) / (255.0 * 3.0);
            let rnd2 = f32::from(
                (((i / 5) % 255) + ((i / 17) % 255) + ((i / 2) % 255) + (((i + 1) / 9) % 255))
                    as u16,
            ) / (255.0 * 3.0);

            let pol = if rnd < rnd2 { -1.0 } else { 1.0 };

            let r = rnd * 65.0;
            SphereData::new(
                Vec3::new(
                    (((i * 40) % WINDOW_WIDTH) as f32).max(r),
                    (((((WINDOW_HEIGHT / ((i % 45) + 1)) + 100) % WINDOW_HEIGHT) as f32) * 2.5)
                        % WINDOW_HEIGHT as f32,
                    (85.2 + (pol * rnd * (WINDOW_HEIGHT as f32 / 1.66))).max(r),
                ),
                r,
                ColorType::new(
                    ((i / 12) % 255) as u8,
                    ((i / 3) % 255) as u8,
                    ((i * 8) % 255) as u8,
                )
                .into_format(),
            )
        })
        .collect_array::<N_RANDOM_SPHERES>()
        .expect("N_RANDOM_SPHERES")
});

static RENDER_RAY_DIRECTION: Vec3 = Vec3::new(0.0, 0.0, 1.0);

static LIGHT_1: LazyLock<SphereData<Vec3, f32>> = LazyLock::new(|| {
    SphereData::new(
        Vec3::new(-100.0, 1000.0, -10.0),
        100.0,
        ColorType::new(0.822, 0.675, 0.45),
    )
});

static LIGHT_2: LazyLock<SphereData<Vec3, f32>> = LazyLock::new(|| {
    SphereData::new(
        Vec3::new(1.0, -1.0, 100.0),
        100.0,
        ColorType::new(0.0, 0.675, 0.9),
    )
});

static LIGHT_3: LazyLock<SphereData<Vec3, f32>> = LazyLock::new(|| {
    SphereData::new(
        Vec3::new(0.0, 0.0, -100000.0),
        100.0,
        ColorType::new(0., 0., 0.),
    )
});

static LIGHT_4: LazyLock<SphereData<Vec3, f32>> = LazyLock::new(|| {
    SphereData::new(
        Vec3::new((WINDOW_WIDTH / 2) as f32, (WINDOW_HEIGHT / 2) as f32, 120.0),
        100.0,
        ColorType::new(0.7, 0.6, 0.5),
    )
});

static LIGHTS: [&LazyLock<SphereData<Vec3, f32>, fn() -> SphereData<Vec3, f32>>; 4] =
    [&LIGHT_1, &LIGHT_2, &LIGHT_3, &LIGHT_4];

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

        let intersection @ RayIntersection {
            distance,
            incident_angle_cos,
            intersection: intersection_point,
            normal: intersection_normal,
            ..
        } = intersection_object.intersect(&ray, &nearest_intersection)?;

        // println!("cos(theta) = {}", incident_angle_cos);
        let d = distance / intersection_object.c.mag();

        let ambient_lighting_amount = 0.06;
        let mut direct_lighting_amount = 0.0;
        let mut light_color = ColorType::new(0., 0., 0.);

        for light in LIGHTS.iter() {
            let light_to_intersection = light.c - intersection_point;
            let light_distance = light_to_intersection.mag();

            let incident_light_angle_cos =
                light_to_intersection.dot(intersection_normal) / light_distance;

            if incident_light_angle_cos >= 0.0 {
                light_color = light_color.mix(
                    light.color,
                    incident_light_angle_cos.abs() + 1.0 / (light_distance),
                );
                direct_lighting_amount = (direct_lighting_amount
                    + (incident_light_angle_cos.abs() + 1.0 / (light_distance)))
                    .min(0.89);
            }
        }

        // raytrace light back to surce
        // let reflected_ray = intersection.to_reflected_ray();
        // if let Some(_) = LIGHT.check_intersection(&reflected_ray, ()) {
        //     direct_lighting_amount = incident_angle_cos.abs();
        //     light_color = LIGHT.color.clone();
        // }

        let ambient_color = intersection_object
            .color
            .darken_fixed(1.0 - ambient_lighting_amount * (0.25 / d).min(1.1));

        Some(Pixel(ambient_color.mix(
            ambient_color.soft_light(light_color),
            direct_lighting_amount,
        )))
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
                ($xn: ident, $xs: ident, $ys: ident, $zs: ident, $idxs: ident, $set_pixel: ident) => {
                    type VecType = concat_idents!(Vec3, $xn);
                    type F32Type = concat_idents!(f32, $xn);
                    type MaskType = concat_idents!(m32, $xn);
                    type RayType = concat_idents!(Ray, $xn);

                    let coords = VecType::new(F32Type::from($xs), F32Type::from($ys), F32Type::from($zs));
                    let ray = RayType::new(
                        coords,
                        VecType::unit_z(),
                    );

                    let nearest_intersection = SPHERES.iter().fold(None, |previous_intersection: Option<_>, sphere| {
                        let sphere = SphereData::<VecType, F32Type>::splat(sphere);
                        let new_intersection: RayIntersectionCandidate<F32Type, _, MaskType> = sphere.check_intersection(&ray, Cow::Owned(sphere));

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


                        let new_is_nearer = previous_intersection.estimated_distance_sq.cmp_ge(new_intersection.estimated_distance_sq);

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
                        let merged_t = pick_old_mask.blend(previous_intersection.t, new_intersection.t);
                        let merged_est_dist =
                            pick_old_mask.blend(previous_intersection.estimated_distance_sq, new_intersection.estimated_distance_sq);
                        let merged_payload = SphereData::<VecType, F32Type>::blend(
                            pick_old_mask,
                            &previous_intersection.payload,
                            &new_intersection.payload,
                        );

                        // 3) The new valid lanes are old OR new
                        let merged_valid = previous_valid | new_valid;

                        let merged_candidate = RayIntersectionCandidate::new(
                            merged_t,
                            merged_est_dist,
                            Cow::Owned(merged_payload),
                            merged_valid,
                        );

                        Some(merged_candidate)
                    });

                    let Some(nearest_intersection) = nearest_intersection else {
                        return;
                    };

                    if nearest_intersection.valid_mask.none() {
                        return;
                    }


                    let intersected_object: Cow<SphereData<VecType, F32Type>> = nearest_intersection.payload;
                    let RayIntersection {
                        distance,
                        incident_angle_cos,
                        intersection: intersection_point,
                        valid_mask,
                        normal: intersection_normal,
                        ..
                    } = intersected_object.intersect(&ray, &RayIntersectionCandidate::new(nearest_intersection.t, nearest_intersection.estimated_distance_sq, intersected_object.as_ref(), nearest_intersection.valid_mask));





                    //////////////////////


                    let d = distance / intersected_object.c.mag();

                    let ambient_lighting_amount = F32Type::splat(0.06);
                    let mut direct_lighting_amount = F32Type::ZERO;
                    let mut light_color = Srgb::<F32Type>::new(F32Type::ZERO, F32Type::ZERO, F32Type::ZERO);

                    for light in LIGHTS.iter() {
                        let light = SphereData::<VecType, F32Type>::splat(light);
                        let light_to_intersection = light.c - intersection_point;
                        let light_distance = light_to_intersection.mag();

                        let incident_light_angle_cos =
                            light_to_intersection.dot(intersection_normal) / light_distance;

                        let incident_angle_pos = incident_light_angle_cos.cmp_ge(F32Type::ZERO);

                        let mixed_color = light_color.mix(
                                light.color,
                                incident_light_angle_cos.abs() + F32Type::ONE / (light_distance),
                            );

                        // tod extract that to a trait or something
                        light_color = Srgb::<F32Type>::new(
                            incident_angle_pos.blend(
                                mixed_color.red,
                                light_color.red,
                            ),
                            incident_angle_pos.blend(
                                mixed_color.green,
                                light_color.green,
                            ),
                           incident_angle_pos.blend(
                                mixed_color.blue,
                                light_color.blue,
                            ),
                        );


                        direct_lighting_amount = incident_angle_pos.blend(
                            (direct_lighting_amount
                            + (incident_light_angle_cos.abs() + F32Type::ONE / (light_distance)))
                            .min(F32Type::splat(0.89)),
                            direct_lighting_amount
                        );
                    }

                    // raytrace light back to surce
                    // let reflected_ray = intersection.to_reflected_ray();
                    // if let Some(_) = LIGHT.check_intersection(&reflected_ray, ()) {
                    //     direct_lighting_amount = incident_angle_cos.abs();
                    //     light_color = LIGHT.color.clone();
                    // }

                    let ambient_color = intersected_object.color
                        .darken_fixed(F32Type::splat(1.0) - ambient_lighting_amount * (F32Type::splat(0.25) / d).min(F32Type::splat(1.1)));

                    let colors =ambient_color.mix(
                        ambient_color.soft_light(light_color),
                        direct_lighting_amount,
                    );

                    //////////////////////






                    let x = colors.extract_values(Some(valid_mask));

                    for (pixel_index, &c) in x.iter().enumerate().filter_map(|(i, v)| {
                        let Some(v) = v else {
                            return None;
                        };
                        Some((i, v))
                    }) {
                        $set_pixel($idxs[pixel_index], Pixel(c));
                    }
                };
            }

            if likely(len == 8) {
                impl_render_pixel_colors_simd!(x8, xs, ys, zs, idxs, set_pixel);
            } else if unlikely(len == 4) {
                impl_render_pixel_colors_simd!(x4, xs, ys, zs, idxs, set_pixel);
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
