use crate::extensions::SrgbColorConvertExt;
use crate::helpers::{ColorType, Pixel};
use crate::image_buffer::ImageBuffer;
use crate::output::OutputColorEncoder;
use crate::renderer::{RenderCoordinates, RenderCoordinatesVectorized, Renderer};
use crate::{WINDOW_HEIGHT, WINDOW_WIDTH};
use itertools::{Itertools, izip};

use crate::scalar::Scalar;
use crate::vector::{NormalizableVector, ReflectableVector, SimdCapableVector, VectorOperations};

use num_traits::{Float, NumOps, One, Zero};
use palette::blend::{Blend, Premultiply};
use palette::bool_mask::{HasBoolMask, LazySelect};
use palette::cast::ArrayCast;
use palette::stimulus::StimulusColor;
use palette::{Darken, Mix, Srgb};
use simba::scalar::{SubsetOf, SupersetOf};
use simba::simd::{SimdBool, SimdComplexField, SimdPartialOrd, SimdRealField, SimdValue};
use std::borrow::Cow;
use std::fmt::Debug;
use std::hint::{likely, unlikely};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::LazyLock;
use ultraviolet::{Vec3, Vec3x8, f32x8};

// Todo optimization-idea: ensure that ray direction & normals (on intersection) are unit vectors in constructors. This way we can simplify the maths in certain cases

#[derive(Clone, Debug, Copy)]
struct RayIntersectionCandidate<Scalar, Payload>
where
    Scalar: SimdValue + crate::scalar::Scalar,
    Payload: ?Sized,
{
    /// Distance from ray origin
    t: Scalar,
    estimated_distance_sq: Scalar,
    valid_mask: Scalar::SimdBool,
    payload: Payload,
}

impl<Scalar, Payload> RayIntersectionCandidate<Scalar, Payload>
where
    Scalar: SimdValue + crate::scalar::Scalar,
    Payload: Sized,
{
    const fn new(
        t: Scalar,
        estimated_distance_sq: Scalar,
        payload: Payload,
        valid_mask: Scalar::SimdBool,
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
struct RayIntersection<Vector>
where
    Vector: crate::vector::Vector,
    <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool: Debug,
{
    intersection: Vector,
    intersection_direction: Vector,
    normal: Vector,
    distance: Vector::Scalar,
    incident_angle_cos: Vector::Scalar,
    valid_mask: <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool,
}

impl<Vector> RayIntersection<Vector>
where
    Vector: crate::vector::Vector,
    <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool: Debug,
{
    pub const fn new(
        intersection: Vector,
        intersection_direction: Vector,
        normal: Vector,
        distance: Vector::Scalar,
        incident_angle_cos: Vector::Scalar,
        valid_mask: <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool,
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

    pub fn to_reflected_ray(&self) -> Ray<Vector>
    where
        Vector: VectorOperations + ReflectableVector + NormalizableVector + Copy,
        <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool: From<bool>,
        [(); <Vector as crate::vector::Vector>::LANES]:,
    {
        //let sin = (1.0 - self.incident_angle_cos).sqrt();

        Ray::new(
            self.intersection,
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
            true,
        );

        let reflection = intersection.to_reflected_ray();

        println!("reflection: {:?}", reflection);
    }
}

#[derive(Clone, Debug, Copy)]
struct Ray<Vector>
where
    Vector: crate::vector::Vector,
    <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool: Debug,
{
    origin: Vector,
    direction: Vector,
    valid_mask: <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool,
}

impl<Vector> Ray<Vector>
where
    Vector: crate::vector::Vector,
    <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool: Debug,
{
    #[inline]
    pub fn new(origin: Vector, direction: Vector) -> Self
    where
        Vector: NormalizableVector,
        [(); <Vector as crate::vector::Vector>::LANES]:,
        <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool: From<bool>,
    {
        Self::new_with_mask(origin, direction, true.into())
    }

    #[inline]
    pub fn new_with_mask(
        origin: Vector,
        direction: Vector,
        valid_mask: <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool,
    ) -> Self
    where
        Vector: NormalizableVector,
    {
        Self {
            origin,
            direction: direction.normalized(),
            valid_mask,
        }
    }

    #[inline(always)]
    pub fn at(&self, t: Vector::Scalar) -> Vector
    where
        Vector: VectorOperations + Copy,
    {
        self.direction.mul_add(Vector::broadcast(t), self.origin)
    }
    //pub fn reflect_at_intersection(&mut self, intersection: &RayIntersection<Vec3, f32, ()>) {}
}

#[cfg(test)]
mod test_ray {
    use super::*;

    #[test]
    fn test_ray_at() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(ray.at(0.0), Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(ray.at(1.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(ray.at(2.23), Vec3::new(2.23, 0.0, 0.0));
    }
}

impl<Vector> Ray<Vector>
where
    Vector: crate::vector::Vector,
    <Vector::Scalar as SimdValue>::Element: Float,
    <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool: Debug,
{
    #[inline(always)]
    pub fn invalid_value() -> <Vector::Scalar as SimdValue>::Element {
        <Vector::Scalar as SimdValue>::Element::infinity()
    }

    #[inline(always)]
    pub fn invalid_value_splatted() -> Vector::Scalar {
        Vector::Scalar::splat(Self::invalid_value())
    }

    #[inline(always)]
    pub fn invalid_vector() -> Vector
    where
        Vector: VectorOperations,
    {
        Vector::broadcast(Self::invalid_value_splatted())
    }
}

trait Intersectable
where
    Self::ScalarType: Scalar,
    Self::MaskType: Sized + Copy,
    <Self::ScalarType as SimdValue>::SimdBool: Debug,
{
    type RayType;
    type ScalarType: Scalar + SimdValue<SimdBool = Self::MaskType>;
    type VectorType: crate::vector::Vector<Scalar = Self::ScalarType>;
    type MaskType;

    type ReturnTypeWrapper<T>;

    fn check_intersection<'a, P>(
        &'a self,
        ray: &'_ Self::RayType,
        payload: P,
    ) -> Self::ReturnTypeWrapper<RayIntersectionCandidate<Self::ScalarType, P>>;

    // FIXME: we might want to move that to the RayIntersectionCandidate type, as you normaly would call this anyyway by candidate.payload.intersect(ray, candidate)?
    // or we want to have a shortcut from RayIntersectionCandidate
    fn intersect<'a>(
        &'a self,
        ray: &'_ Self::RayType,
        candidate: &'_ RayIntersectionCandidate<Self::ScalarType, &'a Self>,
    ) -> Self::ReturnTypeWrapper<RayIntersection<Self::VectorType>>;
}

#[derive(Debug, Copy, Clone)]
struct SphereData<Vector>
where
    Vector: crate::vector::Vector,
{
    c: Vector,
    r_sq: Vector::Scalar,
    r_inv: Vector::Scalar,
    color: ColorType<Vector::Scalar>,
}

impl<Vector> SphereData<Vector>
where
    Vector: crate::vector::Vector,
{
    fn new(c: Vector, r: Vector::Scalar, color: ColorType<Vector::Scalar>) -> Self
    where
        Vector::Scalar: Sized
            + Copy
            + From<f32>
            + std::ops::Div<Output = Vector::Scalar>
            + std::ops::Mul<Output = Vector::Scalar>,
    {
        Self {
            c,
            r_sq: r * r,
            r_inv: <Vector::Scalar as From<f32>>::from(1.0) / r,
            color,
        }
    }
}

impl<Vector> Intersectable for SphereData<Vector>
where
    Vector: crate::vector::Vector
        + Add<Vector, Output = Vector>
        + Sub<Vector, Output = Vector>
        + Mul<Vector, Output = Vector>
        + Copy
        + VectorOperations
        + NormalizableVector,
    Vector::Scalar:
        Zero + Copy + NumOps<Vector::Scalar, Vector::Scalar> + SimdRealField + SimdPartialOrd,
    <Vector::Scalar as SimdValue>::Element: Float + Copy,
    <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool: Debug,
{
    type RayType = Ray<Vector>;
    type ScalarType = Vector::Scalar;
    type VectorType = Vector;
    type MaskType = <Vector::Scalar as SimdValue>::SimdBool;

    type ReturnTypeWrapper<T> = T;

    fn check_intersection<'a, P>(
        &'a self,
        ray: &'_ Self::RayType,
        payload: P,
    ) -> Self::ReturnTypeWrapper<RayIntersectionCandidate<Self::ScalarType, P>> {
        let u = ray.direction;
        let v = ray.origin - self.c;

        // FIXME if  we garantuee ray direction is normalized, we can avoid multiplying by direction_mag_squared here, as it will be 1 anyways => A = 2.0
        // CHATGPT:
        // Diskriminant Δ=(2(u⋅v))^2 − 4(v⋅v−r^2)
        // You can factor out common terms (like the constant 4) and simplify the square root and division. This may let you avoid some multiplications and divisions in the inner loop.
        // this then further down leads to the optimization that we can calculate the inverse of A (maybe use the fast inverse from minmath crate) and convert the division by A futher down t A multiplication by inv_a -> multiplication is generally faster than division
        //let A = 2.0 * ray.direction_mag_squared; // u dot u
        static A: f32 = 2.0; // 2 * (u dot u) => 2 * direction_mag_squared => 2 * 1 => 2
        static A_INV: f32 = A.recip(); // crate::helpers::fast_inverse(A);
        let a_splat = Vector::Scalar::from_subset(&A);
        let two_splat = Vector::Scalar::from_subset(&2.0);
        let a_inv_splat = Vector::Scalar::from_subset(&A_INV);

        let b: Vector::Scalar = two_splat * u.dot(v);
        let c: Vector::Scalar = v.dot(v) - self.r_sq;

        let discriminant: Vector::Scalar = b.simd_mul_add(b, ((two_splat * a_splat) * c).neg());

        let discriminant_pos = discriminant.simd_ge(Vector::Scalar::zero());
        let discriminant_sqrt = discriminant.simd_sqrt();

        // FIXME optimize by replacing / A by * inv_a
        //let t1 = (-b - discriminant_sqrt) / A;
        let minus_b: Vector::Scalar = b.neg();
        let t0: Vector::Scalar = (minus_b - discriminant_sqrt) * a_inv_splat;
        let t1: Vector::Scalar = (minus_b + discriminant_sqrt) * a_inv_splat;

        //let t1_valid = t1.cmp_ge(F32Type::ZERO) & discriminant_pos;

        let t0_valid: <Vector::Scalar as SimdValue>::SimdBool =
            t0.simd_ge(Vector::Scalar::zero()) & discriminant_pos;
        let t1_valid: <Vector::Scalar as SimdValue>::SimdBool =
            t1.simd_ge(Vector::Scalar::zero()) & discriminant_pos;

        // Prefer t0 if it's valid, else t1 if that is valid.
        // If both are valid, t0 is nearer.
        let use_t0: <Vector::Scalar as SimdValue>::SimdBool =
            t0_valid & (!t1_valid | (t0.simd_lt(t1)));
        let use_t1: <Vector::Scalar as SimdValue>::SimdBool = t1_valid & !use_t0;

        // Start with invalid for all lanes
        let mut final_t = Ray::<Vector>::invalid_value_splatted();

        // Where t0 is chosen, blend in t0
        final_t = t0.select(use_t0, final_t);

        // Where t1 is chosen, blend in t1
        let final_t = t1.select(use_t1, final_t);

        // final_t_valid is lanes where we picked something
        let final_t_valid = use_t0 | use_t1;

        let estimated_distance = final_t * final_t;

        RayIntersectionCandidate::new(final_t, estimated_distance, payload, final_t_valid)
    }

    fn intersect<'a>(
        &'a self,
        ray: &'_ Self::RayType,
        candidate: &'_ RayIntersectionCandidate<Self::ScalarType, &'a Self>,
    ) -> Self::ReturnTypeWrapper<RayIntersection<Self::VectorType>> {
        let t = candidate.t;
        let valid_mask = candidate.valid_mask;

        let p = ray.at(t);
        let p_mag = p.mag();
        let r = p - self.c;
        let n = r.normalized();

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

#[cfg(test)]
mod test_sphere_intersection {
    use super::*;
    use assert_float_eq::assert_f32_near;

    #[test]
    fn test_check_intersection_sphere() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        let sphere = SphereData::new(3.0 * v, 0.1, ColorType::new(1., 1., 1.));

        let ray = Ray::new(Vec3::zero(), v);

        let i = sphere.check_intersection(&ray, &sphere);

        assert!(i.valid_mask);
        assert_f32_near!(i.estimated_distance_sq, ((3.0 * v) - 0.1 * v).mag_sq(), 10);

        let i = sphere.intersect(&ray, &i);
        assert!(i.valid_mask);

        assert_f32_near!(i.distance, ((3.0 * v) - 0.1 * v).mag(), 5);
    }

    #[test]
    fn test_check_intersection_sphere_not() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        let sphere = SphereData::new(
            2.0 * Vec3::new(0.0, 1.0, 0.0),
            0.1,
            ColorType::new(1., 1., 1.),
        );

        let ray = Ray::new(Vec3::zero(), v);

        let i = sphere.check_intersection(&ray, ());

        assert!(!i.valid_mask);
    }
}

impl<Vector> SphereData<Vector>
where
    Vector: crate::vector::Vector + SimdCapableVector,
    Vector::Scalar:
        Zero + Clone + NumOps<Vector::Scalar, Vector::Scalar> + SimdRealField + SimdPartialOrd,
{
    fn blend(mask: <Vector::Scalar as SimdValue>::SimdBool, t: &Self, f: &Self) -> Self {
        Self {
            c: Vector::blend(mask, t.c.clone(), f.c.clone()),
            r_inv: t.r_inv.clone().select(mask, f.r_inv.clone()),
            r_sq: t.r_sq.clone().select(mask, f.r_sq.clone()),
            color: ColorType::<Vector::Scalar>::new(
                t.color.red.clone().select(mask, f.color.red.clone()),
                t.color.green.clone().select(mask, f.color.green.clone()),
                t.color.blue.clone().select(mask, f.color.blue.clone()),
            ),
        }
    }

    fn splat(v: &SphereData<<Vector as SimdCapableVector>::SingleValueVector>) -> Self
    where
        <<Vector as SimdCapableVector>::SingleValueVector as crate::vector::Vector>::Scalar:
            SubsetOf<<Vector as crate::vector::Vector>::Scalar>,
    {
        Self {
            c: Vector::splat(v.c.clone()),
            r_inv: Vector::Scalar::from_subset(&v.r_inv),
            r_sq: Vector::Scalar::from_subset(&v.r_sq),
            color: ColorType::<Vector::Scalar>::new(
                Vector::Scalar::from_subset(&v.color.red),
                Vector::Scalar::from_subset(&v.color.green),
                Vector::Scalar::from_subset(&v.color.blue),
            ),
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct PointData<Vector>
where
    Vector: crate::vector::Vector,
    <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool: Debug,
{
    p: Vector,
}

impl<Vector> PointData<Vector>
where
    Vector: crate::vector::Vector,
    <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool: Debug,
{
    const fn new(p: Vector) -> Self {
        Self { p }
    }
}

impl<Vector> Intersectable for PointData<Vector>
where
    Vector: crate::vector::Vector
        + VectorOperations
        + NormalizableVector
        + Copy
        + Add<Vector, Output = Vector>
        + Sub<Vector, Output = Vector>
        + Mul<Vector, Output = Vector>,
    Vector::Scalar: Zero + Copy + SimdRealField,
    <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool: Debug,
{
    type RayType = Ray<Vector>;
    type ScalarType = Vector::Scalar;
    type VectorType = Vector;
    type MaskType = <Vector::Scalar as SimdValue>::SimdBool;

    type ReturnTypeWrapper<T> = T;

    fn check_intersection<'a, P>(
        &'a self,
        ray: &'_ Self::RayType,
        payload: P,
    ) -> Self::ReturnTypeWrapper<RayIntersectionCandidate<Self::ScalarType, P>> {
        let p = self.p;
        let v = p - ray.origin;
        let t = v.dot(ray.direction);

        let intersection_valid = (ray.at(t) - p)
            .mag_sq()
            .simd_lt(Vector::Scalar::from_subset(&0.001));

        let estimated_distance = t * t;

        RayIntersectionCandidate::new(t, estimated_distance, payload, intersection_valid)
    }

    fn intersect<'a>(
        &'a self,
        ray: &'_ Self::RayType,
        candidate: &'_ RayIntersectionCandidate<Self::ScalarType, &'a Self>,
    ) -> Self::ReturnTypeWrapper<RayIntersection<Self::VectorType>> {
        let t = candidate.t;
        let valid_mask = candidate.valid_mask;

        let p = ray.at(t);
        let n = ray.at(t.neg()).normalized();

        RayIntersection::new(
            p,
            ray.direction.clone(),
            n,
            (p - ray.origin).mag(),
            Vector::Scalar::from_subset(&1.0),
            valid_mask,
        )
    }
}

impl<Vector> PointData<Vector>
where
    Vector: crate::vector::Vector + SimdCapableVector,
    Vector::Scalar:
        Zero + Clone + NumOps<Vector::Scalar, Vector::Scalar> + SimdRealField + SimdPartialOrd,
    <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool: Debug,
{
    fn blend(mask: <Vector::Scalar as SimdValue>::SimdBool, t: &Self, f: &Self) -> Self {
        Self {
            p: Vector::blend(mask, t.p.clone(), f.p.clone()),
        }
    }

    fn splat(v: &PointData<<Vector as SimdCapableVector>::SingleValueVector>) -> Self
    where
        <<Vector as SimdCapableVector>::SingleValueVector as crate::vector::Vector>::Scalar:
        SubsetOf<<Vector as crate::vector::Vector>::Scalar>,
        <<<Vector as SimdCapableVector>::SingleValueVector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool: Debug,
    {
        Self {
            p: Vector::splat(v.p.clone()),
        }
    }
}

#[cfg(test)]
mod test_point_intersection {
    use super::*;
    use palette::bool_mask::BoolMask;

    #[test]
    fn test_check_intersection_point() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        let point = PointData::new(2.0 * v);

        let ray = Ray::new(Vec3::zero(), v);

        let i = point.check_intersection(&ray, ());

        assert_eq!(i.estimated_distance_sq, (2.0 * v).mag_sq());
    }

    #[test]
    fn test_check_intersection_point_not() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        let point = PointData::new(2.0 * Vec3::new(0.0, 1.0, 0.0));

        let ray = Ray::new(Vec3::zero(), v);

        let i = point.check_intersection(&ray, ());

        assert!(i.valid_mask.is_false());
    }
}

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

const N_RANDOM_SPHERES: usize = 2_000;
static SPHERES_RANDOM: LazyLock<[SphereData<Vec3>; N_RANDOM_SPHERES]> = LazyLock::new(|| {
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

static LIGHTS: LazyLock<[SphereData<Vec3>; 4]> = LazyLock::new(|| {
    [
        SphereData::new(
            Vec3::new(-100.0, 1000.0, -10.0),
            100.0,
            ColorType::new(0.822, 0.675, 0.45),
        ),
        SphereData::new(
            Vec3::new(1.0, -1.0, 100.0),
            100.0,
            ColorType::new(0.0, 0.675, 0.9),
        ),
        SphereData::new(
            Vec3::new(0.0, 0.0, -100000.0),
            100.0,
            ColorType::new(0., 0., 0.),
        ),
        SphereData::new(
            Vec3::new((WINDOW_WIDTH / 2) as f32, (WINDOW_HEIGHT / 2) as f32, 120.0),
            100.0,
            ColorType::new(0.7, 0.6, 0.5),
        ),
    ]
});

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct TestRenderer3DLightColorSW03<C: OutputColorEncoder>(PhantomData<C>);

impl<C: OutputColorEncoder> TestRenderer3DLightColorSW03<C> {
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

    fn get_pixel_color_vectorized<'a, Vector>(
        coords: Vector,
        unit_z: Vector,
        spheres: impl IntoIterator<
            Item = &'a SphereData<<Vector as SimdCapableVector>::SingleValueVector>,
        >,
        lights: impl IntoIterator<
            Item = &'a SphereData<<Vector as SimdCapableVector>::SingleValueVector>,
        >,
    ) -> Option<(
        ColorType<Vector::Scalar>,
        <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool,
    )>
    where
        Vector: 'a
            + crate::vector::Vector
            + Copy
            + VectorOperations
            + NormalizableVector
            + SimdCapableVector
            + Add<Vector, Output = Vector>
            + Sub<Vector, Output = Vector>
            + Mul<Vector, Output = Vector>,
        Vector::Scalar: Zero
            + One
            + Copy
            + NumOps<Vector::Scalar, Vector::Scalar>
            + SimdRealField
            + SimdPartialOrd
            + SubsetOf<<Vector as crate::vector::Vector>::Scalar>
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
        ColorType<<Vector as crate::vector::Vector>::Scalar>: Premultiply<Scalar = Vector::Scalar>
            + StimulusColor
            + ArrayCast<Array = [Vector::Scalar; <Vector as crate::vector::Vector>::DIMENSIONS]>,
        <<Vector as crate::vector::Vector>::Scalar as HasBoolMask>::Mask:
            LazySelect<<Vector as crate::vector::Vector>::Scalar>,
        <Vector::Scalar as SimdValue>::Element: Float + Copy,
        <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool:
            Debug + SimdValue<Element = bool>,
        [(); <Vector as crate::vector::Vector>::LANES]:,
        <<Vector as SimdCapableVector>::SingleValueVector as crate::vector::Vector>::Scalar:
            SubsetOf<<Vector as crate::vector::Vector>::Scalar>,
    {
        let ray = Ray::<Vector>::new_with_mask(
            coords,
            unit_z,
            <Vector::Scalar as SimdValue>::SimdBool::splat(true),
        );

        let nearest_intersection =
            spheres
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

                    let new_is_nearer = previous_intersection
                        .estimated_distance_sq
                        .simd_ge(new_intersection.estimated_distance_sq);

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
                    let merged_est_dist = previous_intersection.estimated_distance_sq.select(
                        pick_old_mask.clone(),
                        new_intersection.estimated_distance_sq,
                    );
                    let merged_payload = SphereData::<Vector>::blend(
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
                nearest_intersection.estimated_distance_sq,
                intersected_object.as_ref(),
                nearest_intersection.valid_mask,
            ),
        );

        //////////////////////

        let d = distance / intersected_object.c.mag();

        let zero = Vector::Scalar::zero();
        let one = Vector::Scalar::one();

        let ambient_lighting_amount = Vector::Scalar::from_subset(&0.06);
        let mut direct_lighting_amount = zero;
        let mut light_color = Srgb::<Vector::Scalar>::new(zero, zero, zero);

        for light in lights.into_iter() {
            let light = SphereData::<Vector>::splat(light);
            let light_to_intersection = light.c - intersection_point;
            let light_distance = light_to_intersection.mag();

            let incident_light_angle_cos =
                light_to_intersection.dot(intersection_normal) / light_distance;

            let incident_angle_pos: <Vector::Scalar as SimdValue>::SimdBool =
                incident_light_angle_cos.simd_ge(zero);

            let mixed_color = light_color.mix(
                light.color,
                incident_light_angle_cos.simd_abs() + one / (light_distance),
            );

            // todo extract that to a trait or something
            light_color = Srgb::<Vector::Scalar>::new(
                mixed_color.red.select(incident_angle_pos, light_color.red),
                mixed_color
                    .green
                    .select(incident_angle_pos, light_color.green),
                mixed_color
                    .blue
                    .select(incident_angle_pos, light_color.blue),
            );

            let new_light_value: Vector::Scalar = direct_lighting_amount
                + (incident_light_angle_cos.simd_abs() + one / (light_distance));

            let new_light_value = new_light_value.simd_min(Vector::Scalar::from_subset(&0.89));

            direct_lighting_amount =
                new_light_value.select(incident_angle_pos, direct_lighting_amount);
        }

        // raytrace light back to surce
        // let reflected_ray = intersection.to_reflected_ray();
        // if let Some(_) = LIGHT.check_intersection(&reflected_ray, ()) {
        //     direct_lighting_amount = incident_angle_cos.abs();
        //     light_color = LIGHT.color.clone();
        // }

        let ambient_color = intersected_object.color.darken_fixed(
            one - ambient_lighting_amount
                * (Vector::Scalar::from_subset(&0.25) / d)
                    .simd_min(Vector::Scalar::from_subset(&1.1)),
        );

        Some((
            ambient_color.mix(
                ambient_color.soft_light(light_color),
                direct_lighting_amount,
            ),
            valid_mask,
        ))
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
