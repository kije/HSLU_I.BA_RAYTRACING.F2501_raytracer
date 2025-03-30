use crate::color::ColorSimdExt;
use crate::geometry::Ray;
use crate::helpers::{ColorType, Splatable};
use crate::raytracing::{Intersectable, RayIntersection, RayIntersectionCandidate};
use crate::scene::Lightable;
use crate::vector::{
    CommonVecOperations, CommonVecOperationsFloat, CommonVecOperationsSimdOperations, VectorAware,
};
use num_traits::{Float, NumOps, Zero};
use simba::scalar::{SubsetOf, SupersetOf};
use simba::simd::{SimdComplexField, SimdPartialOrd, SimdRealField, SimdValue};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Copy, Clone)]
pub(crate) struct SphereData<Vector>
where
    Vector: crate::vector::Vector,
{
    pub(crate) center: Vector,
    r_sq: Vector::Scalar,
    r_inv: Vector::Scalar,
    pub(crate) color: ColorType<Vector::Scalar>,
}

impl<Vector> SphereData<Vector>
where
    Vector: crate::vector::Vector,
{
    pub(crate) fn new(c: Vector, r: Vector::Scalar, color: ColorType<Vector::Scalar>) -> Self
    where
        Vector::Scalar:
            Sized + Copy + From<f32> + Div<Output = Vector::Scalar> + Mul<Output = Vector::Scalar>,
    {
        Self {
            center: c,
            r_sq: r * r,
            r_inv: <Vector::Scalar as From<f32>>::from(1.0) / r,
            color,
        }
    }

    #[inline(always)]
    pub(crate) fn get_surface_normal_at_point(&self, p: Vector) -> Vector
    where
        Vector: std::ops::Sub<Vector, Output = Vector> + CommonVecOperationsFloat,
    {
        // FIXME this can & should bec cached?
        let r = p - self.center.clone();
        r.normalized()
    }
}

impl<Vector> SphereData<Vector>
where
    Vector: crate::vector::Vector + CommonVecOperationsSimdOperations,
{
    pub(crate) fn blend(mask: <Vector::Scalar as SimdValue>::SimdBool, t: &Self, f: &Self) -> Self
    where
        <<Vector as crate::vector::Vector>::Scalar as SimdValue>::Element:
            SubsetOf<<Vector as crate::vector::Vector>::Scalar>,
    {
        Self {
            center: Vector::blend(mask, t.center.clone(), f.center.clone()),
            r_inv: t.r_inv.clone().select(mask, f.r_inv.clone()),
            r_sq: t.r_sq.clone().select(mask, f.r_sq.clone()),
            color: ColorSimdExt::blend(mask, &t.color, &f.color),
        }
    }

    // pub(crate) fn splat(v: &SphereData<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector>) -> Self
    // where
    //     <<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar:
    //         SubsetOf<<Vector as crate::vector::Vector>::Scalar>,
    //     Vector::Scalar:   Clone +  SimdRealField + SupersetOf<<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar>, <<Vector as crate::vector::Vector>::Scalar as SimdValue>::Element: SubsetOf<<Vector as crate::vector::Vector>::Scalar>
    // {
    //     Self {
    //         center: Vector::splat(v.center.clone()),
    //         r_inv: Vector::Scalar::from_subset(&v.r_inv),
    //         r_sq: Vector::Scalar::from_subset(&v.r_sq),
    //         color: ColorSimdExt::splat(&v.color),
    //     }
    // }
}

impl<Vector> Splatable<SphereData<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector>> for SphereData<Vector>
where
    Vector: crate::vector::Vector + CommonVecOperationsSimdOperations,
    Vector::Scalar: Clone + SimdRealField + SupersetOf<<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar> + Splatable<<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar>,
    <<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar: SubsetOf<<Vector as crate::vector::Vector>::Scalar>,
    <<Vector as crate::vector::Vector>::Scalar as SimdValue>::Element: SubsetOf<<Vector as crate::vector::Vector>::Scalar>
{
    fn splat(v: &SphereData<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector>) -> Self  {
        Self {
            center: Vector::splat(v.center.clone()),
            r_inv: Vector::Scalar::from_subset(&v.r_inv),
            r_sq: Vector::Scalar::from_subset(&v.r_sq),
            color: Splatable::splat(&v.color),
        }
    }
}

impl<Vector> VectorAware<Vector> for SphereData<Vector> where Vector: crate::vector::Vector {}

impl<Vector> Intersectable<Vector> for SphereData<Vector>
where
    Vector: crate::vector::Vector
        + Add<Vector, Output = Vector>
        + Sub<Vector, Output = Vector>
        + Mul<Vector, Output = Vector>
        + Copy
        + CommonVecOperations
        + CommonVecOperationsFloat,
    Vector::Scalar: Zero
        + Copy
        + crate::scalar::Scalar
        + SimdValue
        + NumOps<Vector::Scalar, Vector::Scalar>
        + SimdRealField
        + SimdPartialOrd,
    <Vector::Scalar as SimdValue>::Element: Float + Copy,
{
    type RayType = Ray<Vector>;
    type ReturnTypeWrapper<T> = T;

    fn check_intersection<'a, P>(
        &'a self,
        ray: &'_ Self::RayType,
        payload: P,
    ) -> Self::ReturnTypeWrapper<RayIntersectionCandidate<Vector::Scalar, P>> {
        let u = ray.direction;
        let v = ray.origin - self.center;

        // FIXME if  we garantuee ray direction is normalized, we can avoid multiplying by direction_mag_squared here, as it will be 1 anyways => A = 2.0
        // CHATGPT:
        // Diskriminant Δ=(2(u⋅v))^2 − 4(v⋅v−r^2)
        // You can factor out common terms (like the constant 4) and simplify the square root and division. This may let you avoid some multiplications and divisions in the inner loop.
        // this then further down leads to the optimization that we can calculate the inverse of A (maybe use the fast inverse from minmath crate) and convert the division by A futher down t A multiplication by inv_a -> multiplication is generally faster than division
        //let A = 2.0 * ray.direction_mag_squared; // u dot u
        const A: f32 = 2.0; // 2 * (u dot u) => 2 * direction_mag_squared => 2 * 1 => 2
        const A_INV: f32 = A.recip(); // crate::helpers::fast_inverse(A);
        let a_splat_neg = Vector::Scalar::from_subset(&-A);
        let two_splat = Vector::Scalar::from_subset(&2.0);
        let a_inv_splat = Vector::Scalar::from_subset(&A_INV);
        let zero = Vector::Scalar::zero();

        let b: Vector::Scalar = two_splat * u.dot(v);
        let c: Vector::Scalar = v.dot(v) - self.r_sq;

        let discriminant: Vector::Scalar = b.simd_mul_add(b, (two_splat * a_splat_neg) * c);

        let discriminant_pos = discriminant.simd_ge(zero);
        let discriminant_sqrt = discriminant.simd_sqrt();

        // FIXME optimize by replacing / A by * inv_a
        //let t1 = (-b - discriminant_sqrt) / A;
        let minus_b: Vector::Scalar = b.neg();
        let minusb_a_inv = minus_b * a_inv_splat;
        let discriminant_sqr_times_a_inv = discriminant_sqrt * a_inv_splat;
        let t0: Vector::Scalar = minusb_a_inv - discriminant_sqr_times_a_inv;
        let t1: Vector::Scalar = minusb_a_inv + discriminant_sqr_times_a_inv;

        //let t1_valid = t1.cmp_ge(F32Type::ZERO) & discriminant_pos;

        let t0_valid: <Vector::Scalar as SimdValue>::SimdBool = t0.simd_ge(zero) & discriminant_pos;
        let t1_valid: <Vector::Scalar as SimdValue>::SimdBool = t1.simd_ge(zero) & discriminant_pos;

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

        RayIntersectionCandidate::new(final_t, payload, final_t_valid)
    }

    fn intersect<'a>(
        &'a self,
        ray: &'_ Self::RayType,
        candidate: &'_ RayIntersectionCandidate<Vector::Scalar, &'a Self>,
    ) -> Self::ReturnTypeWrapper<RayIntersection<Vector>> {
        let t = candidate.t;
        let valid_mask = candidate.valid_mask;

        let p = ray.at(t);
        let p_mag = p.mag();
        let n = self.get_surface_normal_at_point(p);

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

impl<Vector: crate::vector::Vector + Sub<Vector, Output = Vector>> Sub<Vector>
    for SphereData<Vector>
{
    type Output = Self;

    fn sub(self, rhs: Vector) -> Self::Output {
        Self {
            r_sq: self.r_sq,
            r_inv: self.r_inv,
            color: self.color,
            center: self.center - rhs,
        }
    }
}

impl<Vector: crate::vector::Vector + Add<Vector, Output = Vector>> Add<Vector>
    for SphereData<Vector>
{
    type Output = Self;

    fn add(self, rhs: Vector) -> Self::Output {
        Self {
            r_sq: self.r_sq,
            r_inv: self.r_inv,
            color: self.color,
            center: self.center + rhs,
        }
    }
}

impl<Vector> Lightable<Vector> for SphereData<Vector>
where
    Vector: crate::vector::Vector + Sub<Vector, Output = Vector> + CommonVecOperationsFloat,
{
    fn get_material_color_at(&self, _: Vector) -> ColorType<Vector::Scalar> {
        self.color.clone()
    }

    fn get_surface_normal_at(&self, point: Vector) -> Vector {
        self.get_surface_normal_at_point(point)
    }
}

#[cfg(test)]
mod test_sphere_intersection {
    use super::*;
    use assert_float_eq::assert_f32_near;
    use ultraviolet::Vec3;

    #[test]
    fn test_check_intersection_sphere() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        let sphere = SphereData::new(3.0 * v, 0.1, ColorType::new(1., 1., 1.));

        let ray = Ray::new(Vec3::zero(), v);

        let i = sphere.check_intersection(&ray, &sphere);

        assert!(i.valid_mask);
        assert_f32_near!(i.t, ((3.0 * v) - 0.1 * v).mag(), 10);

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
