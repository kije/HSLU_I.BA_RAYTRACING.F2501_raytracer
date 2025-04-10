use crate::color::ColorSimdExt;
use crate::color_traits::LightCompatibleColor;
use crate::geometry::Ray;
use crate::helpers::{ColorType, Splatable};
use crate::raytracing::{Intersectable, RayIntersection, RayIntersectionCandidate};
use crate::scalar_traits::LightScalar;
use crate::scene::Lightable;
use crate::vector::{NormalizableVector, SimdCapableVector, VectorAware, VectorOperations};
use crate::vector_traits::{BaseVector, RenderingVector};
use num_traits::{Float, NumOps, Zero};
use palette::bool_mask::BoolMask;
use simba::scalar::{SubsetOf, SupersetOf};
use simba::simd::{SimdBool, SimdComplexField, SimdPartialOrd, SimdRealField, SimdValue};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Copy, Clone)]
pub(crate) struct SphereData<V>
where
    V: BaseVector,
{
    pub(crate) center: V,
    r_sq: V::Scalar,
    r_inv: V::Scalar,
    pub(crate) color: ColorType<V::Scalar>,
}

impl<V> SphereData<V>
where
    V: BaseVector,
{
    pub(crate) fn new(c: V, r: V::Scalar, color: ColorType<V::Scalar>) -> Self
    where
        V::Scalar: Sized + Copy + From<f32> + Div<Output = V::Scalar> + Mul<Output = V::Scalar>,
    {
        Self {
            center: c,
            r_sq: r * r,
            r_inv: <V::Scalar as From<f32>>::from(1.0) / r,
            color,
        }
    }

    #[inline(always)]
    pub(crate) fn get_surface_normal_at_point(&self, p: V) -> V
    where
        V: Sub<V, Output = V> + NormalizableVector,
    {
        // FIXME this can & should bec cached?
        let r = p - self.center.clone();
        r.normalized()
    }
}

impl<V> SphereData<V>
where
    V: RenderingVector + SimdCapableVector,
    V::Scalar: LightScalar,
{
    pub(crate) fn blend(mask: <V::Scalar as SimdValue>::SimdBool, t: &Self, f: &Self) -> Self
    where
        <<V as crate::vector::Vector>::Scalar as SimdValue>::Element:
            SubsetOf<<V as crate::vector::Vector>::Scalar>,
    {
        Self {
            center: V::blend(mask, t.center.clone(), f.center.clone()),
            r_inv: t.r_inv.clone().select(mask, f.r_inv.clone()),
            r_sq: t.r_sq.clone().select(mask, f.r_sq.clone()),
            color: ColorSimdExt::blend(mask, &t.color, &f.color),
        }
    }
}

impl<V> Splatable<SphereData<<V as SimdCapableVector>::SingleValueVector>> for SphereData<V>
where
    V: RenderingVector + SimdCapableVector,
    <V as SimdCapableVector>::SingleValueVector: BaseVector,
    V::Scalar: LightScalar
        + SupersetOf<<<V as SimdCapableVector>::SingleValueVector as crate::vector::Vector>::Scalar>
        + Splatable<<<V as SimdCapableVector>::SingleValueVector as crate::vector::Vector>::Scalar>,
    <<V as SimdCapableVector>::SingleValueVector as crate::vector::Vector>::Scalar:
        SubsetOf<<V as crate::vector::Vector>::Scalar>,
    <<V as crate::vector::Vector>::Scalar as SimdValue>::Element:
        SubsetOf<<V as crate::vector::Vector>::Scalar>,
{
    fn splat(v: &SphereData<<V as SimdCapableVector>::SingleValueVector>) -> Self {
        Self {
            center: V::splat(v.center.clone()),
            r_inv: V::Scalar::from_subset(&v.r_inv),
            r_sq: V::Scalar::from_subset(&v.r_sq),
            color: Splatable::splat(&v.color),
        }
    }
}

impl<V> VectorAware<V> for SphereData<V> where V: BaseVector {}

impl<V> Intersectable<V> for SphereData<V>
where
    V: RenderingVector,
    V::Scalar: Zero + LightScalar<SimdBool: SimdBool + BoolMask>,
    <V::Scalar as SimdValue>::Element: Float + Copy,
{
    type RayType = Ray<V>;
    type ReturnTypeWrapper<T> = T;

    fn check_intersection<'a, P>(
        &'a self,
        ray: &'_ Self::RayType,
        payload: P,
    ) -> Self::ReturnTypeWrapper<RayIntersectionCandidate<V::Scalar, P>> {
        let u = ray.direction;
        let v = ray.origin - self.center;

        const A: f32 = 2.0; // 2 * (u dot u) => 2 * direction_mag_squared => 2 * 1 => 2
        const A_INV: f32 = A.recip(); // crate::helpers::fast_inverse(A);
        let a_splat_neg = V::Scalar::from_subset(&-A);
        let two_splat = V::Scalar::from_subset(&2.0);
        let a_inv_splat = V::Scalar::from_subset(&A_INV);
        let zero = V::Scalar::zero();
        let invalid_value = Self::RayType::invalid_value_splatted();

        let b: V::Scalar = two_splat * u.dot(v);
        let c: V::Scalar = v.dot(v) - self.r_sq;

        let discriminant: V::Scalar = b.simd_mul_add(b, (two_splat * a_splat_neg) * c);

        let discriminant_pos = discriminant.simd_ge(zero);

        // shortcircuit
        if discriminant_pos.none() {
            return RayIntersectionCandidate::new(
                invalid_value,
                payload,
                <<V::Scalar as SimdValue>::SimdBool as BoolMask>::from_bool(false),
            );
        }

        let discriminant_sqrt = discriminant.simd_sqrt();

        let minus_b: V::Scalar = b.neg();
        let minusb_a_inv = minus_b * a_inv_splat;
        let discriminant_sqr_times_a_inv = discriminant_sqrt * a_inv_splat;
        let t0: V::Scalar = minusb_a_inv - discriminant_sqr_times_a_inv;
        let t1: V::Scalar = minusb_a_inv + discriminant_sqr_times_a_inv;

        let t0_valid: <V::Scalar as SimdValue>::SimdBool = t0.simd_ge(zero) & discriminant_pos;
        let t1_valid: <V::Scalar as SimdValue>::SimdBool = t1.simd_ge(zero) & discriminant_pos;

        // Prefer t0 if it's valid, else t1 if that is valid.
        // If both are valid, t0 is nearer.
        let use_t0: <V::Scalar as SimdValue>::SimdBool = t0_valid & (!t1_valid | (t0.simd_lt(t1)));
        let use_t1: <V::Scalar as SimdValue>::SimdBool = t1_valid & !use_t0;

        // Start with invalid for all lanes
        let mut final_t = invalid_value;

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
        candidate: &'_ RayIntersectionCandidate<V::Scalar, &'a Self>,
    ) -> Self::ReturnTypeWrapper<RayIntersection<V>> {
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

impl<V: BaseVector + Sub<V, Output = V>> Sub<V> for SphereData<V> {
    type Output = Self;

    fn sub(self, rhs: V) -> Self::Output {
        Self {
            r_sq: self.r_sq,
            r_inv: self.r_inv,
            color: self.color,
            center: self.center - rhs,
        }
    }
}

impl<V: BaseVector + Add<V, Output = V>> Add<V> for SphereData<V> {
    type Output = Self;

    fn add(self, rhs: V) -> Self::Output {
        Self {
            r_sq: self.r_sq,
            r_inv: self.r_inv,
            color: self.color,
            center: self.center + rhs,
        }
    }
}

impl<V> Lightable<V> for SphereData<V>
where
    V: RenderingVector,
    V::Scalar: LightScalar,
{
    fn get_material_color_at(&self, _: V) -> ColorType<V::Scalar> {
        self.color.clone()
    }

    fn get_surface_normal_at(&self, point: V) -> V {
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
