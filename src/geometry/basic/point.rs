use crate::geometry::Ray;
use crate::helpers::Splatable;
use crate::raytracing::{Intersectable, RayIntersection, RayIntersectionCandidate};
use crate::vector::{SimdCapableVector, VectorAware};
use crate::vector_traits::{BaseVector, RenderingVector, SimdRenderingVector};
use simba::scalar::SupersetOf;
use simba::simd::{SimdPartialOrd, SimdValue};
use std::ops::{Add, Neg, Sub};

#[derive(Debug, Copy, Clone)]
#[repr(transparent)]
pub struct PointData<V>
where
    V: BaseVector,
{
    p: V,
}

impl<V> PointData<V>
where
    V: BaseVector,
{
    pub const fn new(p: V) -> Self {
        Self { p }
    }
}

impl<V> PointData<V>
where
    V: SimdRenderingVector,
{
    pub fn blend(mask: <V::Scalar as SimdValue>::SimdBool, t: &Self, f: &Self) -> Self {
        Self {
            p: V::blend(mask, t.p.clone(), f.p.clone()),
        }
    }
}

impl<V> Splatable<PointData<<V as SimdCapableVector>::SingleValueVector>> for PointData<V>
where
    V: SimdRenderingVector,
{
    fn splat(v: &PointData<<V as SimdCapableVector>::SingleValueVector>) -> Self {
        Self {
            p: V::splat(v.p.clone()),
        }
    }
}

impl<V> VectorAware<V> for PointData<V> where V: BaseVector {}
impl<V> BasicGeometry<V> for PointData<V> where V: BaseVector {}

impl<V: BaseVector> Sub<V> for PointData<V> {
    type Output = Self;

    fn sub(self, rhs: V) -> Self::Output {
        Self { p: self.p - rhs }
    }
}

impl<V: BaseVector> Add<V> for PointData<V> {
    type Output = Self;

    fn add(self, rhs: V) -> Self::Output {
        Self { p: self.p + rhs }
    }
}

impl<V> Intersectable<V> for PointData<V>
where
    V: RenderingVector,
{
    type RayType = Ray<V>;

    type ReturnTypeWrapper<T> = T;

    fn check_intersection<'a, P>(
        &'a self,
        ray: &'_ Self::RayType,
        payload: P,
    ) -> Self::ReturnTypeWrapper<RayIntersectionCandidate<V::Scalar, P>> {
        let p = self.p;
        let v = p - ray.origin;
        let t = v.dot(ray.direction);

        let intersection_valid = (ray.at(t) - p)
            .mag_sq()
            .simd_lt(V::Scalar::from_subset(&0.001));

        RayIntersectionCandidate::new(t, payload, intersection_valid)
    }

    fn intersect<'a, P>(
        &'a self,
        ray: &'_ Self::RayType,
        candidate: &'_ RayIntersectionCandidate<V::Scalar, &'a P>,
    ) -> Self::ReturnTypeWrapper<RayIntersection<V>>
    where
        P: Intersectable<V>,
    {
        let t = candidate.t;
        let valid_mask = candidate.valid_mask;

        let p = ray.at(t);
        let n = ray.at(t.neg()).normalized();

        RayIntersection::new(
            p,
            ray.direction.clone(),
            n,
            (p - ray.origin).mag(),
            V::Scalar::from_subset(&1.0),
            valid_mask,
        )
    }
}

#[cfg(test)]
mod test_point_intersection {
    use super::*;
    use palette::bool_mask::BoolMask;
    use ultraviolet::Vec3;

    #[test]
    fn test_check_intersection_point() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        let point = PointData::new(2.0 * v);

        let ray = Ray::new(Vec3::zero(), v);

        let i = point.check_intersection(&ray, ());

        assert_eq!(i.t, (2.0 * v).mag());
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
