use crate::geometry::Ray;
use crate::helpers::Splatable;
use crate::raytracing::{Intersectable, RayIntersection, RayIntersectionCandidate};
use crate::vector::{
    CommonVecOperations, CommonVecOperationsFloat, CommonVecOperationsSimdOperations, VectorAware,
};
use num_traits::{NumOps, Zero};
use simba::scalar::{SubsetOf, SupersetOf};
use simba::simd::{SimdPartialOrd, SimdRealField, SimdValue};
use std::fmt::Debug;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Debug, Copy, Clone)]
#[repr(transparent)]
pub(crate) struct PointData<Vector>
where
    Vector: crate::vector::Vector,
{
    p: Vector,
}

impl<Vector> PointData<Vector>
where
    Vector: crate::vector::Vector,
{
    pub(crate) const fn new(p: Vector) -> Self {
        Self { p }
    }
}

impl<Vector> PointData<Vector>
where
    Vector: crate::vector::Vector + CommonVecOperationsSimdOperations,
    Vector::Scalar:
        Zero + Clone + NumOps<Vector::Scalar, Vector::Scalar> + SimdRealField + SimdPartialOrd,
{
    pub(crate) fn blend(mask: <Vector::Scalar as SimdValue>::SimdBool, t: &Self, f: &Self) -> Self {
        Self {
            p: Vector::blend(mask, t.p.clone(), f.p.clone()),
        }
    }

    // pub(crate) fn splat(v: &PointData<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector>) -> Self
    // where
    //     <<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar:
    //     SubsetOf<<Vector as crate::vector::Vector>::Scalar>
    // {
    //     Self {
    //         p: Vector::splat(v.p.clone()),
    //     }
    // }
}

impl<Vector> Splatable<PointData<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector>> for PointData<Vector>
where
    Vector: crate::vector::Vector + CommonVecOperationsSimdOperations,
    Vector::Scalar: Clone + SimdRealField + SupersetOf<<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar>,
    <<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar: SubsetOf<<Vector as crate::vector::Vector>::Scalar>
{
    fn splat(v: &PointData<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector>) -> Self {
        Self {
            p: Vector::splat(v.p.clone()),
        }
    }
}

impl<Vector> VectorAware<Vector> for PointData<Vector> where Vector: crate::vector::Vector {}

impl<Vector: crate::vector::Vector + Sub<Vector, Output = Vector>> Sub<Vector>
    for PointData<Vector>
{
    type Output = Self;

    fn sub(self, rhs: Vector) -> Self::Output {
        Self { p: self.p - rhs }
    }
}

impl<Vector: crate::vector::Vector + Add<Vector, Output = Vector>> Add<Vector>
    for PointData<Vector>
{
    type Output = Self;

    fn add(self, rhs: Vector) -> Self::Output {
        Self { p: self.p + rhs }
    }
}

impl<Vector> Intersectable<Vector> for PointData<Vector>
where
    Vector: crate::vector::Vector
        + CommonVecOperations
        + CommonVecOperationsFloat
        + Copy
        + Add<Vector, Output = Vector>
        + Sub<Vector, Output = Vector>
        + Mul<Vector, Output = Vector>,
    Vector::Scalar: Zero + Copy + crate::scalar::Scalar + SimdValue + SimdRealField,
{
    type RayType = Ray<Vector>;

    type ReturnTypeWrapper<T> = T;

    fn check_intersection<'a, P>(
        &'a self,
        ray: &'_ Self::RayType,
        payload: P,
    ) -> Self::ReturnTypeWrapper<RayIntersectionCandidate<Vector::Scalar, P>> {
        let p = self.p;
        let v = p - ray.origin;
        let t = v.dot(ray.direction);

        let intersection_valid = (ray.at(t) - p)
            .mag_sq()
            .simd_lt(Vector::Scalar::from_subset(&0.001));

        RayIntersectionCandidate::new(t, payload, intersection_valid)
    }

    fn intersect<'a>(
        &'a self,
        ray: &'_ Self::RayType,
        candidate: &'_ RayIntersectionCandidate<Vector::Scalar, &'a Self>,
    ) -> Self::ReturnTypeWrapper<RayIntersection<Vector>> {
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
