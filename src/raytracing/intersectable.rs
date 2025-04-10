use crate::geometry::Ray;
use crate::vector::{
    CommonVecOperations, CommonVecOperationsFloat, CommonVecOperationsReflectable, VectorAware,
};

use simba::simd::SimdValue;
use std::fmt::Debug;

pub(crate) trait Intersectable<Vector>
where
    Vector: crate::vector::Vector,
    Vector::Scalar: crate::scalar::Scalar + SimdValue,
{
    const LANES: usize = Vector::LANES;

    type RayType: VectorAware<Vector>;

    type ReturnTypeWrapper<T>;

    fn check_intersection<'a, P>(
        &'a self,
        ray: &'_ Self::RayType,
        payload: P,
    ) -> Self::ReturnTypeWrapper<RayIntersectionCandidate<Vector::Scalar, P>>;

    // FIXME: we might want to move that to the RayIntersectionCandidate type, as you normaly would call this anyyway by candidate.payload.intersect(ray, candidate)?
    // or we want to have a shortcut from RayIntersectionCandidate
    fn intersect<'a>(
        &'a self,
        ray: &'_ Self::RayType,
        candidate: &'_ RayIntersectionCandidate<Vector::Scalar, &'a Self>,
    ) -> Self::ReturnTypeWrapper<RayIntersection<Vector>>;
}

#[derive(Clone, Debug, Copy)]
pub(crate) struct RayIntersectionCandidate<Scalar, Payload>
where
    Scalar: SimdValue + crate::scalar::Scalar,
    Payload: ?Sized,
{
    /// Distance from ray origin
    pub(crate) t: Scalar,
    pub(crate) valid_mask: Scalar::SimdBool,
    pub(crate) payload: Payload,
}

impl<Scalar, Payload> RayIntersectionCandidate<Scalar, Payload>
where
    Scalar: SimdValue + crate::scalar::Scalar,
    Payload: Sized,
{
    #[inline(always)]
    pub(crate) const fn new(t: Scalar, payload: Payload, valid_mask: Scalar::SimdBool) -> Self {
        Self {
            t,
            valid_mask,
            payload,
        }
    }

    #[inline(always)]
    pub(crate) fn replace_payload<NewPayload: Sized>(
        self,
        new_payload: NewPayload,
    ) -> RayIntersectionCandidate<Scalar, NewPayload> {
        RayIntersectionCandidate::<Scalar, NewPayload>::new(self.t, new_payload, self.valid_mask)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RayIntersection<Vector>
where
    Vector: crate::vector::Vector,
{
    pub(crate) intersection: Vector,
    intersection_direction: Vector,
    pub(crate) normal: Vector,
    pub(crate) distance: Vector::Scalar,
    incident_angle_cos: Vector::Scalar,
    pub(crate) valid_mask: <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool,
}

impl<Vector> RayIntersection<Vector>
where
    Vector: crate::vector::Vector,
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
        Vector:
            CommonVecOperations + CommonVecOperationsReflectable + CommonVecOperationsFloat + Copy,
        <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool: From<bool>,
        [(); <Vector as crate::vector::Vector>::LANES]:,
    {
        Ray::new(
            self.intersection,
            self.intersection_direction.reflected(self.normal),
        )
    }
}

impl<Vector> VectorAware<Vector> for RayIntersection<Vector> where Vector: crate::vector::Vector {}

#[cfg(test)]
mod test_ray_intersection_struct {
    use super::*;

    use ultraviolet::Vec3;

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
