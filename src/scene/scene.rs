use crate::geometry::{Ray, SphereData};
use crate::helpers::ColorType;
use crate::raytracing::{Intersectable, RayIntersection, RayIntersectionCandidate};
use crate::vector::{CommonVecOperations, CommonVecOperationsFloat, Vector, VectorAware};
use minifb::Key::T;
use num_traits::{Float, NumOps, Zero};
use simba::simd::{SimdPartialOrd, SimdRealField, SimdValue};
use std::ops::{Add, Mul, Sub};
use ultraviolet::Vec3;

#[derive(Clone, Debug)]
#[repr(transparent)]
pub(crate) enum SceneObject<Vector>
where
    Vector: crate::vector::Vector,
    Vector::Scalar: crate::scalar::Scalar + SimdValue,
{
    Sphere(SphereData<Vector>),
}

impl<Vector> VectorAware<Vector> for SceneObject<Vector>
where
    Vector: crate::vector::Vector,
    Vector::Scalar: crate::scalar::Scalar + SimdValue,
{
}

impl<Vector> Intersectable<Vector> for SceneObject<Vector>
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
        match self {
            SceneObject::Sphere(sphere_data) => sphere_data.check_intersection(ray, payload),
        }
    }

    fn intersect<'a>(
        &'a self,
        ray: &'_ Self::RayType,
        candidate: &'_ RayIntersectionCandidate<Vector::Scalar, &'a Self>,
    ) -> Self::ReturnTypeWrapper<RayIntersection<Vector>> {
        match self {
            SceneObject::Sphere(sphere_data) => {
                sphere_data.intersect(ray, &candidate.replace_payload(sphere_data))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Scene {
    objects: Vec<SceneObject<Vec3>>,
}
