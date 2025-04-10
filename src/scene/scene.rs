use crate::geometry::{Ray, SphereData};
use crate::raytracing::{Intersectable, RayIntersection, RayIntersectionCandidate};
use crate::scalar_traits::LightScalar;
use crate::vector::VectorAware;
use crate::vector_traits::{Vector3D, VectorBasic};
use num_traits::{Float, NumOps};
use palette::bool_mask::BoolMask;
use simba::simd::{SimdBool, SimdValue};
use ultraviolet::Vec3;

#[derive(Clone, Debug)]
#[repr(transparent)]
pub(crate) enum SceneObject<Vector>
where
    Vector: VectorBasic,
    Vector::Scalar: crate::scalar::Scalar + SimdValue,
{
    Sphere(SphereData<Vector>),
}

impl<Vector> VectorAware<Vector> for SceneObject<Vector>
where
    Vector: VectorBasic,
    Vector::Scalar: crate::scalar::Scalar + SimdValue,
{
}

impl<Vector> Intersectable<Vector> for SceneObject<Vector>
where
    Vector: Vector3D + crate::vector::Vector,
    Vector::Scalar:
        LightScalar<SimdBool: SimdBool + BoolMask> + NumOps<Vector::Scalar, Vector::Scalar>,
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
