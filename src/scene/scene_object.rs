use crate::geometry::{PointData, Ray, SphereData};
use crate::raytracing::{Intersectable, RayIntersection, RayIntersectionCandidate};
use crate::vector::VectorAware;
use crate::vector_traits::{BaseVector, RenderingVector};

#[derive(Clone, Debug)]
pub(crate) enum SceneObject<Vector>
where
    Vector: BaseVector,
{
    Sphere(SphereData<Vector>),
    Point(PointData<Vector>),
}

impl<Vector> VectorAware<Vector> for SceneObject<Vector> where Vector: RenderingVector {}

impl<Vector> Intersectable<Vector> for SceneObject<Vector>
where
    Vector: RenderingVector,
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
            SceneObject::Point(point_data) => point_data.check_intersection(ray, payload),
        }
    }

    fn intersect<'a, P>(
        &'a self,
        ray: &'_ Self::RayType,
        candidate: &'_ RayIntersectionCandidate<Vector::Scalar, &'a P>,
    ) -> Self::ReturnTypeWrapper<RayIntersection<Vector>>
    where
        P: Intersectable<Vector>,
    {
        match self {
            SceneObject::Sphere(sphere_data) => {
                sphere_data.intersect(ray, &candidate.replace_payload(sphere_data))
            }
            SceneObject::Point(point_data) => {
                point_data.intersect(ray, &candidate.replace_payload(point_data))
            }
        }
    }
}
