use crate::float_ext::AbsDiffEq;
use crate::geometry::basic::BasicGeometry;
use crate::geometry::{HasRenderObjectId, Ray, RenderObjectId, SphereData};
use crate::helpers::{ColorType, Splatable};
use crate::matrix::{MatrixFixedDimensions, MatrixOperations};
use crate::raytracing::Intersectable;
use crate::raytracing::Material;
use crate::raytracing::SurfaceInteraction;
use crate::simd_compat::SimdValueRealSimplified;
use crate::vector::{SimdCapableVector, Vector, Vector3DAccessor, VectorAssociations, VectorAware};
use crate::vector_traits::{RenderingVector, SimdRenderingVector};
use num_traits::One;
use num_traits::Zero;
use simba::scalar::SupersetOf;
use simba::simd::SimdRealField;
use simba::simd::{SimdBool, SimdPartialOrd};
use simba::simd::{SimdComplexField, SimdValue};
use std::ops::Neg;

/// Represents a triangle in 3D space
#[derive(Debug, Copy, Clone)]
pub struct TriangleData<V: Vector<Scalar: SimdValueRealSimplified>> {
    /// The three vertices of the triangle
    ///        v3
    ///         *
    ///         |\
    ///         | \
    ///         |  \
    ///         |   \
    ///      e2 |    \
    ///         |     \
    ///         |      \
    ///         *-------*
    ///        v1   e1  v2
    pub vertex1: V,
    pub vertex2: V,
    pub vertex3: V,

    edge1: V,
    edge2: V,

    /// Pre-computed face normal
    pub normal: V,

    /// Material for the triangle's surface
    pub material: Material<V::Scalar>,
    object_id: RenderObjectId<V::Scalar>,
}

impl<V> VectorAware<V> for TriangleData<V> where V: Vector<Scalar: SimdValueRealSimplified> {}

impl<V: RenderingVector> TriangleData<V> {
    pub fn new(vertex1: V, vertex2: V, vertex3: V, color: ColorType<V::Scalar>) -> Self {
        Self::with_material(vertex1, vertex2, vertex3, Material::diffuse(color))
    }

    /// Create a triangle with a custom material
    pub fn with_material(
        vertex1: V,
        vertex2: V,
        vertex3: V,
        material: Material<V::Scalar>, // fixme: allow material to either be owned or referenced (e.g. keep a list of materials in the scene and reference it here) -> possible issues with multi-threading & lifetimes? Might no bring too much anyways due to splatting and blending
    ) -> Self {
        // Calculate the face normal
        let edge1 = vertex2 - vertex1;
        let edge2 = vertex3 - vertex1;
        let normal = edge1.cross(edge2).normalized();

        Self {
            vertex1,
            vertex2,
            vertex3,
            edge1,
            edge2,
            normal,
            material,
            object_id: RenderObjectId::new(),
        }
    }

    pub fn with_material_and_normal(
        vertex1: V,
        vertex2: V,
        vertex3: V,
        normal: V,
        material: Material<V::Scalar>,
    ) -> Self {
        // Calculate the face normal
        let edge1 = vertex2 - vertex1;
        let edge2 = vertex3 - vertex1;

        Self {
            vertex1,
            vertex2,
            vertex3,
            edge1,
            edge2,
            normal,
            material,
            object_id: RenderObjectId::new(),
        }
    }

    pub fn get_center(&self) -> V {
        (self.vertex1 + self.vertex2 + self.vertex3)
            * V::broadcast(V::Scalar::from_subset(&3.0).simd_recip())
    }

    pub(in super::super) fn with_object_id(mut self, object_id: RenderObjectId<V::Scalar>) -> Self {
        self.object_id = object_id;
        self
    }
}

impl<V> HasRenderObjectId<V::Scalar> for TriangleData<V>
where
    V: Vector<Scalar: SimdValueRealSimplified>,
{
    fn get_render_object_id(&self) -> RenderObjectId<V::Scalar> {
        self.object_id
    }
}

impl<V> Splatable<TriangleData<<V as SimdCapableVector>::SingleValueVector>> for TriangleData<V>
where
    V: SimdRenderingVector,
{
    fn splat(v: &TriangleData<<V as SimdCapableVector>::SingleValueVector>) -> Self {
        Self {
            vertex1: V::splat(v.vertex1),
            vertex2: V::splat(v.vertex2),
            vertex3: V::splat(v.vertex3),
            edge1: V::splat(v.edge1),
            edge2: V::splat(v.edge2),
            normal: V::splat(v.normal),
            material: Splatable::splat(&v.material),
            object_id: RenderObjectId::from(V::Scalar::from_subset(&v.object_id.id())),
        }
    }
}

impl<
    V: SimdRenderingVector
        + Vector3DAccessor
        + Neg<Output = V>
        + VectorAssociations<Matrix: MatrixFixedDimensions<3> + MatrixOperations>,
> Intersectable<V> for TriangleData<V>
{
    fn intersect(&self, ray: &Ray<V>) -> Option<SurfaceInteraction<V>> {
        let zero = V::Scalar::zero();
        let one = V::Scalar::one();
        let epsilon = V::Scalar::simd_default_epsilon();

        let is_visible_face = if cfg!(feature = "backface_culling") {
            let is_visible_face = ray
                .direction
                .dot(self.normal)
                .simd_lt(V::Scalar::from_subset(&(0.75)))
                | self.material.transmission.mask();

            if is_visible_face.none() {
                return None;
            }

            is_visible_face
        } else {
            true.into()
        };

        let edge1 = self.edge1;
        let edge2 = self.edge2;
        let b = self.vertex1 - ray.origin;

        let mat = V::Matrix::from_columns([ray.direction, -edge1, -edge2]);

        // fixme can we avoid calculating the determinant twice (once via inversed() and once in determinant())?
        let mat_inverse = mat.inversed();

        let mat_det = mat.determinant();

        // [t, u, v]^T
        let tuv = mat_inverse * b;

        let t = tuv.x();
        let u = tuv.y();
        let v = tuv.z();

        // Check if t is positive (intersection in front of ray)
        let t_invalid = t.simd_le(epsilon);

        // Check if u, v are in valid range for barycentric coordinates
        let uv_invalid = u.simd_lt(zero) | v.simd_lt(zero) | (u + v).simd_ge(one);
        let valid_mask =
            is_visible_face & !(t_invalid | uv_invalid) & !mat_det.abs_diff_eq_default(&zero);

        if valid_mask.none() {
            return None;
        }

        // Compute intersection point
        let intersection_point = ray.at(t);

        // Create surface interaction
        Some(SurfaceInteraction::new(
            intersection_point,
            self.normal,
            t,
            self.material.clone(),
            valid_mask,
            self.object_id,
        ))
    }
}

impl<V> BasicGeometry<V> for TriangleData<V> where
    V: SimdRenderingVector
        + Vector3DAccessor
        + Neg<Output = V>
        + VectorAssociations<Matrix: MatrixFixedDimensions<3> + MatrixOperations>
{
}
