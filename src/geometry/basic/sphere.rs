use crate::geometry::basic::BasicGeometry;
use crate::geometry::{HasRenderObjectId, Ray, RenderObjectId};
use crate::helpers::{ColorType, Splatable};
use crate::raytracing::Intersectable;
use crate::raytracing::Material;
use crate::raytracing::SurfaceInteraction;
use crate::simd_compat::SimdValueRealSimplified;
use crate::vector::{NormalizableVector, SimdCapableVector, Vector, VectorAware};
use crate::vector_traits::{BaseVector, RenderingVector, SimdRenderingVector};
use num_traits::One;
use num_traits::Zero;
use simba::scalar::SupersetOf;
use simba::simd::SimdComplexField;
use simba::simd::{SimdBool, SimdPartialOrd, SimdValue};
use std::hint::unlikely;
use std::ops::Neg;

/// Represents a sphere in 3D space
#[derive(Debug, Copy, Clone)]
pub struct SphereData<V: Vector<Scalar: SimdValueRealSimplified>> {
    /// Center of the sphere
    pub center: V,

    r_sq: V::Scalar,
    r_inv: V::Scalar,

    /// Material for the sphere's surface
    pub material: Material<V::Scalar>,
    object_id: RenderObjectId<V::Scalar>,
}

impl<V> VectorAware<V> for SphereData<V> where V: Vector<Scalar: SimdValueRealSimplified> {}

impl<V: Vector<Scalar: SimdValueRealSimplified>> SphereData<V> {
    pub fn new(center: V, radius: V::Scalar, color: ColorType<V::Scalar>) -> Self {
        Self::with_material(center, radius, Material::diffuse(color))
    }

    /// Create a sphere with a custom material
    pub fn with_material(center: V, radius: V::Scalar, material: Material<V::Scalar>) -> Self {
        Self {
            center,
            r_sq: radius * radius,
            r_inv: V::Scalar::one() / radius,
            material,
            object_id: RenderObjectId::new(),
        }
    }
}

impl<V> HasRenderObjectId<V::Scalar> for SphereData<V>
where
    V: Vector<Scalar: SimdValueRealSimplified>,
{
    fn get_render_object_id(&self) -> RenderObjectId<V::Scalar> {
        self.object_id
    }
}

impl<V> Splatable<SphereData<<V as SimdCapableVector>::SingleValueVector>> for SphereData<V>
where
    V: SimdRenderingVector,
{
    fn splat(v: &SphereData<<V as SimdCapableVector>::SingleValueVector>) -> Self {
        Self {
            center: V::splat(v.center),
            r_inv: V::Scalar::from_subset(&v.r_inv),
            r_sq: V::Scalar::from_subset(&v.r_sq),
            material: Splatable::splat(&v.material),
            object_id: RenderObjectId::from(V::Scalar::from_subset(&v.object_id.id())),
        }
    }
}

impl<V: RenderingVector + NormalizableVector + SimdCapableVector> Intersectable<V>
    for SphereData<V>
{
    fn intersect(&self, ray: &Ray<V>) -> Option<SurfaceInteraction<V>> {
        let u = ray.direction;
        let v = ray.origin - self.center;

        const A: f32 = 2.0; // 2 * (u dot u) => 2 * direction_mag_squared => 2 * 1 => 2
        const A_INV: f32 = const { A.recip() }; // crate::helpers::fast_inverse(A);
        let a_splat_neg = V::Scalar::from_subset(&-A);
        let two_splat = V::Scalar::from_subset(&2.0);
        let zero = V::Scalar::zero();
        let invalid_value = Ray::<V>::invalid_value_splatted();

        let b: V::Scalar = two_splat * u.dot(v);
        let c: V::Scalar = v.dot(v) - self.r_sq;

        let discriminant: V::Scalar = b.simd_mul_add(b, (two_splat * a_splat_neg) * c);

        let discriminant_pos = discriminant.simd_ge(zero);

        // shortcircuit
        if unlikely(discriminant_pos.none()) {
            return None;
        }

        let a_inv_splat = V::Scalar::from_subset(&A_INV);

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

        // Compute the intersection point
        let intersection_point = ray.at(final_t);

        // Compute the surface normal at the intersection point
        let normal = (intersection_point - self.center).normalized();

        // Create the surface interaction
        Some(SurfaceInteraction::new(
            intersection_point,
            normal,
            final_t,
            self.material.clone(),
            final_t_valid,
            self.object_id,
        ))
    }
}

impl<V> BasicGeometry<V> for SphereData<V> where
    V: RenderingVector + NormalizableVector + SimdCapableVector
{
}
