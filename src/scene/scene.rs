use crate::geometry::Ray;
use crate::geometry::{SphereData, TriangleData};
use crate::raytracing::Intersectable;
use crate::vector::Vector;
use crate::vector_traits::{RenderingVector, SimdRenderingVector};
use simba::simd::SimdValue;

pub struct Scene<V: RenderingVector> {
    pub spheres: Vec<SphereData<V>>,
    pub triangles: Vec<TriangleData<V>>,
    // Add other geometry types as needed
}

impl<V: RenderingVector> Scene<V> {
    pub fn new() -> Self {
        Self {
            spheres: Vec::new(),
            triangles: Vec::new(),
        }
    }

    pub fn add_sphere(&mut self, sphere: SphereData<V>) {
        self.spheres.push(sphere);
    }

    pub fn add_triangle(&mut self, triangle: TriangleData<V>) {
        self.triangles.push(triangle);
    }
}

// For SIMD vector types, we need specialized scene handling
pub struct SceneSimd<V: SimdRenderingVector> {
    pub spheres: Vec<SphereData<V::SingleValueVector>>,
    pub triangles: Vec<TriangleData<V::SingleValueVector>>,
    // Add other geometry types as needed
}

impl<V: SimdRenderingVector> SceneSimd<V>
where
    V::SingleValueVector: RenderingVector,
{
    /// Find the nearest intersection across all geometry types
    pub fn find_nearest_intersection(&self, ray: &Ray<V>) -> Option<RayIntersectionResult<V>> {
        // First check spheres
        let sphere_intersection = self.find_nearest_sphere_intersection(ray);

        // Then check triangles
        let triangle_intersection = self.find_nearest_triangle_intersection(ray);

        // Combine results, taking the nearest one
        match (sphere_intersection, triangle_intersection) {
            (None, None) => None,
            (Some(s), None) => Some(s),
            (None, Some(t)) => Some(t),
            (Some(s), Some(t)) => {
                // Compare distances to determine which is closer
                let s_is_closer = s.distance <= t.distance;

                if s_is_closer.all() {
                    Some(s)
                } else if s_is_closer.none() {
                    Some(t)
                } else {
                    // Mix results based on which is closer in each SIMD lane
                    Some(RayIntersectionResult::blend(s_is_closer, &s, &t))
                }
            }
        }
    }

    fn find_nearest_sphere_intersection(&self, ray: &Ray<V>) -> Option<RayIntersectionResult<V>> {
        // Process all spheres
        self.spheres.iter().fold(None, |acc, sphere| {
            let sphere_simd = SphereData::<V>::splat(sphere);
            let candidate = sphere_simd.check_intersection(ray, sphere_simd);

            merge_intersection_results(acc, candidate.into())
        })
    }

    fn find_nearest_triangle_intersection(&self, ray: &Ray<V>) -> Option<RayIntersectionResult<V>> {
        // Process all triangles
        self.triangles.iter().fold(None, |acc, triangle| {
            let triangle_simd = TriangleData::<V>::splat(triangle);
            let candidate = triangle_simd.check_intersection(ray, triangle_simd);

            merge_intersection_results(acc, candidate.into())
        })
    }
}

/// Represents the result of a ray intersection with any geometry type
pub enum RayIntersectionResult<V: SimdRenderingVector> {
    Sphere(RayIntersectionCandidate<V::Scalar, SphereData<V>>),
    Triangle(RayIntersectionCandidate<V::Scalar, TriangleData<V>>),
    // Add other geometry types as needed
}

impl<V: SimdRenderingVector> RayIntersectionResult<V> {
    pub fn distance(&self) -> V::Scalar {
        match self {
            RayIntersectionResult::Sphere(candidate) => candidate.t,
            RayIntersectionResult::Triangle(candidate) => candidate.t,
        }
    }

    pub fn valid_mask(&self) -> <<V as Vector>::Scalar as SimdValue>::SimdBool {
        match self {
            RayIntersectionResult::Sphere(candidate) => candidate.valid_mask,
            RayIntersectionResult::Triangle(candidate) => candidate.valid_mask,
        }
    }

    /// Blend two intersection results based on a boolean mask
    pub fn blend(mask: <<V as Vector>::Scalar as SimdValue>::SimdBool, a: &Self, b: &Self) -> Self {
        // This is a simplified implementation - a real one would need to properly
        // blend the geometry types based on the mask
        if mask.all() { a.clone() } else { b.clone() }
    }
}

impl<V: SimdRenderingVector> Clone for RayIntersectionResult<V> {
    fn clone(&self) -> Self {
        match self {
            RayIntersectionResult::Sphere(candidate) => {
                RayIntersectionResult::Sphere(candidate.clone())
            }
            RayIntersectionResult::Triangle(candidate) => {
                RayIntersectionResult::Triangle(candidate.clone())
            }
        }
    }
}

impl<V: SimdRenderingVector> From<RayIntersectionCandidate<V::Scalar, SphereData<V>>>
    for RayIntersectionResult<V>
{
    fn from(candidate: RayIntersectionCandidate<V::Scalar, SphereData<V>>) -> Self {
        RayIntersectionResult::Sphere(candidate)
    }
}

impl<V: SimdRenderingVector> From<RayIntersectionCandidate<V::Scalar, TriangleData<V>>>
    for RayIntersectionResult<V>
{
    fn from(candidate: RayIntersectionCandidate<V::Scalar, TriangleData<V>>) -> Self {
        RayIntersectionResult::Triangle(candidate)
    }
}

/// Helper function to merge intersection results
fn merge_intersection_results<V: SimdRenderingVector>(
    a: Option<RayIntersectionResult<V>>,
    b: RayIntersectionResult<V>,
) -> Option<RayIntersectionResult<V>> {
    match a {
        None => Some(b),
        Some(a) => {
            let a_mask = a.valid_mask();
            let b_mask = b.valid_mask();

            // If b has no valid intersections, keep a
            if b_mask.none() {
                return Some(a);
            }

            // If a has no valid intersections, use b
            if a_mask.none() {
                return Some(b);
            }

            // Compare distances
            let b_is_closer = a.distance() >= b.distance();

            // Simple cases
            if b_is_closer.none() {
                Some(a)
            } else if b_is_closer.all() {
                Some(b)
            } else {
                // Blend results
                Some(RayIntersectionResult::blend(b_is_closer, &b, &a))
            }
        }
    }
}
