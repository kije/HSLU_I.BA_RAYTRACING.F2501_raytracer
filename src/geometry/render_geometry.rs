use crate::geometry::Ray;
use crate::geometry::basic::{BasicGeometry, sphere::SphereData, triangle::TriangleData};
use crate::matrix::{MatrixFixedDimensions, MatrixOperations};
use crate::raytracing::{Intersectable, Material, SurfaceInteraction};
use crate::vector::{SimdCapableVector, Vector, Vector3DAccessor, VectorAssociations};
use crate::vector_traits::SimdRenderingVector;

use crate::helpers::Splatable;
use enumcapsulate::{Encapsulate, VariantDiscriminant};
use std::collections::HashMap;
use std::ops::{Deref, Index, Neg};

/// Enum that represents all possible basic geometry types
#[derive(Debug, Clone, Encapsulate, VariantDiscriminant)]
#[enumcapsulate(discriminant(name = RenderGeometryKind))]
pub enum RenderGeometry<V>
where
    V: Vector,
{
    Sphere(SphereData<V>),
    Triangle(TriangleData<V>),
}

impl<V> Intersectable<V> for RenderGeometry<V>
where
    V: SimdRenderingVector
        + Vector3DAccessor
        + Neg<Output = V>
        + VectorAssociations<Matrix: MatrixFixedDimensions<3> + MatrixOperations>,
{
    fn intersect(&self, ray: &Ray<V>) -> Option<SurfaceInteraction<V>> {
        match self {
            RenderGeometry::Sphere(sphere) => sphere.intersect(ray),
            RenderGeometry::Triangle(triangle) => triangle.intersect(ray),
        }
    }
}

impl<V> Splatable<RenderGeometry<<V as SimdCapableVector>::SingleValueVector>> for RenderGeometry<V>
where
    V: SimdRenderingVector,
{
    fn splat(v: &RenderGeometry<<V as SimdCapableVector>::SingleValueVector>) -> Self {
        match v {
            RenderGeometry::Sphere(sphere) => RenderGeometry::new_sphere(SphereData::splat(sphere)),
            RenderGeometry::Triangle(triangle) => {
                RenderGeometry::new_triangle(TriangleData::splat(triangle))
            }
        }
    }
}

impl<V> RenderGeometry<V>
where
    V: Vector,
{
    /// Create a new RenderGeometry from a sphere
    pub fn new_sphere(sphere: SphereData<V>) -> Self {
        RenderGeometry::Sphere(sphere)
    }

    /// Create a new RenderGeometry from a triangle
    pub fn new_triangle(triangle: TriangleData<V>) -> Self {
        RenderGeometry::Triangle(triangle)
    }

    #[inline]
    pub fn get_material(&self) -> Material<V::Scalar> {
        match self {
            RenderGeometry::Sphere(sphere) => sphere.material,
            RenderGeometry::Triangle(triangle) => triangle.material,
        }
    }
}

impl<V: Vector> From<SphereData<V>> for RenderGeometry<V> {
    fn from(sphere: SphereData<V>) -> Self {
        RenderGeometry::Sphere(sphere)
    }
}

impl<V: Vector> From<TriangleData<V>> for RenderGeometry<V> {
    fn from(triangle: TriangleData<V>) -> Self {
        RenderGeometry::Triangle(triangle)
    }
}

impl<V> BasicGeometry<V> for RenderGeometry<V> where
    V: SimdRenderingVector
        + Vector3DAccessor
        + Neg<Output = V>
        + VectorAssociations<Matrix: MatrixFixedDimensions<3> + MatrixOperations>
{
}

/// A collection of geometries organized by kind for efficient access
#[derive(Debug, Clone)]
pub struct GeometryCollection<V: Vector> {
    geometries: HashMap<RenderGeometryKind, Vec<RenderGeometry<V>>>,
}

impl<V> GeometryCollection<V>
where
    V: Vector,
{
    /// Create a new empty geometry collection
    pub fn new() -> Self {
        Self {
            geometries: HashMap::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            geometries: HashMap::with_capacity(capacity),
        }
    }

    /// Add a geometry to the collection
    pub fn add(&mut self, geometry: RenderGeometry<V>) {
        let kind = geometry.variant_discriminant();
        self.geometries
            .entry(kind)
            .or_insert_with(Vec::new)
            .push(geometry);
    }

    /// Get all geometries of a specific kind
    pub fn get_by_kind(&self, kind: RenderGeometryKind) -> &[RenderGeometry<V>] {
        self.geometries
            .get(&kind)
            .map(|g| g.as_slice())
            .unwrap_or_else(|| &[])
    }

    /// Get a mutable reference to all geometries of a specific kind
    pub fn get_by_kind_mut(&mut self, kind: RenderGeometryKind) -> &mut Vec<RenderGeometry<V>> {
        self.geometries
            .get_mut(&kind)
            .unwrap_or_else(|| panic!("No geometries found for kind: {:?}", kind))
    }

    /// Get all geometries
    pub fn get_all(&self) -> impl Iterator<Item = &RenderGeometry<V>> {
        self.geometries.values().flat_map(|v| v.iter())
    }
}

impl<V: Vector> Deref for GeometryCollection<V> {
    type Target = HashMap<RenderGeometryKind, Vec<RenderGeometry<V>>>;

    fn deref(&self) -> &Self::Target {
        &self.geometries
    }
}

impl<V: Vector> Index<RenderGeometryKind> for GeometryCollection<V> {
    type Output = [RenderGeometry<V>];

    fn index(&self, index: RenderGeometryKind) -> &Self::Output {
        self.get_by_kind(index)
    }
}

// Example code showcasing the use of GeometryCollection
#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::ColorType;
    use ultraviolet::Vec3;

    #[test]
    fn test_geometry_collection() {
        // Create some basic geometries
        let red = ColorType::<f32>::new(1.0, 0.0, 0.0);
        let green = ColorType::<f32>::new(0.0, 1.0, 0.0);

        // Create a sphere
        let sphere = SphereData::new(Vec3::new(0.0, 0.0, 0.0), 1.0, red);

        // Create a triangle
        let triangle = TriangleData::new(
            Vec3::new(-1.0, -1.0, 2.0),
            Vec3::new(1.0, -1.0, 2.0),
            Vec3::new(0.0, 1.0, 2.0),
            green,
        );

        // Create a collection to store our geometries
        let mut collection = GeometryCollection::<Vec3>::new();

        // Add geometries to the collection
        collection.add(RenderGeometry::new_sphere(sphere));
        collection.add(RenderGeometry::new_triangle(triangle));

        // Check that we have the correct number of each geometry type
        assert_eq!(collection.get_by_kind(RenderGeometryKind::Sphere).len(), 1);
        assert_eq!(
            collection.get_by_kind(RenderGeometryKind::Triangle).len(),
            1
        );
    }
}
