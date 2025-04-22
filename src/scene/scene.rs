use crate::geometry::GeometryCollection;
use crate::geometry::{SphereData, TriangleData};
use crate::vector::Vector;

pub struct Scene<V: Vector> {
    pub scene_objects: GeometryCollection<V>,
}

impl<V: Vector> Scene<V> {
    pub fn new() -> Self {
        Self {
            scene_objects: GeometryCollection::new(),
        }
    }

    pub fn with_capacities(scene_objects: usize) -> Self {
        Self {
            scene_objects: GeometryCollection::with_capacity(scene_objects),
        }
    }

    pub fn add_sphere(&mut self, sphere: SphereData<V>) {
        self.scene_objects.add(sphere.into());
    }

    pub fn add_triangle(&mut self, triangle: TriangleData<V>) {
        self.scene_objects.add(triangle.into());
    }
}
