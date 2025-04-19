mod basic;
mod composite;
pub mod ray;

pub use basic::sphere::SphereData;
pub use basic::triangle::TriangleData;
pub use composite::{BoundedPlane, CompositeGeometry};
pub use ray::Ray;
