mod basic;
mod composite;
pub mod ray;
pub mod render_geometry;

pub use basic::sphere::SphereData;
pub use basic::triangle::TriangleData;
pub use composite::{BoundedPlane, CompositeGeometry};
pub use ray::Ray;
pub use render_geometry::{GeometryCollection, RenderGeometry, RenderGeometryKind};
