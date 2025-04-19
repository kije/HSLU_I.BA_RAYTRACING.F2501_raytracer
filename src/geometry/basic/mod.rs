use crate::raytracing::Intersectable;
use crate::vector::Vector;

//pub mod plane;
pub mod sphere;
pub mod triangle;

pub trait BasicGeometry<V: Vector>: Intersectable<V> {}
