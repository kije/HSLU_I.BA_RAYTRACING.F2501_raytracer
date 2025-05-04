mod basic;
mod composite;
pub mod ray;
pub mod render_geometry;

use crate::helpers::Splatable;
use crate::random::pseudo_rng;
use crate::raytracing::Material;
use crate::vector::{SimdCapableVector, Vector};
use crate::vector_traits::SimdRenderingVector;
pub use basic::sphere::SphereData;
pub use basic::triangle::TriangleData;
pub use composite::{BoundedPlane, CompositeGeometry};
use num_traits::{One, Zero};
use rand::Rng;
pub use ray::Ray;
pub use render_geometry::{GeometryCollection, RenderGeometry, RenderGeometryKind};
use simba::scalar::SupersetOf;
use simba::simd::SimdValue;
use std::fmt::Debug;
use std::ops::Add;

pub trait HasRenderObjectId<
    S: Sized + Copy + Clone + PartialEq + Debug + SupersetOf<f32> + SimdValue,
>
{
    fn get_render_object_id(&self) -> RenderObjectId<S>;
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(transparent)]
pub struct RenderObjectId<S: Sized + Copy + Clone + PartialEq + Debug + SupersetOf<f32> + SimdValue>(
    S,
);

impl<S: Sized + Copy + Clone + PartialEq + Debug + SupersetOf<f32> + SimdValue> RenderObjectId<S> {
    pub fn new() -> Self {
        let mut rng = pseudo_rng();

        Self(S::from_subset(&rng.r#gen()))
    }

    pub fn id(&self) -> S {
        self.0
    }

    pub fn blend(mask: <S as SimdValue>::SimdBool, a: &Self, b: &Self) -> Self {
        Self(a.0.select(mask, b.0))
    }
}

impl<S: Sized + Copy + Clone + PartialEq + Debug + SupersetOf<f32> + SimdValue> From<S>
    for RenderObjectId<S>
{
    fn from(value: S) -> Self {
        Self(value)
    }
}

impl<S: Sized + Copy + Clone + PartialEq + Debug + Default + SupersetOf<f32> + SimdValue> Default
    for RenderObjectId<S>
{
    fn default() -> Self {
        Self(S::default())
    }
}
