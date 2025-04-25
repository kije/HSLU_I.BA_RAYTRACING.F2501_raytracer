mod bounded_plane;

use crate::geometry::basic::BasicGeometry;
use crate::vector::Vector;

use crate::simd_compat::SimdValueRealSimplified;
pub use bounded_plane::BoundedPlane;

pub trait CompositeGeometry<V: Vector<Scalar: SimdValueRealSimplified>> {
    type BasicGeometry: BasicGeometry<V>;

    fn to_basic_geometries(self) -> Vec<Self::BasicGeometry>;
}
