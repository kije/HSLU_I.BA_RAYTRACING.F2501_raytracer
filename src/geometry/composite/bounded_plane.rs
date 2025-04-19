use crate::geometry::TriangleData;
use crate::geometry::composite::CompositeGeometry;
use crate::helpers::ColorType;
use crate::raytracing::Material;
use crate::vector::{
    NormalizableVector, Vector, Vector3DOperations, VectorFixedDimensions, VectorOperations,
};
use crate::vector_traits::{RenderingVector, SimdRenderingVector};
use num_traits::Zero;
use simba::scalar::SupersetOf;
use simba::simd::SimdBool;
use simba::simd::{SimdPartialOrd, SimdValue};

pub struct BoundedPlane<V: Vector> {
    normal: V,
    center: V,
    up: V,
    left: V,
    width: V::Scalar,
    height: V::Scalar,
    pub material: Material<V::Scalar>,
}

impl<V: RenderingVector> BoundedPlane<V> {
    pub fn new(
        normal: V,
        center: V,
        up: V,
        width: <V as Vector>::Scalar,
        height: <V as Vector>::Scalar,
        color: ColorType<<V as Vector>::Scalar>,
    ) -> Self {
        Self::with_material(normal, center, up, width, height, Material::diffuse(color))
    }

    pub fn with_material(
        normal: V,
        center: V,
        up: V,
        width: <V as Vector>::Scalar,
        height: <V as Vector>::Scalar,
        material: Material<<V as Vector>::Scalar>,
    ) -> Self {
        assert!(width.simd_gt(<V as Vector>::Scalar::zero()).all());
        assert!(height.simd_gt(<V as Vector>::Scalar::zero()).all());
        assert!(normal.dot(up).simd_eq(<V as Vector>::Scalar::zero()).all());

        Self {
            normal,
            center,
            up,
            left: normal.cross(up).normalized(),
            width,
            height,
            material,
        }
    }
}

impl<V> CompositeGeometry<V> for BoundedPlane<V>
where
    V: SimdRenderingVector + VectorFixedDimensions<3>,
{
    type BasicGeometry = TriangleData<V>;

    fn to_basic_geometries(self) -> Vec<Self::BasicGeometry> {
        let x = V::broadcast(self.width / <V as Vector>::Scalar::from_subset(&2.0)) * -self.left;
        let y = V::broadcast(self.height / <V as Vector>::Scalar::from_subset(&2.0)) * self.up;

        let c = self.center;

        //   p0            p1
        //   *-------------*
        //   | \           |
        //   |   \         |
        //   |     \       |
        //   |      *      |
        //   |      c \    |
        //   |          \  |
        //   |            \|
        //   *-------------*
        //   p2            p3

        let p0 = -x + y;
        let p1 = x + y;
        let p2 = -x - y;
        let p3 = x - y;

        vec![
            TriangleData::with_material(c + p1, c + p0, c + p3, self.material),
            TriangleData::with_material(c + p2, c + p3, c + p0, self.material),
        ]
    }
}
