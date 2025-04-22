use crate::geometry::TriangleData;
use crate::geometry::composite::CompositeGeometry;
use crate::helpers::ColorType;
use crate::raytracing::Material;
use crate::vector::{Vector, VectorFixedDimensions};
use crate::vector_traits::{RenderingVector, SimdRenderingVector};
use num_traits::Zero;
use simba::scalar::SupersetOf;
use simba::simd::SimdBool;
use simba::simd::SimdPartialOrd;

pub struct BoundedPlane<V: Vector> {
    center: V,
    up: V,
    left: V,
    normal: V,
    width: V::Scalar,
    height: V::Scalar,
    pub material: Material<V::Scalar>,
}

impl<V: RenderingVector> BoundedPlane<V> {
    #[track_caller]
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

    #[track_caller]
    pub fn with_material(
        normal: V,
        center: V,
        up: V,
        width: <V as Vector>::Scalar,
        height: <V as Vector>::Scalar,
        material: Material<<V as Vector>::Scalar>,
    ) -> Self {
        assert!(
            width.simd_gt(<V as Vector>::Scalar::zero()).all(),
            "width must be positive"
        );
        assert!(
            height.simd_gt(<V as Vector>::Scalar::zero()).all(),
            "height must be positive"
        );
        assert!(
            normal.dot(up).simd_eq(<V as Vector>::Scalar::zero()).all(),
            "up must be orthogonal to normal"
        );

        Self {
            // normal,
            center,
            up,
            left: normal.cross(up).normalized(),
            normal,
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
            TriangleData::with_material_and_normal(
                c + p1,
                c + p0,
                c + p3,
                self.normal,
                self.material,
            ),
            TriangleData::with_material_and_normal(
                c + p2,
                c + p3,
                c + p0,
                self.normal,
                self.material,
            ),
        ]
    }
}
