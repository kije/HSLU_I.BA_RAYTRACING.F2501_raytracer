use crate::float_ext::AbsDiffEq;
use crate::geometry::composite::CompositeGeometry;
use crate::geometry::{HasRenderObjectId, RenderObjectId, TriangleData};
use crate::helpers::ColorType;
use crate::raytracing::Material;
use crate::simd_compat::SimdValueRealSimplified;
use crate::vector::{Vector, VectorFixedDimensions};
use crate::vector_traits::{RenderingVector, SimdRenderingVector};
use num_traits::Zero;
use simba::scalar::SupersetOf;
use simba::simd::SimdBool;
use simba::simd::SimdPartialOrd;

pub struct BoundedPlane<V: Vector<Scalar: SimdValueRealSimplified>> {
    center: V,
    up: V,
    left: V,
    normal: V,
    width: V::Scalar,
    height: V::Scalar,
    depth: V::Scalar,
    pub material: Material<V::Scalar>,
    object_id: RenderObjectId<V::Scalar>,
}

impl<V: RenderingVector> BoundedPlane<V> {
    #[track_caller]
    pub fn new(
        normal: V,
        center: V,
        up: V,
        width: <V as Vector>::Scalar,
        height: <V as Vector>::Scalar,
        depth: <V as Vector>::Scalar,
        color: ColorType<<V as Vector>::Scalar>,
    ) -> Self {
        Self::with_material(
            normal,
            center,
            up,
            width,
            height,
            depth,
            Material::diffuse(color),
        )
    }

    #[track_caller]
    pub fn with_material(
        normal: V,
        center: V,
        up: V,
        width: <V as Vector>::Scalar,
        height: <V as Vector>::Scalar,
        depth: <V as Vector>::Scalar,
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
            normal
                .dot(up)
                .abs_diff_eq_default(&<V as Vector>::Scalar::zero())
                .all(),
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
            depth,
            material,
            object_id: RenderObjectId::new(),
        }
    }

    /// Triangulates the plane into two triangles.
    ///
    ///   p0            p1
    ///   *-------------*
    ///   | \           |
    ///   |   \         |
    ///   |     \       |
    ///   |      *      |
    ///   |      c \    |
    ///   |          \  |
    ///   |            \|
    ///   *-------------*
    ///   p2            p3
    ///
    /// first vertex of each returned triangle are p1 and p2 respectively.
    pub fn triangulate(&self) -> ((V, V, V), (V, V, V)) {
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

        ((c + p1, c + p0, c + p3), (c + p2, c + p3, c + p0))
    }
}

impl<V> HasRenderObjectId<V::Scalar> for BoundedPlane<V>
where
    V: Vector<Scalar: SimdValueRealSimplified>,
{
    fn get_render_object_id(&self) -> RenderObjectId<V::Scalar> {
        self.object_id
    }
}

impl<V> CompositeGeometry<V> for BoundedPlane<V>
where
    V: SimdRenderingVector + VectorFixedDimensions<3>,
{
    type BasicGeometry = TriangleData<V>;

    fn to_basic_geometries(self) -> Vec<Self::BasicGeometry> {
        let (triangle1, triangle2) = self.triangulate();

        // object must have depth, so we need two planes for front & back, and 4 planes for the sides
        // make sure normals always point outwards, otherwise we'll later get wired behaviour during transmission, as this indicates from which medium we are traveling from/to
        let mut triangles = Vec::with_capacity(8);
        let depth_offset_direction = self.normal;

        let half = V::Scalar::from_subset(&0.5);

        // front & back plates
        for (depth_offset, normal) in [
            (-(self.depth * half), -self.normal),
            (self.depth * half, self.normal),
        ] {
            let offset = depth_offset_direction * V::broadcast(depth_offset);
            triangles.push(
                TriangleData::with_material_and_normal(
                    triangle1.0 + offset,
                    triangle1.1 + offset,
                    triangle1.2 + offset,
                    normal,
                    self.material,
                )
                .with_object_id(self.object_id),
            );

            triangles.push(
                TriangleData::with_material_and_normal(
                    triangle2.0 + offset,
                    triangle2.1 + offset,
                    triangle2.2 + offset,
                    normal,
                    self.material,
                )
                .with_object_id(self.object_id),
            );
        }

        // side plates
        for (dir, dir_offset, width, normal) in [
            (self.up, self.height, self.width, self.up),
            (self.left, self.width, self.height, self.left),
            (-self.up, self.height, self.width, -self.up),
            (-self.left, self.width, self.height, -self.left),
        ] {
            let plate_center = dir.mul_add(V::broadcast(dir_offset * half), self.center);

            let (t1, t2) = Self::with_material(
                normal,
                plate_center,
                depth_offset_direction,
                width,
                self.depth,
                V::Scalar::zero(),
                self.material,
            )
            .triangulate();

            triangles.push(
                TriangleData::with_material_and_normal(t1.0, t1.1, t1.2, normal, self.material)
                    .with_object_id(self.object_id),
            );

            triangles.push(
                TriangleData::with_material_and_normal(t2.0, t2.1, t2.2, normal, self.material)
                    .with_object_id(self.object_id),
            );
        }

        triangles
    }
}
