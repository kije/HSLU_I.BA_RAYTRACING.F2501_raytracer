use crate::simd_compat::SimdValueBoolExt;
use crate::vector::{NormalizableVector, VectorAware, VectorOperations};
use num_traits::Float;
use simba::simd::SimdValue;
use std::fmt::Debug;

// todo: idea impl mul<scalar> / div<scalar> trait to shrink/stretch ray by a factor (simply calling ray.at(factor))? Or do we need a line primitive for that? -> needs propper handling of validity mask if we would also support wide/Simd data types?
#[derive(Clone, Debug, Copy)]
pub(crate) struct Ray<Vector>
where
    Vector: crate::vector::Vector,
{
    pub(crate) origin: Vector,
    pub(crate) direction: Vector, // todo introduce a "Direction" newtype that garatuees already a normalized vector
    pub(crate) valid_mask: <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool,
}

impl<Vector> Ray<Vector>
where
    Vector: crate::vector::Vector,
{
    #[inline]
    pub(crate) fn new(origin: Vector, direction: Vector) -> Self
    where
        Vector: NormalizableVector,
        [(); <Vector as crate::vector::Vector>::LANES]:,
    {
        Self::new_with_mask(origin, direction, Vector::Scalar::create_mask(true))
    }

    #[inline]
    pub(crate) fn new_with_mask(
        origin: Vector,
        direction: Vector,
        valid_mask: <<Vector as crate::vector::Vector>::Scalar as SimdValue>::SimdBool,
    ) -> Self
    where
        Vector: NormalizableVector,
    {
        Self {
            origin,
            direction: direction.normalized(),
            valid_mask,
        }
    }

    #[inline(always)]
    pub(crate) fn at(&self, t: Vector::Scalar) -> Vector
    where
        Vector: VectorOperations + Copy,
    {
        self.direction.mul_add(Vector::broadcast(t), self.origin)
    }

    // todo method extend_to(inetrsectable: Intersectable) -> Vector?
}

// fixme: shouldn't that (invalid values) be a property of a Vector?
impl<Vector> Ray<Vector>
where
    Vector: crate::vector::Vector,
    <Vector::Scalar as SimdValue>::Element: Float,
{
    #[inline(always)]
    pub(crate) fn invalid_value() -> <Vector::Scalar as SimdValue>::Element {
        <Vector::Scalar as SimdValue>::Element::infinity()
    }

    #[inline(always)]
    pub(crate) fn invalid_value_splatted() -> Vector::Scalar {
        Vector::Scalar::splat(Self::invalid_value())
    }

    #[inline(always)]
    pub(crate) fn invalid_vector() -> Vector
    where
        Vector: VectorOperations,
    {
        Vector::broadcast(Self::invalid_value_splatted())
    }
}

impl<Vector> VectorAware<Vector> for Ray<Vector> where Vector: crate::vector::Vector {}

#[cfg(test)]
mod test_ray {
    use super::*;
    use ultraviolet::Vec3;

    #[test]
    fn test_ray_at() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(ray.at(0.0), Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(ray.at(1.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(ray.at(2.23), Vec3::new(2.23, 0.0, 0.0));
    }
}
