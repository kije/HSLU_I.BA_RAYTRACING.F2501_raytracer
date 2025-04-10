use crate::scalar_traits::LightScalar;
use palette::bool_mask::{HasBoolMask, LazySelect};
use simba::simd::SimdValue;
use std::fmt::Debug;

/// A trait for vectors with mask operations that support lazy selection
pub trait VectorWithLazySelectMask: crate::vector::Vector
where
    Self::Scalar: LightScalar,
    <<Self as crate::vector::Vector>::Scalar as HasBoolMask>::Mask: LazySelect<Self::Scalar>,
    <<Self as crate::vector::Vector>::Scalar as SimdValue>::SimdBool:
        Debug + SimdValue<Element = bool>,
{
}

// Blanket implementation
impl<V> VectorWithLazySelectMask for V
where
    V: crate::vector::Vector,
    V::Scalar: LightScalar,
    <<V as crate::vector::Vector>::Scalar as HasBoolMask>::Mask: LazySelect<V::Scalar>,
    <<V as crate::vector::Vector>::Scalar as SimdValue>::SimdBool:
        Debug + SimdValue<Element = bool>,
{
}
