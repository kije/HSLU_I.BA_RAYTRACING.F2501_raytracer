use crate::helpers::ColorType;
use crate::scalar::Scalar;
use simba::simd::SimdValue;

/// A trait for types that can be converted to and from SIMD representations
pub trait SimdCompatible: Sized {
    type SimdType;

    fn to_simd(&self) -> Self::SimdType;
    fn from_simd(simd: &Self::SimdType) -> Self;
}

// Implementation for ColorType
impl<S: Scalar + SimdValue> SimdCompatible for ColorType<S> {
    type SimdType = ColorType<S>;

    fn to_simd(&self) -> Self::SimdType {
        self.clone()
    }

    fn from_simd(simd: &Self::SimdType) -> Self {
        simd.clone()
    }
}

/// Safely blend two values based on a mask
pub fn blend_values<S: SimdValue>(mask: S::SimdBool, a: S, b: S) -> S {
    a.select(mask, b)
}

/// Convert a scalar value to a SIMD-compatible wide type
pub fn splat_to_wide<T, W>(scalar: T) -> W
where
    T: Copy,
    W: SimdValue<Element = T>,
{
    W::splat(scalar)
}
