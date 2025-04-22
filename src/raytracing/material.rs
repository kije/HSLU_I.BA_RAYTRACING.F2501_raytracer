use crate::color::ColorSimdExt;
use crate::helpers::{ColorType, Splatable};
use crate::simd_compat::SimdValueSimplified;
use simba::simd::SimdValue;

/// Surface material properties
#[derive(Debug, Copy, Clone)]
pub struct Material<S: SimdValueSimplified> {
    /// Surface color
    pub color: ColorType<S>,

    /// Reflectivity (0.0 = diffuse, 1.0 = mirror)
    pub reflectivity: S,

    /// Shininess of the surface (affects specular highlight) (0.0 = rough surface,  1.0 = very shiny)
    pub shininess: S,
    // Additional parameters could be added like:
    // - metalness
    // - transparency
    // - refraction index
    // - emission
}

impl<S: SimdValueSimplified> Material<S> {
    /// Create a new material with basic properties
    pub fn new(color: ColorType<S>, reflectivity: S, shininess: S) -> Self {
        Self {
            color,
            reflectivity,
            shininess,
        }
    }

    /// Create a simple diffuse material
    pub fn diffuse(color: ColorType<S>) -> Self {
        Self::new(color, S::zero(), S::zero())
    }

    /// Blend two materials based on a mask
    pub fn blend(mask: <S as SimdValue>::SimdBool, a: &Self, b: &Self) -> Self {
        Self {
            color: ColorSimdExt::blend(mask.clone(), &a.color, &b.color),
            reflectivity: a.reflectivity.select(mask.clone(), b.reflectivity),
            shininess: a.shininess.select(mask, b.shininess),
        }
    }
}

impl<SourceScalar: SimdValueSimplified, Scalar: SimdValueSimplified + Splatable<SourceScalar>>
    Splatable<Material<SourceScalar>> for Material<Scalar>
{
    #[inline(always)]
    fn splat(
        Material {
            color,
            reflectivity,
            shininess,
            ..
        }: &Material<SourceScalar>,
    ) -> Self {
        Material::new(
            Splatable::splat(color),
            Splatable::splat(reflectivity),
            Splatable::splat(shininess),
        )
    }
}
