use crate::color::ColorSimdExt;
use crate::helpers::{ColorType, Splatable};
use crate::simd_compat::SimdValueRealSimplified;
use simba::simd::{SimdOption, SimdValue};

#[derive(Debug, Copy, Clone, Default)]
pub struct TransmissionProperties<S: SimdValueRealSimplified> {
    pub refraction_index: S,
    pub opacity: SimdOption<S>,
}

/// Surface material properties
#[derive(Debug, Copy, Clone, Default)]
pub struct Material<S: SimdValueRealSimplified> {
    /// Surface color
    pub color: ColorType<S>,

    /// Reflectivity (0.0 = diffuse, 1.0 = mirror)
    pub reflectivity: S,

    /// Shininess of the surface (affects specular highlight) (0.0 = rough surface,  1.0 = very shiny)
    pub shininess: S,

    pub refraction_index: SimdOption<S>,
    // Additional parameters could be added like:
    // - metalness
    // - transparency
    // - refraction index
    // - emission
}

impl<S: SimdValueRealSimplified> Material<S> {
    /// Create a new material with basic properties
    pub fn new(
        color: ColorType<S>,
        reflectivity: S,
        shininess: S,
        refraction_index: SimdOption<S>,
    ) -> Self {
        Self {
            color,
            reflectivity,
            shininess,
            refraction_index,
        }
    }

    /// Create a simple diffuse material
    pub fn diffuse(color: ColorType<S>) -> Self {
        Self::new(
            color,
            <S as palette::num::Zero>::zero(),
            <S as palette::num::Zero>::zero(),
            SimdOption::none(),
        )
    }

    pub fn translucent(color: ColorType<S>, refraction_index: S) -> Self {
        Self::new(
            color,
            <S as palette::num::Zero>::zero(),
            <S as palette::num::Zero>::zero(),
            SimdOption::new(refraction_index, true.into()),
        )
    }

    /// Blend two materials based on a mask
    pub fn blend(mask: <S as SimdValue>::SimdBool, a: &Self, b: &Self) -> Self {
        Self {
            color: ColorSimdExt::blend(mask.clone(), &a.color, &b.color),
            reflectivity: a.reflectivity.select(mask.clone(), b.reflectivity),
            shininess: a.shininess.select(mask, b.shininess),
            refraction_index: SimdOption::new(
                a.refraction_index
                    .value()
                    .select(mask, *b.refraction_index.value()),
                a.refraction_index
                    .mask()
                    .select(mask, b.refraction_index.mask()),
            ),
        }
    }
}

impl<
    SourceScalar: SimdValueRealSimplified,
    Scalar: SimdValueRealSimplified<SimdBool: Splatable<SourceScalar::SimdBool>> + Splatable<SourceScalar>,
> Splatable<Material<SourceScalar>> for Material<Scalar>
{
    #[inline(always)]
    fn splat(
        Material {
            color,
            reflectivity,
            shininess,
            refraction_index,
            ..
        }: &Material<SourceScalar>,
    ) -> Self {
        Material::new(
            Splatable::splat(color),
            Splatable::splat(reflectivity),
            Splatable::splat(shininess),
            SimdOption::new(
                Splatable::splat(refraction_index.value()),
                Splatable::splat(&refraction_index.mask()),
            ),
        )
    }
}
