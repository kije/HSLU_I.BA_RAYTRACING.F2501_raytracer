use crate::color::ColorSimdExt;
use crate::helpers::{ColorType, Splatable};
use crate::simd_compat::SimdValueRealSimplified;
use simba::simd::{SimdOption, SimdValue};

#[derive(Debug, Copy, Clone)]
pub struct TransmissionProperties<S: SimdValueRealSimplified> {
    refraction_index: S,
    opacity: SimdOption<S>,
}

impl<S: SimdValueRealSimplified> TransmissionProperties<S> {
    pub(crate) fn new(opacity: S, refraction_index: S) -> TransmissionProperties<S> {
        TransmissionProperties {
            refraction_index,
            opacity: SimdOption::new(opacity, true.into()),
        }
    }
    pub(crate) fn none() -> TransmissionProperties<S> {
        TransmissionProperties {
            refraction_index: <S as palette::num::Zero>::zero(),
            opacity: SimdOption::none(),
        }
    }

    pub(crate) fn mask(&self) -> S::SimdBool {
        self.opacity.mask()
            & !self
                .opacity
                .value()
                .abs_diff_eq_default(&<S as num_traits::Zero>::zero())
    }

    pub(crate) fn refraction_index(&self) -> SimdOption<S> {
        SimdOption::new(self.refraction_index, self.mask())
    }

    pub(crate) fn opacity(&self) -> SimdOption<S> {
        self.opacity
    }
}

impl<S: SimdValueRealSimplified> Default for TransmissionProperties<S> {
    fn default() -> Self {
        TransmissionProperties {
            refraction_index: <S as palette::num::Zero>::zero(),
            opacity: SimdOption::none(),
        }
    }
}

/// Surface material properties
#[derive(Debug, Copy, Clone)]
pub struct Material<S: SimdValueRealSimplified> {
    /// Surface color
    pub color: ColorType<S>,

    /// Reflectivity (0.0 = diffuse, 1.0 = mirror)
    pub reflectivity: S,

    /// Shininess of the surface (affects specular highlight) (0.0 = rough surface,  1.0 = very shiny)
    pub shininess: S,

    pub transmission: TransmissionProperties<S>,
    // Additional parameters could be added like:
    // - metalness
    // - transparency
    // - refraction index
    // - emission
}

impl<S: SimdValueRealSimplified> Default for Material<S> {
    fn default() -> Self {
        let zero = <S as palette::num::Zero>::zero();
        Self {
            color: ColorType::new(zero, zero, zero),
            reflectivity: zero,
            shininess: zero,
            transmission: TransmissionProperties::default(),
        }
    }
}

impl<S: SimdValueRealSimplified> Material<S> {
    /// Create a new material with basic properties
    pub fn new(
        color: ColorType<S>,
        reflectivity: S,
        shininess: S,
        transmission: TransmissionProperties<S>,
    ) -> Self {
        Self {
            color,
            reflectivity,
            shininess,
            transmission,
        }
    }

    /// Create a simple diffuse material
    pub fn diffuse(color: ColorType<S>) -> Self {
        Self {
            color,
            ..Default::default()
        }
    }

    pub fn translucent(color: ColorType<S>, opacity: S, refraction_index: S) -> Self {
        Self {
            color,
            transmission: TransmissionProperties {
                refraction_index,
                opacity: SimdOption::new(opacity, true.into()),
            },
            ..Default::default()
        }
    }

    /// Blend two materials based on a mask
    pub fn blend(mask: <S as SimdValue>::SimdBool, a: &Self, b: &Self) -> Self {
        Self {
            color: ColorSimdExt::blend(mask.clone(), &a.color, &b.color),
            reflectivity: a.reflectivity.select(mask.clone(), b.reflectivity),
            shininess: a.shininess.select(mask.clone(), b.shininess),
            transmission: TransmissionProperties {
                refraction_index: a
                    .transmission
                    .refraction_index
                    .select(mask.clone(), b.transmission.refraction_index),
                opacity: SimdOption::new(
                    a.transmission
                        .opacity
                        .value()
                        .select(mask.clone(), *b.transmission.opacity.value()),
                    a.transmission
                        .opacity
                        .mask()
                        .select(mask, b.transmission.opacity.mask()),
                ),
            },
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
            transmission,
            ..
        }: &Material<SourceScalar>,
    ) -> Self {
        Material {
            color: Splatable::splat(color),
            reflectivity: Splatable::splat(reflectivity),
            shininess: Splatable::splat(shininess),
            transmission: TransmissionProperties {
                refraction_index: Splatable::splat(&transmission.refraction_index),
                opacity: SimdOption::new(
                    Splatable::splat(transmission.opacity.value()),
                    Splatable::splat(&transmission.opacity.mask()),
                ),
            },
        }
    }
}
