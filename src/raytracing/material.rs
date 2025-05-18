use crate::color::ColorSimdExt;
use crate::color_traits::LightCompatibleColor;
use crate::helpers::{ColorType, Splatable};
use crate::simd_compat::SimdValueRealSimplified;
use crate::vector::{SimdCapableVector, Vector, VectorLerp};
use crate::vector_traits::{RenderingVector, SimdRenderingVector};
use num_traits::{One, Zero};
use palette::blend::Premultiply;
use simba::simd::SimdComplexField;
use simba::simd::{SimdBool, SimdPartialOrd};
use simba::simd::{SimdOption, SimdValue};
use std::ops::BitXor;

#[derive(Debug, Copy, Clone)]
pub struct TransmissionProperties<S: SimdValueRealSimplified> {
    refraction_index: S,
    opacity: SimdOption<S>,
    boost: S,
}

impl<S: SimdValueRealSimplified> TransmissionProperties<S> {
    pub fn new(opacity: S, refraction_index: S) -> TransmissionProperties<S> {
        TransmissionProperties {
            refraction_index,
            opacity: SimdOption::new(opacity, true.into()),
            boost: <S as palette::num::Zero>::zero(),
        }
    }
    pub fn new_with_boost(opacity: S, refraction_index: S, boost: S) -> TransmissionProperties<S> {
        TransmissionProperties {
            refraction_index,
            opacity: SimdOption::new(opacity, true.into()),
            boost,
        }
    }
    pub fn none() -> TransmissionProperties<S> {
        TransmissionProperties {
            refraction_index: <S as palette::num::Zero>::zero(),
            opacity: SimdOption::none(),
            boost: <S as palette::num::Zero>::zero(),
        }
    }

    pub fn mask(&self) -> S::SimdBool {
        self.opacity.mask()
            & !self
                .opacity
                .value()
                .abs_diff_eq_default(&<S as num_traits::Zero>::zero())
    }

    pub fn refraction_index(&self) -> SimdOption<S> {
        // fixme: also return something for metallic surfaces?
        SimdOption::new(self.refraction_index, self.mask())
    }

    pub fn opacity(&self) -> SimdOption<S> {
        self.opacity
    }

    pub fn boost(&self) -> SimdOption<S> {
        SimdOption::new(self.boost, self.mask())
    }
}

impl<S: SimdValueRealSimplified> Default for TransmissionProperties<S> {
    fn default() -> Self {
        TransmissionProperties {
            refraction_index: <S as palette::num::One>::one(),
            opacity: SimdOption::none(),
            boost: <S as palette::num::Zero>::zero(),
        }
    }
}

/// Surface material properties
#[derive(Debug, Copy, Clone)]
pub struct Material<S: SimdValueRealSimplified> {
    /// Surface color
    pub color: ColorType<S>,

    /// Reflectivity (0.0 = diffuse, 1.0 = mirror)
    pub metallic: S,

    /// Shininess of the surface (affects specular highlight) (0.0 = rough surface,  1.0 = very shiny)
    shininess: S, // fixme refactor to "roughness", more common way to store/refer to this

    pub transmission: TransmissionProperties<S>,
    // Additional parameters could be added like:
    // - metalness
    // - transparency
    // - refraction index
    // - emission
}

impl From<&tobj::Material> for Material<f32> {
    fn from(value: &tobj::Material) -> Self {
        let illumination_model = value.illumination_model.unwrap_or(0);
        let color = ColorType::from(value.diffuse.unwrap_or([Default::default(); 3]));
        let metallic = match illumination_model {
            3 => value
                .unknown_param
                .get("Pm")
                .unwrap_or(&"0.0".to_string())
                .parse()
                .unwrap_or(0.0),
            _ => 0.0,
        };
        let shininess = match illumination_model {
            3 | 2 | 0 => value
                .unknown_param
                .get("Ps")
                .unwrap_or(&"0.0".to_string())
                .parse()
                .unwrap_or(0.0),
            _ => 0.0,
        };
        //println!("Material {:?} ", value.unknown_param);
        Material {
            color,
            metallic,
            shininess,
            ..Default::default()
        }
    }
}

impl<S: SimdValueRealSimplified> Default for Material<S> {
    fn default() -> Self {
        let zero = <S as palette::num::Zero>::zero();
        Self {
            color: ColorSimdExt::zero(),
            metallic: zero,
            shininess: zero,
            transmission: TransmissionProperties::default(),
        }
    }
}

impl<S: SimdValueRealSimplified> Material<S> {
    /// Create a new material with basic properties
    pub fn new(
        color: ColorType<S>,
        metallic: S,
        shininess: S,
        transmission: TransmissionProperties<S>,
    ) -> Self {
        Self {
            color,
            metallic,
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
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Blend two materials based on a mask
    pub fn blend(mask: <S as SimdValue>::SimdBool, a: &Self, b: &Self) -> Self {
        Self {
            color: ColorSimdExt::blend(mask.clone(), &a.color, &b.color),
            metallic: a.metallic.select(mask.clone(), b.metallic),
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
                boost: a
                    .transmission
                    .boost
                    .select(mask.clone(), b.transmission.boost),
            },
        }
    }

    pub fn shininess(&self) -> S {
        self.shininess
    }

    pub fn roughness(&self) -> S {
        <S as One>::one() - self.shininess
    }

    pub fn absorption(&self) -> ColorType<S>
    where
        ColorType<S>: LightCompatibleColor<S>,
    {
        let one = <S as One>::one();
        let zero = <S as Zero>::zero();

        let refraction_opacity = self
            .transmission
            .opacity()
            .simd_unwrap_or(|| one)
            .simd_clamp(zero, one - S::default_epsilon());
        let material_absorption = self
            .color
            .premultiply(one - refraction_opacity)
            .into_linear();

        material_absorption
    }

    /// Bidirectional scattering distribution function
    ///
    /// - `n` - surface normal vector
    /// - `wo` - unit direction vector toward the viewer
    /// - `wi` - unit direction vector toward the incident ray
    ///
    /// This works for both opaque and transmissive materials, based on a Beckmann
    /// microfacet distribution model, Cook-Torrance shading for the specular component,
    /// and Lambertian shading for the diffuse component. Useful references:
    ///
    /// - http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
    /// - https://computergraphics.stackexchange.com/q/4394
    /// - https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
    /// - http://www.pbr-book.org/3ed-2018/Materials/BSDFs.html
    /// - https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    pub fn bsdf<V: SimdRenderingVector<Scalar = S>>(
        &self,
        n: V,
        wo: V,
        wi: V,
        other_ior: V::Scalar,
    ) -> ColorType<S> {
        let n_dot_wi = n.dot(wi);
        let n_dot_wo = n.dot(wo);
        let wi_outside = n_dot_wi.is_simd_positive();
        let wo_outside = n_dot_wo.is_simd_positive();

        let is_opaque = !self.transmission.mask() & (!wi_outside | !wo_outside);
        if is_opaque.all() {
            // println!("OP");
            // Opaque materials do not transmit light
            return ColorSimdExt::zero();
        }

        let wi_wo_same_direction = !(wi_outside.bitxor(wo_outside)); // XNOR

        let is_btdf_domain = !is_opaque & !wi_wo_same_direction;
        if is_btdf_domain.all() {
            return self.btdf_internal(n, wo, wi, n_dot_wi, n_dot_wo, wo_outside, other_ior);
        }

        let is_brdf_domain = !is_opaque & wi_wo_same_direction;
        if is_brdf_domain.all() {
            return self.brdf_internal(n, wo, wi, n_dot_wi, n_dot_wo, wi_outside, other_ior);
        }

        let btdf_color = self.btdf_internal(n, wo, wi, n_dot_wi, n_dot_wo, wo_outside, other_ior);
        let brdf_color = self.brdf_internal(n, wo, wi, n_dot_wi, n_dot_wo, wi_outside, other_ior);

        //println!("brdf: {:?} / btdf: {:?}", brdf_color, btdf_color);

        ColorSimdExt::blend(
            is_btdf_domain,
            &btdf_color,
            &ColorSimdExt::blend(is_brdf_domain, &brdf_color, &ColorSimdExt::zero()),
        )
    }

    pub fn btdf<V: SimdRenderingVector<Scalar = S>>(
        &self,
        n: V,
        wo: V,
        wi: V,
        other_ior: V::Scalar,
    ) -> ColorType<S> {
        let n_dot_wi = n.dot(wi);
        let n_dot_wo = n.dot(wo);
        let wi_outside = n_dot_wi.is_simd_positive();
        let wo_outside = n_dot_wo.is_simd_positive();

        let is_opaque = !self.transmission.mask() & (!wi_outside | !wo_outside);

        let wi_wo_same_direction = !(wi_outside.bitxor(wo_outside)); // XNOR

        let is_btdf_domain = !is_opaque & !wi_wo_same_direction;
        if is_btdf_domain.none() {
            // Opaque materials do not transmit light
            return ColorSimdExt::zero();
        }

        let final_color = self.btdf_internal(n, wo, wi, n_dot_wi, n_dot_wo, wo_outside, other_ior);

        ColorSimdExt::blend(is_btdf_domain, &final_color, &ColorSimdExt::zero())
    }

    pub fn brdf<V: SimdRenderingVector<Scalar = S>>(
        &self,
        n: V,
        wo: V,
        wi: V,
        other_ior: V::Scalar,
    ) -> ColorType<S> {
        let n_dot_wi = n.dot(wi);
        let n_dot_wo = n.dot(wo);
        let wi_outside = n_dot_wi.is_simd_positive();
        let wo_outside = n_dot_wo.is_simd_positive();

        let is_opaque = !self.transmission.mask() & (!wi_outside | !wo_outside);

        let wi_wo_same_direction = !(wi_outside.bitxor(wo_outside)); // XNOR

        let is_brdf_domain = !is_opaque & wi_wo_same_direction;
        if is_brdf_domain.none() {
            // Opaque materials do not transmit light
            return ColorSimdExt::zero();
        }

        let final_color = self.brdf_internal(n, wo, wi, n_dot_wi, n_dot_wo, wi_outside, other_ior);

        ColorSimdExt::blend(is_brdf_domain, &final_color, &ColorSimdExt::zero())
    }

    fn btdf_internal<V: SimdRenderingVector<Scalar = S>>(
        &self,
        n: V,
        wo: V,
        wi: V,
        n_dot_wi: S,
        n_dot_wo: S,
        wo_outside: S::SimdBool,
        other_ior: V::Scalar,
    ) -> ColorType<S> {
        let one = <S as One>::one();
        let two = S::from_subset(&2.0);
        let zero = <S as Zero>::zero();
        let pi = S::simd_pi();

        let color_as_vec = self.color.to_vec3();

        let refractive_index = self.transmission.refraction_index;
        // Ratio of refractive indices, η_i / η_o
        let eta_t = (refractive_index / other_ior).select(wo_outside, other_ior / refractive_index);
        let h = wi.mul_add(V::broadcast(eta_t), wo).normalized(); // halfway vector
        let wi_dot_h = wi.dot(h);
        let wo_dot_h = wo.dot(h);
        let n_dot_h = n.dot(h);
        let nh2 = n_dot_h.simd_powi(2);

        // d: microfacet distribution function
        // D = exp(((n • h)^2 - 1) / (m^2 (n • h)^2)) / (π m^2 (n • h)^4)
        let m2 = self.roughness() * self.roughness();
        let d = ((nh2 - one) / (m2 * nh2)).simd_exp() / (m2 * pi * nh2 * nh2);

        // fixme dry up fresnel with implementation below
        // f: fresnel, schlick's approximation
        // F = F0 + (1 - F0)(1 - wi • h)^5
        let f0 = ((refractive_index - other_ior) / (refractive_index + other_ior)).simd_powi(2);
        let f0 = VectorLerp::lerp(&V::broadcast(f0), color_as_vec, self.metallic);
        let f = f0 + (V::one() - f0) * V::broadcast((one - wo_dot_h).simd_powi(5));

        // g: geometry function, microfacet shadowing
        // G = min(1, 2(n • h)(n • wo)/(wo • h), 2(n • h)(n • wi)/(wo • h))
        let g = (n_dot_wi * n_dot_h)
            .simd_abs()
            .simd_min((n_dot_wo * n_dot_h).simd_abs());
        let g = (two * g) / wo_dot_h.simd_abs();
        let g = g.simd_min(one);

        // BTDF: putting it all together
        // Cook-Torrance = |h • wi|/|n • wi| * |h • wo|/|n • wo|
        //                  * η_o^2 (1 - F)DG / (η_i (h • wi) + η_o (h • wo))^2
        // let btdf = V::broadcast((wi_dot_h * wo_dot_h / (n_dot_wi * n_dot_wo)).simd_abs())
        //     * (V::broadcast(d) * (V::one() - f) * V::broadcast(g)
        //         / V::broadcast((eta_t * wi_dot_h + wo_dot_h).simd_powi(2)));
        let jacobian =
            (wo_dot_h * wo_dot_h) / ((eta_t * wi_dot_h + wo_dot_h) * (eta_t * wi_dot_h + wo_dot_h));
        let btdf = V::broadcast(jacobian * (n_dot_wi * n_dot_wo).simd_abs().simd_recip())
            * (V::broadcast(d) * (V::one() - f) * V::broadcast(g));

        let btdf = btdf * V::broadcast(eta_t * eta_t);

        ColorSimdExt::from_vec3(&(btdf * color_as_vec))
    }
    fn brdf_internal<V: SimdRenderingVector<Scalar = S>>(
        &self,
        n: V,
        wo: V,
        wi: V,
        n_dot_wi: S,
        n_dot_wo: S,
        wi_outside: S::SimdBool,
        other_ior: V::Scalar,
    ) -> ColorType<S> {
        let one = <S as One>::one();
        let two = S::from_subset(&2.0);
        let zero = <S as Zero>::zero();
        let pi = S::simd_pi();

        let color_as_vec = self.color.to_vec3();

        let refractive_index = self.transmission.refraction_index;

        let h = (wi + wo).normalized(); // halfway vector
        let wo_dot_h = wo.dot(h);
        let n_dot_h = n.dot(h);
        let nh2 = n_dot_h.simd_powi(2);

        // d: microfacet distribution function
        // D = exp(((n • h)^2 - 1) / (m^2 (n • h)^2)) / (π m^2 (n • h)^4)
        let m2 = self.roughness() * self.roughness();
        let d = ((nh2 - one) / (m2 * nh2)).simd_exp() / (m2 * pi * nh2 * nh2);

        // f: fresnel, schlick's approximation
        // F = F0 + (1 - F0)(1 - wi • h)^5
        let is_total_internal_reflection = !wi_outside
            & (wo_dot_h.simd_mul_add(wo_dot_h, -one).simd_sqrt() * refractive_index).simd_gt(one);

        let f0 = ((refractive_index - other_ior) / (refractive_index + other_ior)).simd_powi(2);
        let f0 = VectorLerp::lerp(&V::broadcast(f0), color_as_vec, self.metallic);
        let f = f0 + (V::one() - f0) * V::broadcast((one - wo_dot_h).simd_powi(5));

        let f = <V as SimdCapableVector>::blend(is_total_internal_reflection, V::one(), f);

        // g: geometry function, microfacet shadowing
        // G = min(1, 2(n • h)(n • wo)/(wo • h), 2(n • h)(n • wi)/(wo • h))
        let g = (n_dot_wi * n_dot_h).simd_min(n_dot_wo * n_dot_h);
        let g = (two * g) / wo_dot_h;
        let g = g.simd_min(one);

        // BRDF: putting it all together
        // Cook-Torrance = DFG / (4(n • wi)(n • wo))
        // Lambert = (1 - F) * c / π
        let specular = V::broadcast(d) * f * V::broadcast(g)
            / V::broadcast(S::from_subset(&4.0) * n_dot_wo * n_dot_wi);

        let diffuse = (V::one() - f) * color_as_vec / V::broadcast(pi);

        ColorSimdExt::from_vec3(&<V as SimdCapableVector>::blend(
            self.transmission.mask(),
            specular,
            specular + diffuse,
        ))
    }

    /// Compute Fresnel reflectance using Schlick's approximation
    pub fn compute_fresnel<V: SimdRenderingVector<Scalar = S>>(
        &self,
        normal: V,
        view_dir: V,
        other_ior: V::Scalar,
    ) -> (ColorType<V::Scalar>, ColorType<V::Scalar>)
    where
        ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
    {
        let one = <S as One>::one();
        let zero = <S as Zero>::zero();

        let is_reflective = self.metallic.simd_gt(zero);
        let is_transmissive = self.transmission.mask();

        if is_transmissive.none() {
            return (
                <ColorType<V::Scalar> as ColorSimdExt<V::Scalar>>::one() * self.metallic,
                <ColorType<V::Scalar> as ColorSimdExt<V::Scalar>>::zero(),
            );
        }

        let refractive_index = self.transmission.refraction_index;

        let n_dot_v = normal.dot(view_dir);
        // Cosine of angle between normal and view direction
        let cos_theta = n_dot_v.simd_abs();

        // Check for total internal reflection
        let is_inside = n_dot_v.simd_lt(zero);
        let eta_t = (refractive_index / other_ior).select(is_inside, other_ior / refractive_index);
        let sin2_t = eta_t * eta_t * (one - cos_theta * cos_theta);
        let is_tir = (self.transmission.mask() & is_inside & sin2_t.simd_gt(one)) | is_reflective;

        let color_as_vec = self.color.to_vec3();

        // For metals, F0 is tinted by the material color
        let f0 = ((other_ior - refractive_index) / (other_ior + refractive_index)).simd_powi(2);
        let f0 = VectorLerp::lerp(&V::broadcast(f0), color_as_vec, self.metallic);
        let fresnel = f0 + (V::one() - f0) * V::broadcast((one - cos_theta).simd_powi(5));

        let reflected_amount = ColorType::<V::Scalar>::blend(
            is_reflective,
            &(<ColorType<V::Scalar> as ColorSimdExt<V::Scalar>>::one() * self.metallic),
            &<ColorType<V::Scalar> as ColorSimdExt<V::Scalar>>::one(),
        );
        // In case of total internal reflection, return full reflection
        let f = ColorType::<V::Scalar>::blend(
            is_tir,
            &reflected_amount,
            &ColorSimdExt::from_vec3(&fresnel),
        );

        (
            f,
            <ColorType<V::Scalar> as ColorSimdExt<V::Scalar>>::one() - f,
        )
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
            metallic,
            shininess,
            transmission,
            ..
        }: &Material<SourceScalar>,
    ) -> Self {
        Material {
            color: Splatable::splat(color),
            metallic: Splatable::splat(metallic),
            shininess: Splatable::splat(shininess),
            transmission: TransmissionProperties {
                refraction_index: Splatable::splat(&transmission.refraction_index),
                opacity: SimdOption::new(
                    Splatable::splat(transmission.opacity.value()),
                    Splatable::splat(&transmission.opacity.mask()),
                ),
                boost: Splatable::splat(&transmission.boost),
            },
        }
    }
}
