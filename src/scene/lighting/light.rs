use crate::color::ColorSimdExt;
use crate::color_traits::LightCompatibleColor;
use crate::helpers::{ColorType, Splatable};
use crate::scalar_traits::LightScalar;
use crate::scene::lighting::Lightable;
use crate::vector::{SimdCapableVector, VectorAware};
use crate::vector_traits::{RenderingVector, SimdRenderingVector};
use enum_dispatch::enum_dispatch;
use num_traits::{One, Zero};
use palette::bool_mask::BoolMask;
use simba::scalar::SupersetOf;
use simba::simd::{SimdPartialOrd, SimdValue};

#[derive(Debug, Copy, Clone)]
pub(crate) struct LightContribution<S: LightScalar> {
    pub(crate) color: ColorType<S>,
    pub(crate) intensity: S,
    pub(crate) valid_mask: S::SimdBool,
}

impl<S: LightScalar> LightContribution<S> {
    pub(crate) const fn new(color: ColorType<S>, intensity: S, valid_mask: S::SimdBool) -> Self {
        Self {
            color,
            intensity,
            valid_mask,
        }
    }
}

impl<S: LightScalar> palette::num::Zero for LightContribution<S> {
    #[inline]
    fn zero() -> Self {
        Self {
            color: ColorType::new(
                <S as num_traits::Zero>::zero(),
                <S as num_traits::Zero>::zero(),
                <S as num_traits::Zero>::zero(),
            ),
            intensity: <S as num_traits::Zero>::zero(),
            valid_mask: S::SimdBool::from_bool(false),
        }
    }
}

impl<S: LightScalar> Default for LightContribution<S>
where
    Self: palette::num::Zero,
{
    #[inline]
    fn default() -> Self {
        use palette::num::Zero;
        Self::zero()
    }
}

#[enum_dispatch]
/// Base trait for all light types
pub(crate) trait Light<V>
where
    V: RenderingVector,
{
    /// Calculate the lighting contribution at a point for a lightable object
    fn calculate_contribution_at(
        &self,
        lightable: &impl Lightable<V>,
        position: V,
        ray_from_direction: V,
    ) -> LightContribution<V::Scalar>;
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct AmbientLight<V>
where
    V: RenderingVector,
{
    pub(crate) color: ColorType<V::Scalar>,
    pub(crate) intensity: V::Scalar,
}

impl<V> AmbientLight<V>
where
    V: RenderingVector,
{
    pub const fn new(color: ColorType<V::Scalar>, intensity: V::Scalar) -> Self {
        Self { color, intensity }
    }
}

impl<V> AmbientLight<V>
where
    V: SimdRenderingVector,
{
    pub(crate) fn blend(mask: <V::Scalar as SimdValue>::SimdBool, t: &Self, f: &Self) -> Self {
        Self {
            color: ColorSimdExt::blend(mask, &t.color, &f.color),
            intensity: t.intensity.clone().select(mask, f.intensity.clone()),
        }
    }
}

impl<V> Splatable<AmbientLight<<V as SimdCapableVector>::SingleValueVector>> for AmbientLight<V>
where
    V: SimdRenderingVector,
{
    fn splat(v: &AmbientLight<<V as SimdCapableVector>::SingleValueVector>) -> Self {
        Self {
            intensity: V::Scalar::from_subset(&v.intensity),
            color: Splatable::splat(&v.color),
        }
    }
}

impl<V: RenderingVector> VectorAware<V> for AmbientLight<V> {}

impl<V> Light<V> for AmbientLight<V>
where
    V: RenderingVector,
    ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
{
    fn calculate_contribution_at(
        &self,
        lightable: &impl Lightable<V>,
        point: V,
        ray_from_direction: V,
    ) -> LightContribution<V::Scalar> {
        // Apply ambient lighting to the material color
        let material = lightable.get_material_color_at(point);

        let one = V::Scalar::one();
        let zero = V::Scalar::zero();

        let normal = lightable.get_surface_normal_at(point);

        let incident_light_angle_cos = -ray_from_direction.dot(normal);

        let rescaled_incident_angle =
            (incident_light_angle_cos.clone() + one) / V::Scalar::from_subset(&2.75);

        let intensity = rescaled_incident_angle * self.intensity.clone();

        LightContribution::new(
            material * self.color.clone(),
            intensity,
            incident_light_angle_cos.simd_gt(zero),
        )
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct PointLight<V>
where
    V: RenderingVector,
{
    pub(crate) position: V,
    pub(crate) color: ColorType<V::Scalar>,
    pub(crate) intensity: V::Scalar,
}

impl<V> PointLight<V>
where
    V: RenderingVector,
{
    pub(crate) const fn new(
        position: V,
        color: ColorType<V::Scalar>,
        intensity: V::Scalar,
    ) -> Self {
        Self {
            position,
            color,
            intensity,
        }
    }
}

impl<V> PointLight<V>
where
    V: SimdRenderingVector,
{
    pub(crate) fn blend(mask: <V::Scalar as SimdValue>::SimdBool, t: &Self, f: &Self) -> Self {
        Self {
            position: V::blend(mask, t.position.clone(), f.position.clone()),
            color: ColorSimdExt::blend(mask, &t.color, &f.color),
            intensity: t.intensity.clone().select(mask, f.intensity.clone()),
        }
    }
}

impl<V> Splatable<PointLight<<V as SimdCapableVector>::SingleValueVector>> for PointLight<V>
where
    V: SimdRenderingVector,
{
    fn splat(v: &PointLight<<V as SimdCapableVector>::SingleValueVector>) -> Self {
        Self {
            position: V::splat(v.position.clone()),
            intensity: V::Scalar::from_subset(&v.intensity),
            color: Splatable::splat(&v.color),
        }
    }
}

impl<V: RenderingVector<Scalar: LightScalar>> VectorAware<V> for PointLight<V> {}

impl<V> Light<V> for PointLight<V>
where
    V: SimdRenderingVector,
    ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
{
    fn calculate_contribution_at(
        &self,
        lightable: &impl Lightable<V>,
        point_position: V,
        ray_from_direction: V,
    ) -> LightContribution<V::Scalar> {
        let zero = V::Scalar::zero();
        let one = V::Scalar::one();
        let epsilon = V::Scalar::from_subset(&f32::EPSILON);
        let half = V::Scalar::from_subset(&0.15);
        let thousand = V::Scalar::from_subset(&35000.0); // Fixme strange hardcoded constant

        let normal = lightable.get_surface_normal_at(point_position);
        let material = lightable.get_material_color_at(point_position);

        let light_to_point = self.position - point_position;
        let light_distance = light_to_point.mag() + epsilon;

        // Calculate incident angle
        let incident_light_angle_cos = light_to_point.dot(normal) / light_distance;
        // Check if light is on the right side of the surface
        let incident_angle_pos = incident_light_angle_cos.simd_gt(zero);

        let attenuation =
            thousand / (epsilon + one * light_distance + one * light_distance * light_distance);

        // Calculate light intensity based on angle and distance
        let light_factor =
            incident_light_angle_cos * self.intensity * attenuation.simd_clamp(epsilon, one + half);

        LightContribution::new(
            ColorType::blend(
                incident_angle_pos,
                &(material * self.color),
                &ColorType::new(zero, zero, zero),
            ),
            light_factor.select(incident_angle_pos, zero),
            incident_angle_pos,
        )
    }
}

#[enum_dispatch(Light<V>)]
pub(crate) enum SceneLightSource<V>
where
    V: SimdRenderingVector,
    ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
{
    PointLight(PointLight<V>),
}
