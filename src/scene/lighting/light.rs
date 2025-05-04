use crate::color::{ColorSimdExt, maximize_value};
use crate::color_traits::LightCompatibleColor;
use crate::helpers::{ColorType, Splatable};
use crate::scene::lighting::Lightable;
use crate::simd_compat::SimdValueRealSimplified;
use crate::vector::{SimdCapableVector, Vector, VectorAware};
use crate::vector_traits::{RenderingVector, SimdRenderingVector};
use crate::{
    WINDOW_TO_SCENE_DEPTH_FACTOR, WINDOW_TO_SCENE_HEIGHT_FACTOR, WINDOW_TO_SCENE_WIDTH_FACTOR,
};
use fast_poisson::{Poisson2D, Poisson3D};
use itertools::Itertools;
use num_traits::{One, Zero};
use palette::bool_mask::BoolMask;
use rand::distributions::{Distribution, Standard};
use simba::scalar::SupersetOf;
use simba::simd::SimdComplexField;
use simba::simd::SimdRealField;
use simba::simd::{SimdPartialOrd, SimdValue};
use ultraviolet::Vec3;

#[derive(Debug, Copy, Clone)]
pub struct LightContribution<S: SimdValueRealSimplified> {
    pub color: ColorType<S>,
    pub intensity: S,
    pub valid_mask: S::SimdBool,
}

impl<S: SimdValueRealSimplified> LightContribution<S> {
    pub const fn new(color: ColorType<S>, intensity: S, valid_mask: S::SimdBool) -> Self {
        Self {
            color,
            intensity,
            valid_mask,
        }
    }
}

impl<S: SimdValueRealSimplified> palette::num::Zero for LightContribution<S> {
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

impl<S: SimdValueRealSimplified> Default for LightContribution<S>
where
    Self: palette::num::Zero,
{
    #[inline]
    fn default() -> Self {
        use palette::num::Zero;
        Self::zero()
    }
}

/// Base trait for all light types
pub trait Light<V>
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
pub struct AmbientLight<V>
where
    V: RenderingVector,
{
    pub color: ColorType<V::Scalar>,
    pub intensity: V::Scalar,
}

impl<V> AmbientLight<V>
where
    V: RenderingVector,
{
    pub fn new(color: ColorType<V::Scalar>, intensity: V::Scalar) -> Self {
        Self {
            color: maximize_value(color),
            intensity,
        }
    }
}

impl<V> AmbientLight<V>
where
    V: SimdRenderingVector,
{
    pub fn blend(mask: <V::Scalar as SimdValue>::SimdBool, t: &Self, f: &Self) -> Self {
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
pub struct PointLight<V>
where
    V: RenderingVector,
{
    pub position: V,
    pub color: ColorType<V::Scalar>,
    pub intensity: V::Scalar,
}

impl<V> PointLight<V>
where
    V: RenderingVector,
{
    pub fn new(position: V, color: ColorType<V::Scalar>, intensity: V::Scalar) -> Self {
        Self {
            position,
            color: maximize_value(color),
            intensity,
        }
    }

    pub fn to_point_light_cloud<const N: usize>(&self) -> [Self; N]
    where
        V: SimdCapableVector<SingleValueVector = Vec3>,
        Standard: Distribution<<V as Vector>::Scalar>,
    {
        if N == 1 {
            return [self.clone(); N];
        }

        let cloud_radius = (1.375 + (N as f32 / 56.0));

        let scale = V::Scalar::from_subset(&(1.0 / N as f32));
        let cloud_radius_v = V::broadcast(V::Scalar::from_subset(&cloud_radius));

        let window_to_scene_scale = V::unit_x().mul_add(
            V::broadcast(V::Scalar::from_subset(&WINDOW_TO_SCENE_WIDTH_FACTOR)),
            V::unit_y().mul_add(
                V::broadcast(V::Scalar::from_subset(&WINDOW_TO_SCENE_HEIGHT_FACTOR)),
                V::unit_z() * V::broadcast(V::Scalar::from_subset(&WINDOW_TO_SCENE_DEPTH_FACTOR)),
            ),
        );
        let random_points = Poisson3D::new()
            .with_dimensions([cloud_radius, cloud_radius, cloud_radius], (1.0 / N as f32))
            .with_samples(N as u32);

        (0..N)
            .zip(
                random_points
                    .iter()
                    .map(|p| V::splat(Vec3::new(p[0], p[1], p[2])))
                    .pad_using(N, |_| V::sample_random() * cloud_radius_v),
            )
            .map(|(_, random_point)| {
                let mut l = self.clone();

                l.position = l.position + (random_point * window_to_scene_scale);
                l.intensity = scale * l.intensity;

                l
            })
            .collect_array::<{ N }>()
            .expect("Failed to collect array.")
    }
}

impl<V> PointLight<V>
where
    V: SimdRenderingVector,
{
    pub fn blend(mask: <V::Scalar as SimdValue>::SimdBool, t: &Self, f: &Self) -> Self {
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

impl<V: RenderingVector<Scalar: SimdValueRealSimplified>> VectorAware<V> for PointLight<V> {}

impl<V> Light<V> for PointLight<V>
where
    V: SimdRenderingVector,
    ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
{
    fn calculate_contribution_at(
        &self,
        lightable: &impl Lightable<V>,
        point_position: V,
        _: V,
    ) -> LightContribution<V::Scalar> {
        let zero = V::Scalar::zero();
        let one = V::Scalar::one();
        let epsilon = V::Scalar::simd_default_epsilon();
        let thousand = V::Scalar::from_subset(&0.95); // Fixme strange hardcoded constant

        let normal = lightable.get_surface_normal_at(point_position);
        let material = lightable.get_material_color_at(point_position);

        let light_to_point = self.position - point_position;
        let light_distance = light_to_point.mag() + epsilon;

        // Calculate incident angle
        let incident_light_angle_cos = light_to_point.dot(normal) / light_distance;
        // Check if light is on the right side of the surface
        let incident_angle_pos = incident_light_angle_cos.simd_gt(zero);

        let attenuation = thousand * (epsilon + light_distance + light_distance * light_distance);

        let att_sigmoid = attenuation.simd_tanh();
        // Calculate light intensity based on angle and distance
        let light_factor =
            incident_light_angle_cos * self.intensity * att_sigmoid.simd_clamp(zero, one);

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

pub enum SceneLightSource<V>
where
    V: SimdRenderingVector,
    ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
{
    PointLight(PointLight<V>),
}

impl<V> Light<V> for SceneLightSource<V>
where
    V: SimdRenderingVector,
    ColorType<V::Scalar>: LightCompatibleColor<V::Scalar>,
{
    fn calculate_contribution_at(
        &self,
        lightable: &impl Lightable<V>,
        position: V,
        ray_from_direction: V,
    ) -> LightContribution<V::Scalar> {
        match self {
            SceneLightSource::PointLight(light) => {
                light.calculate_contribution_at(lightable, position, ray_from_direction)
            }
        }
    }
}
