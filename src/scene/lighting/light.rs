use crate::color::ColorSimdExt;
use crate::helpers::ColorType;
use crate::light_utils::{calculate_attenuation, calculate_basic_light_factors};
use crate::light_vector_traits::{LightVectorOps, Vec3Helper};
use crate::scene::lighting::Lightable;
use crate::vector::{Vector, VectorAware};
use enum_dispatch::enum_dispatch;
use simba::simd::{SimdBool, SimdValue};

/// The contribution of a light source at a specific point
#[derive(Debug, Copy, Clone)]
pub(crate) struct LightContribution<Scalar>
where
    Scalar: SimdValue,
{
    pub(crate) color: ColorType<Scalar>,
    pub(crate) intensity: Scalar,
    pub(crate) valid_mask: Scalar::SimdBool,
}

impl<Scalar: SimdValue> LightContribution<Scalar> {
    pub(crate) const fn new(
        color: ColorType<Scalar>,
        intensity: Scalar,
        valid_mask: Scalar::SimdBool,
    ) -> Self {
        Self {
            color,
            intensity,
            valid_mask,
        }
    }
}

// A simplified zero implementation for LightContribution
impl<Scalar: SimdValue> Default for LightContribution<Scalar> 
where Scalar: Default,
      Scalar::SimdBool: Default
{
    fn default() -> Self {
        Self {
            color: ColorType::new(Default::default(), Default::default(), Default::default()),
            intensity: Default::default(),
            valid_mask: Default::default(),
        }
    }
}

#[enum_dispatch]
/// Base trait for all light types
pub(crate) trait Light<V: Vector> {
    /// Calculate the lighting contribution at a point for a lightable object
    fn calculate_contribution_at(
        &self,
        lightable: &impl Lightable<V>,
        position: V,
        ray_from_direction: V,
    ) -> LightContribution<V::Scalar>;
}

/// Ambient light that illuminates all surfaces equally
#[derive(Debug, Copy, Clone)]
pub(crate) struct AmbientLight<V: Vector> {
    pub(crate) color: ColorType<V::Scalar>,
    pub(crate) intensity: V::Scalar,
}

impl<V: Vector> AmbientLight<V> {
    pub const fn new(color: ColorType<V::Scalar>, intensity: V::Scalar) -> Self {
        Self { color, intensity }
    }
}

impl<V: Vector> VectorAware<V> for AmbientLight<V> {}

// Specialized implementation for Vec3 using our helper
impl Light<ultraviolet::Vec3> for AmbientLight<ultraviolet::Vec3> {
    fn calculate_contribution_at(
        &self,
        lightable: &impl Lightable<ultraviolet::Vec3>,
        point: ultraviolet::Vec3,
        ray_from_direction: ultraviolet::Vec3,
    ) -> LightContribution<f32> {
        // Apply ambient lighting to the material color
        let material = lightable.get_material_color_at(point);

        let one = Vec3Helper::one();
        let zero = Vec3Helper::zero();

        let normal = lightable.get_surface_normal_at(point);

        let incident_light_angle_cos = Vec3Helper::negate(Vec3Helper::dot(ray_from_direction, normal));

        let rescaled_incident_angle =
            Vec3Helper::div(
                Vec3Helper::add(incident_light_angle_cos, one),
                Vec3Helper::from_f32(2.75)
            );

        let intensity = Vec3Helper::mul(rescaled_incident_angle, self.intensity);
        let incident_angle_pos = Vec3Helper::is_greater_than(incident_light_angle_cos, zero);

        LightContribution::new(
            material * self.color,
            intensity,
            incident_angle_pos,
        )
    }
}

/// A point light that emits light in all directions from a single point
#[derive(Debug, Copy, Clone)]
pub(crate) struct PointLight<V: Vector> {
    pub(crate) position: V,
    pub(crate) color: ColorType<V::Scalar>,
    pub(crate) intensity: V::Scalar,
}

impl<V: Vector> PointLight<V> {
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

impl<V: Vector> VectorAware<V> for PointLight<V> {}

// Specialized implementation for Vec3 using our helper
impl Light<ultraviolet::Vec3> for PointLight<ultraviolet::Vec3> {
    fn calculate_contribution_at(
        &self,
        lightable: &impl Lightable<ultraviolet::Vec3>,
        point_position: ultraviolet::Vec3,
        ray_from_direction: ultraviolet::Vec3,
    ) -> LightContribution<f32> {
        let zero = Vec3Helper::zero();
        let one = Vec3Helper::one();
        let epsilon = Vec3Helper::from_f32(f32::EPSILON);
        let half = Vec3Helper::from_f32(0.15);
        let thousand = Vec3Helper::from_f32(35000.0);

        let normal = lightable.get_surface_normal_at(point_position);
        let material = lightable.get_material_color_at(point_position);

        let light_to_point = Vec3Helper::subtract(self.position, point_position);
        let light_distance = Vec3Helper::add(Vec3Helper::mag(light_to_point), epsilon);

        // Use the utility function to calculate basic light factors
        let (incident_light_angle_cos, incident_angle_pos) =
            calculate_basic_light_factors::<Vec3Helper>(normal, light_to_point, light_distance);

        // Calculate attenuation with the utility function
        let attenuation = calculate_attenuation::<Vec3Helper>(light_distance, thousand);

        // Calculate light intensity based on angle and distance
        let clamped_attenuation = Vec3Helper::clamp(attenuation, epsilon, Vec3Helper::add(one, half));
        let temp = Vec3Helper::mul(incident_light_angle_cos, self.intensity);
        let light_factor = Vec3Helper::mul(temp, clamped_attenuation);

        // Create the zero color for blending
        let zero_color = ColorType::new(zero, zero, zero);
        
        // Blend the colors based on the incident angle
        let blended_color = Vec3Helper::blend_colors(
            incident_angle_pos,
            material * self.color,
            zero_color,
        );

        // Select the light factor based on incident angle
        let final_light_factor = Vec3Helper::select(incident_angle_pos, light_factor, zero);

        LightContribution::new(
            blended_color,
            final_light_factor,
            incident_angle_pos,
        )
    }
}

// Use a concrete enum that doesn't rely on complex generic bounds
#[enum_dispatch(Light<ultraviolet::Vec3>)]
pub(crate) enum StandardLightSource {
    PointLight(PointLight<ultraviolet::Vec3>),
    AmbientLight(AmbientLight<ultraviolet::Vec3>),
}

// Similar enums could be created for SIMD vector types as needed