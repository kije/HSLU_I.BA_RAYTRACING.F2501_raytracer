use crate::color::ColorSimdExt;
use crate::helpers::{ColorType, Splatable};
use crate::scalar::Scalar;
use crate::scene::lighting::Lightable;
use crate::vector::{
    CommonVecOperations, CommonVecOperationsFloat, CommonVecOperationsSimdOperations, Vector,
    VectorAware,
};
use enum_dispatch::enum_dispatch;
use num_traits::{One, Zero};
use palette::blend::Premultiply;
use palette::bool_mask::{BoolMask, HasBoolMask, LazySelect};
use palette::cast::ArrayCast;
use palette::stimulus::StimulusColor;
use palette::{Darken, Lighten, Mix, Srgb};
use simba::scalar::SubsetOf;
use simba::scalar::SupersetOf;
use simba::simd::{SimdBool, SimdComplexField, SimdPartialOrd, SimdRealField, SimdValue};

#[derive(Debug, Copy, Clone)]
pub(crate) struct LightContribution<Scalar>
where
    Scalar: crate::scalar::Scalar + SimdValue,
{
    pub(crate) color: ColorType<Scalar>,
    pub(crate) intensity: Scalar,
    pub(crate) valid_mask: Scalar::SimdBool,
}

impl<Scalar: crate::scalar::Scalar + SimdValue> LightContribution<Scalar> {
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

impl<Scalar: crate::scalar::Scalar + palette::num::Zero + SimdValue<SimdBool: SimdBool + BoolMask>>
    palette::num::Zero for LightContribution<Scalar>
{
    #[inline]
    fn zero() -> Self {
        Self {
            color: ColorType::new(Scalar::zero(), Scalar::zero(), Scalar::zero()),
            intensity: Scalar::zero(),
            valid_mask: Scalar::SimdBool::from_bool(false),
        }
    }
}

impl<Scalar: crate::scalar::Scalar + palette::num::Zero + SimdValue<SimdBool: SimdBool + BoolMask>>
    Default for LightContribution<Scalar>
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
pub(crate) trait Light<Vector>
where
    Vector: crate::vector::Vector,
    Vector::Scalar: SimdValue + Scalar,
{
    /// Calculate the lighting contribution at a point for a lightable object
    fn calculate_contribution_at(
        &self,
        lightable: &impl Lightable<Vector>,
        position: Vector,
        ray_from_direction: Vector,
    ) -> LightContribution<Vector::Scalar>;
}

// fixme this does not depend on Vector, just scalar
#[derive(Debug, Copy, Clone)]
pub(crate) struct AmbientLight<Vector>
where
    Vector: crate::vector::Vector,
    Vector::Scalar: SimdValue + Scalar,
{
    pub(crate) color: ColorType<Vector::Scalar>,
    pub(crate) intensity: Vector::Scalar,
}

impl<Vector> AmbientLight<Vector>
where
    Vector: crate::vector::Vector,
    Vector::Scalar: SimdValue + Scalar,
{
    pub const fn new(color: ColorType<Vector::Scalar>, intensity: Vector::Scalar) -> Self {
        Self { color, intensity }
    }
}

impl<Vector> AmbientLight<Vector>
where
    Vector: crate::vector::Vector + CommonVecOperationsSimdOperations,
    Vector::Scalar: SimdValue + Scalar,
{
    pub(crate) fn blend(mask: <Vector::Scalar as SimdValue>::SimdBool, t: &Self, f: &Self) -> Self
    where
        <<Vector as crate::vector::Vector>::Scalar as SimdValue>::Element:
            SubsetOf<<Vector as crate::vector::Vector>::Scalar>,
    {
        Self {
            color: ColorSimdExt::blend(mask, &t.color, &f.color),
            intensity: t.intensity.clone().select(mask, f.intensity.clone()),
        }
    }

    // pub(crate) fn splat(v: &PointLight<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector>) -> Self
    // where
    //     <<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar:
    //     SubsetOf<<Vector as crate::vector::Vector>::Scalar>,
    //     Vector::Scalar:   Clone +  SimdRealField + SupersetOf<<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar>,
    //     <<Vector as crate::vector::Vector>::Scalar as SimdValue>::Element: SubsetOf<<Vector as crate::vector::Vector>::Scalar>
    // {
    //     Self {
    //         intensity: Vector::Scalar::from_subset(&v.intensity),
    //         color: ColorSimdExt::splat(&v.color),
    //     }
    // }
}

impl<Vector> Splatable<AmbientLight<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector>> for AmbientLight<Vector>
where
    Vector: crate::vector::Vector + CommonVecOperationsSimdOperations,
    Vector::Scalar: Clone + SimdRealField + SupersetOf<<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar> + Splatable<<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar>,
    <<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar: SubsetOf<<Vector as crate::vector::Vector>::Scalar>,
    <<Vector as crate::vector::Vector>::Scalar as SimdValue>::Element: SubsetOf<<Vector as crate::vector::Vector>::Scalar>,
{
    fn splat(v: &AmbientLight<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector>) -> Self {
        Self {
            intensity: Vector::Scalar::from_subset(&v.intensity),
            color: Splatable::splat(&v.color),
        }
    }
}

impl<Vector> VectorAware<Vector> for AmbientLight<Vector> where Vector: crate::vector::Vector {}

impl<Vector> Light<Vector> for AmbientLight<Vector>
where
    Vector: crate::vector::Vector + CommonVecOperations + Copy,
    Vector::Scalar: Scalar
        + SimdValue
        + SimdRealField
        + palette::num::Real
        + palette::num::Zero
        + palette::num::One
        + palette::num::Arithmetics
        + palette::num::Clamp
        + palette::num::Sqrt
        + palette::num::Abs
        + palette::num::PartialCmp
        + HasBoolMask
        + palette::num::MinMax,
    <<Vector as crate::vector::Vector>::Scalar as HasBoolMask>::Mask:
        LazySelect<<Vector as crate::vector::Vector>::Scalar>,
    ColorType<<Vector as crate::vector::Vector>::Scalar>: Premultiply<Scalar = Vector::Scalar>
        + StimulusColor
        + ArrayCast<Array = [Vector::Scalar; <Vector as crate::vector::Vector>::DIMENSIONS]>,
{
    fn calculate_contribution_at(
        &self,
        lightable: &impl Lightable<Vector>,
        point: Vector,
        ray_from_direction: Vector,
    ) -> LightContribution<Vector::Scalar> {
        // Apply ambient lighting to the material color
        let material = lightable.get_material_color_at(point);

        let one = Vector::Scalar::one();
        let zero = Vector::Scalar::zero();

        let normal = lightable.get_surface_normal_at(point);

        let incident_light_angle_cos = -ray_from_direction.dot(normal);

        let rescaled_incident_angle =
            (incident_light_angle_cos.clone() + one) / Vector::Scalar::from_subset(&2.75);

        let intensity = rescaled_incident_angle * self.intensity.clone();

        LightContribution::new(
            material * self.color.clone(),
            intensity,
            incident_light_angle_cos.simd_gt(zero),
        )

        //(material.darken_fixed(Vector::Scalar::one() - intensity)) * self.color.clone()
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct PointLight<Vector>
where
    Vector: crate::vector::Vector,
    Vector::Scalar: SimdValue + Scalar,
{
    pub(crate) position: Vector,
    pub(crate) color: ColorType<Vector::Scalar>,
    pub(crate) intensity: Vector::Scalar,
}

impl<Vector> PointLight<Vector>
where
    Vector: crate::vector::Vector,
    Vector::Scalar: SimdValue + Scalar,
{
    pub(crate) const fn new(
        position: Vector,
        color: ColorType<Vector::Scalar>,
        intensity: Vector::Scalar, // fixme how do we get this to be always between 0.0 and 1.0?
    ) -> Self {
        Self {
            position,
            color,
            intensity,
        }
    }
}

impl<Vector> PointLight<Vector>
where
    Vector: crate::vector::Vector + CommonVecOperationsSimdOperations,
    Vector::Scalar: SimdValue + Scalar,
{
    pub(crate) fn blend(mask: <Vector::Scalar as SimdValue>::SimdBool, t: &Self, f: &Self) -> Self
    where
        <<Vector as crate::vector::Vector>::Scalar as SimdValue>::Element:
            SubsetOf<<Vector as crate::vector::Vector>::Scalar>,
    {
        Self {
            position: Vector::blend(mask, t.position.clone(), f.position.clone()),
            color: ColorSimdExt::blend(mask, &t.color, &f.color),
            intensity: t.intensity.clone().select(mask, f.intensity.clone()),
        }
    }

    // pub(crate) fn splat(v: &PointLight<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector>) -> Self
    // where
    //     <<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar:
    //     SubsetOf<<Vector as crate::vector::Vector>::Scalar>,
    //     Vector::Scalar:   Clone +  SimdRealField + SupersetOf<<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar>,
    //     <<Vector as crate::vector::Vector>::Scalar as SimdValue>::Element: SubsetOf<<Vector as crate::vector::Vector>::Scalar>
    // {
    //     Self {
    //         position: Vector::splat(v.position.clone()),
    //         intensity: Vector::Scalar::from_subset(&v.intensity),
    //         color: ColorSimdExt::splat(&v.color),
    //     }
    // }
}

impl<Vector> Splatable<PointLight<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector>> for PointLight<Vector>
where
    Vector: crate::vector::Vector + CommonVecOperationsSimdOperations,
    Vector::Scalar: Clone + SimdRealField + SupersetOf<<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar> + Splatable<<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar>,
    <<Vector as CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar: SubsetOf<<Vector as crate::vector::Vector>::Scalar>,
    <<Vector as crate::vector::Vector>::Scalar as SimdValue>::Element: SubsetOf<<Vector as crate::vector::Vector>::Scalar>,
{
    fn splat(v: &PointLight<<Vector as CommonVecOperationsSimdOperations>::SingleValueVector>) -> Self {
        Self {
            position: Vector::splat(v.position.clone()),
            intensity: Vector::Scalar::from_subset(&v.intensity),
            color: Splatable::splat(&v.color),
        }
    }
}

impl<Vector: crate::vector::Vector> VectorAware<Vector> for PointLight<Vector> {}

impl<Vector> Light<Vector> for PointLight<Vector>
where
    Vector: crate::vector::Vector
        + CommonVecOperations
        + CommonVecOperationsSimdOperations
        + Copy
        + std::ops::Sub<Vector, Output = Vector>,
    Vector::Scalar: SimdValue
        + Copy
        + Scalar
        + SimdRealField
        + palette::num::Real
        + palette::num::Zero
        + palette::num::One
        + palette::num::Arithmetics
        + palette::num::Clamp
        + palette::num::Sqrt
        + palette::num::Abs
        + palette::num::PartialCmp
        + HasBoolMask
        + palette::num::MinMax
        + std::ops::Sub<Vector::Scalar, Output = Vector::Scalar>,
    <<Vector as crate::vector::Vector>::Scalar as HasBoolMask>::Mask:
        LazySelect<<Vector as crate::vector::Vector>::Scalar>,
    <<Vector as crate::vector::Vector>::Scalar as SimdValue>::Element:
        SubsetOf<<Vector as crate::vector::Vector>::Scalar>,
    ColorType<<Vector as crate::vector::Vector>::Scalar>: Premultiply<Scalar = Vector::Scalar>
        + StimulusColor
        + ArrayCast<Array = [Vector::Scalar; <Vector as crate::vector::Vector>::DIMENSIONS]>,
{
    fn calculate_contribution_at(
        &self,
        lightable: &impl Lightable<Vector>,
        point_position: Vector,
        ray_from_direction: Vector,
    ) -> LightContribution<Vector::Scalar> {
        let zero = Vector::Scalar::zero();
        let one = Vector::Scalar::one();
        let epsilon = Vector::Scalar::from_subset(&f32::EPSILON);
        let half = Vector::Scalar::from_subset(&0.15);
        let thousand = Vector::Scalar::from_subset(&35000.0); // Fixme strange hardcoded constant

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
pub(crate) trait SceneLightSourceVector:
    Vector<
        Scalar: SimdValue<Element: SubsetOf<Self::Scalar>>
                    + Copy
                    + Scalar
                    + SimdRealField
                    + palette::num::Real
                    + palette::num::Zero
                    + palette::num::One
                    + palette::num::Arithmetics
                    + palette::num::Clamp
                    + palette::num::Sqrt
                    + palette::num::Abs
                    + palette::num::PartialCmp
                    + HasBoolMask<Mask: LazySelect<Self::Scalar>>
                    + palette::num::MinMax
                    + std::ops::Sub<Self::Scalar, Output = Self::Scalar>,
    > + CommonVecOperations
    + CommonVecOperationsSimdOperations
    + Copy
    + std::ops::Sub<Self, Output = Self>
{
}
#[enum_dispatch(Light<Vector>)]
pub(crate) enum SceneLightSource<Vector>
where
    Vector: SceneLightSourceVector,
    ColorType<Vector::Scalar>: Premultiply<Scalar = Vector::Scalar>
        + StimulusColor
        + ArrayCast<Array = [Vector::Scalar; <Vector as crate::vector::Vector>::DIMENSIONS]>,
    [(); <Vector as crate::vector::Vector>::DIMENSIONS]:,
{
    PointLight(PointLight<Vector>),
}
