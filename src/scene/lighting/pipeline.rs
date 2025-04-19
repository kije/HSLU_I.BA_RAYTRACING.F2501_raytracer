use crate::helpers::{ColorType, Splatable};
use crate::scalar_traits::LightScalar;

use crate::scene::lighting::{Light, Lightable, SceneLightSource};
use crate::scene::{AmbientLight, PointLight};
use crate::vector::{SimdCapableVector, Vector, VectorOperations};
use crate::vector_traits::RenderingVector;
use num_traits::Zero;
use palette::{Darken, Mix, Srgb};
use simba::scalar::{SubsetOf, SupersetOf};
use simba::simd::{SimdRealField, SimdValue};

/// Pipeline for calculating lighting on objects
#[derive(Debug, Clone, Copy)]
pub struct ShadingPipeline;

impl ShadingPipeline {
    /// Calculate lighting for a lightable object using all provided lights
    pub fn calculate_shading<'a, Vector, L>(
        lightable: &impl Lightable<Vector>,
        ambient: &impl Light<Vector>,
        lights: impl IntoIterator<
            Item = &'a SceneLightSource<<Vector as SimdCapableVector>::SingleValueVector>,
        > + Clone,
        position: Vector,
        ray_from_direction: Vector,
    ) -> ColorType<Vector::Scalar>
    where
        Vector: 'a + SimdRenderingVector,
        Vector::Scalar: LightScalar,
        <Vector::Scalar as SimdValue>::Element: SubsetOf<Vector::Scalar>,
        <Vector as SimdCapableVector>::SingleValueVector: SimdCapableVector + RenderingVector,
        <<Vector as SimdCapableVector>::SingleValueVector as crate::vector::Vector>::Scalar: LightScalar,
        <<<Vector as SimdCapableVector>::SingleValueVector as crate::vector::Vector>::Scalar as SimdValue>::Element: SubsetOf<<<Vector as SimdCapableVector>::SingleValueVector as crate::vector::Vector>::Scalar>,
        ColorType<Vector::Scalar>: Mix<Scalar = Vector::Scalar> + Darken<Scalar = Vector::Scalar>,
        [(); <Vector as crate::vector::Vector>::DIMENSIONS]:,
    {
        let zero = Vector::Scalar::zero();

        // Start with black
        let mut result_color = ColorType::new(zero, zero, zero);
        let mut direct_lighting_amount = zero;

        // Calculate contribution from each light
        for light in lights {
            let light = Splatable::splat(light);
            let light_contribution =
                light.calculate_contribution_at(lightable, position, ray_from_direction);

            // Blend the light contribution with existing color
            // This part would depend on the actual blending strategy you want
            // result_color =
            //     result_color.mix(light_contribution.color, Vector::Scalar::from_subset(&0.5));
            // direct_lighting_amount += Vector::Scalar::from_subset(&0.2);
        }

        // Final color adjustment
        // This would be similar to what you have in the original code
        // but extracted and parameterized
        result_color
    }
}
