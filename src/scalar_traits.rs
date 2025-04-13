use crate::simd_compat::SimdValueRealSimplified;

/// A consolidated trait for scalar types used in lighting calculations
///
/// This combines all the common requirements for scalars used in light calculations
/// to simplify trait bounds throughout the codebase.
#[deprecated = "Use `SimdValueRealSimplified` directly instead."]
pub trait LightScalar: SimdValueRealSimplified {}

// Blanket implementation for any type that satisfies the requirements
impl<T> LightScalar for T where T: SimdValueRealSimplified {}
