use crate::scalar_traits::LightScalar;
use palette::blend::Premultiply;
use palette::cast::ArrayCast;
use palette::stimulus::StimulusColor;

/// A consolidated trait for color types compatible with lighting calculations
///
/// This combines all the common requirements for colors used in light calculations
/// to simplify trait bounds throughout the codebase.
pub trait LightCompatibleColor<S: LightScalar>:
    Premultiply<Scalar = S> + StimulusColor + ArrayCast<Array = [S; 3]> + Clone
{
}

// Blanket implementation for any color type that satisfies the requirements
impl<S, C> LightCompatibleColor<S> for C
where
    S: LightScalar,
    C: Premultiply<Scalar = S> + StimulusColor + ArrayCast<Array = [S; 3]> + Clone,
{
}
