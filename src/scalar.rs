use simba::simd::SimdValue;
use std::fmt::Debug;

/// The basic scalar type
///
/// This does not make any assumption on the algebraic properties of `Self`.
pub trait Scalar: 'static + Clone + PartialEq + Debug {}

impl<T: 'static + Clone + PartialEq + Debug> Scalar for T {}

pub(crate) trait CheckScalarLanesMatch<const REQUIRED_LANES: usize>: SimdValue {
    const CHECK: ();
}

impl<const REQUIRED_LANES_NUM: usize, T: SimdValue> CheckScalarLanesMatch<REQUIRED_LANES_NUM>
    for T
{
    const CHECK: () = [()][(Self::LANES == REQUIRED_LANES_NUM) as usize];
}
