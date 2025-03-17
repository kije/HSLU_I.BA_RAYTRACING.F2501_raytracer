use std::fmt::Debug;

/// The basic scalar type
///
/// This does not make any assumption on the algebraic properties of `Self`.
pub trait Scalar: 'static + Clone + PartialEq + Debug {}

impl<T: 'static + Clone + PartialEq + Debug> Scalar for T {}
