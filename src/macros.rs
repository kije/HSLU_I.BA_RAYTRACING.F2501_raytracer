/// Macro for implementing splatable functionality for light types
#[macro_export]
macro_rules! impl_splatable_for_light {
    ($light_type:ident <$vec_gen:ident>, { $($field:ident : $splat_method:expr),+ $(,)? }) => {
        impl<$vec_gen> crate::helpers::Splatable<$light_type<<$vec_gen as crate::vector::CommonVecOperationsSimdOperations>::SingleValueVector>>
            for $light_type<$vec_gen>
        where
            $vec_gen: crate::vector::Vector + crate::vector::CommonVecOperationsSimdOperations,
            $vec_gen::Scalar: Clone + simba::simd::SimdRealField
                + simba::scalar::SupersetOf<<<$vec_gen as crate::vector::CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar>
                + crate::helpers::Splatable<<<$vec_gen as crate::vector::CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar>,
            <<$vec_gen as crate::vector::CommonVecOperationsSimdOperations>::SingleValueVector as crate::vector::Vector>::Scalar:
                simba::scalar::SubsetOf<<$vec_gen as crate::vector::Vector>::Scalar>,
            <<$vec_gen as crate::vector::Vector>::Scalar as simba::simd::SimdValue>::Element:
                simba::scalar::SubsetOf<<$vec_gen as crate::vector::Vector>::Scalar>
        {
            fn splat(v: &$light_type<<$vec_gen as crate::vector::CommonVecOperationsSimdOperations>::SingleValueVector>) -> Self {
                Self {
                    $($field: $splat_method(&v.$field),)+
                }
            }
        }
    };
}

/// Macro for simplifying where clauses in light implementations
#[macro_export]
macro_rules! where_light_vector {
    () => {
        where
            Vector: crate::vector_traits::LightSourceVector,
            Vector::Scalar: crate::scalar_traits::LightScalar,
            crate::helpers::ColorType<Vector::Scalar>: crate::color_traits::LightCompatibleColor<Vector::Scalar>,
            [(); <Vector as crate::vector::Vector>::DIMENSIONS]:
    };
}
