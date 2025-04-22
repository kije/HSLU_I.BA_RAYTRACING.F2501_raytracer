use crate::helpers::Splatable;
use crate::matrix::Matrix;
use crate::scalar::Scalar;
use crate::simd_compat::SimdValueSimplified;
use simba::scalar::SupersetOf;
use simba::simd::{SimdValue, WideF32x4, WideF32x8};
use std::fmt::Debug;
use tt_call::tt_if;
use tt_equal::tt_equal;
use ultraviolet::{
    Bivec2, Bivec2x4, Bivec2x8, Bivec3, Bivec3x4, Bivec3x8, IVec2, IVec3, IVec4, Mat2, Mat2x4,
    Mat2x8, Mat3, Mat3x4, Mat3x8, Rotor2, Rotor2x4, Rotor2x8, Rotor3, Rotor3x4, Rotor3x8, UVec2,
    UVec3, UVec4, Vec2, Vec2x4, Vec2x8, Vec3, Vec3x4, Vec3x8, Vec4, Vec4x4, Vec4x8, m32x4, m32x8,
};
use wide::{f32x4, f32x8};

/// The basic scalar type
///
/// This does not make any assumption on the algebraic properties of `Self`.
pub trait Vector: Clone + PartialEq + Debug {
    type Scalar: SimdValueSimplified;
    type InnerScalar: Scalar;

    const LANES: usize = Self::Scalar::LANES;
    const DIMENSIONS: usize;
}

pub trait VectorAssociations: Vector {
    type Bivec;
    type Rotor;

    type Matrix: Matrix<Vector = Self>;
}

pub trait VectorOperations: Vector {
    fn broadcast(val: Self::Scalar) -> Self;

    fn dot(&self, other: Self) -> Self::Scalar;
    fn mag_sq(&self) -> Self::Scalar;

    fn mag(&self) -> Self::Scalar;

    fn mul_add(&self, mul: Self, add: Self) -> Self;

    fn abs(&self) -> Self;

    fn clamp(&mut self, min: Self, max: Self);

    fn clamped(mut self, min: Self, max: Self) -> Self {
        self.clamp(min, max);
        self
    }

    fn map<F>(&self, f: F) -> Self
    where
        F: FnMut(Self::Scalar) -> Self::Scalar;

    fn apply<F>(&mut self, f: F)
    where
        F: FnMut(Self::Scalar) -> Self::Scalar;

    fn max_by_component(self, other: Self) -> Self;

    fn min_by_component(self, other: Self) -> Self;
    fn component_max(&self) -> Self::Scalar;

    fn component_min(&self) -> Self::Scalar;

    fn zero() -> Self;

    fn one() -> Self;

    fn sample_random() -> Self
    where
        rand::distributions::Standard: rand::distributions::Distribution<Self::Scalar>;
}

pub trait Vector3DOperations: VectorAssociations {
    fn cross(&self, other: Self) -> Self;
    fn wedge(&self, other: Self) -> Self::Bivec;
    fn geom(&self, other: Self) -> Self::Rotor;
}

pub trait VectorAccessorX: Vector {
    fn x(&self) -> Self::Scalar;

    fn unit_x() -> Self;
}

pub trait VectorAccessorY: Vector {
    fn y(&self) -> Self::Scalar;

    fn unit_y() -> Self;
}

pub trait VectorAccessorZ: Vector {
    fn z(&self) -> Self::Scalar;

    fn unit_z() -> Self;
}

pub trait Vector3DAccessor: VectorAccessorX + VectorAccessorY + VectorAccessorZ {}

impl<T> Vector3DAccessor for T where T: VectorAccessorX + VectorAccessorY + VectorAccessorZ {}

pub trait NormalizableVector: Vector {
    fn normalize(&mut self);

    #[must_use = "Did you mean to use `.normalize()` to normalize `self` in place?"]
    fn normalized(&self) -> Self {
        let mut r = self.clone();
        r.normalize();
        r
    }
}

pub trait ReflectableVector: Vector {
    fn reflect(&mut self, normal: Self);

    fn reflected(&self, normal: Self) -> Self;
}

pub trait SimdCapableVector:
    Vector<
    Scalar: SupersetOf<<<Self as SimdCapableVector>::SingleValueVector as Vector>::Scalar>
                + Splatable<<<Self as SimdCapableVector>::SingleValueVector as Vector>::Scalar>,
>
{
    type SingleValueVector: Vector;
    /// Blend two vectors together lanewise using `mask` as a mask.
    ///
    /// This is essentially a bitwise blend operation, such that any point where
    /// there is a 1 bit in `mask`, the output will put the bit from `tru`, while
    /// where there is a 0 bit in `mask`, the output will put the bit from `fals`
    fn blend(mask: <Self::Scalar as SimdValue>::SimdBool, tru: Self, fals: Self) -> Self;

    fn splat(vec: Self::SingleValueVector) -> Self;
}

pub type SingleValueVectorScalar<V> =
    <<V as SimdCapableVector>::SingleValueVector as Vector>::Scalar;

pub trait RotatableVector: VectorAssociations {
    fn rotate_by(&mut self, rotor: Self::Rotor);
    fn rotated_by(self, rotor: Self::Rotor) -> Self;
}

pub trait CheckVectorDimensionsMatch<const REQUIRED_DIMENSIONS: usize>: Vector {
    const CHECK: ();
}

impl<const REQUIRED_DIMENSIONS: usize, T: Vector + ?Sized>
    CheckVectorDimensionsMatch<REQUIRED_DIMENSIONS> for T
{
    const CHECK: () = [()][(Self::DIMENSIONS != REQUIRED_DIMENSIONS) as usize];
}

pub trait VectorFixedDimensions<const DIMENSIONS: usize>: Vector {
    const DIMENSIONS: usize = DIMENSIONS;

    fn from_components(components: [Self::Scalar; DIMENSIONS]) -> Self;
}

pub trait VectorFixedLanes<const LANES: usize>: Vector {
    const LANES: usize = LANES;
}

#[inline(always)]
unsafe fn cast_simd_value<Wrapper, Inner>(value: &Wrapper) -> &Inner {
    let x = value as *const Wrapper as *const Inner;
    unsafe { &*(x) }
}

#[cfg(test)]
mod test_cast_simd_value {
    use super::*;
    use num_traits::Zero;

    #[test]
    fn test_cast_simd_value() {
        let a: f32 = 1.0;
        let b: WideF32x4 = WideF32x4::zero();

        let x = unsafe { cast_simd_value::<f32, f32>(&a) };
        let y = unsafe { cast_simd_value::<WideF32x4, f32x4>(&b) };

        let u = unsafe { cast_simd_value::<f32, f32>(x) };
        let v = unsafe { cast_simd_value::<f32x4, WideF32x4>(y) };

        assert_eq!(b.0, *y);
        assert_eq!(a, *x);
        assert_eq!(*u, a);
        assert_eq!(*v, b);
    }
}

macro_rules! if_three {
    ($n: literal => $($tts:tt)*) => {
        tt_if!{
            condition = [{tt_equal}]
            input = [{ $n 3 }]         // The two identifiers are here passed to 'tt_equal'
            true = [{
                $($tts)*
            }]
            false = [{

            }]
        }
    };
}

macro_rules! impl_vector {
    (SIMD_OPS[$low_vec:ty, $mask_wide_type:ty$(,)?] ;; $vec:ident ;; ($($component:ident),+)) => {
          impl crate::vector::SimdCapableVector for $vec {
            type SingleValueVector = $low_vec;

             #[inline(always)]
            fn blend(mask: <Self::Scalar as SimdValue>::SimdBool, tru: Self, fals: Self) -> Self {
                let mask = *unsafe { cast_simd_value::<<Self::Scalar as SimdValue>::SimdBool, $mask_wide_type>(&mask) };
                $vec::blend(mask, tru, fals)
            }

            #[inline(always)]
              fn splat(vec: Self::SingleValueVector) -> Self {
                $(
                    let $component = {
                        let splatted_comp = <Self::Scalar as crate::helpers::Splatable<_>>::splat(&vec.$component);
                        *unsafe { cast_simd_value::<Self::Scalar, Self::InnerScalar>(&splatted_comp) }
                    };
                )+
                Self {
                      $($component),+
                }
            }
        }
    };
    (MANUAL_SIMD_OPS[$low_vec:ty, $mask_wide_type:ty$(,)?] ;; $vec:ident ;; ($($component:ident),+)) => {
          impl crate::vector::SimdCapableVector for $vec {
            type SingleValueVector = $low_vec;

            #[inline(always)]
            fn blend(mask: <Self::Scalar as SimdValue>::SimdBool, tru: Self, fals: Self) -> Self {
                $(
                let $component = {
                    let trucomp = *unsafe { cast_simd_value::<Self::InnerScalar, Self::Scalar>(&tru.$component) };
                    let falscomp = *unsafe { cast_simd_value::<Self::InnerScalar, Self::Scalar>(&fals.$component) };
                    let rescomp = trucomp.select(mask, falscomp);
                    *unsafe { cast_simd_value::<Self::Scalar, Self::InnerScalar>(&rescomp) }
                }; )+

                Self {
                    $($component),+
                }
            }

              #[inline(always)]
              fn splat(vec: Self::SingleValueVector) -> Self {
                $(
                    let $component = {
                        let splatted_comp = <Self::Scalar as crate::helpers::Splatable<_>>::splat(&vec.$component);
                        *unsafe { cast_simd_value::<Self::Scalar, Self::InnerScalar>(&splatted_comp) }
                    };
                )+
                Self {
                      $($component),+
                }
            }
        }
    };
    (REFLECTABLE ;; $vec:ident ;; ($($component:ident),+)) => {
         impl crate::vector::ReflectableVector for $vec {
            #[inline(always)]
            fn reflect(&mut self, normal: Self){
                $vec::reflect(self, normal);
            }

            #[inline(always)]
             fn reflected(&self, normal: Self) -> Self {
                $vec::reflected(self, normal)
            }
        }
    };
    (FLOAT ;; $vec:ident ;; ($($component:ident),+)) => {
          impl crate::vector::NormalizableVector for $vec {
            #[inline(always)]
            fn normalize(&mut self) {
                $vec::normalize(self);
            }
        }
    };

    ($([$($vec:ident ( $scalar_type:ty = $inner_scalar:ty | $lanes: literal ) $(=>   $rotor_type:ty | $bivec_type:ty | $matrix_type:ty )? $({ $(($special_attr: tt $([$($special_attr_args: tt),+])?)),+})?  ),+] => [$dims:literal := $components:tt]),+) => {
        $(
            $(
                impl_vector!($vec, $scalar_type, $inner_scalar, $lanes, $dims, $components $(, $bivec_type, $rotor_type, $matrix_type)?);
                $(
                    $(impl_vector!($special_attr $([$($special_attr_args),+])? ;; $vec ;; $components);)*
                )?
            )+
        )+
    };
    (@cast_simd_value $scalar_type:ty, $inner_scalar:ty, $var:ident) => {
        tt_if!{
            condition = [{tt_equal}]
            input = [{ $scalar_type $inner_scalar }]
            true = [{
                $var
            }]
            false = [{
                *unsafe { cast_simd_value::<$scalar_type, $inner_scalar>(&$var) }
            }]
        }
    };
    ($vec:ident , $scalar_type:ty, $inner_scalar:ty, $lanes: literal, $dims:literal, ($($component:ident),+) $(, $bivec_type:ty, $rotor_type:ty, $matrix_type:ty )?) => {
        impl crate::vector::Vector for $vec {
            type Scalar = $scalar_type;
            type InnerScalar = $inner_scalar;

            const LANES: usize = $lanes;
            const DIMENSIONS: usize = $dims;
        }

        impl crate::vector::VectorOperations for $vec {
            #[inline(always)]
            fn broadcast(val: Self::Scalar) -> Self {
                Self::broadcast(
                    impl_vector!(@cast_simd_value $scalar_type, $inner_scalar, val)
                )
            }

            #[inline(always)]
             fn dot(&self, other: Self) -> Self::Scalar {
                let dot = $vec::dot(
                    self,
                    other
                );

                impl_vector!(@cast_simd_value $inner_scalar,  $scalar_type, dot)
            }

            #[inline(always)]
             fn mag_sq(&self) -> Self::Scalar {
                let mag_sq = $vec::mag_sq(self);

                impl_vector!(@cast_simd_value $inner_scalar,  $scalar_type, mag_sq)
            }

            #[inline(always)]
            fn mag(&self) -> Self::Scalar {
                let mag = $vec::mag(self);

                impl_vector!(@cast_simd_value $inner_scalar,  $scalar_type, mag)
            }


            #[inline(always)]
            fn mul_add(&self, mul: Self, add: Self) -> Self {
                $vec::mul_add(self, mul, add)
            }

            #[inline(always)]
            fn abs(&self) -> Self {
                $vec::abs(self)
            }

            #[inline(always)]
             fn clamp(&mut self, min: Self, max: Self) {
                $vec::clamp(self, min, max);
            }

            #[inline(always)]
            fn map<F>(&self, mut f: F) -> Self where F: FnMut(Self::Scalar) -> Self::Scalar {
                $vec::map(self, |s|{
                    let val =  impl_vector!(@cast_simd_value $inner_scalar,  $scalar_type, s);

                    let result = f(val);

                    impl_vector!(@cast_simd_value $scalar_type,  $inner_scalar, result)
                })
            }

            #[inline(always)]
             fn apply<F>(&mut self, mut f: F) where F: FnMut(Self::Scalar ) -> Self::Scalar {
                $vec::apply(self, |s|{
                    let val =  impl_vector!(@cast_simd_value $inner_scalar,  $scalar_type, s);

                    let result = f(val);

                    impl_vector!(@cast_simd_value $scalar_type,  $inner_scalar, result)
                });
            }

            #[inline(always)]
            fn max_by_component(self, other: Self) -> Self {
                $vec::max_by_component(self, other)
            }

            #[inline(always)]
            fn min_by_component(self, other: Self) -> Self {
                $vec::min_by_component(self, other)
            }

            #[inline(always)]
            fn component_max(&self) -> Self::Scalar {
                let max = $vec::component_max(self);
                impl_vector!(@cast_simd_value $inner_scalar,  $scalar_type, max)
            }

            #[inline(always)]
            fn component_min(&self) -> Self::Scalar {
                let min = $vec::component_min(self);
                impl_vector!(@cast_simd_value $inner_scalar,  $scalar_type, min)
            }

            #[inline(always)]
            fn zero() -> Self {
                Self::zero()
            }

            #[inline(always)]
            fn one() -> Self {
                Self::one()
            }

            fn sample_random() -> Self where rand::distributions::Standard: rand::distributions::Distribution<Self::Scalar> {
                use rand::Rng;
                let mut rng = crate::random::pseudo_rng();
                let random = rng.r#gen::<[$scalar_type; $dims]>();
                let casted_random = impl_vector!(@cast_simd_value [$scalar_type;$dims], [$inner_scalar;$dims], random);
                casted_random.into()
            }
        }

        impl crate::helpers::Splatable<$scalar_type> for $vec {
            #[inline(always)]
            fn splat(source: &$scalar_type) -> Self {
                crate::vector::VectorOperations::broadcast(*source)
            }
        }

        impl crate::vector::VectorFixedDimensions<$dims> for $vec {
            #[inline(always)]
            fn from_components(components: [Self::Scalar; $dims]) -> Self {
                Self::from(
                    impl_vector!(@cast_simd_value [$scalar_type;$dims], [$inner_scalar;$dims], components)
                )
            }
        }
        impl crate::vector::VectorFixedLanes<$lanes> for $vec {

        }

        $(
            tt_if!{
                condition = [{tt_equal}]
                input = [{ $component x }]
                true = [{
                    impl crate::vector::VectorAccessorX for $vec {
                        #[inline(always)]
                        fn x(&self) -> Self::Scalar {
                            let x = self.x;
                            impl_vector!(@cast_simd_value $inner_scalar,  $scalar_type, x)
                        }

                        #[inline(always)]
                        fn unit_x() -> Self {
                            $vec::unit_x()
                        }
                    }
                }]
                false = [{

                }]
            }

            tt_if!{
                condition = [{tt_equal}]
                input = [{ $component y }]
                true = [{
                    impl crate::vector::VectorAccessorY for $vec {
                        #[inline(always)]
                        fn y(&self) -> Self::Scalar {
                            let y = self.y;
                            impl_vector!(@cast_simd_value $inner_scalar,  $scalar_type, y)
                        }

                        #[inline(always)]
                        fn unit_y() -> Self {
                            $vec::unit_y()
                        }
                    }
                }]
                false = [{

                }]
            }

            tt_if!{
                condition = [{tt_equal}]
                input = [{ $component z }]
                true = [{
                    impl crate::vector::VectorAccessorZ for $vec {
                        #[inline(always)]
                        fn z(&self) -> Self::Scalar {
                            let z = self.z;
                            impl_vector!(@cast_simd_value $inner_scalar,  $scalar_type, z)
                        }

                        #[inline(always)]
                        fn unit_z() -> Self {
                            $vec::unit_z()
                        }
                    }
                }]
                false = [{

                }]
            }
        )*

        $(
            impl crate::vector::VectorAssociations for $vec {
                type Bivec = $bivec_type;
                type Rotor = $rotor_type;
                type Matrix = $matrix_type;
            }

            impl crate::vector::RotatableVector for $vec {
               fn rotate_by(&mut self, rotor: Self::Rotor) {
                   $vec::rotate_by(self, rotor);
               }
                fn rotated_by(self, rotor: Self::Rotor) -> Self {
                   $vec::rotated_by(self, rotor)
               }
            }

            if_three!($dims =>
                impl crate::vector::Vector3DOperations for $vec {
                    fn cross(&self, other: Self) -> Self {
                         $vec::cross(
                            self,
                            other
                        )
                    }
                    fn wedge(&self, other: Self) -> Self::Bivec {
                         $vec::wedge(
                            self,
                            other
                        )
                    }
                    fn geom(&self, other: Self) -> Self::Rotor {
                        $vec::geom(
                            self,
                            other
                        )
                    }
                }
            );
        )?
    };
}

impl_vector!(
    [
        Vec2 (f32 = f32 | 1) => Rotor2|Bivec2|Mat2 {(FLOAT), (MANUAL_SIMD_OPS[Vec2, bool])}, IVec2 ( i32 = i32 | 1), UVec2 ( u32 = u32 | 1),
        Vec2x4 (WideF32x4 = f32x4 | 4) => Rotor2x4|Bivec2x4|Mat2x4 {(FLOAT), (SIMD_OPS[Vec2, m32x4])},
        Vec2x8 (WideF32x8 = f32x8 | 8) => Rotor2x8|Bivec2x8|Mat2x8 {(FLOAT), (SIMD_OPS[Vec2, m32x8])}
    ] => [2 := (x,y)],
    [
        Vec3 (f32 = f32 | 1) => Rotor3|Bivec3|Mat3 {(REFLECTABLE), (FLOAT), (MANUAL_SIMD_OPS[Vec3, bool])}, IVec3 ( i32 = i32 | 1) {(REFLECTABLE)}, UVec3 ( u32 = u32 | 1),
        Vec3x4 (WideF32x4 = f32x4 | 4) => Rotor3x4|Bivec3x4|Mat3x4 {(REFLECTABLE), (FLOAT), (SIMD_OPS[Vec3, m32x4])},
        Vec3x8 (WideF32x8 = f32x8 | 8) => Rotor3x8|Bivec3x8|Mat3x8 {(REFLECTABLE), (FLOAT), (SIMD_OPS[Vec3, m32x8])}
    ] => [3 := (x,y,z)],
    [
        Vec4 (f32 = f32 | 1) {(REFLECTABLE), (FLOAT), (MANUAL_SIMD_OPS[Vec4, bool])}, IVec4 ( i32 = i32 | 1) {(REFLECTABLE)}, UVec4 ( u32 = u32 | 1),
        Vec4x4 (WideF32x4 = f32x4 | 4) {(REFLECTABLE), (FLOAT), (SIMD_OPS[Vec4, m32x4])},
        Vec4x8 (WideF32x8 = f32x8 | 8) {(REFLECTABLE), (FLOAT), (SIMD_OPS[Vec4, m32x8])}
    ] => [4 := (x,y,z,w)]
);

pub trait VectorAware<Vector>
where
    Vector: self::Vector,
{
    const LANES: usize = Vector::LANES;
}

// impl vector-aware trait for the vectors itslef
impl<Vector> VectorAware<Vector> for Vector where Vector: self::Vector {}
