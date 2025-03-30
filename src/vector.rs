use crate::scalar::Scalar;
use simba::simd::{SimdValue, WideF32x4, WideF32x8};
use std::fmt::Debug;
use ultraviolet::{
    Bivec2, Bivec2x4, Bivec2x8, Bivec3, Bivec3x4, Bivec3x8, IVec2, IVec3, IVec4, Rotor2, Rotor2x4,
    Rotor2x8, Rotor3, Rotor3x4, Rotor3x8, UVec2, UVec3, UVec4, Vec2, Vec2x4, Vec2x8, Vec3, Vec3x4,
    Vec3x8, Vec4, Vec4x4, Vec4x8, m32x4, m32x8,
};
use wide::{f32x4, f32x8};

/// The basic scalar type
///
/// This does not make any assumption on the algebraic properties of `Self`.
pub trait Vector: Clone + PartialEq + Debug
where
    <Self::Scalar as SimdValue>::SimdBool: Debug,
{
    type Scalar: Scalar + SimdValue;
    type InnerScalar: Scalar;

    const LANES: usize = Self::Scalar::LANES;
    const DIMENSIONS: usize;
}

pub trait VectorAssociations: Vector {
    type Bivec;
    type Rotor;
}

pub trait CommonVecOperations: Vector {
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

pub trait CommonVecOperationsFloat: Vector {
    fn normalize(&mut self);

    #[must_use = "Did you mean to use `.normalize()` to normalize `self` in place?"]
    fn normalized(&self) -> Self {
        let mut r = self.clone();
        r.normalize();
        r
    }
}

pub trait CommonVecOperationsReflectable: Vector {
    fn reflect(&mut self, normal: Self);

    fn reflected(&self, normal: Self) -> Self;
}

pub trait CommonVecOperationsSimdOperations: Vector {
    type SingleValueVector: Vector;
    /// Blend two vectors together lanewise using `mask` as a mask.
    ///
    /// This is essentially a bitwise blend operation, such that any point where
    /// there is a 1 bit in `mask`, the output will put the bit from `tru`, while
    /// where there is a 0 bit in `mask`, the output will put the bit from `fals`
    fn blend(mask: <Self::Scalar as SimdValue>::SimdBool, tru: Self, fals: Self) -> Self;

    fn splat(vec: Self::SingleValueVector) -> Self;
}

pub trait CommonVecOperationsWithAssociations: VectorAssociations {
    fn rotate_by(&mut self, rotor: Self::Rotor);
    fn rotated_by(self, rotor: Self::Rotor) -> Self;
}

pub(crate) trait CheckVectorDimensionsMatch<const REQUIRED_DIMENSIONS: usize>:
    Vector
{
    const CHECK: ();
}

impl<const REQUIRED_DIMENSIONS: usize, T: Vector + ?Sized>
    CheckVectorDimensionsMatch<REQUIRED_DIMENSIONS> for T
{
    const CHECK: () = [()][(Self::DIMENSIONS == REQUIRED_DIMENSIONS) as usize];
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

macro_rules! impl_vector {
    (SIMD_OPS[$low_vec:ty, $mask_wide_type:ty, ($($component:ident),+)] ;; $vec:ident) => {
          impl crate::vector::CommonVecOperationsSimdOperations for $vec {
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
                        let splatted_comp = Self::Scalar::splat(vec.$component);
                        *unsafe { cast_simd_value::<Self::Scalar, Self::InnerScalar>(&splatted_comp) }
                    };
                )+
                Self {
                      $($component),+
                }
            }
        }
    };
    (MANUAL_SIMD_OPS[$low_vec:ty, $mask_wide_type:ty, ($($component:ident),+)] ;; $vec:ident) => {
          impl crate::vector::CommonVecOperationsSimdOperations for $vec {
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
                        let splatted_comp = Self::Scalar::splat(vec.$component);
                        *unsafe { cast_simd_value::<Self::Scalar, Self::InnerScalar>(&splatted_comp) }
                    };
                )+
                Self {
                      $($component),+
                }
            }
        }
    };
    (REFLECTABLE ;; $vec:ident) => {
         impl crate::vector::CommonVecOperationsReflectable for $vec {
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
    (FLOAT ;; $vec:ident) => {
          impl crate::vector::CommonVecOperationsFloat for $vec {
            #[inline(always)]
            fn normalize(&mut self) {
                $vec::normalize(self);
            }
        }
    };

    ($([$($vec:ident ( $scalar_type:ty = $inner_scalar:ty | $lanes: expr ) $(=>   $rotor_type:ty | $bivec_type:ty )? $({ $(($special_attr: tt $([$($special_attr_args: tt),+])?)),+})?  ),+] => $dims:expr),+) => {
        $(
            $(
                impl_vector!($vec, $scalar_type, $inner_scalar, $lanes, $dims $(, $bivec_type, $rotor_type)?);
                $(
                    $(impl_vector!($special_attr $([$($special_attr_args),+])? ;; $vec);)*
                )?
            )+
        )+
    };
    ($vec:ident , $scalar_type:ty, $inner_scalar:ty, $lanes: expr, $dims:expr $(, $bivec_type:ty, $rotor_type:ty )?) => {
        impl crate::vector::Vector for $vec {
            type Scalar = $scalar_type;
            type InnerScalar = $inner_scalar;

            const LANES: usize = $lanes;
            const DIMENSIONS: usize = $dims;
        }

        impl crate::vector::CommonVecOperations for $vec {
            #[inline(always)]
            fn broadcast(val: Self::Scalar) -> Self {
                Self::broadcast(
                    *unsafe { cast_simd_value::<$scalar_type, $inner_scalar>(&val) }
                )
            }

            #[inline(always)]
             fn dot(&self, other: Self) -> Self::Scalar {
                let dot = $vec::dot(
                    self,
                    other
                );

                *unsafe { cast_simd_value::<$inner_scalar, $scalar_type>(&dot) }
            }

            #[inline(always)]
             fn mag_sq(&self) -> Self::Scalar {
                let mag_sq = $vec::mag_sq(self);
                *unsafe { cast_simd_value::<$inner_scalar, $scalar_type>(&mag_sq) }
            }

            #[inline(always)]
            fn mag(&self) -> Self::Scalar {
                let mag = $vec::mag(self);
                *unsafe { cast_simd_value::<$inner_scalar, $scalar_type>(&mag) }
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
                    let val =  *unsafe { cast_simd_value::<$inner_scalar, $scalar_type>(&s) };

                    let result = f(val);

                    *unsafe { cast_simd_value::<$scalar_type, $inner_scalar>(&result) }
                })
            }

            #[inline(always)]
             fn apply<F>(&mut self, mut f: F) where F: FnMut(Self::Scalar ) -> Self::Scalar {
                $vec::apply(self, |s|{
                    let val =  *unsafe { cast_simd_value::<$inner_scalar, $scalar_type>(&s) };

                    let result = f(val);

                    *unsafe { cast_simd_value::<$scalar_type, $inner_scalar>(&result) }
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
                *unsafe { cast_simd_value::<$inner_scalar, $scalar_type>(&max) }
            }

            #[inline(always)]
            fn component_min(&self) -> Self::Scalar {
                let min = $vec::component_min(self);
                *unsafe { cast_simd_value::<$inner_scalar, $scalar_type>(&min) }
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
                let casted_random = (*unsafe { cast_simd_value::<[$scalar_type;$dims],[$inner_scalar;$dims]>(&random) });
                casted_random.into()
            }
        }

        impl crate::helpers::Splatable<$scalar_type> for $vec {
            #[inline(always)]
            fn splat(source: &$scalar_type) -> Self {
                crate::vector::CommonVecOperations::broadcast(*source)
            }
        }

        $(
            impl crate::vector::VectorAssociations for $vec {
                type Bivec = $bivec_type;
                type Rotor = $rotor_type;
            }

            impl crate::vector::CommonVecOperationsWithAssociations for $vec {
               fn rotate_by(&mut self, rotor: Self::Rotor) {
                   $vec::rotate_by(self, rotor);
               }
                fn rotated_by(self, rotor: Self::Rotor) -> Self {
                   $vec::rotated_by(self, rotor)
               }
            }
        )?
    };
}

impl_vector!(
    [
        Vec2 (f32 = f32 | 1) => Rotor2|Bivec2 {(FLOAT), (MANUAL_SIMD_OPS[Vec2, bool, (x,y)])}, IVec2 ( i32 = i32 | 1), UVec2 ( u32 = u32 | 1),
        Vec2x4 (WideF32x4 = f32x4 | 4) => Rotor2x4|Bivec2x4 {(FLOAT), (SIMD_OPS[Vec2, m32x4, (x,y)])},
        Vec2x8 (WideF32x8 = f32x8 | 8) => Rotor2x8|Bivec2x8 {(FLOAT), (SIMD_OPS[Vec2, m32x8, (x,y)])}
    ] => 2,
    [
        Vec3 (f32 = f32 | 1) => Rotor3|Bivec3 {(REFLECTABLE), (FLOAT), (MANUAL_SIMD_OPS[Vec3, bool, (x,y,z)])}, IVec3 ( i32 = i32 | 1) {(REFLECTABLE)}, UVec3 ( u32 = u32 | 1),
        Vec3x4 (WideF32x4 = f32x4 | 4) => Rotor3x4|Bivec3x4 {(REFLECTABLE), (FLOAT), (SIMD_OPS[Vec3, m32x4, (x,y,z)])},
        Vec3x8 (WideF32x8 = f32x8 | 8) => Rotor3x8|Bivec3x8 {(REFLECTABLE), (FLOAT), (SIMD_OPS[Vec3, m32x8, (x,y,z)])}
    ] => 3,
    [
        Vec4 (f32 = f32 | 1) {(REFLECTABLE), (FLOAT), (MANUAL_SIMD_OPS[Vec4, bool, (x,y,z,w)])}, IVec4 ( i32 = i32 | 1) {(REFLECTABLE)}, UVec4 ( u32 = u32 | 1),
        Vec4x4 (WideF32x4 = f32x4 | 4) {(REFLECTABLE), (FLOAT), (SIMD_OPS[Vec4, m32x4, (x,y,z,w)])},
        Vec4x8 (WideF32x8 = f32x8 | 8) {(REFLECTABLE), (FLOAT), (SIMD_OPS[Vec4, m32x8, (x,y,z,w)])}
    ] => 4
);

pub(crate) trait VectorAware<Vector>
where
    Vector: self::Vector,
{
    const LANES: usize = Vector::LANES;
}

// impl vector-aware trait for the vectors itslef
impl<Vector> VectorAware<Vector> for Vector where Vector: self::Vector {}
