use crate::vector::Vector;
use itertools::Itertools;
use std::fmt::Debug;
use std::ops::{Add, Mul};
use tt_call::tt_if;
use tt_equal::tt_equal;

use simba::simd::{WideF32x4, WideF32x8};
use ultraviolet::{
    Mat2, Mat2x4, Mat2x8, Mat3, Mat3x4, Mat3x8, Mat4, Mat4x4, Mat4x8, Vec2, Vec2x4, Vec2x8, Vec3,
    Vec3x4, Vec3x8, Vec4, Vec4x4, Vec4x8,
};
use wide::{f32x4, f32x8};

pub trait Matrix:
    Clone
    + PartialEq
    + Debug
    + Mul<Self, Output = Self>
    + Add<Self, Output = Self>
    + Mul<Self::Vector, Output = Self::Vector>
    + Mul<<Self::Vector as Vector>::InnerScalar, Output = Self>
{
    type Vector: Vector;

    // /// Vector type with one lower dimension than Self::Vector
    // type LowerDimVector: Vector<Scalar = <Self::Vector as Vector>::Scalar>;

    const LANES: usize = <Self::Vector as Vector>::LANES;
    const DIMENSIONS: usize = <Self::Vector as Vector>::DIMENSIONS;
    const COLUMNS: usize = <Self as Matrix>::DIMENSIONS;
    const ROWS: usize = <Self as Matrix>::DIMENSIONS;
}

pub trait MatrixOperations: Matrix {
    fn identity() -> Self;

    fn determinant(&self) -> <Self::Vector as Vector>::Scalar;
    fn adjugate(&self) -> Self;

    fn inverse(&mut self);
    fn inversed(&self) -> Self;
    fn transpose(&mut self);
    fn transposed(&self) -> Self;

    // fn from_scale(scale: <Self::Vector as Vector>::Scalar) -> Self;
    //
    // fn from_nonuniform_scale(scale: Self::Vector) -> Self;

    //
    // fn from_euler_angles(
    //     roll: <Self::Vector as Vector>::Scalar,
    //     pitch: <Self::Vector as Vector>::Scalar,
    //     yaw: <Self::Vector as Vector>::Scalar,
    // ) -> Self;
    // fn from_rotation_x(angle: <Self::Vector as Vector>::Scalar) -> Self;
    //
    // fn from_rotation_y(angle: <Self::Vector as Vector>::Scalar) -> Self;
    // fn from_rotation_z(angle: <Self::Vector as Vector>::Scalar) -> Self;
    // fn from_rotation_around(axis: Self::Vector, angle: <Self::Vector as Vector>::Scalar) -> Self;
}

pub trait MatrixFixedDimensions<const DIMENSIONS: usize>: Matrix {
    const DIMENSIONS: usize = DIMENSIONS;

    fn from_columns(columns: [Self::Vector; DIMENSIONS]) -> Self;
}

#[inline(always)]
unsafe fn cast_simd_value<Wrapper, Inner>(value: &Wrapper) -> &Inner {
    let x = value as *const Wrapper as *const Inner;
    unsafe { &*(x) }
}

macro_rules! impl_matrix {
    ($([$($mat:ident ($vector_type:ty | $scalar_type:ty = $inner_scalar:ty | $lanes: literal )),+] => [$dims:literal,$cols:literal := $components:tt]),+) => {
        $(
            $(
                impl_matrix!($mat, $vector_type, $scalar_type, $inner_scalar, $lanes, $dims, $cols, $components);
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
    ($mat:ident, $vector_type:ty, $scalar_type:ty, $inner_scalar:ty, $lanes: literal, $dims:literal, $cols:literal, ($($component:ident),+)) => {
        impl crate::matrix::Matrix for $mat {
            type Vector = $vector_type;

            const LANES: usize = $lanes;
            const DIMENSIONS: usize = $dims;
            const COLUMNS: usize = $cols;
        }

        impl crate::matrix::MatrixOperations for $mat {
            #[inline(always)]
            fn identity() -> Self {
                $mat::identity()
            }

            #[inline(always)]
            fn determinant(&self) -> <Self::Vector as Vector>::Scalar {
                let det = $mat::determinant(self);
                impl_matrix!(@cast_simd_value $inner_scalar,  $scalar_type, det)
            }

            #[inline(always)]
            fn adjugate(&self) -> Self {
                $mat::adjugate(self)
            }

            #[inline(always)]
            fn inverse(&mut self) {
                $mat::inverse(self);
            }

            #[inline(always)]
            fn inversed(&self) -> Self {
                $mat::inversed(self)
            }

            #[inline(always)]
            fn transpose(&mut self) {
                $mat::transpose(self);
            }

            #[inline(always)]
            fn transposed(&self) -> Self {
                $mat::transposed(self)
            }
        }

        impl crate::matrix::MatrixFixedDimensions<$dims> for $mat {
           const DIMENSIONS: usize = $dims;

           #[inline(always)]
           fn from_columns(columns: [Self::Vector; $cols]) -> Self {
                Self::from(
                    (0..$cols)
                    .map(|i| [$(columns[i].$component),*])
                    .collect_array::<$cols>().expect("Matrix dimensions mismatch.")
                )
           }
       }
    };
}

impl_matrix!(
  [
        Mat2 (Vec2 | f32 = f32 | 1),
        Mat2x4 (Vec2x4 | WideF32x4 = f32x4 | 4),
        Mat2x8 (Vec2x8 | WideF32x8 = f32x8 | 8)
    ] => [2,2 := (x,y)],
    [
        Mat3 (Vec3 | f32 = f32 | 1),
        Mat3x4 (Vec3x4 | WideF32x4 = f32x4 | 4),
        Mat3x8 (Vec3x8 | WideF32x8 = f32x8 | 8)
    ] => [3,3 := (x,y,z)],
    [
        Mat4 (Vec4 | f32 = f32 | 1),
        Mat4x4 (Vec4x4 | WideF32x4 = f32x4 | 4),
        Mat4x8 (Vec4x8 | WideF32x8 = f32x8 | 8)
    ] => [4,4 := (x,y,z,w)]
);
