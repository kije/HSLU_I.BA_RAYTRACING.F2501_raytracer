use crate::helpers::ColorType;
use itertools::{Itertools, izip};
use palette::num::One;
use simba::simd::{WideBoolF32x4, WideBoolF32x8, WideF32x4, WideF32x8};
use ultraviolet::{m32x4, m32x8};
use wide::{f32x4, f32x8};

pub trait SrgbColorConvertExt {
    /// Number of expected output elements
    /// If 0, indicates that number of output elements are not known at compile time
    const NUM_OUTPUT_VALUES: usize;

    type Output;
    type Mask;

    fn extract_values(self, mask: Option<Self::Mask>) -> Self::Output;
}

macro_rules! impl_srgb_color_convert {
    ($x: ident, $n: literal) => {
        impl SrgbColorConvertExt for ColorType<concat_idents!(f32, $x)> {
            const NUM_OUTPUT_VALUES: usize = $n;

            type Output = [Option<ColorType<f32>>; Self::NUM_OUTPUT_VALUES];
            type Mask = concat_idents!(m32, $x);

            fn extract_values(self, mask: Option<Self::Mask>) -> Self::Output {
                let mask = mask.unwrap_or_else(<concat_idents!(m32, $x)>::one);

                izip!(
                    self.red.as_array_ref(),
                    self.green.as_array_ref(),
                    self.blue.as_array_ref(),
                    mask.as_array_ref(),
                )
                .map(|(&r, &g, &b, &mask)| {
                    if mask == 0.0 {
                        return None;
                    }

                    Some(ColorType::new(r, g, b))
                })
                .collect_array::<{ Self::NUM_OUTPUT_VALUES }>()
                .unwrap()
            }
        }

        impl SrgbColorConvertExt for ColorType<concat_idents!(WideF32, $x)> {
            const NUM_OUTPUT_VALUES: usize = $n;

            type Output = [Option<ColorType<f32>>; Self::NUM_OUTPUT_VALUES];
            type Mask = concat_idents!(WideBoolF32, $x);

            fn extract_values(self, mask: Option<Self::Mask>) -> Self::Output {
                let mask = mask.unwrap_or_else(|| {
                    <concat_idents!(WideBoolF32, $x)>::from([true; Self::NUM_OUTPUT_VALUES])
                });

                izip!(
                    self.red.0.as_array_ref(),
                    self.green.0.as_array_ref(),
                    self.blue.0.as_array_ref(),
                    mask.0.as_array_ref(),
                )
                .map(|(&r, &g, &b, &mask)| {
                    if mask == 0.0 {
                        return None;
                    }

                    Some(ColorType::new(r, g, b))
                })
                .collect_array::<{ Self::NUM_OUTPUT_VALUES }>()
                .unwrap()
            }
        }
    };
}

impl_srgb_color_convert!(x8, 8);
impl_srgb_color_convert!(x4, 4);
