use itertools::{Itertools, izip};
use palette::Srgb;
use palette::num::One;
use palette::rgb::Rgb;
use ultraviolet::{Vec3, m32x4, m32x8};
use wide::{f32x4, f32x8};

pub(crate) trait SrgbColorConvertExt {
    /// Number of expected output elements
    /// If 0, indicates that number of output elements are not known at compile time
    const NUM_OUTPUT_VALUES: usize;

    type Output;
    type Mask;

    fn extract_values(self, mask: Option<Self::Mask>) -> Self::Output;
}

macro_rules! impl_srgb_color_convert {
    ($x: ident, $n: literal) => {
        impl SrgbColorConvertExt for Srgb<concat_idents!(f32, $x)> {
            const NUM_OUTPUT_VALUES: usize = $n;

            type Output = [Option<Srgb<f32>>; Self::NUM_OUTPUT_VALUES];
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

                    Some(Srgb::new(r, g, b))
                })
                .collect_array::<{ Self::NUM_OUTPUT_VALUES }>()
                .unwrap()
            }
        }
    };
}

impl_srgb_color_convert!(x8, 8);
impl_srgb_color_convert!(x4, 4);
