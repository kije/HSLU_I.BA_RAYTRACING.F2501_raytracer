use crate::image_buffer::ImageBuffer;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::IntoParallelIterator;
use std::sync::atomic::{AtomicU32, Ordering};

mod file;
mod window;
use crate::WINDOW_SCENE_DEPTH;
use crate::helpers::{Pixel, RenderTiming};
pub use file::{FileColorEncoder, FileOutput};
pub use window::{WindowColorEncoder, WindowOutput};

pub trait OutputColorEncoder {
    fn to_output(pixel: &Pixel) -> u32;
    fn from_output(pixel: u32) -> Pixel;
}

pub trait Output<const W: usize, const H: usize> {
    type ColorEncoder: OutputColorEncoder;

    fn render_buffer(&mut self, buffer: &ImageBuffer<W, H>)
    where
        [(); W * H]:;

    fn get_feature_string() -> String {
        format!(
            "{} | {} | {} | {} | {} ({}×{}×{}) | {}",
            if cfg!(feature = "simd_render") {
                "SIMD"
            } else {
                "Non-SIMD"
            },
            if cfg!(feature = "anti_aliasing") {
                format!(
                    "Antialiasing {} {}",
                    if cfg!(feature = "anti_aliasing_rotation_scale") {
                        "ROS_SCL"
                    } else {
                        ""
                    },
                    if cfg!(feature = "anti_aliasing_randomness") {
                        "RNG"
                    } else {
                        ""
                    }
                )
                .trim()
                .to_string()
            } else {
                "Non-Antialiasing".to_string()
            },
            if cfg!(feature = "reflections") || cfg!(feature = "refractions") {
                if !cfg!(feature = "refractions") {
                    "Reflections"
                } else {
                    "Reflections + Refractions"
                }
            } else {
                "Non-Realistic"
            },
            if cfg!(feature = "high_quality") {
                if cfg!(feature = "extreme_quality") {
                    "Extreme Quality"
                } else {
                    "High Quality"
                }
            } else {
                "Standard Quality"
            },
            if cfg!(feature = "high_resolution") {
                "High Resolution"
            } else if cfg!(feature = "medium_resolution") {
                "Medium Resolution"
            } else {
                "Small Resolution"
            },
            W,
            H,
            WINDOW_SCENE_DEPTH,
            if cfg!(feature = "backface_culling") {
                "Backface Culling"
            } else {
                "NO-OPT"
            }
        )
        .trim()
        .to_string()
    }
}

pub trait OutputInteractive<const W: usize, const H: usize> {
    type Output: Output<W, H>;

    fn render_loop<F: FnMut(&mut Self::Output, &RenderTiming)>(&mut self, cb: F);

    fn render_static<F: FnOnce(&mut Self::Output, &RenderTiming)>(
        &mut self,
        cb: F,
        timing: Option<RenderTiming>,
    );
}
