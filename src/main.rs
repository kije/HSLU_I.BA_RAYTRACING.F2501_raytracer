#![feature(generic_const_exprs)]
#![feature(concat_idents)]
#![feature(likely_unlikely)]
#![allow(incomplete_features)]

pub(crate) mod color;
pub(crate) mod extensions;
pub(crate) mod geometry;
pub(crate) mod helpers;
pub(crate) mod image_buffer;
pub(crate) mod math;
pub(crate) mod output;
pub(crate) mod random;
pub(crate) mod raytracing;
pub(crate) mod renderer;
pub(crate) mod scalar;
pub(crate) mod scene;
pub(crate) mod vector;

// New modules for trait simplification
pub(crate) mod color_traits;
pub(crate) mod scalar_traits;
pub(crate) mod simd_compat;
pub(crate) mod vector_traits;

use crate::helpers::RenderTiming;
use crate::image_buffer::ImageBuffer;
use crate::output::{Output, OutputInteractive, WindowColorEncoder, WindowOutput};
use crate::renderer::{RaytracerRenderer, Renderer};
use std::sync::Arc;
use std::thread;

const WINDOW_WIDTH: usize = if cfg!(feature = "high_resolution") {
    1620
} else {
    768
};
const WINDOW_HEIGHT: usize = if cfg!(feature = "high_resolution") {
    1280
} else {
    640
};

fn main() {
    let buffer = Arc::new(ImageBuffer::<WINDOW_WIDTH, WINDOW_HEIGHT>::new());

    let buffer_render = buffer.clone();
    thread::spawn(move || {
        let mut start = RenderTiming::default();
        RaytracerRenderer::<WindowColorEncoder>::default().render(&buffer_render);
        start.next();
        println!("Render timing done! {:?}", start);
    });

    let mut output =
        WindowOutput::<WINDOW_WIDTH, WINDOW_HEIGHT>::new().expect("Unable to open output");

    output.render_loop(|output, _| {
        output.render_buffer(&buffer);
    });
}
