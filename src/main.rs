#![feature(generic_const_exprs)]
#![feature(concat_idents)]
#![allow(incomplete_features)]

mod geometry;
mod helpers;
mod image_buffer;
mod math;
mod output;
mod renderer;

use crate::helpers::RenderTiming;
use crate::image_buffer::ImageBuffer;
use crate::output::{Output, OutputInteractive, WindowColorEncoder, WindowOutput};
use crate::renderer::{Renderer, TestRenderer3DSphereSW02};
use std::sync::Arc;
use std::thread;

const WINDOW_WIDTH: usize = if cfg!(feature = "high_resolution") {
    1600
} else {
    800
};
const WINDOW_HEIGHT: usize = if cfg!(feature = "high_resolution") {
    1200
} else {
    600
};

fn main() {
    let buffer = Arc::new(ImageBuffer::<WINDOW_WIDTH, WINDOW_HEIGHT>::new());

    let buffer_render = buffer.clone();
    thread::spawn(move || {
        let mut start = RenderTiming::default();
        TestRenderer3DSphereSW02::<WindowColorEncoder>::default().render(&buffer_render);
        start.next();
        println!("Render timing done! {:?}", start);
    });

    let mut output =
        WindowOutput::<WINDOW_WIDTH, WINDOW_HEIGHT>::new().expect("Unable to open output");

    output.render_loop(|output, _| {
        output.render_buffer(&buffer);
    });
}
