#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

mod renderer;
mod output;
mod image_buffer;
mod helpers;

use std::time::{Duration, Instant};
use crate::image_buffer::ImageBuffer;
use crate::output::{Output, OutputInteractive, WindowColorEncoder, WindowOutput};
use crate::renderer::{Renderer, TestRenderer, TestRendererVector};

const WINDOW_WIDTH: usize = 1000;
const WINDOW_HEIGHT: usize = 800;


fn main() {
    let mut buffer = ImageBuffer::<WINDOW_WIDTH, WINDOW_HEIGHT>::new();
    TestRendererVector::<WindowColorEncoder>::default().render(&mut buffer);
    
    let mut output = WindowOutput::<WINDOW_WIDTH, WINDOW_HEIGHT>::new().expect("Unable to open output");

    output.render_static(|output,_| {
        output.render_buffer(&mut buffer);
    }, None);
}
