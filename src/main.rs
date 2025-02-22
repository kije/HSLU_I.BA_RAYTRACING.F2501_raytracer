#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

mod renderer;
mod output;
mod image_buffer;
mod helpers;

use crate::image_buffer::ImageBuffer;
use crate::output::{Output, OutputInteractive, WindowOutput};
use crate::renderer::{Renderer, TestRenderer};

const WINDOW_WIDTH: usize = 800;
const WINDOW_HEIGHT: usize = 600;


fn main() {
    let mut buffer = ImageBuffer::<WINDOW_WIDTH, WINDOW_HEIGHT>::new();
    TestRenderer.render(&mut buffer);
    
    let mut output = WindowOutput::<WINDOW_WIDTH, WINDOW_HEIGHT>::new().expect("Unable to open output");

    output.render_static(|output| {
        output.render_buffer(&mut buffer);
    });
}
