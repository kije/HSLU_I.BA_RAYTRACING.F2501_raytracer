#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

mod renderer;
mod output;
mod image_buffer;
mod helpers;

use std::sync::{Arc};
use std::thread;
use crate::image_buffer::ImageBuffer;
use crate::output::{Output, OutputInteractive, WindowColorEncoder, WindowOutput};
use crate::renderer::{Renderer, TestRenderer3DSphereSW02};

const WINDOW_WIDTH: usize = 1000;
const WINDOW_HEIGHT: usize = 800;


fn main() {
    let buffer = Arc::new(ImageBuffer::<WINDOW_WIDTH, WINDOW_HEIGHT>::new());

    let buffer_render = buffer.clone();
    thread::spawn(move || {
        TestRenderer3DSphereSW02::<WindowColorEncoder>::default().render(&buffer_render);
    });

    
    let mut output = WindowOutput::<WINDOW_WIDTH, WINDOW_HEIGHT>::new().expect("Unable to open output");

    output.render_loop(|output,_| {
        output.render_buffer(&buffer);
    });
}
