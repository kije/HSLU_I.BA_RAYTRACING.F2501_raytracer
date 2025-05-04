#![feature(generic_const_exprs)]
#![feature(concat_idents)]
#![feature(likely_unlikely)]
#![allow(incomplete_features)]

use raytracer::renderer::SCENE;
use raytracer::scene::Scene;
use raytracer::{
    WINDOW_HEIGHT, WINDOW_WIDTH,
    helpers::RenderTiming,
    image_buffer::ImageBuffer,
    output::{Output, OutputInteractive, WindowColorEncoder, WindowOutput},
    renderer::{RaytracerRenderer, Renderer},
};
use std::sync::Arc;
use std::thread;
use ultraviolet::Vec3;

fn main() {
    // Uncomment to run benchmarks
    // simd_polygon_triangulation_bench::run_triangulation_benchmarks();

    let scene = Scene::<Vec3>::from_obj("./data/obj/simple-test/cube-cubes.obj").unwrap();

    let buffer = Arc::new(ImageBuffer::<WINDOW_WIDTH, WINDOW_HEIGHT>::new());

    let buffer_render = buffer.clone();
    thread::spawn(move || {
        let mut start = RenderTiming::default();
        RaytracerRenderer::<WindowColorEncoder>::default().render(&buffer_render, &SCENE);
        start.next();
        println!("Render timing done! {:?}", start);
    });

    let mut output =
        WindowOutput::<WINDOW_WIDTH, WINDOW_HEIGHT>::new().expect("Unable to open output");

    output.render_loop(|output, _| {
        output.render_buffer(&buffer);
    });
}
