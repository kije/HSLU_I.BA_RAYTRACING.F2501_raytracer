#![feature(generic_const_exprs)]
#![feature(concat_idents)]
#![feature(likely_unlikely)]
#![feature(path_file_prefix)]
#![allow(incomplete_features)]

use raytracer::helpers::ColorType;
use raytracer::scene::{PointLight, Scene};
use raytracer::{
    SCENE_DEPTH, SCENE_HEIGHT, SCENE_WIDTH, WINDOW_HEIGHT, WINDOW_WIDTH,
    helpers::RenderTiming,
    image_buffer::ImageBuffer,
    output::{Output, OutputInteractive, WindowColorEncoder, WindowOutput},
    renderer::{RaytracerRenderer, Renderer},
};
use std::sync::Arc;
use std::thread;
use ultraviolet::{Isometry3, Rotor3, Similarity3, Vec3};

fn main() {
    // Uncomment to run benchmarks
    // simd_polygon_triangulation_bench::run_triangulation_benchmarks();

    let mut scene = Scene::<Vec3>::from_obj::<_, true>(
        "../data/obj/text/text.obj",
        Some(Similarity3::new(
            Vec3::new(0.15, 0.0, 0.5),
            Rotor3::from_euler_angles(0.25, 0.2, 0.),
            1.05,
        )),
    )
    .unwrap();

    scene.add_light(
        PointLight::new(
            Vec3::new(SCENE_WIDTH / 2.0, SCENE_HEIGHT / 1.9, 0.015 * SCENE_DEPTH),
            ColorType::new(0.825, 0.675, 0.5),
            0.99,
        )
        .into(),
    );
    scene.add_light(
        PointLight::new(
            Vec3::new(SCENE_WIDTH / 2.0, SCENE_HEIGHT / 2.1, 0.85 * SCENE_DEPTH),
            ColorType::new(0.825, 0.275, 0.8),
            0.99,
        )
        .into(),
    );

    let buffer = Arc::new(ImageBuffer::<WINDOW_WIDTH, WINDOW_HEIGHT>::new());

    let buffer_render = buffer.clone();
    thread::spawn(move || {
        let mut start = RenderTiming::default();
        RaytracerRenderer::<WindowColorEncoder>::default().render(&buffer_render, &scene);
        start.next();
        println!("Render timing done! {:?}", start);
    });

    let mut output =
        WindowOutput::<WINDOW_WIDTH, WINDOW_HEIGHT>::new().expect("Unable to open output");

    output.render_loop(|output, _| {
        output.render_buffer(&buffer);
    });
}
