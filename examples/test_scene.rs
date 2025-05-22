#![feature(generic_const_exprs)]
#![feature(concat_idents)]
#![feature(likely_unlikely)]
#![feature(path_file_prefix)]
#![allow(incomplete_features)]

use raytracer::geometry::{BoundedPlane, SphereData, TriangleData};
use raytracer::helpers::ColorType;
use raytracer::raytracing::{Material, TransmissionProperties};
use raytracer::scene::{PointLight, Scene};
use raytracer::{
    SCENE_DEPTH, SCENE_HEIGHT, SCENE_WIDTH, WINDOW_HEIGHT, WINDOW_WIDTH,
    helpers::RenderTiming,
    image_buffer::ImageBuffer,
    output::{Output, OutputInteractive, WindowColorEncoder, WindowOutput},
    renderer::{RaytracerRenderer, Renderer},
};
use std::sync::{Arc, LazyLock};
use std::thread;
use ultraviolet::{Rotor3, Vec3};

pub static SCENE: LazyLock<Scene<Vec3>> = LazyLock::new(|| {
    let mut scene = Scene::<Vec3>::with_capacities(20);

    // Add spheres
    scene.add_sphere(SphereData::new(
        Vec3::new(SCENE_WIDTH / 2.5, SCENE_HEIGHT / 2.75, 0.170 * SCENE_DEPTH),
        0.070 * SCENE_DEPTH,
        ColorType::new(255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0),
    ));

    scene.add_sphere(SphereData::with_material(
        Vec3::new(SCENE_WIDTH / 2.5, SCENE_HEIGHT / 1.5, 0.170 * SCENE_DEPTH),
        0.070 * SCENE_DEPTH,
        Material::new(
            ColorType::new(255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0),
            0.8,
            0.0,
            TransmissionProperties::none(),
        ),
    ));

    // scene.add_sphere(SphereData::new(
    //     Vec3::new(SCENE_WIDTH / 2.5, SCENE_HEIGHT / 2.5, 150.0),
    //     90.0,
    //     ColorType::new(0.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0),
    // ));

    scene.add_sphere(SphereData::with_material(
        Vec3::new(
            1.9 * (SCENE_WIDTH / 2.5),
            SCENE_HEIGHT / 2.8,
            0.160 * SCENE_DEPTH,
        ),
        0.088 * SCENE_DEPTH,
        Material::new(
            ColorType::new(250.0 / 255.0, 255.0 / 255.0, 245.0 / 255.0),
            0.01,
            0.2,
            TransmissionProperties::new(0.85, 1.5),
        ),
    ));
    //
    // scene.add_sphere(SphereData::with_material(
    //     Vec3::new(
    //         2.0 * (SCENE_WIDTH / 2.5),
    //         2.0 * (SCENE_HEIGHT / 2.5),
    //         250.0,
    //     ),
    //     120.0,
    //     Material::new(
    //         ColorType::new(158.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0),
    //         0.85,
    //         0.25,
    //     ),
    // ));
    //
    // scene.add_sphere(SphereData::with_material(
    //     Vec3::new(
    //         1.25 * (SCENE_WIDTH / 2.5),
    //         0.5 * (SCENE_HEIGHT / 2.5),
    //         90.0,
    //     ),
    //     30.0,
    //     Material::new(
    //         ColorType::new(128.0 / 255.0, 210.0 / 255.0, 255.0 / 255.0),
    //         1.0,
    //         0.5,
    //     ),
    // ));
    //
    scene.add_sphere(SphereData::with_material(
        Vec3::new(
            SCENE_WIDTH / 2.5,
            2.1 * (SCENE_HEIGHT / 2.5),
            0.5 * SCENE_DEPTH,
        ),
        0.250 * SCENE_DEPTH,
        Material::new(
            ColorType::new(254.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0),
            0.5,
            0.05,
            TransmissionProperties::none(),
        ),
    ));
    //
    // scene.add_sphere(SphereData::new(
    //     Vec3::new(
    //         SCENE_WIDTH / 4.0,
    //         3.0 * (SCENE_HEIGHT / 4.0),
    //         20.0,
    //     ),
    //     10.0,
    //     ColorType::new(255.0 / 255.0, 55.0 / 255.0, 77.0 / 255.0),
    // ));
    //
    // scene.add_sphere(SphereData::new(
    //     Vec3::new(
    //         SCENE_WIDTH / 3.0,
    //         3.0 * (SCENE_HEIGHT / 6.0),
    //         30.0,
    //     ),
    //     25.0,
    //     ColorType::new(55.0 / 255.0, 230.0 / 255.0, 180.0 / 255.0),
    // ));

    // Add triangles
    scene.add_triangle(TriangleData::with_material(
        Vec3::new(SCENE_WIDTH * 0.05, SCENE_HEIGHT * 0.2, 0.2 * SCENE_DEPTH),
        Vec3::new(SCENE_WIDTH * 0.3, SCENE_HEIGHT * 0.5, 0.2 * SCENE_DEPTH),
        Vec3::new(SCENE_WIDTH * 0.25, SCENE_HEIGHT * 0.15, 0.15 * SCENE_DEPTH),
        Material::new(
            ColorType::new(0.5, 0.7, 0.8),
            0.001,
            0.2,
            TransmissionProperties::new(0.999, 1.8),
        ),
    ));

    scene.add_triangle(TriangleData::with_material(
        Vec3::new(SCENE_WIDTH * 0.55, SCENE_HEIGHT * 0.45, 0.2 * SCENE_DEPTH),
        Vec3::new(SCENE_WIDTH * 0.7, SCENE_HEIGHT * 0.72, 0.2 * SCENE_DEPTH),
        Vec3::new(SCENE_WIDTH * 0.65, SCENE_HEIGHT * 0.35, 0.14 * SCENE_DEPTH),
        Material::new(
            ColorType::new(0.7, 0.7, 0.8),
            0.1,
            0.3,
            TransmissionProperties::none(),
        ),
    ));

    scene.add_triangle(TriangleData::with_material(
        Vec3::new(SCENE_WIDTH * 0.7, SCENE_HEIGHT * 0.90, 0.2 * SCENE_DEPTH),
        Vec3::new(SCENE_WIDTH * 0.55, SCENE_HEIGHT * 0.65, 0.2 * SCENE_DEPTH),
        Vec3::new(SCENE_WIDTH * 0.65, SCENE_HEIGHT * 0.55, 0.14 * SCENE_DEPTH),
        Material::new(
            ColorType::new(0.7, 0.7, 0.8),
            0.1,
            0.3,
            TransmissionProperties::new(1.0, 1.5),
        ),
    ));

    let mut plane_up = Vec3::unit_y();
    let mut plane_normal = -Vec3::unit_z();
    plane_normal.rotate_by(Rotor3::from_rotation_yz(-0.555));
    plane_up.rotate_by(Rotor3::from_rotation_yz(-0.555));

    // Convert BoundedPlane to basic geometries and add them
    let plane_triangles = BoundedPlane::with_material(
        plane_normal,
        Vec3::new(SCENE_WIDTH * 0.5, SCENE_HEIGHT * 0.45, 0.270 * SCENE_DEPTH),
        plane_up,
        SCENE_WIDTH * 0.55,
        SCENE_HEIGHT * 0.55,
        0.01 * SCENE_DEPTH,
        Material::new(
            ColorType::new(0.6, 0.7, 0.5),
            0.075,
            0.07,
            TransmissionProperties::new_with_boost(1.0, 1.5, 0.5),
        ),
    )
    .to_basic_geometries();

    for triangle in plane_triangles {
        scene.add_triangle(triangle);
    }

    let mut plane_up = Vec3::unit_y();
    let mut plane_normal = -Vec3::unit_z();
    plane_normal.rotate_by(Rotor3::from_rotation_xz(-0.9955));
    plane_up.rotate_by(Rotor3::from_rotation_xz(-0.9955));

    let plane_triangles = BoundedPlane::with_material(
        plane_normal,
        Vec3::new(SCENE_WIDTH * 0.82, SCENE_HEIGHT * 0.57, 0.110 * SCENE_DEPTH),
        plane_up,
        SCENE_WIDTH * 0.318,
        SCENE_HEIGHT * 0.35,
        0.007 * SCENE_DEPTH,
        Material::new(
            ColorType::new(0.99, 0.99, 0.99),
            1.0,
            0.2,
            TransmissionProperties::none(),
        ),
    )
    .to_basic_geometries();

    for triangle in plane_triangles {
        scene.add_triangle(triangle);
    }

    let back_plane_triangle = BoundedPlane::with_material(
        -Vec3::unit_z(),
        Vec3::new(SCENE_WIDTH * 0.5, SCENE_HEIGHT * 0.5, SCENE_DEPTH),
        Vec3::unit_y(),
        SCENE_WIDTH,
        SCENE_HEIGHT,
        0.001 * SCENE_DEPTH,
        Material::new(
            ColorType::new(0.5, 0.75, 0.75),
            0.0,
            0.0,
            TransmissionProperties::none(),
        ),
    )
    .to_basic_geometries();

    let bottom_plane_triangle = BoundedPlane::with_material(
        Vec3::unit_y(),
        Vec3::new(SCENE_WIDTH * 0.5, SCENE_HEIGHT, SCENE_DEPTH as f32 * 0.5),
        Vec3::unit_z(),
        SCENE_WIDTH,
        SCENE_DEPTH,
        0.001 * SCENE_DEPTH,
        Material::new(
            ColorType::new(0.75, 0.5, 0.75),
            0.0,
            0.0,
            TransmissionProperties::none(),
        ),
    )
    .to_basic_geometries();

    let top_plane_triangle = BoundedPlane::with_material(
        -Vec3::unit_y(),
        Vec3::new(SCENE_WIDTH * 0.5, 0.0, SCENE_DEPTH as f32 * 0.5),
        Vec3::unit_z(),
        SCENE_WIDTH,
        SCENE_DEPTH,
        0.001 * SCENE_DEPTH,
        Material::new(
            ColorType::new(0.75, 0.5, 0.75),
            0.0,
            0.0,
            TransmissionProperties::none(),
        ),
    )
    .to_basic_geometries();

    let left_plane_triangle = BoundedPlane::with_material(
        Vec3::unit_x(),
        Vec3::new(0.0, SCENE_HEIGHT * 0.5, SCENE_DEPTH as f32 * 0.5),
        Vec3::unit_z(),
        SCENE_HEIGHT,
        SCENE_DEPTH,
        0.001 * SCENE_DEPTH,
        Material::new(
            ColorType::new(0.75, 0.75, 0.5),
            0.0,
            0.0,
            TransmissionProperties::none(),
        ),
    )
    .to_basic_geometries();

    let right_plane_triangle = BoundedPlane::with_material(
        -Vec3::unit_x(),
        Vec3::new(SCENE_WIDTH, SCENE_HEIGHT * 0.5, SCENE_DEPTH as f32 * 0.5),
        -Vec3::unit_z(),
        SCENE_HEIGHT,
        SCENE_DEPTH,
        0.001 * SCENE_DEPTH,
        Material::new(
            ColorType::new(0.75, 0.75, 0.5),
            0.0,
            0.0,
            TransmissionProperties::none(),
        ),
    )
    .to_basic_geometries();

    for triangles in [
        back_plane_triangle,
        bottom_plane_triangle,
        top_plane_triangle,
        left_plane_triangle,
        right_plane_triangle,
    ] {
        for triangle in triangles {
            scene.add_triangle(triangle);
        }
    }

    for light in [
        PointLight::new(
            Vec3::new(SCENE_WIDTH / 2.0, SCENE_HEIGHT / 1.8, 0.016 * SCENE_DEPTH),
            ColorType::new(0.825, 0.675, 0.5),
            0.15,
        ),
        PointLight::new(
            Vec3::new(SCENE_WIDTH / 3.5, SCENE_HEIGHT / 3.75, 0.025 * SCENE_DEPTH),
            ColorType::new(0.825, 0.675, 0.45),
            0.485,
        ),
        PointLight::new(
            Vec3::new(SCENE_WIDTH / 1.22, SCENE_HEIGHT / 2.9, 0.38 * SCENE_DEPTH),
            ColorType::new(0.78, 0.67, 0.45),
            0.6,
        ),
        PointLight::new(
            Vec3::new(SCENE_WIDTH - 80.0, SCENE_HEIGHT / 2.0, 0.125 * SCENE_DEPTH),
            ColorType::new(1.0, 1.0, 1.0),
            0.1,
        ),
        PointLight::new(
            Vec3::new(SCENE_WIDTH / 2.5, (SCENE_HEIGHT / 5.0), 0.175 * SCENE_DEPTH),
            ColorType::new(0.75, 0.56, 0.65),
            0.2,
        ),
        PointLight::new(
            Vec3::new(SCENE_WIDTH / 4.0, (SCENE_HEIGHT / 6.0), 0.01 * SCENE_DEPTH),
            ColorType::new(0.01, 0.5, 0.4),
            0.175,
        ),
    ] {
        scene.add_light(light.into())
    }

    scene
});

fn main() {
    // Uncomment to run benchmarks
    // simd_polygon_triangulation_bench::run_triangulation_benchmarks();

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
