#![feature(generic_const_exprs)]
#![feature(concat_idents)]
#![feature(likely_unlikely)]
#![allow(incomplete_features)]

use raytracer::geometry::{BoundedPlane, CompositeGeometry, SphereData};
use raytracer::helpers::ColorType;
use raytracer::raytracing::{Material, TransmissionProperties};
use raytracer::scene::{PointLight, Scene};
use raytracer::{
    AVERAGE_SCENE_DIMENSION, SCENE_DEPTH, SCENE_HEIGHT, SCENE_WIDTH, WINDOW_HEIGHT, WINDOW_WIDTH,
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
        if cfg!(feature = "high_quality_model") || cfg!(feature = "medium_resolution") {
            "./data/obj/text/text.obj"
        } else {
            "./data/obj/text/text_lowres.obj"
        },
        Some(Similarity3::new(
            Vec3::new(
                0.0135 * SCENE_WIDTH,
                0.145 * SCENE_HEIGHT,
                0.885 * SCENE_DEPTH,
            ),
            Rotor3::from_euler_angles(0.0, -0.015, 0.0),
            1.226 * AVERAGE_SCENE_DIMENSION,
        )),
    )
    .unwrap();

    scene.add_sphere(SphereData::with_material(
        Vec3::new(
            0.475 * SCENE_WIDTH,
            0.385 * SCENE_HEIGHT,
            0.595 * SCENE_DEPTH,
        ),
        0.291 * AVERAGE_SCENE_DIMENSION,
        Material::new(
            ColorType::new(1.0, 0.8, 1.0),
            0.0,
            0.1,
            TransmissionProperties::new(0.99, 1.5),
        ),
    ));

    scene.add_sphere(SphereData::with_material(
        Vec3::new(0.8 * SCENE_WIDTH, 0.76 * SCENE_HEIGHT, 0.2 * SCENE_DEPTH),
        0.07 * AVERAGE_SCENE_DIMENSION,
        Material::new(
            ColorType::new(0.75, 0.5, 1.0),
            0.2,
            0.3,
            TransmissionProperties::new(0.78, 1.5),
        ),
    ));

    scene.add_sphere(SphereData::with_material(
        Vec3::new(0.76 * SCENE_WIDTH, 0.76 * SCENE_HEIGHT, 0.4 * SCENE_DEPTH),
        0.07 * AVERAGE_SCENE_DIMENSION,
        Material::new(
            ColorType::new(0.75, 0.9, 0.8),
            0.001,
            0.35,
            TransmissionProperties::new(0.6, 1.8),
        ),
    ));
    scene.add_sphere(SphereData::with_material(
        Vec3::new(0.73 * SCENE_WIDTH, 0.7 * SCENE_HEIGHT, 0.52 * SCENE_DEPTH),
        0.065 * AVERAGE_SCENE_DIMENSION,
        Material::new(
            ColorType::new(0.75, 0.9, 0.8),
            0.0,
            0.7,
            TransmissionProperties::new(0.78, 1.3),
        ),
    ));

    scene.add_sphere(SphereData::with_material(
        Vec3::new(0.69 * SCENE_WIDTH, 0.76 * SCENE_HEIGHT, 0.3 * SCENE_DEPTH),
        0.07 * AVERAGE_SCENE_DIMENSION,
        Material::new(
            ColorType::new(0.88, 0.9, 0.88),
            0.0,
            0.1,
            TransmissionProperties::new(1.0, 1.4),
        ),
    ));

    scene.add_sphere(SphereData::with_material(
        Vec3::new(0.1 * SCENE_WIDTH, 0.68 * SCENE_HEIGHT, 0.3 * SCENE_DEPTH),
        0.07 * AVERAGE_SCENE_DIMENSION,
        Material::new(
            ColorType::new(0.88, 0.9, 0.88),
            0.2,
            0.7,
            TransmissionProperties::none(),
        ),
    ));

    scene.add_sphere(SphereData::with_material(
        Vec3::new(0.35 * SCENE_WIDTH, 0.76 * SCENE_HEIGHT, 0.25 * SCENE_DEPTH),
        0.07 * AVERAGE_SCENE_DIMENSION,
        Material::new(
            ColorType::new(0.9, 0.2, 0.3),
            0.0,
            0.01,
            TransmissionProperties::none(),
        ),
    ));

    scene.add_sphere(SphereData::with_material(
        Vec3::new(0.2 * SCENE_WIDTH, 0.87 * SCENE_HEIGHT, 0.5 * SCENE_DEPTH),
        0.07 * AVERAGE_SCENE_DIMENSION,
        Material::new(
            ColorType::new(0.88, 0.5, 0.7),
            0.4,
            0.2,
            TransmissionProperties::none(),
        ),
    ));

    scene.add_sphere(SphereData::with_material(
        Vec3::new(0.5 * SCENE_WIDTH, 0.87 * SCENE_HEIGHT, 0.46 * SCENE_DEPTH),
        0.075 * AVERAGE_SCENE_DIMENSION,
        Material::new(
            ColorType::new(1., 1., 1.),
            0.95,
            0.23,
            TransmissionProperties::none(),
        ),
    ));

    let rotor = Rotor3::from_euler_angles(-0.04, 0.125, 0.51);
    let isometry = Isometry3::new(
        Vec3::new(
            0.25 * SCENE_WIDTH,
            0.002 * SCENE_HEIGHT,
            0.037 * SCENE_DEPTH,
        ),
        rotor,
    );

    let back_plane_triangle = BoundedPlane::with_material(
        -Vec3::unit_z().rotated_by(rotor),
        isometry.transform_vec(Vec3::new(
            SCENE_WIDTH * 0.5,
            (SCENE_HEIGHT * 1.1) * 0.5,
            SCENE_DEPTH,
        )),
        Vec3::unit_y().rotated_by(rotor),
        SCENE_WIDTH,
        (SCENE_HEIGHT * 1.1),
        0.01 * SCENE_DEPTH,
        Material::new(
            ColorType::new(0.5, 0.75, 0.75),
            0.0,
            0.0,
            TransmissionProperties::none(),
        ),
    )
    .to_basic_geometries();

    let bottom_plane_triangle = BoundedPlane::with_material(
        Vec3::unit_y().rotated_by(rotor),
        isometry.transform_vec(Vec3::new(
            SCENE_WIDTH * 0.5,
            SCENE_HEIGHT,
            SCENE_DEPTH as f32 * 0.5,
        )),
        Vec3::unit_z().rotated_by(rotor),
        SCENE_WIDTH,
        SCENE_DEPTH,
        0.01 * SCENE_DEPTH,
        Material::new(
            ColorType::new(0.75, 0.5, 0.75),
            0.0,
            0.7,
            TransmissionProperties::new(0.6, 1.125),
        ),
    )
    .to_basic_geometries();

    let bottom_plane_triangle2 = BoundedPlane::with_material(
        Vec3::unit_y().rotated_by(rotor),
        isometry.transform_vec(Vec3::new(
            SCENE_WIDTH * 0.5,
            SCENE_HEIGHT + 0.09,
            SCENE_DEPTH as f32 * 0.5,
        )),
        Vec3::unit_z().rotated_by(rotor),
        SCENE_WIDTH,
        SCENE_DEPTH,
        0.01 * SCENE_DEPTH,
        Material::new(
            ColorType::new(0.75, 0.5, 0.75),
            0.0,
            0.7,
            TransmissionProperties::none(),
        ),
    )
    .to_basic_geometries();

    let right_plane_triangle = BoundedPlane::with_material(
        -Vec3::unit_x().rotated_by(rotor),
        isometry.transform_vec(Vec3::new(
            SCENE_WIDTH,
            (SCENE_HEIGHT * 1.1) * 0.5,
            SCENE_DEPTH as f32 * 0.5,
        )),
        -Vec3::unit_z().rotated_by(rotor),
        (SCENE_HEIGHT * 1.1),
        SCENE_DEPTH,
        0.01 * SCENE_DEPTH,
        Material::new(
            ColorType::new(0.85, 0.85, 0.6),
            0.5,
            0.5,
            TransmissionProperties::none(),
        ),
    )
    .to_basic_geometries();

    for triangles in [
        back_plane_triangle,
        bottom_plane_triangle,
        bottom_plane_triangle2,
        right_plane_triangle,
    ] {
        for triangle in triangles {
            scene.add_triangle(triangle);
        }
    }

    scene.add_light(
        PointLight::new(
            Vec3::new(SCENE_WIDTH / 1.2, 0.0, 0.015 * SCENE_DEPTH),
            ColorType::new(0.825, 0.675, 0.5),
            1.0,
        )
        .into(),
    );

    scene.add_light(
        PointLight::new(
            Vec3::new(SCENE_WIDTH / 2.4, SCENE_HEIGHT * 0.1, 0.08 * SCENE_DEPTH),
            ColorType::new(0.825, 0.675, 0.65),
            0.65,
        )
        .into(),
    );

    scene.add_light(
        PointLight::new(
            Vec3::new(SCENE_WIDTH, SCENE_HEIGHT, 0.01 * SCENE_DEPTH),
            ColorType::new(0.825, 0.35, 0.8),
            0.42,
        )
        .into(),
    );
    scene.add_light(
        PointLight::new(
            isometry.transform_vec(Vec3::new(
                SCENE_WIDTH * 0.5,
                SCENE_HEIGHT + 0.05,
                SCENE_DEPTH as f32 * 0.75,
            )),
            ColorType::new(1.0, 1.0, 1.0),
            0.25,
        )
        .into(),
    );
    scene.add_light(
        PointLight::new(
            Vec3::new(0.2 * SCENE_WIDTH, SCENE_HEIGHT * 0.67, 0.95 * SCENE_DEPTH),
            ColorType::new(0.825, 0.5, 0.7),
            0.25,
        )
        .into(),
    );
    //
    // scene.add_light(
    //     PointLight::new(
    //         Vec3::new(SCENE_WIDTH / 2.0, 0.0, 0.1 * SCENE_DEPTH),
    //         ColorType::new(0.825, 0.5, 0.7),
    //         0.3,
    //     )
    //     .into(),
    // );
    //
    // scene.add_light(
    //     PointLight::new(
    //         Vec3::new(SCENE_WIDTH / 7.0, SCENE_HEIGHT * 0.3, 0.78 * SCENE_DEPTH),
    //         ColorType::new(0.8, 0.45, 0.9),
    //         0.7,
    //     )
    //     .into(),
    // );

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
