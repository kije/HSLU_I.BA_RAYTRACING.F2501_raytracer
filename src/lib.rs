#![feature(generic_const_exprs)]
#![feature(concat_idents)]
#![feature(likely_unlikely)]
#![allow(incomplete_features)]
extern crate core;

use ultraviolet::Vec3;

pub mod color;
pub mod extensions;
pub mod float_ext;
pub mod geometry;
pub mod helpers;
pub mod image_buffer;
pub mod math;
pub mod output;
pub mod random;
pub mod raytracing;
pub mod renderer;
pub mod scalar;
pub mod scene;
pub mod vector;

// New modules for trait simplification
pub mod color_traits;
pub mod matrix;
pub mod simd_compat;
pub mod vector_traits;

pub const CONFIGURED_WINDOW_WIDTH: usize = if cfg!(feature = "high_resolution") {
    1620
} else {
    if cfg!(feature = "medium_resolution") {
        1100
    } else {
        768
    }
};

pub const CONFIGURED_WINDOW_HEIGHT: usize = if cfg!(feature = "high_resolution") {
    1280
} else {
    if cfg!(feature = "medium_resolution") {
        830
    } else {
        640
    }
};

pub const WINDOW_WIDTH: usize = match option_env!("WINDOW_WIDTH") {
    Some(width) => {
        let parsed = usize::from_str_radix(width, 10);

        match parsed {
            Ok(parsed) => parsed,
            Err(_) => CONFIGURED_WINDOW_WIDTH,
        }
    }
    None => CONFIGURED_WINDOW_WIDTH,
};
pub const WINDOW_HEIGHT: usize = match option_env!("WINDOW_HEIGHT") {
    Some(height) => {
        let parsed = usize::from_str_radix(height, 10);

        match parsed {
            Ok(parsed) => parsed,
            Err(_) => CONFIGURED_WINDOW_HEIGHT,
        }
    }
    None => CONFIGURED_WINDOW_HEIGHT,
};

pub const WINDOW_ASPECT_RATIO: f32 = (WINDOW_HEIGHT as f32) / WINDOW_WIDTH as f32;
pub const WINDOW_SCENE_DEPTH: usize = (WINDOW_WIDTH + WINDOW_HEIGHT) / 2;

pub const SCENE_WIDTH: f32 = 1.0;
pub const SCENE_HEIGHT: f32 = 1.0 * WINDOW_ASPECT_RATIO; // Fixme somehow this should also be 1.0 -> e.g. coordinates should be between 0 and 1. Idea: have a pixel size ratio. Coordinates in the scene are 0..1, but then are scaled to pixel-size, where the "pixel" can rectangular instead of quadratic
pub const SCENE_DEPTH: f32 = 1.0;

pub const WINDOW_TO_SCENE_WIDTH_FACTOR: f32 = SCENE_WIDTH / WINDOW_WIDTH as f32;
pub const WINDOW_TO_SCENE_HEIGHT_FACTOR: f32 = SCENE_HEIGHT / WINDOW_HEIGHT as f32;
pub const WINDOW_TO_SCENE_DEPTH_FACTOR: f32 = SCENE_DEPTH / WINDOW_SCENE_DEPTH as f32;

pub const AVERAGE_SCENE_FACTOR: f32 =
    (WINDOW_TO_SCENE_WIDTH_FACTOR + WINDOW_TO_SCENE_HEIGHT_FACTOR + WINDOW_TO_SCENE_DEPTH_FACTOR)
        / 3.0;
pub static RENDER_RAY_FOCUS: Vec3 =
    Vec3::new(SCENE_WIDTH / 2.0, SCENE_HEIGHT / 2.0, -2.0 * SCENE_DEPTH);

// IoR of air
pub const DEFAULT_REFRACTION_INDEX: f32 = 1.000293;
