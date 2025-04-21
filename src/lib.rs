#![feature(generic_const_exprs)]
#![feature(concat_idents)]
#![feature(likely_unlikely)]
#![allow(incomplete_features)]

pub mod color;
pub mod extensions;
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
pub mod scalar_traits;
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
