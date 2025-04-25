use crate::helpers::Pixel;
use crate::image_buffer::ImageBuffer;

use rayon::prelude::*;
use std::sync::atomic::Ordering;

#[cfg(feature = "render_timing_debug")]
use itertools::Itertools;

#[cfg(feature = "render_timing_debug")]
use incr_stats::vec::Stats;
#[cfg(feature = "render_timing_debug")]
use std::sync::{Arc, Mutex};
use std::time::Duration;
#[cfg(feature = "render_timing_debug")]
use std::time::Instant;
use std::{mem, thread};

mod raytracer_renderer;

use crate::output::OutputColorEncoder;
pub use raytracer_renderer::RaytracerRenderer;

pub struct RenderCoordinates {
    x: usize,
    y: usize,
}

pub struct RenderCoordinatesVectorized<'a> {
    i: &'a [usize],
    x: &'a [f32],
    y: &'a [f32],
    z: &'a [f32],
}

#[cfg(feature = "render_timing_debug")]
pub fn print_render_stats(render_times: &[f64]) {
    let mut render_times_stats = Stats::new(&render_times).unwrap();
    let (mid_range, mid_range_len) = if render_times.len() % 2 == 1 {
        ((render_times.len() / 2)..(render_times.len() / 2), 1)
    } else {
        ((render_times.len() / 2)..((render_times.len() / 2) + 1), 2)
    };
    fn make_nan_comparable(f: f64) -> u128 {
        if f.is_nan() {
            1u128
        } else {
            (f * 10.0f64.powf(1_000_000_000.0f64)) as u128
        }
    }
    let median = render_times
        .iter()
        .sorted_by(|&&f1, &&f2| {
            let (f1, f2) = (make_nan_comparable(f1), make_nan_comparable(f2));
            if f1 == f2 {
                core::cmp::Ordering::Equal
            } else if f1 < f2 {
                core::cmp::Ordering::Greater
            } else {
                core::cmp::Ordering::Less
            }
        })
        .get(mid_range)
        .sum::<f64>()
        / mid_range_len as f64;
    println!("Render time per Chunk:");
    println!("Mean: {}", render_times_stats.mean().unwrap());
    println!("Median: {}", median);
    println!(
        "Std: {}",
        render_times_stats.sample_standard_deviation().unwrap()
    );
    println!("Min: {}", render_times_stats.min().unwrap());
    println!("Max: {}", render_times_stats.max().unwrap());
}

pub trait Renderer<const W: usize, const H: usize, C: OutputColorEncoder> {
    const RENDER_STRIDE: usize = const {
        (W / 16)
            .next_multiple_of(64 / mem::size_of::<u32>())
            .next_multiple_of(8)
        // align to vectorized x8 size and cache lines to avoid false sharing
    };

    fn render(&self, buffer: &ImageBuffer<W, H>)
    where
        [(); W * H]:;

    fn render_to_buffer<F>(buffer: &ImageBuffer<W, H>, cb: F)
    where
        [(); W * H]:,
        F: (Fn(RenderCoordinates) -> Option<Pixel>) + Sync,
    {
        #[cfg(feature = "render_timing_debug")]
        let render_times = Arc::new(Mutex::new(Vec::with_capacity(
            buffer.len() / (chunk_size - 1),
        )));
        {
            #[cfg(feature = "render_timing_debug")]
            let render_times = render_times.clone();

            buffer.process_chunks_parallel(Self::RENDER_STRIDE, Self::RENDER_STRIDE / 3, |chunk| {
                chunk.process_rows(|y, _, set_pixel_value| {
                    #[cfg(feature = "render_timing_debug")]
                    let start = Instant::now();

                    for i in (0..chunk.width()) {
                        let (x, y) = chunk.global_coordinates(i, y);

                        if let Some(pixel_color) = cb(RenderCoordinates { x, y }) {
                            set_pixel_value(i, C::to_output(&pixel_color));

                            if cfg!(feature = "simulate_slow_render") {
                                thread::sleep(Duration::from_micros(70));
                            }
                        }
                    }

                    #[cfg(feature = "render_timing_debug")]
                    if let Ok(mut render_times) = render_times.lock() {
                        render_times.push(start.elapsed().as_secs_f64());
                    }
                });
            });
        }

        #[cfg(feature = "render_timing_debug")]
        if let Ok(render_times) = render_times.lock() {
            print_render_stats(&render_times);
        }
    }

    fn render_to_buffer_chunked_inplace<F>(buffer: &ImageBuffer<W, H>, cb: F)
    where
        [(); W * H]:,
        F: Fn(
                RenderCoordinatesVectorized,
                &dyn Fn(usize, Pixel), // todo: callback shall also support depth ob intersected object at that point, enabling us to generate a depth map
            ) + Sync,
    {
        #[cfg(feature = "render_timing_debug")]
        let render_times = Arc::new(Mutex::new(Vec::with_capacity(
            buffer.len() / (Self::RENDER_STRIDE - 1),
        )));
        {
            #[cfg(feature = "render_timing_debug")]
            let render_times = render_times.clone();

            buffer.process_chunks_parallel(Self::RENDER_STRIDE, Self::RENDER_STRIDE / 3, |chunk| {
                // Cache-friendly approach: process row by row
                chunk.process_rows(|y, _, set_pixel_value| {
                    #[cfg(feature = "render_timing_debug")]
                    let start = Instant::now();

                    let z = vec![0.0f32; chunk.width()];
                    let i = (0..chunk.width()).collect::<Vec<_>>();

                    let (x, y): (Vec<_>, Vec<_>) = (0..chunk.width())
                        .map(|x| chunk.global_coordinates(x, y))
                        .map(|(x, y)| (x as f32, y as f32))
                        .collect();

                    let coordinates = RenderCoordinatesVectorized {
                        i: &i,
                        x: &x,
                        y: &y,
                        z: &z,
                    };

                    cb(coordinates, &|i, pixel| {
                        set_pixel_value(i, C::to_output(&pixel));

                        if cfg!(feature = "simulate_slow_render") {
                            thread::sleep(Duration::from_micros(70));
                        }
                    });

                    #[cfg(feature = "render_timing_debug")]
                    if let Ok(mut render_times) = render_times.lock() {
                        render_times.push(start.elapsed().as_secs_f64());
                    }
                });
            });
        }

        #[cfg(feature = "render_timing_debug")]
        if let Ok(render_times) = render_times.lock() {
            print_render_stats(&render_times);
        }
    }
}
