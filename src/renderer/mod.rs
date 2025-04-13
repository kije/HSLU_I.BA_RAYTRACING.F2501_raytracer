use crate::helpers::Pixel;
use crate::image_buffer::ImageBuffer;

use rayon::prelude::*;
use std::sync::atomic::Ordering;

#[cfg(feature = "render_timing_debug")]
use incr_stats::vec::Stats;
#[cfg(feature = "render_timing_debug")]
use std::sync::{Arc, Mutex};
use std::time::Duration;
#[cfg(feature = "render_timing_debug")]
use std::time::Instant;
use std::{mem, thread};

mod raytracer_renderer;
mod test_renderer;
mod test_renderer_3d_sphere_sw02;
mod test_renderer_light_color_sw03;
mod test_renderer_sw03_common_code;
mod test_renderer_vector;

use crate::output::OutputColorEncoder;
pub(crate) use raytracer_renderer::RaytracerRenderer;
pub(crate) use test_renderer_sw03_common_code::TestRenderer3DSW03CommonCode;

pub(crate) struct RenderCoordinates {
    x: usize,
    y: usize,
}

pub(crate) struct RenderCoordinatesVectorized<'a> {
    i: &'a [usize],
    x: &'a [f32],
    y: &'a [f32],
    z: &'a [f32],
}

#[cfg(feature = "render_timing_debug")]
pub(crate) fn print_render_stats(render_times: &[f64]) {
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

pub(crate) trait Renderer<const W: usize, const H: usize, C: OutputColorEncoder> {
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
        let chunk_size = Self::RENDER_STRIDE;

        #[cfg(feature = "render_timing_debug")]
        let render_times = Arc::new(Mutex::new(Vec::with_capacity(
            buffer.len() / (chunk_size - 1),
        )));
        {
            #[cfg(feature = "render_timing_debug")]
            let render_times = render_times.clone();

            buffer
                .par_chunks(chunk_size)
                .enumerate()
                .for_each(|(chunk_index, p)| {
                    #[cfg(feature = "render_timing_debug")]
                    let start = Instant::now();
                    let chunk_offset = chunk_index * chunk_size;
                    for (j, pixel) in p.iter().enumerate() {
                        let pixel_offset = chunk_offset + j;
                        let x = pixel_offset % W;
                        let y = pixel_offset / W;

                        if let Some(pixel_color) = cb(RenderCoordinates { x, y }) {
                            pixel.store(C::to_output(&pixel_color), Ordering::Relaxed);
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

            buffer
                .par_chunks(Self::RENDER_STRIDE)
                .enumerate()
                .for_each(|(chunk_index, p)| {
                    #[cfg(feature = "render_timing_debug")]
                    let start = Instant::now();

                    let chunk_offset = chunk_index * Self::RENDER_STRIDE;

                    let indexes = (0..p.len()).map(|j| (j, chunk_offset + j));

                    let (i, x, y): (Vec<_>, Vec<_>, Vec<_>) = indexes
                        .map(|(j, pixel_offset)| {
                            (j, (pixel_offset % W) as f32, (pixel_offset / W) as f32)
                        })
                        .collect();

                    let z = vec![0.0f32; p.len()];

                    let coordinates = RenderCoordinatesVectorized {
                        i: &i,
                        x: &x,
                        y: &y,
                        z: &z,
                    };

                    cb(coordinates, &|i, pixel| {
                        p[i].store(C::to_output(&pixel), Ordering::Relaxed);

                        if cfg!(feature = "simulate_slow_render") {
                            thread::sleep(Duration::from_micros(70));
                        }
                    });

                    #[cfg(feature = "render_timing_debug")]
                    if let Ok(mut render_times) = render_times.lock() {
                        render_times.push(start.elapsed().as_secs_f64());
                    }
                });
        }

        #[cfg(feature = "render_timing_debug")]
        if let Ok(render_times) = render_times.lock() {
            print_render_stats(&render_times);
        }
    }
}
