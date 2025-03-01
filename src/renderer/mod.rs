use crate::helpers::Pixel;
use crate::image_buffer::ImageBuffer;
use itertools::Itertools;
use minifb::Key::Z;
use mint::Point2;
use rayon::max_num_threads;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};
use std::{mem, thread};

mod test_renderer;
mod test_renderer_3d_sphere_sw02;
mod test_renderer_vector;

use crate::output::OutputColorEncoder;
pub(crate) use test_renderer::TestRenderer;
pub(crate) use test_renderer_3d_sphere_sw02::TestRenderer3DSphereSW02;
pub(crate) use test_renderer_vector::TestRendererVector;

pub(crate) type RenderCoordinates = Point2<usize>;

pub(crate) struct RenderCoordinatesVectorized<'a> {
    i: &'a [usize],
    x: &'a [f32],
    y: &'a [f32],
    z: &'a [f32],
}

pub(crate) trait Renderer<const W: usize, const H: usize, C: OutputColorEncoder> {
    fn render(&self, buffer: &ImageBuffer<W, H>)
    where
        [(); W * H]:;

    fn render_chunk_size() -> usize {
        (W / 32).next_multiple_of(64 / mem::size_of::<u32>()) // align to cache lines to avoid false sharing
    }

    fn render_to_buffer<F>(buffer: &ImageBuffer<W, H>, cb: F)
    where
        [(); W * H]:,
        F: (Fn(RenderCoordinates) -> Option<Pixel>) + Sync,
    {
        let chunk_size = Self::render_chunk_size();

        buffer
            .par_chunks(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, p)| {
                let chunk_offset = chunk_index * chunk_size;
                //let start = Instant::now();
                for (j, pixel) in p.iter().enumerate() {
                    let pixel_offset = chunk_offset + j;
                    let x = pixel_offset % W;
                    let y = pixel_offset / W;

                    if let Some(pixel_color) = cb([x, y].into()) {
                        pixel.store(C::to_output(&pixel_color), Ordering::Relaxed);
                        if cfg!(feature = "simulate_slow_render") {
                            thread::sleep(Duration::from_nanos(500));
                        }
                    }
                }
                //println!("{:?}", start.elapsed());
            });
    }

    fn render_to_buffer_chunked_inplace<F>(buffer: &ImageBuffer<W, H>, cb: F)
    where
        [(); W * H]:,
        F: Fn(RenderCoordinatesVectorized, &dyn Fn(usize, Pixel)) + Sync,
    {
        let chunk_size = Self::render_chunk_size();

        buffer
            .par_chunks(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, p)| {
                //let start = Instant::now();

                let chunk_offset = chunk_index * chunk_size;

                let indexes = p.iter().enumerate().map(|(j, _)| j);
                let pixel_offsets = indexes.clone().map(|j| chunk_offset + j);

                let i = indexes.collect_vec();
                let x = pixel_offsets
                    .clone()
                    .map(|pixel_offset| (pixel_offset % W) as f32)
                    .collect_vec();
                let y = pixel_offsets
                    .clone()
                    .map(|pixel_offset| (pixel_offset / W) as f32)
                    .collect_vec();
                let z = vec![0.0; p.len()];

                let c = RenderCoordinatesVectorized {
                    i: &i,
                    x: &x,
                    y: &y,
                    z: &z,
                };

                cb(c, &|i, pixel| {
                    p[i].store(C::to_output(&pixel), Ordering::Relaxed);
                    if cfg!(feature = "simulate_slow_render") {
                        thread::sleep(Duration::from_nanos(500));
                    }
                });
                //println!("{:?}", start.elapsed());
            });
    }
}
