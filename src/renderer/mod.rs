use std::sync::atomic::{AtomicU32, Ordering};
use std::{mem, thread};
use std::time::Duration;
use mint::Point2;
use rayon::max_num_threads;
use rayon::prelude::*;
use crate::helpers::Pixel;
use crate::image_buffer::ImageBuffer;

mod test_renderer;
mod test_renderer_vector;
mod test_renderer_3d_sphere_sw02;

pub(crate) use test_renderer::TestRenderer;
pub(crate) use test_renderer_vector::TestRendererVector;
pub(crate) use test_renderer_3d_sphere_sw02::TestRenderer3DSphereSW02;
use crate::output::OutputColorEncoder;

pub(crate) type RenderCoordinates = Point2<usize>;




pub(crate) trait Renderer<const W: usize, const H: usize, C: OutputColorEncoder> {
    fn render(&self, buffer: &ImageBuffer<W, H>) where [(); W*H]:;

    fn render_chunk_size() -> usize {
        (W/32).next_multiple_of(64/mem::size_of::<u32>())  // align to cache lines to avoid false sharing
    }

    fn render_to_buffer<F>(buffer: &ImageBuffer<W, H>, cb: F) where [(); W*H]:, F : (Fn(RenderCoordinates) -> Option<Pixel>) + Sync  {
        let chunk_size = Self::render_chunk_size();

        buffer.par_chunks(chunk_size).enumerate().for_each(|(chunk_index, p)| {
            let chunk_offset = chunk_index * chunk_size;
            for (j, pixel) in p.iter().enumerate() {
                let pixel_offset =  chunk_offset + j;
                let x = pixel_offset % W;
                let y =  pixel_offset / W;

                if let Some(pixel_color) = cb([x, y].into()) {
                    pixel.store(C::to_output(&pixel_color), Ordering::Relaxed);
                }
            }
        });
    }

    fn render_to_buffer_chunked_inplace<F>(buffer: &ImageBuffer<W, H>, cb: F) where [(); W*H]:, F: Fn(&[(usize,RenderCoordinates)],  &dyn Fn(usize,Pixel)) + Sync  {
        let chunk_size = Self::render_chunk_size();

        buffer.par_chunks(chunk_size).enumerate().for_each(|(chunk_index, p)| {
            let chunk_offset = chunk_index * chunk_size;
            let coordinates: Vec<_> = p.iter().enumerate().map(|(j, pixel)|{
                let pixel_offset =  chunk_offset + j;
                let x = pixel_offset % W;
                let y =  pixel_offset / W;

                (j,RenderCoordinates { x, y })
            }).collect();

            cb(
                &coordinates,
                &|i,pixel|{
                    p[i].store(C::to_output(&pixel), Ordering::Relaxed);
                }
            );
        });
    }
}