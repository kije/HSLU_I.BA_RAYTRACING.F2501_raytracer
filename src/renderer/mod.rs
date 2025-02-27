use std::sync::atomic::Ordering;
use mint::Point2;
use rayon::prelude::*;
use crate::helpers::Pixel;
use crate::image_buffer::ImageBuffer;

mod test_renderer;
mod test_renderer_vector;

pub(crate) use test_renderer::TestRenderer;
pub(crate) use test_renderer_vector::TestRendererVector;
use crate::output::OutputColorEncoder;

pub(crate) type RenderCoordinates = Point2<usize>;




pub(crate) trait Renderer<const W: usize, const H: usize, C: OutputColorEncoder> {
    fn render(&self, buffer: &ImageBuffer<W, H>) where [(); W*H]:;

    fn render_to_buffer<F>(buffer: &ImageBuffer<W, H>, cb: F) where [(); W*H]:, F : (Fn(RenderCoordinates) -> Option<Pixel>) + Sync  {
        buffer.par_iter().enumerate().for_each(|(i, p)| {
            let x = i % W;
            let y = i / W;

            if let Some(pixel_color) = cb([x, y].into()) {
                p.store(C::to_output(&pixel_color), Ordering::Relaxed);
            }
        });
    }
}