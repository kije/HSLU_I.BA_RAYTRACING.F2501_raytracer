use rayon::prelude::*;
use crate::helpers::Pixel;
use crate::image_buffer::ImageBuffer;

mod test_renderer;

pub(crate) use test_renderer::TestRenderer;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Default)]
pub(crate) struct RenderCoordinates{x: usize, y: usize}

impl From<(usize, usize)> for RenderCoordinates {
    fn from((x, y): (usize, usize)) -> Self {
        Self{x, y}
    }
}

pub(crate) trait Renderer<const W: usize, const H: usize> {
    fn render(&self, buffer: &mut ImageBuffer<W, H>) where [(); W*H]:;

    fn render_to_buffer<F>(buffer: &mut ImageBuffer<W, H>, cb: F) where [(); W*H]:, F : (Fn(RenderCoordinates) -> Option<Pixel>) + Sync  {
        buffer.par_iter_mut().enumerate().for_each(|(i, p)| {
            let x = i % W;
            let y = i / W;

            if let Some(pixel_color) = cb((x, y).into()) {
                *p = pixel_color.0;
            }
        });
    }
}