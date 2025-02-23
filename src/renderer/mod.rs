use mint::Point2;
use rayon::prelude::*;
use crate::helpers::Pixel;
use crate::image_buffer::ImageBuffer;

mod test_renderer;

pub(crate) use test_renderer::TestRenderer;
use crate::output::OutputColorEncoder;

pub(crate) type RenderCoordinates = Point2<usize>;




pub(crate) trait Renderer<const W: usize, const H: usize, C: OutputColorEncoder> {
    fn render(&self, buffer: &mut ImageBuffer<W, H>) where [(); W*H]:;

    fn render_to_buffer<F>(buffer: &mut ImageBuffer<W, H>, cb: F) where [(); W*H]:, F : (Fn(RenderCoordinates) -> Option<Pixel>) + Sync  {
        buffer.par_iter_mut().enumerate().for_each(|(i, p)| {
            let x = i % W;
            let y = i / W;

            if let Some(pixel_color) = cb([x, y].into()) {
                *p = C::to_output(&pixel_color);
            }
        });
    }
}