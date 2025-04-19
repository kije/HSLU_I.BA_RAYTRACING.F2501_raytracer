use crate::image_buffer::ImageBuffer;

mod window;
use crate::helpers::{Pixel, RenderTiming};
pub use window::{WindowColorEncoder, WindowOutput};

pub trait OutputColorEncoder {
    fn to_output(pixel: &Pixel) -> u32;
}

pub trait Output<const W: usize, const H: usize> {
    type ColorEncoder: OutputColorEncoder;

    fn render_buffer(&mut self, buffer: &ImageBuffer<W, H>)
    where
        [(); W * H]:;
}

pub trait OutputInteractive<const W: usize, const H: usize> {
    type Output: Output<W, H>;

    fn render_loop<F: FnMut(&mut Self::Output, &RenderTiming)>(&mut self, cb: F);

    fn render_static<F: FnOnce(&mut Self::Output, &RenderTiming)>(
        &mut self,
        cb: F,
        timing: Option<RenderTiming>,
    );
}
