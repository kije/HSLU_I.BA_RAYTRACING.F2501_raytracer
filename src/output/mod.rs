use crate::image_buffer::ImageBuffer;

mod window;
pub(crate) use window::{WindowOutput, WindowColorEncoder};
use crate::helpers::Pixel;

pub(crate) trait OutputColorEncoder {
    fn to_output(pixel: &Pixel) -> u32;
}

pub(crate) trait Output<const W: usize, const H: usize>{
    type ColoEncoder: OutputColorEncoder;
    
    fn render_buffer(&mut self, buffer: &ImageBuffer<W, H>) where [(); W*H]:;
}

pub(crate) trait OutputInteractive<const W: usize, const H: usize>{
    type Output: Output<W, H>;
    
    fn render_loop<F: FnMut(&mut Self::Output)>(&mut self, cb: F);
    
    fn render_static<F: FnOnce(&mut Self::Output)>(&mut self, cb: F);
}