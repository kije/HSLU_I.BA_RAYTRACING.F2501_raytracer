use crate::image_buffer::ImageBuffer;

mod window;
pub(crate) use window::WindowOutput;

pub(crate) trait Output<const W: usize, const H: usize>{
    fn render_buffer(&mut self, buffer: &ImageBuffer<W, H>) where [(); W*H]:;
}

pub(crate) trait OutputInteractive<const W: usize, const H: usize>{
    type Output: Output<W, H>;
    
    fn render_loop<F: FnMut(&mut Self::Output)>(&mut self, cb: F);
    fn render_static<F: FnOnce(&mut Self::Output)>(&mut self, cb: F);
}