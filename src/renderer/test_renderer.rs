
use crate::image_buffer::ImageBuffer;
use crate::renderer::{RenderCoordinates, Renderer};
use crate::{WINDOW_HEIGHT, WINDOW_WIDTH};
use crate::helpers::Pixel;

const CIRCLE_X0: usize = WINDOW_WIDTH / 2;
const CIRCLE_Y0: usize = WINDOW_HEIGHT / 2;
const CIRCLE_RADIUS: usize = 80;

pub(crate) struct TestRenderer;

impl TestRenderer {
    fn get_pixel_color(RenderCoordinates { x, y}: RenderCoordinates) -> Option<Pixel> {
        let rel_x = x.abs_diff(CIRCLE_X0);
        let rel_y = y.abs_diff(CIRCLE_Y0);

        if rel_x.pow(2) + rel_y.pow(2) <= CIRCLE_RADIUS.pow(2) {
            return Some((((x as u32%255 ) << 16) | ((y as u32 %255 ) << 8) | 0).into());
        }

        None
    }
}

impl<const W: usize, const H: usize> Renderer<W, H> for TestRenderer {
    fn render(&self, buffer: &mut ImageBuffer<W, H>)
    where
        [(); W * H]:
    {
        Self::render_to_buffer(buffer, Self::get_pixel_color)
    }
}