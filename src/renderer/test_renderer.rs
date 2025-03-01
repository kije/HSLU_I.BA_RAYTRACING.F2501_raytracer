use crate::helpers::Pixel;
use crate::image_buffer::ImageBuffer;
use crate::output::OutputColorEncoder;
use crate::renderer::{RenderCoordinates, Renderer};
use crate::{WINDOW_HEIGHT, WINDOW_WIDTH};
use std::marker::PhantomData;

const CIRCLE_X0: usize = WINDOW_WIDTH / 2;
const CIRCLE_Y0: usize = WINDOW_HEIGHT / 2;
const CIRCLE_RADIUS: usize = WINDOW_WIDTH / 3;

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct TestRenderer<C: OutputColorEncoder>(PhantomData<C>);

impl<C: OutputColorEncoder> TestRenderer<C> {
    fn get_pixel_color(RenderCoordinates { x, y }: RenderCoordinates) -> Option<Pixel> {
        let rel_x = x.abs_diff(CIRCLE_X0);
        let rel_y = y.abs_diff(CIRCLE_Y0);

        if rel_x.pow(2) + rel_y.pow(2) <= CIRCLE_RADIUS.pow(2) {
            return Some(Pixel::new(
                (x % 255) as u8,
                (y % 255) as u8,
                (((x * y) / 7) % 255) as u8,
            ));
        }

        None
    }
}

impl<const W: usize, const H: usize, C: OutputColorEncoder> Renderer<W, H, C> for TestRenderer<C> {
    fn render(&self, buffer: &ImageBuffer<W, H>)
    where
        [(); W * H]:,
    {
        Self::render_to_buffer(buffer, Self::get_pixel_color)
    }
}
