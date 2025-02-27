use std::marker::PhantomData;
use ultraviolet::{UVec2, Vec2};
use crate::image_buffer::ImageBuffer;
use crate::renderer::{RenderCoordinates, Renderer};
use crate::{WINDOW_HEIGHT, WINDOW_WIDTH};
use crate::helpers::Pixel;
use crate::output::OutputColorEncoder;

const CIRCLE_X0: u32 = WINDOW_WIDTH as u32 / 2;
const CIRCLE_Y0: u32 = WINDOW_HEIGHT  as u32 / 2;
const CIRCLE_RADIUS: u32 = WINDOW_WIDTH  as u32 / 3;

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct TestRendererVector<C: OutputColorEncoder>(PhantomData<C>);

impl< C: OutputColorEncoder> TestRendererVector<C> {
    fn get_pixel_color(RenderCoordinates { x, y}: RenderCoordinates) -> Option<Pixel> {
        let p = UVec2::new(x as u32,y as u32);
        let c = UVec2::new(CIRCLE_X0 , CIRCLE_Y0 );
        let d = p - c;


        if d.mag() <= CIRCLE_RADIUS {
            return Some(Pixel::new((x%255) as u8, (y %255) as u8 , (((x*y) / 7) %255) as u8));
        }

        None
    }
}

impl<const W: usize, const H: usize, C: OutputColorEncoder> Renderer<W, H, C> for TestRendererVector<C> {
    fn render(&self, buffer: &mut ImageBuffer<W, H>)
    where
        [(); W * H]:
    {
        Self::render_to_buffer(buffer, Self::get_pixel_color)
    }
}