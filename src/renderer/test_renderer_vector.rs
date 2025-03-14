use crate::helpers::{ColorType, Pixel};
use crate::image_buffer::ImageBuffer;
use crate::output::OutputColorEncoder;
use crate::renderer::{RenderCoordinates, Renderer};
use crate::{WINDOW_HEIGHT, WINDOW_WIDTH};
use std::marker::PhantomData;
use ultraviolet::UVec2;

struct CircleData {
    c: UVec2,
    r: u32,
    color: ColorType,
}

impl CircleData {
    fn is_inside(&self, p: UVec2) -> bool {
        let d = p - self.c;

        d.mag() <= self.r
    }
}

static CIRCLE_1: CircleData = CircleData {
    c: UVec2::new(WINDOW_WIDTH as u32 / 2, WINDOW_HEIGHT as u32 / 3),
    r: WINDOW_WIDTH as u32 / 4,
    color: ColorType::new(255.0 / 255.0, 0. / 255.0, 0. / 255.0),
};

static CIRCLE_2: CircleData = CircleData {
    c: UVec2::new(WINDOW_WIDTH as u32 / 3, 2 * (WINDOW_HEIGHT as u32 / 3)),
    r: WINDOW_WIDTH as u32 / 4,
    color: ColorType::new(0.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0),
};

static CIRCLE_3: CircleData = CircleData {
    c: UVec2::new(
        2 * (WINDOW_WIDTH as u32 / 3),
        2 * (WINDOW_HEIGHT as u32 / 3),
    ),
    r: WINDOW_WIDTH as u32 / 4,
    color: ColorType::new(0.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0),
};

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct TestRendererVector<C: OutputColorEncoder>(PhantomData<C>);

impl<C: OutputColorEncoder> TestRendererVector<C> {
    fn get_pixel_color(RenderCoordinates { x, y }: RenderCoordinates) -> Option<Pixel> {
        let p = UVec2::new(x as u32, y as u32);

        let mut pixel_color: Option<ColorType> = None;

        for circle in [&CIRCLE_1, &CIRCLE_2, &CIRCLE_3] {
            if circle.is_inside(p) {
                pixel_color = pixel_color.map_or(Some(circle.color), |c| Some(c + circle.color));
            }
        }

        pixel_color.map(|c| Pixel(c))
    }
}

impl<const W: usize, const H: usize, C: OutputColorEncoder> Renderer<W, H, C>
    for TestRendererVector<C>
{
    fn render(&self, buffer: &ImageBuffer<W, H>)
    where
        [(); W * H]:,
    {
        Self::render_to_buffer(buffer, Self::get_pixel_color)
    }
}
