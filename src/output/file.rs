use crate::helpers::{ColorType, Pixel, RenderTiming};
use crate::image_buffer::ImageBuffer;
use crate::output::{Output, OutputColorEncoder, OutputInteractive};
use crate::{WINDOW_HEIGHT, WINDOW_SCENE_DEPTH, WINDOW_WIDTH};
use minifb::{
    CursorStyle, HasWindowHandle, Key, Result as WindowResult, Scale, ScaleMode, Window,
    WindowOptions,
};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::thread;

#[derive(Debug)]
pub struct FileOutputInner<const W: usize, const H: usize> {
    path: Path,
}

#[derive(Debug)]
pub struct FileOutput<const W: usize, const H: usize, P: AsRef<Path> + Sized> {
    path: P,
}

impl<P: AsRef<Path> + Sized, const W: usize, const H: usize> Output<W, H> for FileOutput<W, H, P> {
    type ColorEncoder = FileColorEncoder;

    fn render_buffer(&mut self, buffer: &ImageBuffer<W, H>)
    where
        [(); W * H]:,
    {
        //self.path.file_prefix()
        let file = File::create(self.path.as_ref()).unwrap();
        let ref mut w = BufWriter::new(file);

        let mut encoder = png::Encoder::new(w, W as u32, H as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);

        let mut writer = encoder.write_header().unwrap();

        let data: Vec<_> = buffer
            .get_u32_slice()
            .iter()
            .map(|&p| FileColorEncoder::from_output(p).into_format::<u8>())
            .flat_map(|p| [p.red, p.green, p.blue])
            .collect();

        writer.write_image_data(&data).unwrap(); // Save
    }
}

impl<const W: usize, const H: usize, P: AsRef<Path> + Sized> FileOutput<W, H, P> {
    pub fn new(path: P) -> Self {
        Self { path }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FileColorEncoder;

impl OutputColorEncoder for FileColorEncoder {
    #[inline(always)]
    fn to_output(pixel: &Pixel) -> u32 {
        pixel.0.into_format::<u8>().into()
    }

    #[inline(always)]
    fn from_output(pixel: u32) -> Pixel {
        ColorType::from(pixel).into_format::<f32>().into()
    }
}
