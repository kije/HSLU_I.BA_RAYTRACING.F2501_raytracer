use crate::output::{Output, OutputColorEncoder, OutputInteractive};
use minifb::{Key, Window, WindowOptions, Result as WindowResult, Scale};
use crate::helpers::Pixel;
use crate::image_buffer::ImageBuffer;

#[derive(Debug)]
pub(crate) struct WindowOutputInner<const W: usize, const H: usize> {
    window: Window
}

impl<const W: usize, const H: usize> Output<W, H> for WindowOutputInner<W, H> {
    type ColoEncoder = WindowColorEncoder;

    fn render_buffer(&mut self, buffer: &ImageBuffer<W, H>) where [(); W*H]: {
        self.window.update_with_buffer(buffer.as_ref(), W, H).unwrap()
    }
}

#[derive(Debug)]
pub(crate) struct WindowOutput<const W: usize, const H: usize> {
    inner: WindowOutputInner<W, H>,
}

impl<const W: usize, const H: usize> WindowOutput<W, H> {
    pub(crate) fn new() -> WindowResult<Self> {
        Ok(Self {
            inner: WindowOutputInner::<W, H> {
                window: Window::new(
                    "Minimal Raytracer - Rust",
                    W,
                    H,
                    WindowOptions {
                        resize: false,
                        borderless: true,
                        scale: Scale::FitScreen,
                        transparency: true,
                        ..WindowOptions::default()
                    },
                )?
            }
        })
    }
}

impl<const W: usize, const H: usize> OutputInteractive<W, H> for WindowOutput<W, H> {
    type Output = WindowOutputInner<W, H>;

    fn render_loop<F: FnMut(&mut Self::Output)>(&mut self, mut cb: F) {
        self.inner.window.update();

        while self.inner.window.is_open() && !self.inner.window.is_key_down(Key::Escape) {
            cb(&mut self.inner)
        }
    }

    fn render_static<F: FnOnce(&mut Self::Output)>(&mut self, cb: F) {
        self.inner.window.update();

        cb(&mut self.inner);

        while self.inner.window.is_open() && !self.inner.window.is_key_down(Key::Escape) {
            self.inner.window.update();
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct WindowColorEncoder;

impl OutputColorEncoder for WindowColorEncoder {
    #[inline(always)]
    fn to_output(pixel: &Pixel) -> u32 {
        let mut x = pixel.0.to_rgba8().to_u8_array();
        x.rotate_right(1);
        u32::from_be_bytes(x)
    }
}