use crate::helpers::{Pixel, RenderTiming};
use crate::image_buffer::ImageBuffer;
use crate::output::{Output, OutputColorEncoder, OutputInteractive};
use minifb::{Key, Result as WindowResult, Scale, ScaleMode, Window, WindowOptions};
use palette::{Srgb, rgb};

#[derive(Debug)]
pub(crate) struct WindowOutputInner<const W: usize, const H: usize> {
    window: Window,
}

impl<const W: usize, const H: usize> Output<W, H> for WindowOutputInner<W, H> {
    type ColorEncoder = WindowColorEncoder;

    fn render_buffer(&mut self, buffer: &ImageBuffer<W, H>)
    where
        [(); W * H]:,
    {
        self.window
            .update_with_buffer(buffer.get_u32_slice(), W, H)
            .unwrap()
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
                        title: false,
                        resize: false,
                        borderless: true,
                        scale: Scale::FitScreen,
                        scale_mode: ScaleMode::AspectRatioStretch,
                        transparency: true,
                        ..WindowOptions::default()
                    },
                )?,
            },
        })
    }
}

impl<const W: usize, const H: usize> OutputInteractive<W, H> for WindowOutput<W, H> {
    type Output = WindowOutputInner<W, H>;

    fn render_loop<F: FnMut(&mut Self::Output, &RenderTiming)>(&mut self, mut cb: F) {
        self.inner.window.update();

        let mut timing = RenderTiming::default();
        while self.inner.window.is_open() && !self.inner.window.is_key_down(Key::Escape) {
            cb(&mut self.inner, &timing);
            timing.next();
        }
    }

    fn render_static<F: FnOnce(&mut Self::Output, &RenderTiming)>(
        &mut self,
        cb: F,
        timing: Option<RenderTiming>,
    ) {
        self.inner.window.update();

        cb(&mut self.inner, &timing.unwrap_or_default());

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
        u32::from(pixel.0.into_format::<u8>())
    }
}
