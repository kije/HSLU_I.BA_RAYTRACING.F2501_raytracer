use crate::helpers::{Pixel, RenderTiming};
use crate::image_buffer::ImageBuffer;
use crate::output::{Output, OutputColorEncoder, OutputInteractive};
use minifb::{
    CursorStyle, HasWindowHandle, Key, Result as WindowResult, Scale, ScaleMode, Window,
    WindowOptions,
};
use std::thread;

#[derive(Debug)]
pub struct WindowOutputInner<const W: usize, const H: usize> {
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
pub struct WindowOutput<const W: usize, const H: usize> {
    inner: WindowOutputInner<W, H>,
}

impl<const W: usize, const H: usize> WindowOutput<W, H> {
    pub fn new() -> WindowResult<Self> {
        let mut window = Self {
            inner: WindowOutputInner::<W, H> {
                window: Window::new(
                    "Raytracer - Rust",
                    W,
                    H,
                    WindowOptions {
                        title: true,
                        resize: true,
                        borderless: true,
                        scale: Scale::FitScreen,
                        scale_mode: ScaleMode::AspectRatioStretch,
                        transparency: true,
                        ..WindowOptions::default()
                    },
                )?,
            },
        };

        window.inner.window.set_background_color(0, 0, 0);
        window.inner.window.set_cursor_style(CursorStyle::Crosshair);
        window.inner.window.set_title(
            format!(
                "Raytracer - {} | {}",
                if cfg!(feature = "simd_render") {
                    "SIMD"
                } else {
                    "Non-SIMD"
                },
                if cfg!(feature = "anti_aliasing") {
                    format!(
                        "Antialiasing {} {}",
                        if cfg!(feature = "anti_aliasing_rotation_scale") {
                            "ROS_SCL"
                        } else {
                            ""
                        },
                        if cfg!(feature = "anti_aliasing_randomness") {
                            "RNG"
                        } else {
                            ""
                        }
                    )
                    .trim()
                    .to_string()
                } else {
                    "Non-Antialiasing".to_string()
                }
            )
            .trim(),
        );

        Ok(window)
    }
}

impl<const W: usize, const H: usize> OutputInteractive<W, H> for WindowOutput<W, H> {
    type Output = WindowOutputInner<W, H>;

    fn render_loop<F: FnMut(&mut Self::Output, &RenderTiming)>(&mut self, mut cb: F) {
        self.inner.window.update();

        self.inner.window.set_target_fps(60);

        let mut timing = RenderTiming::default();
        while self.inner.window.is_open() && !self.inner.window.is_key_down(Key::Escape) {
            cb(&mut self.inner, &timing);
            timing.next();
            thread::yield_now();
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
            thread::yield_now();
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct WindowColorEncoder;

impl OutputColorEncoder for WindowColorEncoder {
    #[inline(always)]
    fn to_output(pixel: &Pixel) -> u32 {
        u32::from(pixel.0.into_format::<u8>())
    }
}
