use palette::rgb::Srgb;
use std::ops::Deref;
use std::time::{Duration, Instant};

pub(crate) type ColorType<T = f32> = Srgb<T>;

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(transparent)]
pub(crate) struct Pixel(pub ColorType);

impl Pixel {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self(ColorType::new(
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
        ))
    }
}

impl From<(u8, u8, u8)> for Pixel {
    fn from((r, g, b): (u8, u8, u8)) -> Self {
        Self::new(r, g, b)
    }
}

impl Deref for Pixel {
    type Target = Srgb;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct RenderTiming {
    pub iteration: u128,
    pub elapsed_time_since_start: Duration,
    pub delta: Duration,
    last_update_time_since_start: Duration,
    start_time: Instant,
}

impl Default for RenderTiming {
    fn default() -> Self {
        Self {
            start_time: Instant::now(),
            iteration: Default::default(),
            elapsed_time_since_start: Default::default(),
            delta: Default::default(),
            last_update_time_since_start: Default::default(),
        }
    }
}

impl RenderTiming {
    #[inline(always)]
    pub fn next(&mut self) {
        self.elapsed_time_since_start = Instant::now() - self.start_time;
        self.delta = self.elapsed_time_since_start - self.last_update_time_since_start;
        self.last_update_time_since_start = self.elapsed_time_since_start.clone();
        self.iteration += 1;
    }
}

#[inline(always)]
pub(crate) const fn fast_inverse(value: f32) -> f32 {
    debug_assert!(value >= 0.0);
    f32::from_bits(0x7f00_0000 - value.to_bits())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_fast_inverse() {
        for i in 1..1000_000 {
            let v = i as f32 / 9.25;
            let v = v * 20.0;

            let v_inv = 1.0 / v;
            let v_inv_fast = fast_inverse(v);

            assert!(
                (v_inv - v_inv_fast).abs() < 0.05,
                "{} != {} (episoln {})",
                v_inv,
                v_inv_fast,
                (v_inv - v_inv_fast).abs()
            );
        }
    }
}
