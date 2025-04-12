use palette::rgb::Srgb;
use simba::simd::{
    SimdValue, WideBoolF32x4, WideBoolF32x8, WideBoolF64x4, WideF32x4, WideF32x8, WideF64x4,
};
use std::ops::Deref;
use std::time::{Duration, Instant};
use wide::{
    f32x4, f32x8, f64x2, f64x4, i8x16, i8x32, i16x8, i16x16, i32x4, i32x8, i64x2, i64x4, u8x16,
    u16x8, u16x16, u32x4, u32x8, u64x2, u64x4,
};

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

// fixme move this somewhere else / other module
pub(crate) trait Splatable<Source> {
    /// Create a new instance by "splatting" the source value across all SIMD lanes
    fn splat(source: &Source) -> Self;
}

macro_rules! impl_splatable_primitives {
    ($($t: ty),*) => {
      $(
        impl crate::helpers::Splatable<$t> for $t {
            #[inline(always)]
            fn splat(source: &$t) -> Self {
                *source
            }
        }
      )*
    };
}

impl_splatable_primitives!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64, bool
);

macro_rules! impl_splatable_wide {
    ($($t: ty = $ts: ty $( as $cast: ident)?),*) => {
      $(
        impl crate::helpers::Splatable<$ts> for $t {
            #[inline(always)]
            fn splat(source: &$ts) -> Self {
                <$t $(as $cast)?>::splat(*source)
            }
        }
      )*
    };
}

impl_splatable_wide!(
    f32x4 = f32,
    f32x8 = f32,
    WideF32x4 = f32 as SimdValue,
    WideF32x8 = f32 as SimdValue,
    f64x2 = f64,
    f64x4 = f64,
    WideF64x4 = f64 as SimdValue,
    i8x16 = i8,
    i8x32 = i8,
    i16x8 = i16,
    i16x16 = i16,
    i32x4 = i32,
    i32x8 = i32,
    i64x2 = i64,
    i64x4 = i64,
    u8x16 = u8,
    u16x8 = u16,
    u16x16 = u16,
    u32x4 = u32,
    u32x8 = u32,
    u64x2 = u64,
    u64x4 = u64,
    WideBoolF64x4 = bool as SimdValue,
    WideBoolF32x4 = bool as SimdValue,
    WideBoolF32x8 = bool as SimdValue
);

#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct RenderTiming {
    pub iteration: u128,
    pub elapsed_time_since_start: Duration,
    pub delta: Duration,
    last_update_time_since_start: Duration,
    start_time: Instant,
}

impl Default for RenderTiming {
    #[inline]
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
