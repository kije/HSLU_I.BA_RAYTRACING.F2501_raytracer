use std::ops::Deref;
use std::time::{Duration, Instant};
use color::{OpaqueColor, Srgb};

#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct Pixel(pub OpaqueColor<Srgb>);

impl Pixel {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self(OpaqueColor::from_rgb8(r,g,b))
    }
}


impl From<(u8,u8,u8)> for Pixel {
    fn from((r,g,b): (u8,u8,u8)) -> Self {
        Self(OpaqueColor::from_rgb8(r,g,b))
    }
}


impl Deref for Pixel {
    type Target = OpaqueColor<Srgb>;
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
    start_time: Instant
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