use std::ops::Deref;
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