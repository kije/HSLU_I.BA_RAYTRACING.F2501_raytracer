use std::ops::{Deref, DerefMut};

pub(crate) struct ImageBuffer<const W: usize, const H: usize> where [(); W*H]: {
    buffer: [u32; W*H],
}

impl<const W: usize, const H: usize> ImageBuffer<W, H> where [(); W*H]: {
    pub fn new() -> Self {
        ImageBuffer {
            buffer: [0x000000; W*H]
        }
    }
}

impl<const W: usize, const H: usize> Deref for ImageBuffer<W, H> where [(); W*H]: {
    type Target = [u32; W*H];
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<const W: usize, const H: usize> DerefMut for ImageBuffer<W, H> where [(); W*H]: {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}