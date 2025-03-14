use std::ops::{Deref, DerefMut};
use std::sync::atomic::AtomicU32;

pub(crate) struct ImageBuffer<const W: usize, const H: usize>
where
    [(); W * H]:,
{
    buffer: [AtomicU32; W * H],
}

impl<const W: usize, const H: usize> ImageBuffer<W, H>
where
    [(); W * H]:,
{
    pub const fn new() -> Self {
        ImageBuffer {
            buffer: [const { AtomicU32::new(0) }; W * H],
        }
    }

    pub const fn new_with_color<const COLOR: u32>() -> Self {
        ImageBuffer {
            buffer: [const { AtomicU32::new(COLOR) }; W * H],
        }
    }

    pub(crate) const fn get_u32_slice(&self) -> &[u32] {
        unsafe {
            // Convert the pointer of the atomic array to a pointer of u32.
            std::slice::from_raw_parts(self.buffer.as_ptr() as *const u32, self.buffer.len())
        }
    }
}

impl<const W: usize, const H: usize> Deref for ImageBuffer<W, H>
where
    [(); W * H]:,
{
    type Target = [AtomicU32; W * H];
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<const W: usize, const H: usize> DerefMut for ImageBuffer<W, H>
where
    [(); W * H]:,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}
