
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Default)]
pub(crate) struct Pixel(pub u32);

impl From<u32> for Pixel {
    fn from(value: u32) -> Self {
        Self(value)
    }
}