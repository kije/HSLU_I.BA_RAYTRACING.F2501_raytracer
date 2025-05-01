use rayon::prelude::*;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Debug)]
#[repr(transparent)]
pub struct ImageBuffer<const W: usize, const H: usize>
where
    [(); W * H]:,
{
    buffer: [AtomicU32; W * H],
}

impl<const W: usize, const H: usize> ImageBuffer<W, H>
where
    [(); W * H]:,
{
    // Constants for cache-friendly processing
    // A typical cache line is 64 bytes which fits 16 u32 values
    const CACHE_LINE_SIZE_BYTES: usize = 64;
    const U32_SIZE_BYTES: usize = std::mem::size_of::<u32>();
    const U32_PER_CACHE_LINE: usize = Self::CACHE_LINE_SIZE_BYTES / Self::U32_SIZE_BYTES;

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

    pub const fn get_u32_slice(&self) -> &[u32] {
        unsafe {
            // Convert the pointer of the atomic array to a pointer of u32.
            std::slice::from_raw_parts(self.buffer.as_ptr() as *const u32, self.buffer.len())
        }
    }

    // Process the image in parallel chunks of chunk_w x chunk_h size
    // with cache-line aware distribution
    pub fn process_chunks_parallel<F>(&self, chunk_w: usize, chunk_h: usize, f: F)
    where
        F: Fn(ChunkView<W, H>) + Send + Sync,
    {
        // Ensure chunk width is cache-line aligned to minimize false sharing
        // Each row of a chunk should ideally start at a cache line boundary
        let aligned_chunk_w =
            if chunk_w % Self::U32_PER_CACHE_LINE != 0 && chunk_w > Self::U32_PER_CACHE_LINE {
                // Round up to nearest multiple of U32_PER_CACHE_LINE
                ((chunk_w + Self::U32_PER_CACHE_LINE - 1) / Self::U32_PER_CACHE_LINE)
                    * Self::U32_PER_CACHE_LINE
            } else {
                chunk_w
            };

        // Calculate number of chunks in each dimension
        let chunks_x = (W + aligned_chunk_w - 1) / aligned_chunk_w;
        let chunks_y = (H + chunk_h - 1) / chunk_h;

        // Process all chunks in parallel
        // Use chunks_y as the outer loop to maximize row-wise locality
        // FIXME: only one parallel iterator 0..(chunks_y * chunks_x) and then calculate chunk_y & chunk_x index via division and modulo
        (0..(chunks_y * chunks_x))
            .into_par_iter()
            .for_each(|chunk_index| {
                let chunk_x = chunk_index % chunks_x;
                let chunk_y = chunk_index / chunks_x;
                //(0..chunks_y).into_par_iter().for_each(|chunk_y| {
                // (0..chunks_x).into_par_iter().for_each(|chunk_x| {
                // Calculate actual chunk dimensions (may be smaller at edges)
                let start_x = chunk_x * aligned_chunk_w;
                let start_y = chunk_y * chunk_h;
                let end_x = (start_x + aligned_chunk_w).min(W);
                let end_y = (start_y + chunk_h).min(H);

                // Create a view of this chunk
                let chunk_view = ChunkView {
                    buffer: &self.buffer,
                    width: W,
                    start_x,
                    start_y,
                    end_x,
                    end_y,
                };

                // Process the chunk
                f(chunk_view);
            });
        //});
    }

    // Alternative API: process the image with a specific chunk size and return results
    pub fn process_chunks_parallel_with_result<F, R>(
        &self,
        chunk_w: usize,
        chunk_h: usize,
        f: F,
    ) -> Vec<R>
    where
        F: Fn(ChunkView<W, H>) -> R + Send + Sync + Clone,
        R: Send,
    {
        // Calculate number of chunks in each dimension
        let chunks_x = (W + chunk_w - 1) / chunk_w;
        let chunks_y = (H + chunk_h - 1) / chunk_h;

        // Process all chunks in parallel and collect results
        (0..chunks_y)
            .into_par_iter()
            .flat_map(|chunk_y| {
                let f = f.clone();
                (0..chunks_x).into_par_iter().map(move |chunk_x| {
                    // Calculate actual chunk dimensions (may be smaller at edges)
                    let start_x = chunk_x * chunk_w;
                    let start_y = chunk_y * chunk_h;
                    let end_x = (start_x + chunk_w).min(W);
                    let end_y = (start_y + chunk_h).min(H);

                    // Create a view of this chunk
                    let chunk_view = ChunkView {
                        buffer: &self.buffer,
                        width: W,
                        start_x,
                        start_y,
                        end_x,
                        end_y,
                    };

                    // Process the chunk and return result
                    f(chunk_view)
                })
            })
            .collect()
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

// A view into a chunk of the image buffer
pub struct ChunkView<'a, const W: usize, const H: usize>
where
    [(); W * H]:,
{
    buffer: &'a [AtomicU32; W * H],
    width: usize,   // Total width of the whole image buffer
    start_x: usize, // Starting x-coordinate of this chunk
    start_y: usize, // Starting y-coordinate of this chunk
    end_x: usize,   // Ending x-coordinate of this chunk (exclusive)
    end_y: usize,   // Ending y-coordinate of this chunk (exclusive)
}

impl<'a, const W: usize, const H: usize> ChunkView<'a, W, H>
where
    [(); W * H]:,
{
    // Get the width of this chunk
    pub fn width(&self) -> usize {
        self.end_x - self.start_x
    }

    // Get the height of this chunk
    pub fn height(&self) -> usize {
        self.end_y - self.start_y
    }

    // Get chunk coordinates
    pub fn start_x(&self) -> usize {
        self.start_x
    }
    pub fn start_y(&self) -> usize {
        self.start_y
    }
    pub fn end_x(&self) -> usize {
        self.end_x
    }
    pub fn end_y(&self) -> usize {
        self.end_y
    }

    #[inline(always)]
    pub fn global_coordinates(&self, x: usize, y: usize) -> (usize, usize) {
        let global_x = self.start_x + x;
        let global_y = self.start_y + y;

        (global_x, global_y)
    }

    // Calculate buffer index from coordinates with proper bounds checking
    #[inline(always)]
    fn index_from_coords(&self, x: usize, y: usize) -> usize {
        let (global_x, global_y) = self.global_coordinates(x, y);

        assert!(
            global_x < self.end_x && global_y < self.end_y,
            "Coordinates out of chunk bounds"
        );

        global_y * self.width + global_x
    }

    // Get a pixel value
    #[inline(always)]
    pub fn get(&self, x: usize, y: usize) -> u32 {
        let index = self.index_from_coords(x, y);
        self.buffer[index].load(Ordering::Relaxed)
    }

    // Set a pixel value
    #[inline(always)]
    pub fn set(&self, x: usize, y: usize, value: u32) {
        let index = self.index_from_coords(x, y);
        self.buffer[index].store(value, Ordering::Relaxed)
    }

    // Modify a pixel with a function
    pub fn modify<F>(&self, x: usize, y: usize, f: F)
    where
        F: FnOnce(u32) -> u32,
    {
        let index = self.index_from_coords(x, y);

        // Simple approach: load, modify, store
        // Note: For more complex atomic operations, you might want to use
        // compare_exchange or fetch_add
        let old_value = self.buffer[index].load(Ordering::Relaxed);
        let new_value = f(old_value);
        self.buffer[index].store(new_value, Ordering::Relaxed);
    }

    // Iterate over all coordinates in this chunk in a cache-friendly manner
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(usize, usize, u32),
    {
        // Process row by row for better cache locality
        for y in 0..self.height() {
            // Process each row in chunks of 16 elements (typical cache line)
            let row_width = self.width();

            // Process elements within each row
            for x in 0..row_width {
                let value = self.get(x, y);
                f(x, y, value);
            }
        }
    }

    // Process the chunk row by row without creating separate row objects
    // This is safer and avoids lifetime issues
    pub fn process_rows<F>(&self, mut f: F)
    where
        F: FnMut(usize, &dyn Fn(usize) -> u32, &dyn Fn(usize, u32)),
    {
        for y in 0..self.height() {
            // Create closure for getting values from this row
            let get_row = |x: usize| self.get(x, y);

            // Create closure for setting values in this row
            let set_row = |x: usize, value: u32| self.set(x, y, value);

            // Process this row
            f(y, &get_row, &set_row);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_chunked_access() {
        // Create a 1024x768 image
        let image = ImageBuffer::<100, 100>::new_with_color::<0>();

        // Process in 64x64 chunks, optimized for cache-line boundaries
        image.process_chunks_parallel(64, 64, |chunk| {
            // Cache-friendly approach: process row by row
            chunk.process_rows(|y, get_row, set_row| {
                for x in 0..chunk.width() {
                    let value = get_row(x);
                    set_row(x, value + 1);
                }
            });
        });

        assert_eq!(
            image.get_u32_slice(),
            ImageBuffer::<100, 100>::new_with_color::<1>().get_u32_slice()
        );
    }
}
