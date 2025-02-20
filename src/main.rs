use std::thread;
use std::thread::sleep;
use std::time::Duration;
use minifb::{Key, Window, WindowOptions};
use rayon::prelude::*;

const WINDOW_WIDTH: usize = 800;
const WINDOW_HEIGHT: usize = 600;

const CIRCLE_X0: usize = WINDOW_WIDTH / 2;
const CIRCLE_Y0: usize = WINDOW_HEIGHT / 2;
const CIRCLE_RADIUS: usize = 80;

fn get_pixel_color(x: usize, y: usize) -> Option<u32> {
    let rel_x = x.abs_diff(CIRCLE_X0);
    let rel_y = y.abs_diff(CIRCLE_Y0);

    if rel_x.pow(2) + rel_y.pow(2) <= CIRCLE_RADIUS.pow(2) {
        return Some(((x as u32%255 ) << 16) | ((y as u32 %255 ) << 8) | 0);
    }

    None
}

fn render_to_buffer(buffer: &mut [u32; WINDOW_WIDTH * WINDOW_HEIGHT]) {
    buffer.par_iter_mut().enumerate().for_each(|(i, p)| {
        let x = i % WINDOW_WIDTH;
        let y = i / WINDOW_WIDTH;

        if let Some(pixel_color) = get_pixel_color(x, y) {
            *p = pixel_color;
        }
    });
}

fn main() {
    let mut buffer: [u32; WINDOW_WIDTH * WINDOW_HEIGHT] = [0x000000; WINDOW_WIDTH * WINDOW_HEIGHT];
     render_to_buffer(&mut buffer);

    let mut window = Window::new(
        "Minimal Raytracer - Rust",
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .expect("Unable to open window");

    //let mut i = 0;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        window.update_with_buffer(&buffer, WINDOW_WIDTH, WINDOW_HEIGHT).unwrap();
        // i += 1;
        // buffer[i] = 0x00FF00;
        // thread::sleep(Duration::from_nanos(1_000_000_000 / 60));
    }
}
