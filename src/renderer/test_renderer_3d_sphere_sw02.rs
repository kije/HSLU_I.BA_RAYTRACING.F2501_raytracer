use std::marker::PhantomData;
use color::{OpaqueColor, Srgb};
use ultraviolet::{Vec3};
use crate::image_buffer::ImageBuffer;
use crate::renderer::{RenderCoordinates, Renderer};
use crate::{WINDOW_HEIGHT, WINDOW_WIDTH};
use crate::helpers::Pixel;
use crate::output::OutputColorEncoder;

struct Ray {
    origin: Vec3,
    direction: Vec3,
}

trait Intersectable {
    fn intersect(&self, ray: &Ray) -> Option<(Vec3, f32)>;
}


struct SphereData {
    c: Vec3,
    r: f32,
    color: OpaqueColor<Srgb>
}

impl Intersectable for  SphereData {
    fn intersect(&self, ray: &Ray) -> Option<(Vec3, f32)> {
        let u = ray.direction;
        let v = ray.origin - self.c;

        let a = u.dot(u);
        let b = 2.0 * u.dot(v);
        let c = v.dot(v) - self.r * self.r;

        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            return None;
        }

        let t1 = (-b - discriminant.sqrt()) / (2.0 * a);

        let q1 = (t1 * u) + ray.origin;

        Some((q1, t1))
    }
}

static SPHERE_1: SphereData = SphereData {
    c: Vec3::new(WINDOW_WIDTH as f32 / 2.0, WINDOW_HEIGHT as f32 / 2.0, 150.0),
    r: 70.0,
    color: OpaqueColor::from_rgb8(255,0,0)
};

static SPHERE_2: SphereData = SphereData {
    c: Vec3::new(WINDOW_WIDTH as f32 / 2.5, WINDOW_HEIGHT as f32 / 2.5, 150.0),
    r: 90.0,
    color: OpaqueColor::from_rgb8(0,255,0)
};

static SPHERE_3: SphereData = SphereData {
    c: Vec3::new(2.0 * (WINDOW_WIDTH as f32 / 2.5), WINDOW_HEIGHT as f32 / 2.5, 150.0),
    r: 90.0,
    color: OpaqueColor::from_rgb8(0,0,255)
};

static SPHERE_4: SphereData = SphereData {
    c: Vec3::new(2.0 * (WINDOW_WIDTH as f32 / 2.5), 2.0 * (WINDOW_HEIGHT as f32 / 2.5), 250.0),
    r: 120.0,
    color: OpaqueColor::from_rgb8(128,0,255)
};

static SPHERE_5: SphereData = SphereData {
    c: Vec3::new(1.25 * (WINDOW_WIDTH as f32 / 2.5), 0.5 * (WINDOW_HEIGHT as f32 / 2.5), 90.0),
    r: 30.0,
    color: OpaqueColor::from_rgb8(128,200,255)
};

static SPHERE_6: SphereData = SphereData {
    c: Vec3::new((WINDOW_WIDTH as f32 / 2.5), 2.25 * (WINDOW_HEIGHT as f32 / 2.5), 500.0),
    r: 250.0,
    color: OpaqueColor::from_rgb8(200,200,255)
};


#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct TestRenderer3DSphereSW02<C: OutputColorEncoder>(PhantomData<C>);

impl< C: OutputColorEncoder> TestRenderer3DSphereSW02<C> {
    fn get_pixel_color(RenderCoordinates { x, y}: RenderCoordinates) -> Option<Pixel> {

        let ray = Ray {
            origin: Vec3::new(x as f32, y as f32, 0.0),
            direction: Vec3::new(0.0, 0.0, 1.0)
        };

        let mut pixel_color: Option<OpaqueColor<Srgb>> = None;

        for sphere in [&SPHERE_1, &SPHERE_2, &SPHERE_3, &SPHERE_4, &SPHERE_5, &SPHERE_6] {
            if let Some((_, distance)) = sphere.intersect(&ray) {
                let d = distance / sphere.c.mag();
                let color = sphere.color.map_lightness(|l| l - 2.0*d);
                pixel_color = pixel_color.map_or(Some(color), |c| Some( color));
            }
        }

        pixel_color.map(|c| Pixel(c))
    }
}

impl<const W: usize, const H: usize, C: OutputColorEncoder> Renderer<W, H, C> for TestRenderer3DSphereSW02<C> {
    fn render(&self, buffer: &ImageBuffer<W, H>)
    where
        [(); W * H]:
    {
        Self::render_to_buffer(buffer, Self::get_pixel_color)
    }
}