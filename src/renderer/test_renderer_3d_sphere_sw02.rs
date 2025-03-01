use std::marker::PhantomData;
use color::{OpaqueColor, Srgb};
use itertools::{Chunk, Itertools};
use ultraviolet::{f32x4, Vec2x4, Vec3, Vec3x4, Vec3x8};
use wide::{f32x8, CmpGt};
use crate::image_buffer::ImageBuffer;
use crate::renderer::{RenderCoordinates, Renderer};
use crate::{WINDOW_HEIGHT, WINDOW_WIDTH};
use crate::helpers::Pixel;
use crate::output::OutputColorEncoder;


struct Ray {
    origin: Vec3,
    direction: Vec3,
}

struct Rayx4 {
    origin: Vec3x4,
    direction: Vec3x4,
}

struct Rayx8 {
    origin: Vec3x8,
    direction: Vec3x8,
}

trait Intersectable {
    fn intersect(&self, ray: &Ray) -> Option<(Vec3, f32)>;
    fn intersect_x4(&self, ray: &Rayx4) -> (Vec3x4, f32x4);
    fn intersect_x8(&self, ray: &Rayx8) -> (Vec3x8, f32x8);
}


struct SphereData {
    c: Vec3,
    r: f32,
    color: OpaqueColor<Srgb>
}

macro_rules! intersect_sphere_simd_impl {
    ($fnname: ident,$x: ident) => {
        fn $fnname(&self, ray: &concat_idents!(Ray, $x)) -> (concat_idents!(Vec3, $x), concat_idents!(f32, $x)) {
            type VecType = concat_idents!(Vec3, $x);
            type F32Type = concat_idents!(f32, $x);

            let u = ray.direction;
            let v = ray.origin - VecType::splat(self.c);

            let a = u.dot(u);
            let b = 2.0 * u.dot(v);
            let c = v.dot(v) - self.r * self.r;

            let discriminant = b * b - 4.0 * a * c;

            let discriminant_pos = discriminant.cmp_gt(F32Type::splat(0.0));
            let discriminant_sqrt = discriminant.sqrt();


            let t1 = (-b - discriminant_sqrt) / (2.0 * a);
            let t1_valid = t1.cmp_gt(F32Type::splat(0.0)) & discriminant_pos;

            // let t2 = (-b + discriminant_sqrt) / (2.0 * a);
            // let t2_valid = t2.cmp_gt(F32Type::splat(0.0)) & discriminant_pos;
            //
            // let t = t2_valid.blend(t2, F32Type::splat(-f32::MIN));
            // let t = t1_valid.blend(t1, t);

            let t = t1_valid.blend(t1, F32Type::splat(f32::MIN));

            let q1 = (t * u) + ray.origin;

            (q1, t)
        }
    };
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

        let mut t = (-b - discriminant.sqrt()) / (2.0 * a);

        if t <= 0.0 {
             let t2 = (-b + discriminant.sqrt()) / (2.0 * a);
            if t2 > 0.0 {
                t = t2;
            } else {
                return None;
            }
        }

        let q1 = (t * u) + ray.origin;

        Some((q1, t))
    }

    intersect_sphere_simd_impl!(intersect_x4, x4);
    intersect_sphere_simd_impl!(intersect_x8, x8);
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
    color: OpaqueColor::from_rgb8(158,0,255)
};

static SPHERE_5: SphereData = SphereData {
    c: Vec3::new(1.25 * (WINDOW_WIDTH as f32 / 2.5), 0.5 * (WINDOW_HEIGHT as f32 / 2.5), 90.0),
    r: 30.0,
    color: OpaqueColor::from_rgb8(128,210,255)
};

static SPHERE_6: SphereData = SphereData {
    c: Vec3::new((WINDOW_WIDTH as f32 / 2.5), 2.25 * (WINDOW_HEIGHT as f32 / 2.5), 500.0),
    r: 250.0,
    color: OpaqueColor::from_rgb8(254,255,255)
};

static SPHERE_7: SphereData = SphereData {
    c: Vec3::new(WINDOW_WIDTH as f32 / 4.0, 3.0* (WINDOW_HEIGHT as f32 / 4.0), 20.0),
    r: 10.0,
    color: OpaqueColor::from_rgb8(255,55,77)
};

static SPHERE_8: SphereData = SphereData {
    c: Vec3::new(WINDOW_WIDTH as f32 / 3.0, 3.0* (WINDOW_HEIGHT as f32 / 6.0), 30.0),
    r: 25.0,
    color: OpaqueColor::from_rgb8(55,230,180)
};

static SPHERES: [&'static SphereData; 8] = [&SPHERE_1, &SPHERE_2, &SPHERE_3, &SPHERE_4, &SPHERE_5, &SPHERE_6, &SPHERE_7, &SPHERE_8];

static RENDER_RAY_DIRECTION: Vec3 = Vec3::new(0.0, 0.0, 1.0);

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct TestRenderer3DSphereSW02<C: OutputColorEncoder>(PhantomData<C>);

impl< C: OutputColorEncoder> TestRenderer3DSphereSW02<C> {
    fn get_pixel_color(RenderCoordinates { x, y}: RenderCoordinates) -> Option<Pixel> {
        let ray = Ray {
            origin: Vec3::new(x as f32, y as f32, 0.0),
            direction: RENDER_RAY_DIRECTION
        };

        let mut pixel_color: Option<OpaqueColor<Srgb>> = None;

        for sphere in SPHERES.into_iter().rev(){
            if let Some((_, distance)) = sphere.intersect(&ray) {
                let d = distance / sphere.c.mag();
                let color = sphere.color.map_lightness(|l| l - 2.0*d);
                pixel_color = Some(color);
                break;
            }
        }

        pixel_color.map(|c| Pixel(c))
    }

    fn render_pixel_colors(coords: &[(usize,RenderCoordinates)], set_pixel: &dyn Fn(usize,Pixel))  {
        coords.into_iter().chunks(8).into_iter().enumerate().for_each(|(chunk_index, chunk)| {
            let (idxs, xs, ys, zs): (Vec<usize>, Vec<f32>,Vec<f32>,Vec<f32>) = chunk.into_iter().map(|(i, coord)| (i, coord.x as f32, coord.y as f32, 0.0)).collect();
            
            macro_rules! impl_render_pixel_colors_simd {
                ($fnname: ident, $xn: ident, $xs: ident, $ys: ident, $zs: ident, $idxs: ident, $set_pixel: ident) => {
                    type VecType = concat_idents!(Vec3, $xn);
                    type F32Type = concat_idents!(f32, $xn);
                    type RayType = concat_idents!(Ray, $xn);
                    
                    let ray = RayType {
                        origin: VecType::new(F32Type::from($xs.as_slice()), F32Type::from($ys.as_slice()), F32Type::from($zs.as_slice())),
                        direction: VecType::splat(RENDER_RAY_DIRECTION)
                    };
            
                    for sphere in SPHERES {
                        let (_, distances) = sphere.$fnname(&ray);
            
                        let d = distances / sphere.c.mag();
            
                        for (i, v) in d.as_array_ref().iter().enumerate() {
                            if *v < 0.0 {
                                continue;
                            }
            
                            $set_pixel($idxs[i], Pixel(sphere.color.map_lightness(|l| l - 2.0 * v)))
                        }
                    }
                };
            }

            if xs.len() == 8 {
                impl_render_pixel_colors_simd!(intersect_x8, x8, xs, ys, zs, idxs, set_pixel);
            } else if xs.len() == 4 {
                impl_render_pixel_colors_simd!(intersect_x4, x4, xs, ys, zs, idxs, set_pixel);
            } else {
                todo!("Todo impl");
                //xs.into_iter().zip(ys.into_iter()).map(|(x,y)|RenderCoordinates { x: x.floor() as usize, y: y.floor() as usize}).map(Self::get_pixel_color).collect::<Vec<_>>()
            }
        });
    }
}

impl<const W: usize, const H: usize, C: OutputColorEncoder> Renderer<W, H, C> for TestRenderer3DSphereSW02<C> {
    fn render(&self, buffer: &ImageBuffer<W, H>)
    where
        [(); W * H]:
    {
        //Self::render_to_buffer(buffer, Self::get_pixel_color)
        Self::render_to_buffer_chunked_inplace(buffer, Self::render_pixel_colors)
    }
}