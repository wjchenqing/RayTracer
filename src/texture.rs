pub use crate::hittable::*;
pub use crate::perlin::*;
pub use crate::randomtool::*;
pub use crate::ray::*;
pub use crate::vec3::*;
pub use image::GenericImageView;
pub use rand::random;
pub use std::cmp::Ordering;
pub use std::f64::consts::PI;
pub use std::f64::INFINITY;
use std::path::Path;
pub use std::sync::Arc;

pub trait Texture: Sync + Send {
    fn value(&self, u: f64, v: f64, p: &Vec3) -> Vec3;
}
#[derive(Clone)]
pub struct SolidColor {
    pub color_value: Vec3,
}
impl SolidColor {
    pub fn new(c: &Vec3) -> Self {
        Self { color_value: *c }
    }
}
impl Texture for SolidColor {
    fn value(&self, _u: f64, _v: f64, _p: &Vec3) -> Vec3 {
        self.color_value
    }
}
#[derive(Clone)]
pub struct CheckerTexture {
    pub odd: Arc<dyn Texture>,
    pub even: Arc<dyn Texture>,
}
impl CheckerTexture {
    pub fn new(t0: &Arc<dyn Texture>, t1: &Arc<dyn Texture>) -> Self {
        Self {
            even: t0.clone(),
            odd: t1.clone(),
        }
    }
    pub fn new_from_color(c1: &Vec3, c2: &Vec3) -> Self {
        Self {
            even: Arc::new(SolidColor::new(c1)),
            odd: Arc::new(SolidColor::new(c2)),
        }
    }
}
impl Texture for CheckerTexture {
    fn value(&self, u: f64, v: f64, p: &Vec3) -> Vec3 {
        let sines = (3.90 * p.x).sin() * (3.90 * (p.y + 0.11)).sin() * (3.90 * p.z).sin();
        if sines < 0.0 {
            self.odd.value(u, v, p)
        } else {
            self.even.value(u, v, p)
        }
    }
}
#[derive(Clone)]
pub struct NoiseTexture {
    pub noise: Perlin,
    pub scale: f64,
}
impl Texture for NoiseTexture {
    fn value(&self, _u: f64, _v: f64, p: &Vec3) -> Vec3 {
        Vec3::new(1.0, 1.0, 1.0)
            * 0.5
            * (1.0 + (self.scale * p.z + 10.0 * self.noise.turb(p, 7)).sin())
    }
}
impl Default for NoiseTexture {
    fn default() -> Self {
        Self::new()
    }
}
impl NoiseTexture {
    pub fn new() -> Self {
        Self {
            noise: Perlin::new(),
            scale: 1.0,
        }
    }
    pub fn new_from_f64(sc: f64) -> Self {
        Self {
            noise: Perlin::new(),
            scale: sc,
        }
    }
}
pub fn clamp(x: f64, min: f64, max: f64) -> f64 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}
#[derive(Clone)]
pub struct ImageTexture {
    pub my_image: image::DynamicImage,
    pub width: u32,
    pub height: u32,
}
impl ImageTexture {
    pub fn new(input_path: &str) -> Self {
        let my_image = image::open(&Path::new(input_path)).unwrap();
        Self {
            my_image: my_image.clone(),
            width: my_image.dimensions().0,
            height: my_image.dimensions().1,
        }
    }
}
impl Texture for ImageTexture {
    fn value(&self, u: f64, v: f64, _p: &Vec3) -> Vec3 {
        let u = clamp(u, 0.0, 1.0);
        let v = 1.0 - clamp(v, 0.0, 1.0);
        let mut i = (u * self.width as f64) as u32;
        let mut j = (v * self.height as f64) as u32;
        if i >= self.width {
            i = self.width - 1;
        }
        if j >= self.height {
            j = self.height - 1;
        }
        let color_scale = 1.0 / 255.0;
        let pixel = self.my_image.get_pixel(i as u32, j as u32);
        Vec3::new(
            pixel[0] as f64 * color_scale,
            pixel[1] as f64 * color_scale,
            pixel[2] as f64 * color_scale,
        )
    }
}
