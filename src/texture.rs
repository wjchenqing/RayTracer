pub use crate::hittable::*;
pub use crate::randomtool::*;
pub use crate::ray::*;
pub use crate::vec3::*;
pub use rand::random;
pub use std::cmp::Ordering;
pub use std::f64::consts::PI;
pub use std::f64::INFINITY;
pub use std::sync::Arc;

pub trait Texture: Sync + Send {
    fn value(&self, u: f64, v: f64, p: &Vec3) -> Vec3;
}
pub struct SolidColor {
    pub color_value: Vec3,
}
impl SolidColor {
    pub fn new(c: Vec3) -> Self {
        Self { color_value: c }
    }
}
impl Texture for SolidColor {
    fn value(&self, _u: f64, _v: f64, _p: &Vec3) -> Vec3 {
        self.color_value
    }
}
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
            even: Arc::new(SolidColor::new(*c1)),
            odd: Arc::new(SolidColor::new(*c2)),
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
