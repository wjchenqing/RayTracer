pub use crate::hittable::*;
pub use crate::randomtool::*;
pub use crate::ray::*;
pub use crate::texture::*;
pub use crate::vec3::*;
pub use rand::random;
pub use std::cmp::Ordering;
pub use std::f64::consts::PI;
pub use std::f64::INFINITY;
pub use std::sync::Arc;

pub struct ONB {
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
}
impl ONB {
    pub fn local_f64(&self, a: &f64, b: &f64, c: &f64) -> Vec3 {
        self.u * *a + self.v * *b + self.w * *c
    }
    pub fn local_vec(&self, a: &Vec3) -> Vec3 {
        self.u * a.x + self.v * a.y + self.w * a.z
    }
    pub fn build_from_w(n: &Vec3) -> Self {
        let w = n.unit();
        let a;
        if w.x.abs() > 0.9 {
            a = Vec3::new(0.0, 1.0, 0.0);
        } else {
            a = Vec3::new(1.0, 0.0, 0.0);
        }
        let v = Vec3::cross(w, a).unit();
        Self {
            v,
            u: Vec3::cross(w, v),
            w,
        }
    }
}
