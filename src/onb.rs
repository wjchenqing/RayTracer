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
    pub axis1: Vec3,
    pub axis2: Vec3,
    pub axis3: Vec3,
}
impl ONB {
    pub fn local_f64(&self, a: &f64, b: &f64, c: &f64) -> Vec3 {
        self.axis1 * *a + self.axis2 * *b + self.axis3 * *c
    }
    pub fn local_vec(&self, a: &Vec3) -> Vec3 {
        self.axis1 * a.x + self.axis2 * a.y + self.axis3 * a.z
    }
    pub fn build_from_w(n: &Vec3) -> Self {
        let axis3 = n.unit();
        let a;
        if axis3.x.abs() > 0.9 {
            a = Vec3::new(0.0, 1.0, 0.0);
        } else {
            a = Vec3::new(1.0, 0.0, 0.0);
        }
        let axis2 = Vec3::cross(axis3, a).unit();
        Self {
            axis2,
            axis1: Vec3::cross(axis3, axis2),
            axis3,
        }
    }
}
