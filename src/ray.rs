use crate::vec3::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub ori: Vec3,
    pub dir: Vec3,
}

impl Ray {
    pub fn new(ori: Vec3, dir: Vec3) -> Self {
        Self { ori, dir }
    }
    pub fn zero() -> Self {
        Self::new(Vec3::zero(), Vec3::zero())
    }
    pub fn at(&self, t: f64) -> Vec3 {
        self.ori + self.dir * t
    }
}
