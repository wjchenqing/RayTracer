use crate::vec3::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Firework {
    pub ori: Vec3,
    pub dir: Vec3,
}

impl Firework {
    pub fn new(ori: Vec3, dir: Vec3) -> Self {
        Self { ori, dir }
    }
    pub fn zero() -> Self {
        Self::new(Vec3::zero(), Vec3::zero())
    }
    pub fn origin(&self) -> Vec3 {
        self.ori
    }
    pub fn direction(&self) -> Vec3 {
        self.dir
    }
    pub fn at(&self, t: f64) -> Vec3 {
        Vec3::new(
            self.ori.x + self.dir.x * t,
            self.ori.y + self.dir.y * t,
            self.ori.z - self.dir.z * t + (self.dir.squared_length() as f64) * t * t / (900 as f64),
        )
    }
}
