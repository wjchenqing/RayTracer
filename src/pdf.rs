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

pub trait PDF: Sync + Send {
    fn value(&self, dir: &Vec3) -> f64;
    fn generate(&self) -> Vec3;
}

pub struct CosinePDF {
    pub uvw: ONB,
}
impl CosinePDF {
    pub fn new(w: &Vec3) -> Self {
        Self {
            uvw: ONB::build_from_w(w),
        }
    }
}
impl PDF for CosinePDF {
    fn value(&self, dir: &Vec3) -> f64 {
        let cosine = dir.unit() * self.uvw.w;
        if cosine < 0.0 {
            0.0
        } else {
            cosine / PI
        }
    }
    fn generate(&self) -> Vec3 {
        self.uvw.local_vec(&random_cosine_direction())
    }
}

pub struct HittablePDF {
    pub o: Vec3,
    pub ptr: Arc<dyn Hittable>,
}
impl PDF for HittablePDF {
    fn value(&self, dir: &Vec3) -> f64 {
        self.ptr.pdf_value(&self.o, dir)
    }
    fn generate(&self) -> Vec3 {
        self.ptr.random(&self.o)
    }
}

pub struct MixtruePDF {
    pub p1: Arc<dyn PDF>,
    pub p2: Arc<dyn PDF>,
}
impl PDF for MixtruePDF {
    fn value(&self, dir: &Vec3) -> f64 {
        0.5 * self.p1.value(dir) + 0.5 * self.p2.value(dir)
    }
    fn generate(&self) -> Vec3 {
        if random::<f64>() < 0.8 {
            self.p1.generate()
        } else {
            self.p2.generate()
        }
    }
}
