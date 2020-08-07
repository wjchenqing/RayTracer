pub use crate::hittable::*;
pub use crate::randomtool::*;
pub use crate::ray::*;
pub use crate::texture::*;
pub use crate::vec3::*;
pub use rand::random;
pub use rand::Rng;
pub use std::cmp::Ordering;
pub use std::f64::consts::PI;
pub use std::f64::INFINITY;
pub use std::sync::Arc;

pub struct BvhNode {
    pub left: Arc<dyn Hittable>,
    pub right: Arc<dyn Hittable>,
    pub _box: AABB,
}
impl Hittable for BvhNode {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        if self._box.hit(ray, &t_min, &t_max) == None {
            return None;
        }
        if let Some(tmp1) = self.left.hit(ray, t_min, t_max) {
            if let Some(tmp2) = self.right.hit(ray, t_min, t_max) {
                return Some(tmp2);
            } else {
                return Some(tmp1);
            }
        } else if let Some(tmp2) = self.right.hit(ray, t_min, t_max) {
            return Some(tmp2);
        }
        None
    }
    fn bounding_box(&self, _t0: f64, _t1: f64) -> Option<AABB> {
        Some(self._box)
    }
}
impl BvhNode {
    fn box_x_compare(a: &Arc<dyn Hittable>, b: &Arc<dyn Hittable>) -> Ordering {
        if let Some(box_a) = a.bounding_box(0.0, 0.0) {
            if let Some(box_b) = b.bounding_box(0.0, 0.0) {
                if let Some(tmp) = box_a._min.x.partial_cmp(&box_b._min.x) {
                    return tmp;
                }
            }
        }
        panic!();
    }
    fn box_y_compare(a: &Arc<dyn Hittable>, b: &Arc<dyn Hittable>) -> Ordering {
        if let Some(box_a) = a.bounding_box(0.0, 0.0) {
            if let Some(box_b) = b.bounding_box(0.0, 0.0) {
                if let Some(tmp) = box_a._min.y.partial_cmp(&box_b._min.y) {
                    return tmp;
                }
            }
        }
        panic!();
    }
    fn box_z_compare(a: &Arc<dyn Hittable>, b: &Arc<dyn Hittable>) -> Ordering {
        if let Some(box_a) = a.bounding_box(0.0, 0.0) {
            if let Some(box_b) = b.bounding_box(0.0, 0.0) {
                if let Some(tmp) = box_a._min.z.partial_cmp(&box_b._min.z) {
                    return tmp;
                }
            }
        }
        panic!();
    }
    pub fn new(
        objects: &mut Vec<Arc<dyn Hittable>>,
        start: usize,
        end: usize,
        time0: f64,
        time1: f64,
    ) -> Self {
        let left;
        let right;
        let _box;
        let axis = rand::thread_rng().gen_range(0, 2);
        match axis {
            0 => objects[start..end].sort_by(|a, b| Self::box_x_compare(a, b)),
            1 => objects[start..end].sort_by(|a, b| Self::box_y_compare(a, b)),
            _ => objects[start..end].sort_by(|a, b| Self::box_z_compare(a, b)),
        };
        let len = end - start;
        if len == 1 {
            left = objects[start].clone();
            right = objects[start].clone();
        } else if len == 2 {
            left = objects[start].clone();
            right = objects[end].clone();
        } else {
            let mid = start + len / 2;
            left = Arc::new(BvhNode::new(objects, start, mid, time0, time1));
            right = Arc::new(BvhNode::new(objects, mid, end, time0, time1));
        }
        if let Some(box_left) = left.bounding_box(time0, time1) {
            if let Some(box_right) = left.bounding_box(time0, time1) {
                _box = surrounding_box(box_left, box_right);
                return Self { left, right, _box };
            }
        }
        panic!();
    }
}

#[derive(Clone, Copy)]
pub struct AABB {
    pub _min: Vec3,
    pub _max: Vec3,
}
impl AABB {
    pub fn new(a: &Vec3, b: &Vec3) -> Self {
        Self { _min: *a, _max: *b }
    }
    pub fn hit(&self, ray: &Ray, tmin: &f64, tmax: &f64) -> Option<(f64, f64)> {
        let mut t_min = *tmin;
        let mut t_max = *tmax;

        let inv = 1.0 / ray.dir.x;
        let mut t0 =
            ((self._min.x - ray.ori.x) / ray.dir.x).min((self._max.x - ray.ori.x) / ray.dir.x);
        let mut t1 =
            ((self._min.x - ray.ori.x) / ray.dir.x).max((self._max.x - ray.ori.x) / ray.dir.x);
        if inv < 0.0 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t_min = t0.max(t_min);
        t_max = t1.min(t_max);
        if t_max <= t_min {
            return None;
        }

        let inv = 1.0 / ray.dir.y;
        let mut t0 =
            ((self._min.y - ray.ori.y) / ray.dir.y).min((self._max.y - ray.ori.y) / ray.dir.y);
        let mut t1 =
            ((self._min.y - ray.ori.y) / ray.dir.y).max((self._max.y - ray.ori.y) / ray.dir.y);
        if inv < 0.0 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t_min = t0.max(t_min);
        t_max = t1.min(t_max);
        if t_max <= t_min {
            return None;
        }

        let inv = 1.0 / ray.dir.z;
        let mut t0 =
            ((self._min.z - ray.ori.z) / ray.dir.z).min((self._max.z - ray.ori.z) / ray.dir.z);
        let mut t1 =
            ((self._min.z - ray.ori.z) / ray.dir.z).max((self._max.z - ray.ori.z) / ray.dir.z);
        if inv < 0.0 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t_min = t0.max(t_min);
        t_max = t1.min(t_max);
        if t_max <= t_min {
            return None;
        }

        Some((t_min, t_max))
    }
}