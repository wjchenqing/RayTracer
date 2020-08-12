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
        let mut t_min = t_min;
        let mut t_max = t_max;
        if !self._box.hit(ray, &mut t_min, &mut t_max) {
            // println!("a");
            return None;
        }
        if let Some(tmp1) = self.left.hit(ray, t_min, t_max) {
            if let Some(tmp2) = self.right.hit(ray, t_min, t_max) {
                // println!("22222222222222");
                return Some(tmp2);
            } else {
                // println!("11111111111111");
                return Some(tmp1);
            }
        } else if let Some(tmp2) = self.right.hit(ray, t_min, t_max) {
            // println!("2222");
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
    pub fn new_from_list(list: &mut HittableList, time0: f64, time1: f64) -> Self {
        let len = list.objects.len();
        Self::new(&mut list.objects, 0, len, time0, time1)
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
        let axis = rand::random::<usize>() % 3;
        let comparator = match axis {
            0 => Self::box_x_compare,
            1 => Self::box_y_compare,
            _ => Self::box_z_compare,
        };
        let len = end - start;
        if len == 1 {
            left = objects[start].clone();
            right = objects[start].clone();
        } else if len == 2 {
            if comparator(&objects[start], &objects[start + 1]) == Ordering::Less {
                right = objects[start + 1].clone();
                left = objects[start].clone();
            } else {
                left = objects[start + 1].clone();
                right = objects[start].clone();
            }
        } else {
            let obj = &mut objects[start..end];
            obj.sort_by(|a, b| comparator(a, b));

            let mid = (start + end) / 2;
            left = Arc::new(BvhNode::new(objects, start, mid, time0, time1));
            right = Arc::new(BvhNode::new(objects, mid, end, time0, time1));
        }
        if let Some(box_left) = left.bounding_box(time0, time1) {
            if let Some(box_right) = left.bounding_box(time0, time1) {
                _box = surrounding_box(box_left, box_right);
                return Self { left, right, _box };
            }
        }
        panic!("aaaaaaaaaaaaaaaaaa");
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
    pub fn hit(&self, ray: &Ray, t_min: &mut f64, t_max: &mut f64) -> bool {
        let mut t1 = (self._min.x - ray.ori.x) / ray.dir.x;
        let mut t2 = (self._max.x - ray.ori.x) / ray.dir.x;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2)
        }
        // println!("{}, {}, {}, {}", t1, t_min, t2, t_max);
        *t_min = if t1 > *t_min { t1 } else { *t_min };
        *t_max = if t2 < *t_max { t2 } else { *t_max };
        // if t_max.min(t2) <= t_min.max(t1) {
        //     return false;
        // }
        let mut t1 = (self._min.y - ray.ori.y) / ray.dir.y;
        let mut t2 = (self._max.y - ray.ori.y) / ray.dir.y;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2)
        }
        *t_min = if t1 > *t_min { t1 } else { *t_min };
        *t_max = if t2 < *t_max { t2 } else { *t_max };
        // if t_max.min(t2) <= t_min.max(t1) {
        //     return false;
        // }
        let mut t1 = (self._min.z - ray.ori.z) / ray.dir.z;
        let mut t2 = (self._max.z - ray.ori.z) / ray.dir.z;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2)
        }
        *t_min = if t1 > *t_min { t1 } else { *t_min };
        *t_max = if t2 < *t_max { t2 } else { *t_max };
        // if t_max.min(t2) <= t_min.max(t1) {
        //     return false;
        // }
        true
    }
}
