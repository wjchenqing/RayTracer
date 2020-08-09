pub use crate::bvh::*;
pub use crate::material::*;
pub use crate::pdf::*;
pub use crate::vec3::*;

#[derive(Clone)]
pub struct HitRecord {
    pub pos: Vec3,
    pub nor: Vec3,
    pub t: f64,
    pub u: f64,
    pub v: f64,
    pub front_face: bool,
    pub mat_ptr: Arc<dyn Material>,
}
pub trait Hittable: Sync + Send {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
    fn bounding_box(&self, t0: f64, t1: f64) -> Option<AABB>;
    fn pdf_value(&self, _o: &Vec3, _v: &Vec3) -> f64 {
        0.0
    }
    fn random(&self, _o: &Vec3) -> Vec3 {
        Vec3::new(1.0, 0.0, 0.0)
    }
}
#[derive(Clone)]
pub struct HittableList {
    pub objects: Vec<Arc<dyn Hittable>>,
}
impl HittableList {
    pub fn add(&mut self, hittable: Arc<dyn Hittable>) {
        self.objects.push(hittable);
    }
}
impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut hit_anything: Option<HitRecord> = None;
        let mut closest_so_far = t_max;
        for i in self.objects.iter() {
            let tmp_rec = i.hit(ray, t_min, closest_so_far);
            if let Some(tmp) = tmp_rec {
                hit_anything = Some(tmp.clone());
                closest_so_far = tmp.t;
            }
        }
        hit_anything
    }
    fn bounding_box(&self, t0: f64, t1: f64) -> Option<AABB> {
        if self.objects.is_empty() {
            return None;
        }
        let mut output_box;
        if let Some(tmp_box) = self.objects[0].bounding_box(t0, t1) {
            output_box = tmp_box;
        } else {
            return None;
        }

        for i in 1..self.objects.len() {
            if let Some(tmp_box) = self.objects[i].bounding_box(t0, t1) {
                output_box = surrounding_box(output_box, tmp_box);
            } else {
                return None;
            }
        }
        Some(output_box)
    }
    fn pdf_value(&self, o: &Vec3, v: &Vec3) -> f64 {
        let weight = 1.0 / self.objects.len() as f64;
        let mut sum = 0.0;
        for object in self.objects.iter() {
            sum += weight * object.pdf_value(o, v);
        }
        sum
    }
    fn random(&self, o: &Vec3) -> Vec3 {
        self.objects[rand::random::<usize>() % self.objects.len()].random(o)
    }
}
pub fn surrounding_box(box0: AABB, box1: AABB) -> AABB {
    AABB {
        _min: Vec3 {
            x: box0._min.x.min(box1._min.x),
            y: box0._min.y.min(box1._min.y),
            z: box0._min.z.min(box1._min.z),
        },
        _max: Vec3 {
            x: box0._max.x.max(box1._max.x),
            y: box0._max.y.max(box1._max.y),
            z: box0._max.z.max(box1._max.z),
        },
    }
}

fn get_sphere_uv(p: &Vec3) -> (f64, f64) {
    let phi = p.z.atan2(p.x);
    let theta = p.y.asin();
    (1.0 - (phi + PI) / 2.0 / PI, (theta + PI / 2.0) / PI)
}
#[derive(Clone)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
    pub mat_ptr: Arc<dyn Material>,
}
impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc: Vec3 = ray.ori - self.center;
        let a = ray.dir.squared_length();
        let _b = oc * ray.dir;
        let c = oc.squared_length() - self.radius * self.radius;
        let discriminant = _b * _b - a * c;

        if discriminant > 0.0 {
            let root = discriminant.sqrt();
            let tmp = (-_b - root) / a;
            if (tmp < t_max) && (tmp > t_min) {
                let pos = ray.at(tmp);
                let mut nor = (pos - self.center) / self.radius;
                let flag = (ray.dir * nor) < 0.0;
                if !flag {
                    nor = -nor;
                }
                let (u, v) = get_sphere_uv(&((pos - self.center) / self.radius));
                return Some(HitRecord {
                    t: tmp,
                    pos,
                    nor,
                    front_face: flag,
                    mat_ptr: self.mat_ptr.clone(),
                    u,
                    v,
                });
            } else {
                let tmp = (-_b + root) / a;
                if (tmp < t_max) && (tmp > t_min) {
                    let pos = ray.at(tmp);
                    let mut nor = (pos - self.center) / self.radius;
                    let flag = (ray.dir * nor) < 0.0;
                    if !flag {
                        nor = -nor;
                    }
                    let (u, v) = get_sphere_uv(&((pos - self.center) / self.radius));
                    return Some(HitRecord {
                        t: tmp,
                        pos,
                        nor,
                        front_face: flag,
                        mat_ptr: self.mat_ptr.clone(),
                        u,
                        v,
                    });
                }
            }
        }
        None
    }
    fn bounding_box(&self, _t0: f64, _t1: f64) -> Option<AABB> {
        Some(AABB::new(
            &(self.center - Vec3::new(self.radius, self.radius, self.radius)),
            &(self.center + Vec3::new(self.radius, self.radius, self.radius)),
        ))
    }
    fn pdf_value(&self, o: &Vec3, v: &Vec3) -> f64 {
        if let Some(_rec) = self.hit(&Ray::new(*o, *v), 0.001, f64::MAX) {
            let cos_theta_max =
                (1.0 - self.radius * self.radius / (self.center - *o).squared_length()).sqrt();
            let solid_angle = 2.0 * PI * (1.0 - cos_theta_max);
            1.0 / solid_angle
        } else {
            0.0
        }
    }
    fn random(&self, o: &Vec3) -> Vec3 {
        let dir = self.center - *o;
        let distance_squared = dir.squared_length();
        let uvw = ONB::build_from_w(&dir);
        uvw.local_vec(&random_to_sphere(self.radius, distance_squared))
    }
}
pub struct XyRect {
    pub mp: Arc<dyn Material>,
    pub x0: f64,
    pub x1: f64,
    pub y0: f64,
    pub y1: f64,
    pub k: f64,
}
impl XyRect {
    pub fn new(x0: &f64, x1: &f64, y0: &f64, y1: &f64, k: &f64, mp: &Arc<dyn Material>) -> Self {
        Self {
            x0: *x0,
            x1: *x1,
            y0: *y0,
            y1: *y1,
            k: *k,
            mp: mp.clone(),
        }
    }
}
impl Hittable for XyRect {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let t = (self.k - ray.ori.z) / ray.dir.z;
        if t < t_min || t > t_max {
            return None;
        }
        let x = ray.ori.x + t * ray.dir.x;
        let y = ray.ori.y + t * ray.dir.y;
        if x < self.x0 || x > self.x1 || y < self.y0 || y > self.y1 {
            return None;
        }
        let mut nor = Vec3::new(0.0, 0.0, 1.0);
        let flag = (ray.dir * nor) < 0.0;
        if !flag {
            nor = -nor;
        }
        Some(HitRecord {
            u: (x - self.x0) / (self.x1 - self.x0),
            v: (y - self.y0) / (self.y1 - self.y0),
            t,
            nor,
            front_face: flag,
            mat_ptr: self.mp.clone(),
            pos: ray.at(t),
        })
    }
    fn bounding_box(&self, _t0: f64, _t1: f64) -> Option<AABB> {
        Some(AABB::new(
            &Vec3::new(self.x0, self.y0, self.k - 0.0001),
            &Vec3::new(self.x1, self.y1, self.k + 0.0001),
        ))
    }
}

pub struct XzRect {
    pub mp: Arc<dyn Material>,
    pub x0: f64,
    pub x1: f64,
    pub z0: f64,
    pub z1: f64,
    pub k: f64,
}
impl XzRect {
    pub fn new(x0: &f64, x1: &f64, z0: &f64, z1: &f64, k: &f64, mp: &Arc<dyn Material>) -> Self {
        Self {
            x0: *x0,
            x1: *x1,
            z0: *z0,
            z1: *z1,
            k: *k,
            mp: mp.clone(),
        }
    }
}
impl Hittable for XzRect {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let t = (self.k - ray.ori.y) / ray.dir.y;
        if t < t_min || t > t_max {
            return None;
        }
        let x = ray.ori.x + t * ray.dir.x;
        let z = ray.ori.z + t * ray.dir.z;
        if x < self.x0 || x > self.x1 || z < self.z0 || z > self.z1 {
            return None;
        }
        let mut nor = Vec3::new(0.0, 1.0, 0.0);
        let flag = (ray.dir * nor) < 0.0;
        if !flag {
            nor = -nor;
        }
        Some(HitRecord {
            u: (x - self.x0) / (self.x1 - self.x0),
            v: (z - self.z0) / (self.z1 - self.z0),
            t,
            nor,
            front_face: flag,
            mat_ptr: self.mp.clone(),
            pos: ray.at(t),
        })
    }
    fn bounding_box(&self, _t0: f64, _t1: f64) -> Option<AABB> {
        Some(AABB::new(
            &Vec3::new(self.x0, self.k - 0.0001, self.z0),
            &Vec3::new(self.x1, self.k + 0.0001, self.z1),
        ))
    }
    fn pdf_value(&self, o: &Vec3, v: &Vec3) -> f64 {
        if let Some(rec) = self.hit(&Ray::new(*o, *v), 0.001, f64::MAX) {
            let area = (self.x1 - self.x0) * (self.z1 - self.z0);
            let distance_squared = rec.t * rec.t * v.squared_length();
            let cosine = (*v * rec.nor).abs() / v.length();
            distance_squared / (cosine * area)
        } else {
            0.0
        }
    }
    fn random(&self, o: &Vec3) -> Vec3 {
        let random_point = Vec3::new(
            self.x0 + (self.x1 - self.x0) * random::<f64>(),
            self.k,
            self.z0 + (self.z1 - self.z0) * random::<f64>(),
        );
        random_point - *o
    }
}

pub struct YzRect {
    pub mp: Arc<dyn Material>,
    pub z0: f64,
    pub z1: f64,
    pub y0: f64,
    pub y1: f64,
    pub k: f64,
}
impl YzRect {
    pub fn new(y0: &f64, y1: &f64, z0: &f64, z1: &f64, k: &f64, mp: &Arc<dyn Material>) -> Self {
        Self {
            y0: *y0,
            y1: *y1,
            z0: *z0,
            z1: *z1,
            k: *k,
            mp: mp.clone(),
        }
    }
}
impl Hittable for YzRect {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let t = (self.k - ray.ori.x) / ray.dir.x;
        if t < t_min || t > t_max {
            return None;
        }
        let z = ray.ori.z + t * ray.dir.z;
        let y = ray.ori.y + t * ray.dir.y;
        if z < self.z0 || z > self.z1 || y < self.y0 || y > self.y1 {
            return None;
        }
        let mut nor = Vec3::new(1.0, 0.0, 0.0);
        let flag = (ray.dir * nor) < 0.0;
        if !flag {
            nor = -nor;
        }
        Some(HitRecord {
            u: (z - self.z0) / (self.z1 - self.z0),
            v: (y - self.y0) / (self.y1 - self.y0),
            t,
            nor,
            front_face: flag,
            mat_ptr: self.mp.clone(),
            pos: ray.at(t),
        })
    }
    fn bounding_box(&self, _t0: f64, _t1: f64) -> Option<AABB> {
        Some(AABB::new(
            &Vec3::new(self.k - 0.0001, self.y0, self.z0),
            &Vec3::new(self.k + 0.0001, self.y1, self.z1),
        ))
    }
}

pub struct Box {
    pub box_min: Vec3,
    pub box_max: Vec3,
    pub sides: HittableList,
}
impl Box {
    pub fn new(p0: &Vec3, p1: &Vec3, ptr: Arc<dyn Material>) -> Self {
        let box_min = *p0;
        let box_max = *p1;
        let mut sides = HittableList { objects: vec![] };
        sides.add(Arc::new(XyRect {
            x0: p0.x,
            x1: p1.x,
            y0: p0.y,
            y1: p1.y,
            k: p1.z,
            mp: ptr.clone(),
        }));
        sides.add(Arc::new(XyRect {
            x0: p0.x,
            x1: p1.x,
            y0: p0.y,
            y1: p1.y,
            k: p0.z,
            mp: ptr.clone(),
        }));
        sides.add(Arc::new(XzRect {
            x0: p0.x,
            x1: p1.x,
            z0: p0.z,
            z1: p1.z,
            k: p1.y,
            mp: ptr.clone(),
        }));
        sides.add(Arc::new(XzRect {
            x0: p0.x,
            x1: p1.x,
            z0: p0.z,
            z1: p1.z,
            k: p0.y,
            mp: ptr.clone(),
        }));
        sides.add(Arc::new(YzRect {
            y0: p0.y,
            y1: p1.y,
            z0: p0.z,
            z1: p1.z,
            k: p0.x,
            mp: ptr.clone(),
        }));
        sides.add(Arc::new(YzRect {
            y0: p0.y,
            y1: p1.y,
            z0: p0.z,
            z1: p1.z,
            k: p1.x,
            mp: ptr.clone(),
        }));
        Self {
            box_min,
            box_max,
            sides,
        }
    }
}
impl Hittable for Box {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        self.sides.hit(ray, t_min, t_max)
    }
    fn bounding_box(&self, _t0: f64, _t1: f64) -> Option<AABB> {
        Some(AABB::new(&self.box_min, &self.box_max))
    }
}

pub struct Translate {
    pub ptr: Arc<dyn Hittable>,
    pub offset: Vec3,
}
impl Translate {
    pub fn new(p: &Arc<RotateY>, displacement: &Vec3) -> Self {
        Self {
            ptr: p.clone(),
            offset: *displacement,
        }
    }
}
impl Hittable for Translate {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let moved_r = Ray::new(ray.ori - self.offset, ray.dir);
        if let Some(mut rec) = self.ptr.hit(&moved_r, t_min, t_max) {
            let flag = (moved_r.dir * rec.nor) < 0.0;
            if !flag {
                rec.nor = -rec.nor;
            }
            return Some(HitRecord {
                pos: rec.pos + self.offset,
                nor: rec.nor,
                front_face: flag,
                mat_ptr: rec.mat_ptr,
                u: rec.u,
                t: rec.t,
                v: rec.v,
            });
        }
        None
    }
    fn bounding_box(&self, t0: f64, t1: f64) -> Option<AABB> {
        if let Some(tmp) = self.ptr.bounding_box(t0, t1) {
            return Some(AABB::new(
                &(tmp._min + self.offset),
                &(tmp._max + self.offset),
            ));
        }
        None
    }
}

pub struct RotateY {
    pub ptr: Arc<Box>,
    pub sin_theta: f64,
    pub cos_theta: f64,
    pub has_box: bool,
    pub bbox: AABB,
}
impl RotateY {
    pub fn new(ptr: &Arc<Box>, angle: f64) -> Self {
        let radians = angle / 180.0 * PI;
        let sin_theta = radians.sin();
        let cos_theta = radians.cos();
        let mut _min = Vec3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut _max = Vec3::new(-f64::MAX, -f64::MAX, -f64::MAX);
        if let Some(bbox) = ptr.bounding_box(0.0, 1.0) {
            let has_box = true;
            for i in 0..1 {
                for j in 0..1 {
                    for k in 0..1 {
                        let x = i as f64 * bbox._max.x + (1 - i) as f64 * bbox._min.x;
                        let y = j as f64 * bbox._max.y + (1 - j) as f64 * bbox._min.y;
                        let z = k as f64 * bbox._max.z + (1 - k) as f64 * bbox._min.z;
                        let new_x = cos_theta * x + sin_theta * z;
                        let new_z = -sin_theta * x + cos_theta * z;
                        _min.x = _min.x.min(new_x);
                        _min.y = _min.y.min(y);
                        _min.z = _min.z.min(new_z);
                        _max.x = _max.x.max(new_x);
                        _max.y = _max.y.max(y);
                        _max.z = _max.z.max(new_z);
                    }
                }
            }
            let bbox = AABB::new(&_min, &_max);
            Self {
                ptr: ptr.clone(),
                sin_theta,
                cos_theta,
                bbox,
                has_box,
            }
        } else {
            panic!();
        }
    }
}
impl Hittable for RotateY {
    fn bounding_box(&self, _t0: f64, _t1: f64) -> Option<AABB> {
        if self.has_box {
            Some(self.bbox)
        } else {
            None
        }
    }
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut ori = ray.ori;
        let mut dir = ray.dir;
        ori.x = self.cos_theta * ray.ori.x - self.sin_theta * ray.ori.z;
        ori.z = self.sin_theta * ray.ori.x + self.cos_theta * ray.ori.z;
        dir.x = self.cos_theta * ray.dir.x - self.sin_theta * ray.dir.z;
        dir.z = self.sin_theta * ray.dir.x + self.cos_theta * ray.dir.z;
        let rotated_r = Ray::new(ori, dir);
        if let Some(rec) = self.ptr.hit(&rotated_r, t_min, t_max) {
            let mut pos = rec.pos;
            let mut nor = rec.nor;
            pos.x = self.cos_theta * rec.pos.x + self.sin_theta * rec.pos.z;
            pos.z = -self.sin_theta * rec.pos.x + self.cos_theta * rec.pos.z;
            nor.x = self.cos_theta * rec.nor.x + self.sin_theta * rec.nor.z;
            nor.z = -self.sin_theta * rec.nor.x + self.cos_theta * rec.nor.z;
            let flag = (rotated_r.dir * rec.nor) < 0.0;
            if !flag {
                nor = -nor;
            }
            return Some(HitRecord {
                pos,
                nor,
                front_face: flag,
                mat_ptr: rec.mat_ptr,
                u: rec.u,
                t: rec.t,
                v: rec.v,
            });
        }
        None
    }
}
pub struct FlipFace {
    pub ptr: Arc<dyn Hittable>,
}
impl Hittable for FlipFace {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        if let Some(mut rec) = self.ptr.hit(ray, t_min, t_max) {
            rec.front_face = !rec.front_face;
            Some(rec)
        } else {
            None
        }
    }
    fn bounding_box(&self, t0: f64, t1: f64) -> Option<AABB> {
        self.ptr.bounding_box(t0, t1)
    }
}
