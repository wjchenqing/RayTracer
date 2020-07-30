#![allow(clippy::float_cmp)]
mod ray;
mod vec3;
use image::{ImageBuffer, RgbImage};
use indicatif::ProgressBar;
pub use rand::random;
pub use rand::Rng;
pub use std::cmp::Ordering;
pub use std::f64::consts::PI;
pub use std::f64::INFINITY;
pub use std::sync::Arc;

pub use ray::Ray;
pub use vec3::Vec3;

fn random_num() -> f64 {
    random::<f64>()
}
fn random_positive_unit() -> Vec3 {
    let x = random::<f64>().abs();
    let y = random::<f64>().abs();
    let z = random::<f64>().abs();
    let tmp = Vec3::new(x, y, z);
    if tmp.length() > 1.0 {
        return random_positive_unit();
    }
    tmp.unit()
}
fn random_unit() -> Vec3 {
    let x = random::<i32>();
    let y = random::<i32>();
    let z = random::<i32>();
    let tmp = Vec3::new(x as f64, y as f64, z as f64);
    if tmp.length() == 0.0 {
        return random_unit();
    }
    -tmp.unit()
}
fn random_vec(nor: &Vec3) -> Vec3 {
    let x = random::<i32>();
    let y = random::<i32>();
    let z = random::<i32>();
    let tmp = Vec3::new(x as f64, y as f64, z as f64);
    if tmp.length() == 0.0 {
        return random_vec(nor);
    }
    if tmp * *nor > 0.0 {
        return tmp.unit();
    }
    -tmp.unit()
}
fn random_in_unit_disk() -> Vec3 {
    let p = Vec3::new(random::<f64>(), random::<f64>(), 0.0);
    if p.squared_length() >= 1.0 {
        return random_in_unit_disk();
    }
    p
}

pub trait Material: Sync + Send {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<(Vec3, Ray)>;
    fn emitted(&self, u: f64, v: f64, p: &Vec3) -> Vec3;
}
pub struct Lambertian {
    pub albedo: Arc<dyn Texture>,
}
impl Material for Lambertian {
    fn scatter(&self, _r_in: &Ray, rec: &HitRecord) -> Option<(Vec3, Ray)> {
        let scatter_direction = rec.nor + random_vec(&rec.nor);
        let scattered = Ray::new(rec.pos, scatter_direction);
        Some((self.albedo.value(rec.u, rec.v, &rec.pos), scattered))
    }
    fn emitted(&self, _u: f64, _v: f64, _p: &Vec3) -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}
impl Lambertian {
    pub fn new(albedo: Vec3) -> Self {
        Self {
            albedo: Arc::new(SolidColor::new(albedo)),
        }
    }
    pub fn new_from_arc(a: Arc<dyn Texture>) -> Self {
        Self { albedo: a }
    }
}
fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - n * (v * n) * 2.0
}
fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = -*uv * *n;
    let r_out_perp: Vec3 = (*uv + *n * cos_theta) * etai_over_etat;
    let tmp = 1.0 - r_out_perp.squared_length().abs();
    let r_out_parallel = *n * (-tmp.sqrt());
    r_out_parallel + r_out_perp
}
fn schlick(cosine: f64, ref_idx: f64) -> f64 {
    let r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    let r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}
pub struct Dielectric {
    pub ref_idx: f64,
}
impl Dielectric {
    pub fn new(ri: f64) -> Self {
        Self { ref_idx: ri }
    }
}
impl Material for Dielectric {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<(Vec3, Ray)> {
        let attenuation = Vec3::new(1.0, 1.0, 1.0);
        let etai_over_etat = if rec.front_face {
            1.0 / self.ref_idx
        } else {
            self.ref_idx
        };
        let unit_dir = r_in.dir.unit();
        let cos_theta = if (-unit_dir) * rec.nor < 1.0 {
            (-unit_dir) * rec.nor
        } else {
            1.0
        };
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        if (etai_over_etat * sin_theta) > 1.0 {
            let reflected = reflect(unit_dir, rec.nor);
            Some((attenuation, Ray::new(rec.pos, reflected)))
        } else {
            let reflect_prob = schlick(cos_theta, etai_over_etat);
            if random::<f64>() < reflect_prob {
                let reflected = reflect(unit_dir, rec.nor);
                Some((attenuation, Ray::new(rec.pos, reflected)))
            } else {
                let refracted = refract(&unit_dir, &rec.nor, etai_over_etat);
                Some((attenuation, Ray::new(rec.pos, refracted)))
            }
        }
    }
    fn emitted(&self, _u: f64, _v: f64, _p: &Vec3) -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

pub struct Metal {
    pub albedo: Vec3,
    pub fuzz: f64,
}
impl Metal {
    pub fn new(albedo: Vec3, mut f: f64) -> Self {
        if f > 1.0 {
            f = 1.0;
        }
        Self { albedo, fuzz: f }
    }
}
impl Material for Metal {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<(Vec3, Ray)> {
        let reflected = reflect(r_in.dir.unit(), rec.nor);
        let scattered = Ray::new(rec.pos, reflected + random_unit() * self.fuzz);
        if reflected * rec.nor > 0.0 {
            return Some((self.albedo, scattered));
        }
        None
    }
    fn emitted(&self, _u: f64, _v: f64, _p: &Vec3) -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

pub struct DiffuseLight {
    pub emit: Arc<dyn Texture>,
}
impl DiffuseLight {
    pub fn new(a: Arc<dyn Texture>) -> Self {
        Self { emit: a.clone() }
    }
    pub fn new_from_color(c: &Vec3) -> Self {
        Self {
            emit: Arc::new(SolidColor::new(*c)),
        }
    }
}
impl Material for DiffuseLight {
    fn scatter(&self, _r_in: &Ray, _rec: &HitRecord) -> Option<(Vec3, Ray)> {
        None
    }
    fn emitted(&self, u: f64, v: f64, p: &Vec3) -> Vec3 {
        self.emit.value(u, v, p)
    }
}

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
}
pub struct HittableList {
    pub objects: Vec<Box<dyn Hittable>>,
}
impl HittableList {
    pub fn add(&mut self, hittable: Box<dyn Hittable>) {
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
}
fn surrounding_box(box0: AABB, box1: AABB) -> AABB {
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
}

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

fn random_scene() -> HittableList {
    let mut world = HittableList { objects: vec![] };

    // let material_ground = Arc::new(Lambertian::new(Vec3::new(0.5, 0.5, 0.5)));
    // world.add(Box::new(Sphere {
    //     center: Vec3::new(0.0, -1000.0, -1.0),
    //     radius: 1000.0,
    //     mat_ptr: material_ground,
    // }));
    let checker = Arc::new(CheckerTexture::new_from_color(
        &Vec3::new(0.2, 0.3, 0.1),
        &Vec3::new(0.9, 0.9, 0.9),
    ));
    world.add(Box::new(Sphere {
        center: Vec3::new(0.0, -1000.0, -1.0),
        radius: 1000.0,
        mat_ptr: Arc::new(Lambertian::new_from_arc(checker)),
    }));

    for a in -21..21 {
        for b in -21..21 {
            let choose_mat = random::<f64>();
            let center = Vec3::new(
                a as f64 + 0.9 * random::<f64>().abs(),
                random::<f64>().abs() / 3.0 + 0.1,
                b as f64 + 0.9 * random::<f64>().abs(),
            );
            if ((center - Vec3::new(0.0, 0.2, 0.0)) as Vec3).length() > 0.9 {
                if choose_mat < 0.5 {
                    let albedo = random_positive_unit() * 0.6 + Vec3::new(0.4, 0.25, 0.35);
                    let sphere_material = Arc::new(DiffuseLight::new_from_color(&albedo));
                    world.add(Box::new(Sphere {
                        center,
                        radius: center.y,
                        mat_ptr: sphere_material,
                    }))
                } else if choose_mat < 0.85 {
                    let albedo = random_positive_unit() / 2.0 + Vec3::new(0.5, 0.5, 0.5);
                    let fuzz = random::<f64>().abs() / 2.0;
                    let sphere_material = Arc::new(Metal::new(albedo, fuzz));
                    world.add(Box::new(Sphere {
                        center,
                        radius: center.y,
                        mat_ptr: sphere_material,
                    }))
                } else {
                    let sphere_material = Arc::new(Dielectric::new(1.5));
                    world.add(Box::new(Sphere {
                        center,
                        radius: center.y,
                        mat_ptr: sphere_material,
                    }))
                }
            }
        }
    }
    // let material1 = Arc::new(DiffuseLight::new_from_color(&Vec3::new(0.7, 0.6, 0.5)));
    // world.add(Box::new(Sphere {
    //     center: Vec3::new(0.0, 0.8, 0.0),
    //     radius: 0.79,
    //     mat_ptr: material1,
    // }));
    let checker1 = Arc::new(CheckerTexture::new_from_color(
        &Vec3::new(1.0, 0.45, 1.0),
        &Vec3::new(1.0, 0.9, 1.0),
    ));
    world.add(Box::new(Sphere {
        center: Vec3::new(0.0, 1.5, 0.0),
        radius: 1.4,
        mat_ptr: Arc::new(DiffuseLight::new(checker1)),
    }));
    let material2 = Arc::new(Dielectric::new(1.5));
    world.add(Box::new(Sphere {
        center: Vec3::new(0.0, 2.0, 0.0),
        radius: 2.0,
        mat_ptr: material2,
    }));
    let material = Arc::new(Dielectric::new(1.5));
    world.add(Box::new(Sphere {
        center: Vec3::new(0.0, 1.9, 0.0),
        radius: -1.8,
        mat_ptr: material,
    }));

    // let material3 = Arc::new(Metal::new(Vec3::new(0.7, 0.6, 0.5), 0.0));
    // world.add(Box::new(Sphere {
    //     center: Vec3::new(0.0, 0.8, 0.0),
    //     radius: 0.79,
    //     mat_ptr: material3,
    // }));

    world
}

fn ray_color(ray: &Ray, background: &Vec3, world: &dyn Hittable, depth: i32) -> Vec3 {
    if depth <= 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }
    let tmp = world.hit(ray, 0.001, f64::MAX);
    if let Some(rec) = tmp {
        let tmp = rec.mat_ptr.scatter(ray, &rec);
        if let Some((attenuation, scattered)) = tmp {
            return rec.mat_ptr.emitted(rec.u, rec.v, &rec.pos)
                + vec3::Vec3::elemul(
                    attenuation,
                    ray_color(&scattered, background, world, depth - 1),
                );
        }
        return rec.mat_ptr.emitted(rec.u, rec.v, &rec.pos);
    }
    // let unit_dir = (ray.dir).unit();
    // let t = 0.5 * (unit_dir.y + 1.0);
    // Vec3::new(255.0 - 127.5 * t, 255.0 - 76.5 * t, 255.0) / 255.0
    *background
}

pub struct Camera {
    pub aspect_ratio: f64,     // = 16.0 / 9.0;
    pub v_h: f64,              // = 2.0;
    pub v_w: f64,              // = v_h * aspect_ratio;
    pub f_l: f64,              // = 1.0;
    pub ori: Vec3,             // = Vec3::zero();
    pub horizontal: Vec3,      // = Vec3::new(v_w, 0.0, 0.0);
    pub vertical: Vec3,        // = Vec3::new(0.0, v_h, 0.0);
    pub low_left_corner: Vec3, // = ori - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, f_l);
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
    pub len_radius: f64,
}
impl Camera {
    pub fn new(
        lookform: &Vec3,
        lookat: &Vec3,
        vup: &Vec3,
        vfov: f64,
        aspect_ratio: f64,
        aperture: f64,
        focus_dist: f64,
    ) -> Self {
        let theta = vfov / 180.0 * PI;
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w: Vec3 = (*lookform - *lookat).unit();
        let u = vec3::Vec3::cross(*vup, w).unit();
        let v = vec3::Vec3::cross(w, u);

        let v_h = 2.0;
        let v_w = v_h * aspect_ratio;
        let f_l = 1.0;

        let ori = *lookform;
        let horizontal: Vec3 = u * (viewport_width * focus_dist);
        let vertical: Vec3 = v * (viewport_height * focus_dist);
        let low_left_corner = ori - horizontal / 2.0 - vertical / 2.0 - w * focus_dist;

        Self {
            ori,
            horizontal,
            vertical,
            low_left_corner,
            aspect_ratio,
            v_h,
            v_w,
            f_l,
            u,
            v,
            w,
            len_radius: aperture / 2.0,
        }
    }
    pub fn get_ray(&self, s: f64, t: f64) -> Ray {
        let rd: Vec3 = random_in_unit_disk() * self.len_radius;
        let offset = self.u * rd.x * rd.y;
        Ray::new(
            self.ori + offset,
            self.low_left_corner + self.horizontal * s + self.vertical * t - self.ori - offset,
        )
    }
}
fn sphere() {
    let i_h = 1080;
    let i_w = 1920;
    let samples_per_pixel = 500;
    let max_depth = 80;
    let mut img: RgbImage = ImageBuffer::new(i_w, i_h);
    let bar = ProgressBar::new(i_h as u64);

    let world = random_scene();

    let lookfrom = Vec3::new(16.0, 2.5, 10.0);
    let lookat = Vec3::new(0.0, 0.0, 0.0);
    let vup = Vec3::new(0.0, 1.0, 0.0);
    let dist_to_focus = 20.0;
    let aperture = 1.0;
    let background = Vec3::new(0.0, 0.0, 0.0);
    let camera = Camera::new(
        &lookfrom,
        &lookat,
        &vup,
        30.0,
        i_w as f64 / i_h as f64,
        aperture,
        dist_to_focus,
    );

    for j in 0..i_h {
        for i in 0..i_w {
            let mut color = Vec3::new(0.0, 0.0, 0.0);
            for _s in 1..samples_per_pixel {
                let u = (i as f64 + random_num()) / ((i_w - 1) as f64);
                let v = ((i_h - j) as f64 - random_num()) / ((i_h - 1) as f64);
                let r = camera.get_ray(u, v);
                color += ray_color(&r, &background, &world, max_depth);
            }
            color = color / (samples_per_pixel as f64);
            let pixel = img.get_pixel_mut(i, j);
            *pixel = image::Rgb([
                (color.x.sqrt() * 255.0) as u8,
                (color.y.sqrt() * 255.0) as u8,
                (color.z.sqrt() * 255.0) as u8,
            ]);
        }
        bar.inc(1);
    }

    img.save("output/test.png").unwrap();
    bar.finish();
}

fn main() {
    sphere();
}
