#[allow(clippy::float_cmp)]
mod ray;
mod vec3;
use image::{ImageBuffer, RgbImage};
use indicatif::ProgressBar;
pub use rand::random;
pub use std::f64::consts;
pub use std::f64::INFINITY;

pub use ray::Ray;
pub use vec3::Vec3;

fn random_num() -> f64 {
    let tmp = random::<f64>();
    tmp / f64::MAX
}
fn random_vec() -> Vec3 {
    let x = random::<i8>();
    let y = random::<i8>();
    let z = random::<i8>();
    if x == 0 && y == 0 && z==0{
        return random_vec();
    }
    Vec3::new(x as f64, y as f64, z as f64).unit()
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HitRecord {
    pub pos: Vec3,
    pub nor: Vec3,
    pub t: f64,
    pub front_face: bool,
}
pub trait Hittable {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
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
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut hit_anything: Option<HitRecord> = None;
        let mut closest_so_far = t_max;
        for i in self.objects.iter() {
            let tmp_rec = i.hit(r, t_min, closest_so_far);
            if let Some(tmp) = tmp_rec {
                hit_anything = tmp_rec;
                closest_so_far = tmp.t;
            }
        }
        hit_anything
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
}
impl Hittable for Sphere {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc: Vec3 = r.ori.copy() - self.center.copy();
        let a = r.dir.squared_length();
        let _b = oc * r.dir;
        let c = oc.squared_length() - self.radius * self.radius;
        let discriminant = _b * _b - a * c;

        if discriminant > 0.0 {
            let root = discriminant.sqrt();
            let tmp = (-_b - root) / a;
            if (tmp < t_max) && (tmp > t_min) {
                let pos = r.at(tmp);
                let mut nor = (pos - self.center) / self.radius;
                let flag = (r.dir * nor) < 0.0;
                if !flag {
                    nor = -nor;
                }
                return Some(HitRecord {
                    t: tmp,
                    pos,
                    nor,
                    front_face: flag,
                });
            } else {
                let tmp = (-_b + root) / a;
                if (tmp < t_max) && (tmp > t_min) {
                    let pos = r.at(tmp);
                    let mut nor = (pos - self.center) / self.radius;
                    let flag = (r.dir * nor) < 0.0;
                    if !flag {
                        nor = -nor;
                    }
                    return Some(HitRecord {
                        t: tmp,
                        pos,
                        nor,
                        front_face: flag,
                    });
                }
            }
        }
        None
    }
}

fn ray_color(r: &Ray, world: &dyn Hittable, depth: i32) -> Vec3 {
    if depth <= 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }
    let tmp = world.hit(r, 0.0, f64::MAX);
    if let Some(rec) = tmp {
        let dir = rec.nor + random_vec();
        // return (rec.nor + Vec3::new(1.0, 1.0, 1.0)) * 0.5 * 255.0;
        return ray_color(&Ray::new(rec.pos, dir), world, depth - 1) * 0.5;
    }
    let unit_dir = (r.dir).unit();
    let t = 0.5 * (unit_dir.y + 1.0);
    Vec3::new(255.0 - 127.5 * t, 255.0 - 76.5 * t, 255.0)
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
}
impl Camera {
    pub fn new() -> Self {
        let aspect_ratio = 16.0 / 9.0;
        let v_h = 2.0;
        let v_w = v_h * aspect_ratio;
        let f_l = 1.0;

        let ori = Vec3::zero();
        let horizontal = Vec3::new(v_w, 0.0, 0.0);
        let vertical = Vec3::new(0.0, v_h, 0.0);
        let low_left_corner = ori - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, f_l);

        Self {
            ori,
            horizontal,
            vertical,
            low_left_corner,
            aspect_ratio,
            v_h,
            v_w,
            f_l,
        }
    }
    pub fn get_ray(&self, u: f64, v: f64) -> Ray {
        Ray::new(
            self.ori,
            self.low_left_corner + self.horizontal * u + self.vertical * v - self.ori,
        )
    }
}
fn sphere() {
    let i_h = 576;
    let i_w = 1024;
    let samples_per_pixel = 100;
    let max_depth = 50;
    let mut img: RgbImage = ImageBuffer::new(i_w, i_h);
    let bar = ProgressBar::new(i_h as u64);

    let mut world = HittableList { objects: vec![] };
    world.add(Box::new(Sphere {
        center: Vec3::new(0.0, 0.0, -1.0),
        radius: 0.5,
    }));
    world.add(Box::new(Sphere {
        center: Vec3::new(0.0, -100.5, -1.0),
        radius: 100.0,
    }));

    let camera = Camera::new();

    for j in 0..i_h {
        for i in 0..i_w {
            let mut color = Vec3::new(0.0, 0.0, 0.0);
            for _s in 1..samples_per_pixel {
                let u = (i as f64 + random_num()) / ((i_w - 1) as f64);
                let v = ((i_h - j) as f64 - random_num()) / ((i_h - 1) as f64);
                let r = camera.get_ray(u, v);
                color += ray_color(&r, &world, max_depth);
            }
            color = color / (samples_per_pixel as f64);
            let pixel = img.get_pixel_mut(i, j);
            *pixel = image::Rgb([color.x as u8, color.y as u8, color.z as u8]);
        }
        bar.inc(1);
    }

    img.save("output/test.png").unwrap();
    bar.finish();
}

fn main() {
    sphere();
}
