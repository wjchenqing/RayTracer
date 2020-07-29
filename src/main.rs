#[allow(clippy::float_cmp)]
mod ray;
mod vec3;
use image::{ImageBuffer, RgbImage};
use indicatif::ProgressBar;
pub use rand::random;
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
    if tmp * nor.copy() > 0.0 {
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
}
pub struct Lambertian {
    albedo: Vec3,
}
impl Material for Lambertian {
    fn scatter(&self, _r_in: &Ray, rec: &HitRecord) -> Option<(Vec3, Ray)> {
        let scatter_direction = rec.nor + random_vec(&rec.nor);
        let scattered = Ray::new(rec.pos, scatter_direction);
        Some((self.albedo, scattered))
    }
}
impl Lambertian {
    pub fn new(albedo: Vec3) -> Self {
        Self { albedo }
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
}

#[derive(Clone)]
pub struct HitRecord {
    pub pos: Vec3,
    pub nor: Vec3,
    pub t: f64,
    pub front_face: bool,
    pub mat_ptr: Arc<dyn Material>,
}
pub trait Hittable: Sync + Send {
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
                hit_anything = Some(tmp.clone());
                closest_so_far = tmp.t;
            }
        }
        hit_anything
    }
}

#[derive(Clone)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
    pub mat_ptr: Arc<dyn Material>,
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
                    mat_ptr: self.mat_ptr.clone(),
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
                        mat_ptr: self.mat_ptr.clone(),
                    });
                }
            }
        }
        None
    }
}

fn random_scene() -> HittableList {
    let mut world = HittableList { objects: vec![] };

    let material_ground = Arc::new(Lambertian::new(Vec3::new(0.5, 0.5, 0.5)));
    world.add(Box::new(Sphere {
        center: Vec3::new(0.0, -1000.0, -1.0),
        radius: 1000.0,
        mat_ptr: material_ground,
    }));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = random::<f64>();
            let center = Vec3::new(
                a as f64 + 0.9 * random::<f64>().abs(),
                0.2,
                b as f64 + 0.9 * random::<f64>().abs(),
            );
            if ((center - Vec3::new(4.0, 0.2, 0.0)) as Vec3).length() > 0.9 {
                if choose_mat < 0.8 {
                    let albedo = vec3::Vec3::elemul(random_positive_unit(), random_positive_unit());
                    let sphere_material = Arc::new(Lambertian::new(albedo));
                    world.add(Box::new(Sphere {
                        center,
                        radius: 0.2,
                        mat_ptr: sphere_material,
                    }))
                } else if choose_mat < 0.95 {
                    let albedo = random_positive_unit() / 2.0 + Vec3::new(0.5, 0.5, 0.5);
                    let fuzz = random::<f64>().abs() / 2.0;
                    let sphere_material = Arc::new(Metal::new(albedo, fuzz));
                    world.add(Box::new(Sphere {
                        center,
                        radius: 0.2,
                        mat_ptr: sphere_material,
                    }))
                } else {
                    let sphere_material = Arc::new(Dielectric::new(1.5));
                    world.add(Box::new(Sphere {
                        center,
                        radius: 0.2,
                        mat_ptr: sphere_material,
                    }))
                }
            }
        }
    }
    let material1 = Arc::new(Lambertian::new(Vec3::new(0.4, 0.2, 0.1)));
    world.add(Box::new(Sphere {
        center: Vec3::new(-4.0, 1.0, 0.0),
        radius: 1.0,
        mat_ptr: material1,
    }));
    let material2 = Arc::new(Dielectric::new(1.5));
    world.add(Box::new(Sphere {
        center: Vec3::new(0.0, 1.0, 0.0),
        radius: 1.0,
        mat_ptr: material2,
    }));
    let material3 = Arc::new(Metal::new(Vec3::new(0.7, 0.6, 0.5), 0.0));
    world.add(Box::new(Sphere {
        center: Vec3::new(4.0, 1.0, 0.0),
        radius: 1.0,
        mat_ptr: material3,
    }));

    world
}

fn ray_color(r: &Ray, world: &dyn Hittable, depth: i32) -> Vec3 {
    if depth <= 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }
    let tmp = world.hit(r, 0.001, f64::MAX);
    if let Some(rec) = tmp {
        // let dir = rec.nor + random_vec(&rec.nor);
        // return (rec.nor + Vec3::new(1.0, 1.0, 1.0)) * 0.5 * 255.0;
        // return ray_color(&Ray::new(rec.pos, dir), world, depth - 1) * 0.5;
        let tmp = rec.mat_ptr.scatter(r, &rec);
        if let Some((attenuation, scattered)) = tmp {
            return vec3::Vec3::elemul(attenuation, ray_color(&scattered, world, depth - 1));
        }
        return Vec3::new(0.0, 0.0, 0.0);
    }
    let unit_dir = (r.dir).unit();
    let t = 0.5 * (unit_dir.y + 1.0);
    Vec3::new(255.0 - 127.5 * t, 255.0 - 76.5 * t, 255.0) / 255.0
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
    let i_h = 800;
    let i_w = 1200;
    let samples_per_pixel = 500;
    let max_depth = 50;
    let mut img: RgbImage = ImageBuffer::new(i_w, i_h);
    let bar = ProgressBar::new(i_h as u64);

    let world = random_scene();

    // let r = (PI / 4.0).cos();
    // let material_left = Arc::new(Lambertian::new(Vec3::new(0.0, 0.0, 1.0)));
    // let material_right = Arc::new(Lambertian::new(Vec3::new(1.0, 0.0, 0.0)));
    // world.add(Box::new(Sphere {
    //     center: Vec3::new(-r, 0.0, -1.0),
    //     radius: r,
    //     mat_ptr: material_left,
    // }));
    // world.add(Box::new(Sphere {
    //     center: Vec3::new(r, 0.0, -1.0),
    //     radius: r,
    //     mat_ptr: material_right,
    // }));

    // let material_center = Arc::new(Lambertian::new(Vec3::new(0.1, 0.2, 0.5)));
    // let material_ground = Arc::new(Lambertian::new(Vec3::new(0.8, 0.8, 0.0)));
    // let material_left = Arc::new(Dielectric::new(1.5));
    // let material_right = Arc::new(Metal::new(Vec3::new(0.8, 0.6, 0.2), 0.0));
    // world.add(Box::new(Sphere {
    //     center: Vec3::new(0.0, 0.0, -1.0),
    //     radius: 0.5,
    //     mat_ptr: material_center,
    // }));
    // world.add(Box::new(Sphere {
    //     center: Vec3::new(0.0, -100.5, -1.0),
    //     radius: 100.0,
    //     mat_ptr: material_ground,
    // }));
    // world.add(Box::new(Sphere {
    //     center: Vec3::new(-1.0, 0.0, -1.0),
    //     radius: 0.5,
    //     mat_ptr: material_left.clone(),
    // }));
    // world.add(Box::new(Sphere {
    //     center: Vec3::new(-1.0, 0.0, -1.0),
    //     radius: -0.45,
    //     mat_ptr: material_left.clone(),
    // }));
    // world.add(Box::new(Sphere {
    //     center: Vec3::new(1.0, 0.0, -1.0),
    //     radius: 0.5,
    //     mat_ptr: material_right,
    // }));

    let lookfrom = Vec3::new(13.0, 2.0, 3.0);
    let lookat = Vec3::new(0.0, 0.0, 0.0);
    let vup = Vec3::new(0.0, 1.0, 0.0);
    let dist_to_focus = 10.0;
    let aperture = 0.1;
    let camera = Camera::new(
        &lookfrom,
        &lookat,
        &vup,
        20.0,
        3.0 / 2.0,
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
                color += ray_color(&r, &world, max_depth);
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
