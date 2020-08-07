pub use crate::hittable::*;
pub use crate::onb::*;
pub use crate::randomtool::*;
pub use crate::ray::*;
pub use crate::texture::*;
pub use crate::vec3::*;
pub use rand::random;
pub use std::cmp::Ordering;
pub use std::f64::consts::PI;
pub use std::f64::INFINITY;
pub use std::sync::Arc;

pub trait Material: Sync + Send {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<(Vec3, Ray, f64)>;
    fn emitted(&self, r_in: &Ray, rec: &HitRecord, u: f64, v: f64, p: &Vec3) -> Vec3;
    fn scattering_pdf(&self, _r_in: &Ray, _rec: &HitRecord, _scattered: &Ray) -> f64 {
        0.0
    }
}
pub struct Lambertian {
    pub albedo: Arc<dyn Texture>,
}
impl Material for Lambertian {
    fn scatter(&self, _r_in: &Ray, rec: &HitRecord) -> Option<(Vec3, Ray, f64)> {
        let uvw = ONB::build_from_w(&rec.nor);
        let dir = uvw.local_vec(&random_cosine_direction());
        // let scatter_direction = rec.nor + random_unit();
        let scattered = Ray::new(rec.pos, dir.unit());
        Some((
            self.albedo.value(rec.u, rec.v, &rec.pos),
            scattered,
            (rec.nor * dir.unit()) / PI,
        ))
    }
    fn scattering_pdf(&self, _r_in: &Ray, rec: &HitRecord, scattered: &Ray) -> f64 {
        let cosine = rec.nor * scattered.dir.unit();
        if cosine >= 0.0 {
            cosine / PI
        } else {
            0.0
        }
    }
    fn emitted(&self, _r_in: &Ray, _rec: &HitRecord, _u: f64, _v: f64, _p: &Vec3) -> Vec3 {
        Vec3::zero()
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
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<(Vec3, Ray, f64)> {
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
            Some((
                attenuation,
                Ray::new(rec.pos, reflected + random_unit() * 0.25),
                0.0,
            ))
        } else {
            let reflect_prob = schlick(cos_theta, etai_over_etat);
            if random::<f64>() < reflect_prob {
                let reflected = reflect(unit_dir, rec.nor);
                Some((
                    attenuation,
                    Ray::new(rec.pos, reflected + random_unit() * 0.15),
                    0.0,
                ))
            } else {
                let refracted = refract(&unit_dir, &rec.nor, etai_over_etat);
                Some((attenuation, Ray::new(rec.pos, refracted), 0.0))
            }
        }
    }
    fn emitted(&self, _r_in: &Ray, _rec: &HitRecord, _u: f64, _v: f64, _p: &Vec3) -> Vec3 {
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
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<(Vec3, Ray, f64)> {
        let reflected = reflect(r_in.dir.unit(), rec.nor);
        let scattered = Ray::new(rec.pos, reflected + random_unit() * self.fuzz);
        if reflected * rec.nor > 0.0 {
            return Some((self.albedo, scattered, 0.0));
        }
        None
    }
    fn emitted(&self, _r_in: &Ray, _rec: &HitRecord, _u: f64, _v: f64, _p: &Vec3) -> Vec3 {
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
    fn scatter(&self, _r_in: &Ray, _rec: &HitRecord) -> Option<(Vec3, Ray, f64)> {
        // let reflected = Vec3::new(0.0, -1.0, 0.0);//reflect(r_in.dir.unit(), rec.nor);
        // let scattered = Ray::new(rec.pos, reflected + random_vec(&rec.nor));
        // if reflected * rec.nor > 0.0 {
        //     return Some((
        //         self.emit.value(0.0, 0.0, &Vec3::new(0.0, 0.0, 0.0)),
        //         scattered,
        //     ));
        // }
        None
    }
    fn emitted(&self, _r_in: &Ray, rec: &HitRecord, u: f64, v: f64, p: &Vec3) -> Vec3 {
        if rec.front_face {
            self.emit.value(u, v, p)
        } else {
            Vec3::zero()
        }
    }
}