#![allow(clippy::float_cmp)]
mod bvh;
mod hittable;
mod material;
mod onb;
mod randomtool;
mod ray;
mod scene;
mod texture;
mod vec3;
use image::{ImageBuffer, RgbImage};
use indicatif::ProgressBar;
pub use rand::random;
pub use rand::{rngs::SmallRng, Rng};
pub use std::cmp::Ordering;
pub use std::f64::consts::PI;
pub use std::f64::INFINITY;
use std::sync::mpsc::channel;
pub use std::sync::Arc;
use threadpool::ThreadPool;

pub use bvh::*;
pub use hittable::*;
pub use material::*;
pub use onb::*;
pub use randomtool::*;
pub use ray::Ray;
pub use scene::*;
pub use texture::*;
pub use vec3::Vec3;

fn ray_color(ray: &Ray, background: &Vec3, world: &dyn Hittable, depth: i32) -> Vec3 {
    if depth <= 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }
    let tmp = world.hit(ray, 0.001, f64::MAX);
    if let Some(rec) = tmp {
        let tmp = rec.mat_ptr.scatter(ray, &rec);
        if let Some((attenuation, _scattered, _pdf)) = tmp {
            let on_light = Vec3::new(
                213.0 + (343.0 - 213.0) * random::<f64>(),
                554.0,
                227.0 + (332.0 - 227.0) * random::<f64>(),
            );
            let to_light = on_light - rec.pos;
            let distance_squared = to_light.squared_length();
            let to_light = to_light.unit();
            if to_light * rec.nor < 0.0 {
                return rec.mat_ptr.emitted(&ray, &rec, rec.u, rec.v, &rec.pos);
            }
            let light_area = (343.0 - 213.0) * (332.0 - 227.0);
            let light_cosine = to_light.y.abs();
            if light_cosine < 0.000001 {
                return rec.mat_ptr.emitted(&ray, &rec, rec.u, rec.v, &rec.pos);
            }
            let pdf = distance_squared / (light_area * light_cosine);
            let scattered = Ray::new(rec.pos, to_light);
            return rec.mat_ptr.emitted(&ray, &rec, rec.u, rec.v, &rec.pos)
                + vec3::Vec3::elemul(
                    attenuation,
                    ray_color(&scattered, background, world, depth - 1),
                ) * rec.mat_ptr.scattering_pdf(&ray, &rec, &scattered)
                    / pdf;
        }
        return rec.mat_ptr.emitted(&ray, &rec, rec.u, rec.v, &rec.pos);
    }
    // let unit_dir = (ray.dir).unit();
    // let t = 0.5 * (unit_dir.y + 1.0);
    // Vec3::new(255.0 - 127.5 * t, 255.0 - 76.5 * t, 255.0) / 255.0
    *background
}

#[derive(Clone, Copy)]
pub struct Camera {
    pub aspect_ratio: f64,
    pub v_h: f64,
    pub v_w: f64,
    pub f_l: f64,
    pub ori: Vec3,
    pub horizontal: Vec3,
    pub vertical: Vec3,
    pub low_left_corner: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
    pub len_radius: f64,
}
impl Camera {
    pub fn new(
        lookfrom: &Vec3,
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

        let w: Vec3 = (*lookfrom - *lookat).unit();
        let u = vec3::Vec3::cross(*vup, w).unit();
        let v = vec3::Vec3::cross(w, u);

        let v_h = 2.0;
        let v_w = v_h * aspect_ratio;
        let f_l = 1.0;

        let ori = *lookfrom;
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
        // let rd: Vec3 = Vec3::elemul(
        //     random_in_unit_disk() * self.len_radius,
        //     Vec3::new(16.0, 9.0, 0.0).unit(),
        // );
        let rd = random_in_unit_disk() * self.len_radius;
        let offset = self.u * rd.x + self.v * rd.y;
        Ray::new(
            self.ori + offset,
            self.low_left_corner + self.horizontal * s + self.vertical * t - self.ori - offset,
        )
    }
}

fn sphere() {
    let i_h = 600;
    let i_w = 600;
    let (tx, rx) = channel();
    let n_jobs: usize = 32;
    let n_workers = 4;
    let pool = ThreadPool::new(n_workers);

    let samples_per_pixel = 10;
    let max_depth = 50;

    let mut lights = HittableList { objects: vec![] };
    lights.add(Arc::new(XzRect {
        x0: 213.0,
        x1: 343.0,
        z0: 227.0,
        z1: 332.0,
        k: 554.0,
        mp: Arc::new(Lambertian::new(Vec3::zero())),
    }));
    lights.add(Arc::new(Sphere {
        center: Vec3::new(190.0, 90.0, 190.0),
        radius: 90.0,
        mat_ptr: Arc::new(Lambertian::new(Vec3::zero())),
    }));

    // let world = random_scene();
    // let world = simple_light();
    let world = cornell_box();

    let background = Vec3::new(0.0, 0.0, 0.0);
    let camera = Camera::new(
        // &Vec3::new(17.0, 4.0, 12.5),
        // &Vec3::new(-7.8, -1.6, 0.9),
        // &Vec3::new(0.0, 1.0, 0.0),
        // 25.0,
        &Vec3::new(278.0, 278.0, -800.0),
        &Vec3::new(278.0, 278.0, 0.0),
        &Vec3::new(0.0, 1.0, 0.0),
        40.0,
        i_w as f64 / i_h as f64,
        0.0,
        20.0,
    );

    let mut result: RgbImage = ImageBuffer::new(i_w, i_h);
    let bar = ProgressBar::new(n_jobs as u64);

    for i in 0..n_jobs {
        let tx = tx.clone();
        let world = world.clone();
        pool.execute(move || {
            let row_begin = i_h as usize * i / n_jobs;
            let row_end = i_h as usize * (i + 1) / n_jobs;
            let render_height = row_end - row_begin;
            let mut img: RgbImage = ImageBuffer::new(i_w, render_height as u32);
            for x in 0..i_w {
                for (img_y, y) in (row_begin..row_end).enumerate() {
                    let y = y as u32;
                    let mut color = Vec3::new(0.0, 0.0, 0.0);
                    for _s in 1..samples_per_pixel {
                        let u = (x as f64 + random_num()) / ((i_w - 1) as f64);
                        let v = ((i_h - y) as f64 - random_num()) / ((i_h - 1) as f64);
                        let r = camera.get_ray(u, v);
                        color += ray_color(&r, &background, &world, max_depth);
                    }
                    color = color / (samples_per_pixel as f64);
                    let pixel = img.get_pixel_mut(x, img_y as u32);
                    *pixel = image::Rgb([
                        (color.x.sqrt() * 255.0) as u8,
                        (color.y.sqrt() * 255.0) as u8,
                        (color.z.sqrt() * 255.0) as u8,
                    ]);
                }
            }
            tx.send((row_begin..row_end, img))
                .expect("failed to send result");
        });
    }

    // for j in 0..i_h {
    //     for i in 0..i_w {
    //         let mut color = Vec3::new(0.0, 0.0, 0.0);
    //         for _s in 1..samples_per_pixel {
    //             let u = (i as f64 + random_num()) / ((i_w - 1) as f64);
    //             let v = ((i_h - j) as f64 - random_num()) / ((i_h - 1) as f64);
    //             let r = camera.get_ray(u, v);
    //             color += ray_color(&r, &background, &world, max_depth);
    //         }
    //         color = color / (samples_per_pixel as f64);
    //         let pixel = img.get_pixel_mut(i, j);
    //         *pixel = image::Rgb([
    //             (color.x.sqrt() * 255.0) as u8,
    //             (color.y.sqrt() * 255.0) as u8,
    //             (color.z.sqrt() * 255.0) as u8,
    //         ]);
    //     }
    //     bar.inc(1);
    // }

    for (rows, data) in rx.iter().take(n_jobs) {
        for (idx, row) in rows.enumerate() {
            for col in 0..i_w {
                let row = row as u32;
                let idx = idx as u32;
                *result.get_pixel_mut(col, row) = *data.get_pixel(col, idx);
            }
        }
        bar.inc(1);
    }

    result.save("output/test.png").unwrap();
    bar.finish();
}

fn main() {
    sphere();
}
