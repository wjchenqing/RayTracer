#[allow(clippy::float_cmp)]
mod ray;
mod vec3;
use image::{ImageBuffer, RgbImage};
use indicatif::ProgressBar;

pub use ray::Ray;
pub use vec3::Vec3;

fn hit_sphere(center: &Vec3, radius: &f64, r: &Ray) -> bool {
    let oc: Vec3 = r.ori.copy() - center.copy();
    let a = r.dir.squared_length();
    let b = (oc * r.dir) * 2.0;
    let c = oc.squared_length() - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return false;
    }
    true
}

fn ray_color(r: &Ray) -> Vec3 {
    if hit_sphere(&Vec3::new(0.0, 0.0, -1.0), &0.5, r) {
        return Vec3::new(255.0, 0.0, 0.0);
    }
    let unit_dir = (r.dir).unit();
    let t = 0.5 * (unit_dir.y + 1.0);
    Vec3::new(255.0 - 127.5 * t, 255.0 - 76.5 * t, 255.0)
}

fn main() {
    let aspect_ratio = 16.0 / 9.0;
    let i_h = 576;
    let i_w = 1024;
    let mut img: RgbImage = ImageBuffer::new(i_w, i_h);
    let bar = ProgressBar::new(i_h as u64);

    let v_h = 2.0;
    let v_w = v_h * aspect_ratio;
    let f_l = 1.0;

    let ori = Vec3::zero();
    let horizontal = Vec3::new(v_w, 0.0, 0.0);
    let vertical = Vec3::new(0.0, v_h, 0.0);
    let low_left_corner = ori - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, f_l);

    for j in 0..i_h {
        for i in 0..i_w {
            let u = (i as f64) / ((i_w - 1) as f64);
            let v = ((i_h - j) as f64) / ((i_h - 1) as f64);
            let r = Ray::new(ori, low_left_corner + horizontal * u + vertical * v - ori);
            let color = ray_color(&r);
            let pixel = img.get_pixel_mut(i, j);
            *pixel = image::Rgb([color.x as u8, color.y as u8, color.z as u8]);
        }
        bar.inc(1);
    }

    img.save("output/test.png").unwrap();
    bar.finish();
}
