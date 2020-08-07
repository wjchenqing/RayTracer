pub use crate::bvh::*;
pub use crate::hittable::*;
pub use crate::material::*;
pub use crate::randomtool::*;
pub use crate::ray::Ray;
pub use crate::texture::*;
pub use crate::vec3::Vec3;

pub fn cornell_box() -> HittableList {
    let mut objects = HittableList { objects: vec![] };
    let red = Arc::new(Lambertian::new(Vec3::new(0.65, 0.05, 0.05)));
    let white = Arc::new(Lambertian::new(Vec3::new(0.73, 0.73, 0.73)));
    let green = Arc::new(Lambertian::new(Vec3::new(0.12, 0.45, 0.15)));
    let light = Arc::new(DiffuseLight::new_from_color(&Vec3::new(15.0, 15.0, 15.0)));
    objects.add(Arc::new(YzRect {
        y0: 0.0,
        y1: 555.0,
        z0: 0.0,
        z1: 555.0,
        k: 555.0,
        mp: green,
    }));
    objects.add(Arc::new(YzRect {
        y0: 0.0,
        y1: 555.0,
        z0: 0.0,
        z1: 555.0,
        k: 0.0,
        mp: red,
    }));
    objects.add(Arc::new(XzRect {
        x0: 213.0,
        x1: 343.0,
        z0: 227.0,
        z1: 332.0,
        k: 554.0,
        mp: light,
    }));
    // objects.add(Arc::new(Sphere {
    //     center: Vec3::new(278.0, 554.0, 280.0),
    //     radius: 65.0,
    //     mat_ptr: light,
    // }));
    objects.add(Arc::new(XzRect {
        x0: 0.0,
        x1: 555.0,
        z0: 0.0,
        z1: 555.0,
        k: 0.0,
        mp: white.clone(),
    }));
    objects.add(Arc::new(XzRect {
        x0: 0.0,
        x1: 555.0,
        z0: 0.0,
        z1: 555.0,
        k: 555.0,
        mp: white.clone(),
    }));
    objects.add(Arc::new(XyRect {
        x0: 0.0,
        x1: 555.0,
        y0: 0.0,
        y1: 555.0,
        k: 555.0,
        mp: white.clone(),
    }));
    objects.add(Arc::new(Translate::new(
        &Arc::new(RotateY::new(
            &Arc::new(Box::new(
                &Vec3::new(0.0, 0.0, 0.0),
                &Vec3::new(165.0, 330.0, 165.0),
                &white,
            )),
            15.0,
        )),
        &Vec3::new(265.0, 0.0, 295.0),
    )));
    objects.add(Arc::new(Translate::new(
        &Arc::new(RotateY::new(
            &Arc::new(Box::new(
                &Vec3::new(0.0, 0.0, 0.0),
                &Vec3::new(165.0, 165.0, 165.0),
                &white,
            )),
            -18.0,
        )),
        &Vec3::new(130.0, 0.0, 65.0),
    )));
    objects
}
/*pub fn random_scene() -> HittableList {
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
    world.add(Arc::new(Sphere {
        center: Vec3::new(0.0, -2000.0, -1.0),
        radius: 2000.0,
        mat_ptr: Arc::new(Lambertian::new_from_arc(checker)),
    }));

    for a in -11..11 {
        for b in -10..12 {
            let choose_mat = random::<f64>();
            let center = Vec3::new(
                a as f64 + 0.6 * random::<f64>().abs(),
                random::<f64>().abs() * random::<f64>().abs() * 0.47 + 0.06,
                b as f64 + 0.6 * random::<f64>().abs(),
            );
            if ((center - Vec3::new(0.0, 2.0, 0.0)) as Vec3).length() - center.y > 2.2 {
                if choose_mat < 0.8 {
                    let albedo = random_positive_unit() * 0.6 + Vec3::new(0.35, 0.35, 0.2);
                    world.add(Arc::new(Sphere {
                        center,
                        radius: center.y * 0.8,
                        mat_ptr: Arc::new(DiffuseLight::new_from_color(&albedo)),
                    }));
                // world.add(Arc::new(Sphere {
                //     center,
                //     radius: center.y,
                //     mat_ptr: Arc::new(Metal::new(albedo, /*random::<f64>().abs()*/ 1000.0)),
                // }));
                // world.add(Arc::new(Sphere {
                //     center,
                //     radius: center.y,
                //     mat_ptr: Arc::new(Dielectric::new(5.0)),
                // }));
                } else if choose_mat < 0.9 {
                    world.add(Arc::new(Sphere {
                        center,
                        radius: center.y,
                        mat_ptr: Arc::new(Metal::new(
                            random_positive_unit() / 2.0 + Vec3::new(0.5, 0.5, 0.5),
                            random::<f64>().abs() / 4.0,
                        )),
                    }));
                } else {
                    world.add(Arc::new(Sphere {
                        center,
                        radius: center.y,
                        mat_ptr: Arc::new(Dielectric::new(1.5)),
                    }));
                }
            }
        }
    }
    // let material1 = Arc::new(DiffuseLight::new_from_color(&Vec3::new(1.0, 0.53, 0.07)));
    // world.add(Arc::new(Sphere {
    //     center: Vec3::new(0.0, 1.5, 0.0),
    //     radius: 1.4,
    //     mat_ptr: material1,
    // }));
    world.add(Arc::new(Sphere {
        center: Vec3::new(0.0, 1.5, 0.0),
        radius: 1.4,
        mat_ptr: Arc::new(DiffuseLight::new(Arc::new(CheckerTexture::new_from_color(
            &Vec3::new(1.0, 0.53, 0.07),
            &Vec3::new(1.0, 0.81, 0.64),
        )))),
    }));
    world.add(Arc::new(Sphere {
        center: Vec3::new(0.0, 2.0, 0.0),
        radius: 2.0,
        mat_ptr: Arc::new(Dielectric::new(2.3)),
    }));
    world.add(Arc::new(Sphere {
        center: Vec3::new(0.0, 1.85, 0.0),
        radius: -1.75,
        mat_ptr: Arc::new(Dielectric::new(2.3)),
    }));
    // world.add(Arc::new(Sphere {
    //     center: Vec3::new(0.0, 1.5, 0.0),
    //     radius: 1.4001,
    //     mat_ptr: Arc::new(Metal::new(Vec3::new(1.0, 0.9, 1.0), 500.0)),
    // }));

    world
}*/

/*pub fn simple_light() -> HittableList {
    let mut objects = HittableList { objects: vec![] };

    let checker = Arc::new(CheckerTexture::new_from_color(
        &Vec3::new(0.2, 0.3, 0.1),
        &Vec3::new(0.9, 0.9, 0.9),
    ));
    objects.add(Arc::new(Sphere {
        center: Vec3::new(0.0, -1000.0, 0.0),
        radius: 1000.0,
        mat_ptr: Arc::new(Lambertian::new_from_arc(checker.clone())),
    }));
    objects.add(Arc::new(Sphere {
        center: Vec3::new(0.0, 2.0, 0.0),
        radius: 2.0,
        mat_ptr: Arc::new(Lambertian::new_from_arc(checker)),
    }));
    let difflight = Arc::new(DiffuseLight::new_from_color(&Vec3::new(4.0, 4.0, 4.0)));
    objects.add(Arc::new(XyRect {
        x0: 3.0,
        x1: 5.0,
        y0: 1.0,
        y1: 3.0,
        k: -2.0,
        mp: difflight,
    }));

    objects
}*/
