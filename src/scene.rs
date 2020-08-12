pub use crate::bvh::*;
pub use crate::hittable::*;
pub use crate::material::*;
pub use crate::randomtool::*;
pub use crate::ray::Ray;
pub use crate::texture::*;
pub use crate::vec3::Vec3;

pub fn final_scene() -> HittableList {
    let mut objects = HittableList { objects: vec![] };

    let mut box1 = HittableList { objects: vec![] };
    let ground = Arc::new(Lambertian::new(Vec3::new(0.48, 0.83, 0.53)));
    let box_per_side = 19;
    for i in 0..box_per_side {
        for j in 0..box_per_side {
            let w = 100.0;
            let x0 = -1000.0 + i as f64 * w;
            let z0 = -1000.0 + j as f64 * w;
            let y0 = 0.0;
            let x1 = x0 + w;
            let y1 = 1.0 + random::<f64>() * 100.0;
            let z1 = z0 + w;
            box1.add(Arc::new(Box::new(
                &Vec3::new(x0, y0, z0),
                &Vec3::new(x1, y1, z1),
                ground.clone(),
            )));
        }
    }
    // objects.add(Arc::new(BvhNode::new_from_list(&mut box1, 0.0, 1.0)));
    objects.objects.append(&mut box1.objects);

    let light = Arc::new(DiffuseLight::new_from_color(&Vec3::new(7.0, 7.0, 7.0)));
    objects.add(Arc::new(XzRect {
        x0: 123.0,
        x1: 423.0,
        z0: 147.0,
        z1: 412.0,
        k: 554.0,
        mp: light,
    }));

    objects.add(Arc::new(Sphere {
        center: Vec3::new(260.0, 150.0, 45.0),
        radius: 50.0,
        mat_ptr: Arc::new(Dielectric::new(1.5)),
    }));
    objects.add(Arc::new(Sphere {
        center: Vec3::new(0.0, 150.0, 145.0),
        radius: 50.0,
        mat_ptr: Arc::new(Metal::new(Vec3::new(0.8, 0.8, 0.9), 10.0)),
    }));

    let boundary = Arc::new(Sphere {
        center: Vec3::new(360.0, 150.0, 145.0),
        radius: 70.0,
        mat_ptr: Arc::new(Dielectric::new(1.5)),
    });
    objects.add(boundary.clone());
    objects.add(Arc::new(ConstantMedium {
        boundary,
        neg_inv_density: -1.0 / 0.2,
        phase_function: Arc::new(Isotropic {
            albedo: Arc::new(SolidColor::new(Vec3::new(0.2, 0.4, 0.9))),
        }),
    }));
    let boundary = Arc::new(Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 5000.0,
        mat_ptr: Arc::new(Dielectric::new(1.5)),
    });
    objects.add(Arc::new(ConstantMedium {
        boundary,
        neg_inv_density: -1.0 / 0.0001,
        phase_function: Arc::new(Isotropic {
            albedo: Arc::new(SolidColor::new(Vec3::new(1.0, 1.0, 1.0))),
        }),
    }));

    let emat = Arc::new(Lambertian {
        albedo: Arc::new(ImageTexture::new("pikachu/timgD38FGAN2.jpg")),
    });
    objects.add(Arc::new(Sphere {
        center: Vec3::new(400.0, 200.0, 400.0),
        radius: 100.0,
        mat_ptr: emat,
    }));

    let pertext = Arc::new(NoiseTexture::new_from_f64(0.1));
    objects.add(Arc::new(Sphere {
        center: Vec3::new(220.0, 280.0, 300.0),
        radius: 80.0,
        mat_ptr: Arc::new(Lambertian::new_from_arc(pertext)),
    }));

    let mut box2 = HittableList { objects: vec![] };
    let white = Arc::new(Lambertian::new(Vec3::new(0.73, 0.73, 0.73)));
    let ns = 1000;
    for _i in 1..ns {
        box2.add(Arc::new(Sphere {
            center: Vec3::new(
                random::<f64>() * 165.0,
                random::<f64>() * 165.0,
                random::<f64>() * 165.0,
            ),
            radius: 10.0,
            mat_ptr: white.clone(),
        }));
    }
    // objects.add(Arc::new(Translate::new(
    //     &Arc::new(RotateY::new(
    //         Arc::new(BvhNode::new_from_list(&mut box2, 0.0, 1.0)),
    //         15.0,
    //     )),
    //     &Vec3::new(-100.0, 270.0, 395.0),
    // )));
    objects.objects.append(&mut box2.objects);

    objects
}

// pub fn two_perlin_sphere() -> HittableList {
//     let mut objects = HittableList { objects: vec![] };

//     let pertext = Arc::new(NoiseTexture::new());
//     objects.add(Arc::new(Sphere {
//         center: Vec3::new(0.0, -1000.0, 0.0),
//         radius: 1000.0,
//         mat_ptr: Arc::new(Lambertian {
//             albedo: pertext.clone(),
//         }),
//     }));
//     objects.add(Arc::new(Sphere {
//         center: Vec3::new(0.0, 2.0, 0.0),
//         radius: 2.0,
//         mat_ptr: Arc::new(Lambertian { albedo: pertext }),
//     }));
//     objects
// }

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
    objects.add(Arc::new(FlipFace {
        ptr: Arc::new(XzRect {
            x0: 213.0,
            x1: 343.0,
            z0: 227.0,
            z1: 332.0,
            k: 554.0,
            mp: light,
        }),
    }));
    // objects.add(Arc::new(XzRect {
    //     x0: 213.0,
    //     x1: 343.0,
    //     z0: 227.0,
    //     z1: 332.0,
    //     k: 554.0,
    //     mp: light,
    // }));
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
    let box1 = Arc::new(Translate::new(
        &Arc::new(RotateY::new(
            Arc::new(Box::new(
                &Vec3::new(0.0, 0.0, 0.0),
                &Vec3::new(165.0, 330.0, 165.0),
                // Arc::new(Metal::new(Vec3::new(0.8, 0.85, 0.85), 0.0)),
                white,
            )),
            15.0,
        )),
        &Vec3::new(265.0, 0.0, 295.0),
    ));
    // objects.add(box1.clone());
    objects.add(Arc::new(ConstantMedium {
        boundary: box1,
        neg_inv_density: -1.0 / 0.01,
        phase_function: Arc::new(Isotropic {
            albedo: Arc::new(SolidColor::new(Vec3::new(0.0, 0.0, 0.0))),
        }),
    }));
    /*objects.add(Arc::new(Translate::new(
        &Arc::new(RotateY::new(
            &Arc::new(Box::new(
                &Vec3::new(0.0, 0.0, 0.0),
                &Vec3::new(165.0, 165.0, 165.0),
                white,
            )),
            -18.0,
        )),
        &Vec3::new(130.0, 0.0, 65.0),
    )));*/
    objects.add(Arc::new(Sphere {
        center: Vec3::new(190.0, 90.0, 190.0),
        radius: 90.0,
        mat_ptr: Arc::new(Dielectric::new(1.5)),
    }));
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
