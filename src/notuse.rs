fn firework() {
    let N: i32 = 11;
    let M: i32 = 1023;

    let mut img: RgbImage = ImageBuffer::new(1024, 1024);
    let bar = ProgressBar::new((N * M) as u64);

    let ori = Vec3::new(512.0, 512.0, 300.0);
    let r: f64 = 460.0;

    for i in 1..M {
        let t: f64 = (1 as f64) / (M as f64) * (i as f64);
        for n1 in 1..11 {
            let theta: f64 = (2 as f64) * 3.14 / (11 as f64) * ((n1 as f64) + 0.25);
            for n2 in 0..20 {
                let phi: f64 = -(2.0 * 3.14 / (20 as f64) * (n2 as f64));
                let dir = Vec3::new(
                    r * theta.cos() * phi.cos(),
                    r * theta.sin() * phi.cos(),
                    r * phi.sin(),
                );
                let fwk = Firework::new(ori, dir);
                let pos: Vec3 = fwk.at(t);
                let pixel = img.get_pixel_mut(pos.x as u32, pos.z as u32);
                *pixel = image::Rgb([(i / 4) as u8, (i / 4) as u8, (i / 4) as u8]);
            }
            bar.inc(1);
        }
    }

    img.save("output/test.png").unwrap();
    bar.finish();
}

fn rgb() {
    let x = Vec3::new(1.0, 1.0, 1.0);
    println!("{:?}", x);

    let mut img: RgbImage = ImageBuffer::new(256, 256);
    let bar = ProgressBar::new(256);

    for x in 0..256 {
        for y in 0..256 {
            let pixel = img.get_pixel_mut(x, y);
            *pixel = image::Rgb([x as u8, y as u8, 64]);
        }
        bar.inc(1);
    }

    img.save("output/test.png").unwrap();
    bar.finish();
}