use crate::vec3::Vec3;
use rand::random;

pub fn random_num() -> f64 {
    random::<f64>()
}
/*fn random_positive_unit() -> Vec3 {
    let x = random::<f64>().abs();
    let y = random::<f64>().abs();
    let z = random::<f64>().abs();
    let tmp = Vec3::new(x, y, z);
    if tmp.length() > 1.0 {
        return random_positive_unit();
    }
    tmp.unit()
}*/
pub fn random_unit() -> Vec3 {
    let x = random::<f64>() * 2.0 - 1.0;
    let y = random::<f64>() * 2.0 - 1.0;
    let z = random::<f64>() * 2.0 - 1.0;
    let tmp = Vec3::new(x as f64, y as f64, z as f64);
    if tmp.length() == 0.0 || tmp.length() > 1.0 {
        return random_unit();
    }
    tmp.unit()
}
/*fn random_vec(nor: &Vec3) -> Vec3 {
    let x = random::<f64>() * 2.0 - 1.0;
    let y = random::<f64>() * 2.0 - 1.0;
    let z = random::<f64>() * 2.0 - 1.0;
    let tmp = Vec3::new(x as f64, y as f64, z as f64);
    if tmp.length() == 0.0 || tmp.length() > 1.0{
        return random_vec(nor);
    }
    if tmp * *nor > 0.0 {
        return tmp.unit();
    }
    -tmp.unit()
}*/
pub fn random_in_unit_disk() -> Vec3 {
    let p = Vec3::new(random::<f64>(), random::<f64>(), 0.0);
    if p.squared_length() >= 1.0 {
        return random_in_unit_disk();
    }
    p
}
