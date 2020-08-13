use crate::randomtool::*;
use crate::vec3::*;
use rand::*;

pub const POINT_COUNT: usize = 256;

/*pub fn trilinear_interp(c: [[[f64; 2]; 2]; 2], u: f64, v: f64, w: f64) -> f64 {
    let mut accum = 0.0;
    for i in 0..1 {
        for j in 0..1 {
            for k in 0..1 {
                accum += (i as f64 * u + (1 - i) as f64 * (1.0 - u))
                    * (j as f64 * v + (1 - j) as f64 * (1.0 - v))
                    * (k as f64 * w + (1 - k) as f64 * (1.0 - w))
                    * c[i][j][k]
            }
        }
    }
    accum
}*/
#[derive(Clone)]
pub struct Perlin {
    pub ranvec: Vec<Vec3>,
    // pub ranfloat: Vec<f64>,
    pub perm_x: Vec<i32>,
    pub perm_y: Vec<i32>,
    pub perm_z: Vec<i32>,
}
pub fn permute(p: &mut Vec<i32>, n: i32) {
    for i in (0..n).rev() {
        let i = i as usize;
        let target = random::<usize>() % (i + 1);
        (*p).swap(i as usize, target as usize)
    }
}
impl Perlin {
    pub fn perlin_generate_perm() -> Vec<i32> {
        let mut p: Vec<i32> = vec![];
        for i in 0..POINT_COUNT {
            p.push(i as i32);
        }
        permute(&mut p, POINT_COUNT as i32);
        p
    }
    pub fn perlin_interp(c: [[[Vec3; 2]; 2]; 2], u: f64, v: f64, w: f64) -> f64 {
        let uu = u * u * (3.0 - 2.0 * u);
        let vv = v * v * (3.0 - 2.0 * v);
        let ww = w * w * (3.0 - 2.0 * w);
        let mut accum = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let weight_v = Vec3::new(u - i as f64, v - j as f64, w - k as f64);
                    accum += (i as f64 * uu + (1.0 - i as f64) * (1.0 - uu))
                        * (j as f64 * vv + (1.0 - j as f64) * (1.0 - vv))
                        * (k as f64 * ww + (1.0 - k as f64) * (1.0 - ww))
                        * (c[i][j][k] * weight_v)
                }
            }
        }
        accum
    }
    pub fn turb(&self, p: &Vec3, depth: i32) -> f64 {
        let mut accum = 0.0;
        let mut tem_p = *p;
        let mut weight = 1.0;
        for _i in 1..depth {
            accum += weight * self.noise(&tem_p);
            weight *= 0.5;
            tem_p *= 2.0;
        }
        accum.abs()
    }
}
impl Default for Perlin {
    fn default() -> Self {
        Self::new()
    }
}
impl Perlin {
    pub fn new() -> Self {
        // let mut ranfloat: Vec<f64> = vec![];
        let mut ranvec: Vec<Vec3> = vec![];
        for _i in 0..POINT_COUNT {
            ranvec.push(random_unit())
        }
        Self {
            ranvec,
            // ranfloat,
            perm_x: Perlin::perlin_generate_perm(),
            perm_y: Perlin::perlin_generate_perm(),
            perm_z: Perlin::perlin_generate_perm(),
        }
    }
    pub fn noise(&self, pos: &Vec3) -> f64 {
        let dim_u = pos.x - pos.x.floor();
        let dim_v = pos.y - pos.y.floor();
        let dim_w = pos.z - pos.z.floor();
        let dim_i = pos.x.floor() as i32;
        let dim_j = pos.y.floor() as i32;
        let dim_k = pos.z.floor() as i32;
        let mut tmp: [[[Vec3; 2]; 2]; 2] = [[[Vec3::zero(); 2]; 2]; 2];
        for di in 0..2 {
            for dj in 0..2 {
                for dk in 0..2 {
                    tmp[di][dj][dk] = self.ranvec[(self.perm_x[(dim_i + di as i32) as usize & 255]
                        ^ self.perm_y[(dim_j + dj as i32) as usize & 255]
                        ^ self.perm_z[(dim_k + dk as i32) as usize & 255])
                        as usize];
                }
            }
        }
        Perlin::perlin_interp(tmp, dim_u, dim_v, dim_w)
    }
}
