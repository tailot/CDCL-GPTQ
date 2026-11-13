//! Uniform asymmetric quantization grids, bit-packing and the quantized-layer container.

use half::f16;

/// Smallest normal f16: scales are stored as f16, so they are floored here to stay invertible.
pub const F16_MIN_NORMAL: f32 = 6.103_515_6e-5;

#[inline]
pub fn round_f16(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

/// Group parameters `(scale, zero)` for the values `vals` at `bits` precision.
/// The scale is rounded to f16 precision so that the packed format reproduces it exactly.
/// With `mse` a clip-ratio search minimises the L2 rounding error (same grid for every method).
pub fn find_params(vals: &[f32], bits: u32, mse: bool) -> (f32, f32) {
    let maxq = ((1u32 << bits) - 1) as f32;
    let mut xmin = 0f32;
    let mut xmax = 0f32;
    for &v in vals {
        xmin = xmin.min(v);
        xmax = xmax.max(v);
    }
    if xmin == 0.0 && xmax == 0.0 {
        xmin = -1.0;
        xmax = 1.0;
    }
    let mk = |lo: f32, hi: f32| -> (f32, f32) {
        let mut scale = round_f16((hi - lo) / maxq);
        if scale < F16_MIN_NORMAL {
            scale = F16_MIN_NORMAL;
        }
        let zero = (-lo / scale).round().clamp(0.0, maxq);
        (scale, zero)
    };
    if !mse {
        return mk(xmin, xmax);
    }
    const STEPS: usize = 24;
    let mut best = mk(xmin, xmax);
    let mut best_err = f32::INFINITY;
    for i in 0..STEPS {
        let p = 1.0 - 0.02 * i as f32;
        let (s, z) = mk(p * xmin, p * xmax);
        let mut err = 0f32;
        for &v in vals {
            let (_, d) = quant_code(v, s, z, maxq);
            let e = d - v;
            err += e * e;
        }
        if err < best_err {
            best_err = err;
            best = (s, z);
        }
    }
    best
}

/// Round-to-nearest on the grid. Returns `(code, dequantized value)`.
#[inline(always)]
pub fn quant_code(x: f32, scale: f32, zero: f32, maxq: f32) -> (u8, f32) {
    let q = ((x / scale).round() + zero).clamp(0.0, maxq);
    (q as u8, scale * (q - zero))
}

/// Total number of code bits of one row.
pub fn row_bits(n: usize, g: usize, gbits: &[u8]) -> usize {
    (0..n).map(|j| gbits[j / g] as usize).sum()
}

pub fn row_bytes(n: usize, g: usize, gbits: &[u8]) -> usize {
    (row_bits(n, g, gbits) + 7) / 8
}

/// Bit-stream packing (LSB first) with per-group bit widths.
pub fn pack_row(codes: &[u8], g: usize, gbits: &[u8], out: &mut [u8]) {
    let mut acc: u64 = 0;
    let mut nacc = 0u32;
    let mut o = 0;
    for (j, &c) in codes.iter().enumerate() {
        let b = gbits[j / g] as u32;
        acc |= (c as u64) << nacc;
        nacc += b;
        while nacc >= 8 {
            out[o] = (acc & 0xff) as u8;
            o += 1;
            acc >>= 8;
            nacc -= 8;
        }
    }
    if nacc > 0 {
        out[o] = (acc & 0xff) as u8;
    }
}

pub fn unpack_row(data: &[u8], g: usize, gbits: &[u8], codes: &mut [u8]) {
    let mut acc: u64 = 0;
    let mut nacc = 0u32;
    let mut o = 0;
    for (j, c) in codes.iter_mut().enumerate() {
        let b = gbits[j / g] as u32;
        while nacc < b {
            acc |= (data[o] as u64) << nacc;
            o += 1;
            nacc += 8;
        }
        *c = (acc & ((1u64 << b) - 1)) as u8;
        acc >>= b;
        nacc -= b;
    }
}

/// A quantized linear layer. Codes/scales/zeros live in *permuted* column order:
/// permuted position `j` holds original column `perm[j]`; group `gi` = positions `[gi*g, (gi+1)*g)`.
#[derive(Clone, Debug)]
pub struct QLayer {
    pub d_out: usize,
    pub d_in: usize,
    pub group: usize,
    pub perm: Vec<u32>,
    pub gbits: Vec<u8>,
    pub codes: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
}

impl QLayer {
    pub fn ngroups(&self) -> usize {
        self.gbits.len()
    }

    /// Dequantized weights in the ORIGINAL column order (`d_out x d_in`).
    pub fn dequantize(&self) -> Vec<f32> {
        let (n, ng, g) = (self.d_in, self.ngroups(), self.group);
        let mut out = vec![0f32; self.d_out * n];
        for r in 0..self.d_out {
            let row = &mut out[r * n..(r + 1) * n];
            for j in 0..n {
                let gi = j / g;
                let s = self.scales[r * ng + gi];
                let z = self.zeros[r * ng + gi];
                row[self.perm[j] as usize] = s * (self.codes[r * n + j] as f32 - z);
            }
        }
        out
    }

    /// Bit-packed codes, `d_out x row_bytes`.
    pub fn pack(&self) -> (Vec<u8>, usize) {
        let rb = row_bytes(self.d_in, self.group, &self.gbits);
        let mut out = vec![0u8; self.d_out * rb];
        for r in 0..self.d_out {
            pack_row(&self.codes[r * self.d_in..(r + 1) * self.d_in], self.group, &self.gbits, &mut out[r * rb..(r + 1) * rb]);
        }
        (out, rb)
    }

    pub fn from_packed(
        d_out: usize,
        d_in: usize,
        group: usize,
        perm: Vec<u32>,
        gbits: Vec<u8>,
        packed: &[u8],
        scales: Vec<f32>,
        zeros: Vec<f32>,
    ) -> Self {
        let rb = row_bytes(d_in, group, &gbits);
        let mut codes = vec![0u8; d_out * d_in];
        for r in 0..d_out {
            unpack_row(&packed[r * rb..(r + 1) * rb], group, &gbits, &mut codes[r * d_in..(r + 1) * d_in]);
        }
        QLayer { d_out, d_in, group, perm, gbits, codes, scales, zeros }
    }

    /// Average bits per weight spent on codes only.
    pub fn bits_codes(&self) -> f64 {
        row_bits(self.d_in, self.group, &self.gbits) as f64 / self.d_in as f64
    }

    /// Bits per weight including per-(row, group) metadata: f16 scale + 8-bit zero point.
    pub fn bits_total(&self) -> f64 {
        self.bits_codes() + 24.0 / self.group as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn pack_roundtrip_mixed_bits() {
        let mut r = Rng::new(9);
        let g = 32;
        let n = 32 * 5 + 7; // ragged last group
        let gbits: Vec<u8> = vec![8, 3, 3, 4, 2, 3];
        let codes: Vec<u8> = (0..n).map(|j| (r.next_u64() % (1u64 << gbits[j / g])) as u8).collect();
        let rb = row_bytes(n, g, &gbits);
        let mut buf = vec![0u8; rb];
        pack_row(&codes, g, &gbits, &mut buf);
        let mut back = vec![0u8; n];
        unpack_row(&buf, g, &gbits, &mut back);
        assert_eq!(codes, back);
    }

    #[test]
    fn grid_reproduces_zero_and_is_f16_exact() {
        let v: Vec<f32> = (0..64).map(|i| (i as f32 - 20.0) * 0.013).collect();
        let (s, z) = find_params(&v, 3, false);
        assert_eq!(s, round_f16(s));
        let (_, d0) = quant_code(0.0, s, z, 7.0);
        assert!(d0.abs() < 1e-6);
    }
}
