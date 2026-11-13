//! Uniform asymmetric quantization grids, bit-packing and the quantized-layer container.
//!
//! The grid is the classic affine asymmetric one: `w_hat = scale * (code - zero)`, `code` in `[0, 2^bits - 1]`.
//! Everything above the grid (the engine's scale refit, the coordinate descent, the skeleton bit promotion)
//! operates on this representation, so it is kept deliberately small and free of any rounding surprises:
//!
//! * scales are rounded to **f16** before use ([`round_f16`]), so the value the optimizer scores is
//!   bit-identical to the one written to disk. Without that, a refit would "improve" a loss computed at f32
//!   precision and then store a different grid, invalidating the monotonicity guarantee.
//! * zeros are stored as integers (`0..=maxq`) even though they are carried as f32 internally; the packed file
//!   writes them as u8, which is why [`QLayer::bits_total`] charges 16 + 8 = 24 metadata bits per (row, group).
//! * bit widths may differ **per group** ([`pack_row`]/[`unpack_row`] take a `gbits` table), which is what lets
//!   the protected skeleton columns keep more bits without changing the container format.

use half::f16;

/// Smallest normal f16: scales are stored as f16, so they are floored here to stay invertible.
/// A subnormal (or zero) scale would make `w_hat` collapse to 0 and division by it blow up, so the engine
/// treats "scale too small" as a rejected improvement rather than allowing it.
pub const F16_MIN_NORMAL: f32 = 6.103_515_6e-5;

/// Round `x` to the nearest value representable as f16 and return it widened back to f32.
#[inline]
pub fn round_f16(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

/// Group parameters `(scale, zero)` for the values `vals` at `bits` precision.
///
/// Default (min/max): the grid spans exactly `[min, max]`, which is optimal if you only care about *range* and
/// costs no search. The scale is rounded to f16 first and then floored at [`F16_MIN_NORMAL`], so an all-zero
/// group (see the dead-column handling in the engine) still yields a legal, invertible grid; `zero` is snapped
/// to an integer code index because that is what gets packed.
///
/// With `mse` the range is clipped symmetrically by a ratio `p` in `1.0 .. 0.54` (24 steps of 2%) and the L2
/// rounding error over the group is evaluated for each candidate. Clipping trades a few large outliers pinned to
/// the grid ends against smaller error on everything else; at 2-3 bit that trade is worth several percent of
/// loss, which is exactly why `--mse-grid` exists as an option rather than being always on — at 4 bit and above
/// the outliers are rarely clipped by enough to pay for themselves.
pub fn find_params(vals: &[f32], bits: u32, mse: bool) -> (f32, f32) {
    let maxq = ((1u32 << bits) - 1) as f32;
    let mut xmin = 0f32;
    let mut xmax = 0f32;
    for &v in vals {
        xmin = xmin.min(v);
        xmax = xmax.max(v);
    }
    if xmin == 0.0 && xmax == 0.0 {
        // Degenerate all-zero group: any grid would do, but a zero-width one is not invertible. Pick a legal
        // symmetric range so the stored scale/zero pair stays meaningful for an inference kernel.
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
///
/// Returning both is deliberate: the engine needs `w_hat` for its error bookkeeping and the code for storage, and
/// computing the second from the first (or vice versa) would risk a rounding mismatch between what is scored and
/// what is written. The clamp means an input outside the grid saturates instead of wrapping — the case the MSE
/// clip search deliberately creates for outliers.
#[inline(always)]
pub fn quant_code(x: f32, scale: f32, zero: f32, maxq: f32) -> (u8, f32) {
    let q = ((x / scale).round() + zero).clamp(0.0, maxq);
    (q as u8, scale * (q - zero))
}

/// Total number of code bits of one row (sum over columns of their group's bit width).
pub fn row_bits(n: usize, g: usize, gbits: &[u8]) -> usize {
    (0..n).map(|j| gbits[j / g] as usize).sum()
}

pub fn row_bytes(n: usize, g: usize, gbits: &[u8]) -> usize {
    (row_bits(n, g, gbits) + 7) / 8
}

/// Bit-stream packing (LSB first) with per-group bit widths.
///
/// Codes are appended low-bit-first into a 64-bit accumulator and flushed byte by byte, so the format is simply
/// "the codes in order" and an inference kernel can extract column `j` with one shift/mask chain — no alignment
/// padding, which is what makes non-power-of-two widths (3 bit) actually save space. The trailing partial byte is
/// zero-filled. Callers must pass a buffer of exactly [`row_bytes`] length; the widths in `gbits` are at most 8,
/// so one code never straddles more than two bytes and the accumulator can never overflow.
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

/// Exact inverse of [`pack_row`]: refill the accumulator 8 bits at a time and peel `gbits[j/g]` bits per code.
/// Kept as a separate function (rather than derived from the packer) so the round-trip is an *independent* check
/// of the format — see the `cdclq dequant` audit, whose acceptance criterion is `max |dequant - roundtrip| = 0`.
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
///
/// This is the in-memory form of one layer, deliberately kept unpacked and in f32: it is what the engine writes and
/// what the packer reads. The permutation is stored rather than applied, because activation ordering has to happen
/// at inference time (the caller permutes its activations once) instead of being baked into the weights — baking it
/// in would silently corrupt any kernel that does not know about it.
#[derive(Clone, Debug)]
pub struct QLayer {
    pub d_out: usize,
    pub d_in: usize,
    pub group: usize,
    /// `perm[j]` = original column index sitting at permuted position `j` (act-order permutation).
    pub perm: Vec<u32>,
    /// Per-group code width; the leading groups are the promoted skeleton when act-order + skeleton are enabled.
    pub gbits: Vec<u8>,
    /// `d_out x d_in` unpacked codes in permuted order (this crate's working form, not the packed file).
    pub codes: Vec<u8>,
    /// `d_out x ngroups`, already rounded to f16 precision by the grid search and the scale refit.
    pub scales: Vec<f32>,
    /// `d_out x ngroups`, integer-valued in `0..=maxq` (written as u8 on disk).
    pub zeros: Vec<f32>,
}

impl QLayer {
    #[inline]
    pub fn ngroups(&self) -> usize {
        self.gbits.len()
    }

    /// Dequantized weights in the ORIGINAL column order (`d_out x d_in`).
    ///
    /// The scatter `row[perm[j]] = ...` is the only place the permutation is undone, which makes this function the
    /// reference definition of "what a layer means": everything else in the crate works in permuted space. It is
    /// also what `cdclq dequant` writes to `dequant/`, so an evaluation harness that knows nothing about the packed
    /// format can still run the model.
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

    /// Bit-packed codes, `d_out x row_bytes`; the second value is that row stride, which the writer must record so
    /// a reader can locate rows without recomputing it. Note this packs the *permuted* code order: reading column
    /// `j` of the bit-stream gives the weight for original input channel `perm[j]`.
    pub fn pack(&self) -> (Vec<u8>, usize) {
        let rb = row_bytes(self.d_in, self.group, &self.gbits);
        let mut out = vec![0u8; self.d_out * rb];
        for r in 0..self.d_out {
            pack_row(&self.codes[r * self.d_in..(r + 1) * self.d_in], self.group, &self.gbits, &mut out[r * rb..(r + 1) * rb]);
        }
        (out, rb)
    }

    /// Rebuild a layer from the pieces as stored on disk (the `packed/` files). This is the reader half of the
    /// format: it must agree with [`QLayer::pack`] byte for byte, and it is what makes the `dequant` audit
    /// meaningful — the audit compares `dequantize(from_packed(...))` against an independently rounded reference.
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

    /// Average bits per weight spent on codes only (excludes metadata). With a promoted skeleton this is *not* the
    /// nominal `--bits`: it is the weighted average over groups, which is why the README reports measured bpw.
    pub fn bits_codes(&self) -> f64 {
        row_bits(self.d_in, self.group, &self.gbits) as f64 / self.d_in as f64
    }

    /// Bits per weight including per-(row, group) metadata: f16 scale + 8-bit zero point = 24 bits spread over the
    /// group. This is the honest number to compare against fp16: at `group = 128` it adds 0.1875 bit, at `group = 32`
    /// it adds 0.75 bit — which is why a smaller group buys accuracy but also size. Rows are assumed to be full
    /// groups; with a ragged last group the real metadata cost is marginally lower, and the packed file sizes in the
    /// README are the ground truth anyway.
    pub fn bits_total(&self) -> f64 {
        self.bits_codes() + 24.0 / self.group as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// Packing must be lossless with *mixed* widths in one row and a ragged final group: those are the two cases a
    /// fixed-width packer gets wrong (skeleton promotion changes width mid-row, and `d_in` is not always a multiple
    /// of the group).
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

    /// Two properties the rest of the crate depends on: the stored scale is exactly f16-representable (so the loss
    /// the engine scored is the loss the format delivers), and an asymmetric grid represents the value 0 exactly —
    /// important because most weights are near zero, and a grid that cannot hit 0 would add noise everywhere.
    #[test]
    fn grid_reproduces_zero_and_is_f16_exact() {
        let v: Vec<f32> = (0..64).map(|i| (i as f32 - 20.0) * 0.013).collect();
        let (s, z) = find_params(&v, 3, false);
        assert_eq!(s, round_f16(s));
        let (_, d0) = quant_code(0.0, s, z, 7.0);
        assert!(d0.abs() < 1e-6);
    }
}
