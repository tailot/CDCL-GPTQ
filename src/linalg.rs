//! Dense linear algebra kernels (row-major), built on `matrixmultiply` + `rayon`.
//!
//! This module exists so the quantizer has **no BLAS link dependency** (a portable build on Windows/MSVC,
//! Linux and macOS with no external library) while still getting close to vendor performance: the inner
//! kernels come from `matrixmultiply`, which generates blocked SIMD micro-kernels (SSE/AVX/FMA selected at
//! runtime), and parallelism is added by splitting the output matrix across rayon tasks.
//!
//! * [`sgemm`] / [`dgemm`]      : single-thread strided GEMM. Deliberately single-threaded so they are safe
//!   to call from inside a rayon task (the engine calls them per row chunk) — no nested thread explosion.
//! * [`pgemm_f32`] / [`pgemm_f64`]: multi-thread GEMM, splitting `C` along its larger dimension so the
//!   chunks are wide enough to be efficient and disjoint (no synchronisation needed).
//! * [`cholesky_lower`]         : blocked right-looking Cholesky (f64)
//! * [`tri_inverse_lower`]      : blocked inverse of a lower-triangular matrix (f64)
//! * [`upper_chol_of_inverse`]  : the factor `U` (upper) with `H^-1 = U^T U` used by GPTQ
//! * [`gram_accumulate`]        : `H += X^T Y` over token chunks (Hessian capture/validation)
//!
//! All matrices are row-major and every kernel takes explicit row/column strides, because the engine works
//! on sub-blocks of larger buffers (`U[j*n + j..]`, `H[:, c0..c1]`, transposed views such as `U^T`) without
//! ever copying them. A transpose is therefore free: it is just `lda = 1` instead of `lda = n`.
//!
//! # Safety model
//!
//! The raw pointers handed to the kernels must describe valid, non-aliasing strided views (the `C` output
//! must not overlap `A` or `B`). The wrappers themselves are `unsafe fn` for exactly that reason; callers
//! in this crate derive them from slice borrows, which keeps disjointness checked by the borrow checker.

use anyhow::{bail, Result};
use rayon::prelude::*;

/// Raw mutable pointer wrapper that is `Send`/`Sync`, so a rayon closure can capture it. Rust pointers are
/// neither by default, and the split-GEMM below hands disjoint regions of one buffer to different tasks —
/// the disjointness argument lives here, not in the type system.
#[derive(Clone, Copy)]
pub struct SPtr<T>(pub *mut T);
unsafe impl<T> Send for SPtr<T> {}
unsafe impl<T> Sync for SPtr<T> {}
impl<T> SPtr<T> {
    #[inline]
    pub fn get(self) -> *mut T {
        self.0
    }
}

/// Read-only counterpart of [`SPtr`] (see the note there).
#[derive(Clone, Copy)]
pub struct CPtr<T>(pub *const T);
unsafe impl<T> Send for CPtr<T> {}
unsafe impl<T> Sync for CPtr<T> {}
impl<T> CPtr<T> {
    #[inline]
    pub fn get(self) -> *const T {
        self.0
    }
}

macro_rules! make_gemm {
    ($name:ident, $t:ty, $mm:path) => {
        /// `C = alpha * A * B + beta * C` with arbitrary strides (single thread).
        ///
        /// # Safety
        /// All pointers must be valid for the described strided views and `C` must not alias `A`/`B`.
        #[inline]
        pub unsafe fn $name(
            m: usize,
            k: usize,
            n: usize,
            alpha: $t,
            a: *const $t,
            rsa: isize,
            csa: isize,
            b: *const $t,
            rsb: isize,
            csb: isize,
            beta: $t,
            c: *mut $t,
            rsc: isize,
            csc: isize,
        ) {
            if m == 0 || n == 0 {
                return;
            }
            if k == 0 {
                if beta != 1.0 {
                    for i in 0..m {
                        for j in 0..n {
                            let p = c.offset(i as isize * rsc + j as isize * csc);
                            *p = if beta == 0.0 { 0.0 } else { *p * beta };
                        }
                    }
                }
                return;
            }
            $mm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc);
        }
    };
}

make_gemm!(sgemm, f32, matrixmultiply::sgemm);
make_gemm!(dgemm, f64, matrixmultiply::dgemm);

macro_rules! make_pgemm {
    ($name:ident, $t:ty, $base:ident) => {
        /// Multi-threaded `C = alpha * A * B + beta * C` (rayon). Splits along the larger of m/n.
        ///
        /// Small problems run single-threaded: below ~2M multiply-accumulates the rayon dispatch cost
        /// dominates, and many of the engine's GEMMs are exactly that small (per-block lazy updates). The
        /// split axis is chosen so each task writes a contiguous, disjoint slab of `C`, which is why no
        /// locking or reduction is needed.
        ///
        /// # Safety
        /// Same contract as the single-thread kernel.
        pub unsafe fn $name(
            m: usize,
            k: usize,
            n: usize,
            alpha: $t,
            a: *const $t,
            rsa: isize,
            csa: isize,
            b: *const $t,
            rsb: isize,
            csb: isize,
            beta: $t,
            c: *mut $t,
            rsc: isize,
            csc: isize,
        ) {
            let nt = rayon::current_num_threads();
            let work = (m as u128) * (n as u128) * (k as u128);
            if nt <= 1 || work < (1u128 << 21) {
                return $base(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc);
            }
            let (pa, pb, pc) = (CPtr(a), CPtr(b), SPtr(c));
            if m >= n {
                let chunk = ((m + nt * 2 - 1) / (nt * 2)).max(16);
                let starts: Vec<usize> = (0..m).step_by(chunk).collect();
                starts.into_par_iter().for_each(move |i0| {
                    let mc = chunk.min(m - i0);
                    $base(
                        mc,
                        k,
                        n,
                        alpha,
                        pa.get().offset(i0 as isize * rsa),
                        rsa,
                        csa,
                        pb.get(),
                        rsb,
                        csb,
                        beta,
                        pc.get().offset(i0 as isize * rsc),
                        rsc,
                        csc,
                    );
                });
            } else {
                let chunk = ((n + nt * 2 - 1) / (nt * 2)).max(16);
                let starts: Vec<usize> = (0..n).step_by(chunk).collect();
                starts.into_par_iter().for_each(move |j0| {
                    let nc = chunk.min(n - j0);
                    $base(
                        m,
                        k,
                        nc,
                        alpha,
                        pa.get(),
                        rsa,
                        csa,
                        pb.get().offset(j0 as isize * csb),
                        rsb,
                        csb,
                        beta,
                        pc.get().offset(j0 as isize * csc),
                        rsc,
                        csc,
                    );
                });
            }
        }
    };
}

make_pgemm!(pgemm_f32, f32, sgemm);
make_pgemm!(pgemm_f64, f64, dgemm);

/// In-place blocked Cholesky of a symmetric positive-definite `n x n` row-major matrix.
/// On success the lower triangle holds `L` (`A = L L^T`); the upper triangle is garbage.
///
/// Right-looking (LAPACK "right" variant): factor a diagonal block, solve the panel below it, then push the
/// update into the trailing submatrix as one GEMM per row-chunk. Blocking is what makes this competitive —
/// the unblocked textbook loop is memory-bound, while steps 2-3 run inside `dgemm`. The panel copy in step 3
/// exists to break aliasing between the operands and the destination of that GEMM.
///
/// Only the *lower* triangle of the trailing update is touched (`cols = i0 + mrows`), because the upper half
/// is never read back: skipping it halves the arithmetic.
///
/// A non-positive or non-finite pivot is a hard error rather than a silent fallback: with calibration
/// Hessians it means the matrix is singular (too few tokens for `d_in`, or zero damping) and any "repaired"
/// factor would quietly produce garbage quantization. The message tells the user what to change.
pub fn cholesky_lower(a: &mut [f64], n: usize) -> Result<()> {
    assert_eq!(a.len(), n * n);
    const NB: usize = 96;
    const TB: usize = 64;
    let mut k = 0;
    while k < n {
        let kb = NB.min(n - k);
        let kend = k + kb;
        // 1. unblocked factorisation of the diagonal block
        for j in k..kend {
            let mut d = a[j * n + j];
            for t in k..j {
                let v = a[j * n + t];
                d -= v * v;
            }
            if !(d > 0.0) || !d.is_finite() {
                bail!(
                    "matrix is not positive definite (pivot {j} = {d:e}); increase --damp or check the Hessian"
                );
            }
            let d = d.sqrt();
            a[j * n + j] = d;
            for i in (j + 1)..kend {
                let mut s = a[i * n + j];
                for t in k..j {
                    s -= a[i * n + t] * a[j * n + t];
                }
                a[i * n + j] = s / d;
            }
        }
        if kend < n {
            let (top, bottom) = a.split_at_mut(kend * n);
            // 2. panel solve: X L_kk^T = A_panel
            bottom.par_chunks_mut(n).for_each(|row| {
                for j in k..kend {
                    let mut s = row[j];
                    for t in k..j {
                        s -= row[t] * top[j * n + t];
                    }
                    row[j] = s / top[j * n + j];
                }
            });
            // 3. trailing update with a copy of the panel (avoids aliasing)
            let rows_below = n - kend;
            let mut panel = vec![0f64; rows_below * kb];
            for i in 0..rows_below {
                panel[i * kb..(i + 1) * kb].copy_from_slice(&bottom[i * n + k..i * n + kend]);
            }
            let panel_ref = &panel;
            bottom.par_chunks_mut(TB * n).enumerate().for_each(|(ci, chunk)| {
                let i0 = ci * TB;
                let mrows = chunk.len() / n;
                let cols = i0 + mrows; // lower triangle only: columns 0..i1 of the trailing matrix
                unsafe {
                    dgemm(
                        mrows,
                        kb,
                        cols,
                        -1.0,
                        panel_ref.as_ptr().add(i0 * kb),
                        kb as isize,
                        1,
                        panel_ref.as_ptr(),
                        1,
                        kb as isize,
                        1.0,
                        chunk.as_mut_ptr().add(kend),
                        n as isize,
                        1,
                    );
                }
            });
        }
        k = kend;
    }
    Ok(())
}

/// Inverse of a lower-triangular matrix (only the lower triangle of `l` is read).
///
/// Blocked column-panel formulation: invert a diagonal block explicitly, then fill the panel below it with
/// two GEMMs (`Y21 = -L22^-1 * L21 * Y11`). Triangular inversion is O(n^3/3) and dominated by those GEMMs, so
/// blocking matters just as much as in Cholesky. Never reads the strictly upper triangle, which is why the
/// caller may leave it full of Cholesky garbage.
pub fn tri_inverse_lower(l: &[f64], n: usize) -> Vec<f64> {
    const NB: usize = 96;
    let mut y = vec![0f64; n * n];
    let mut s = 0;
    while s < n {
        let e = (s + NB).min(n);
        let w = e - s;
        let mut yii = vec![0f64; w * w];
        for c in 0..w {
            yii[c * w + c] = 1.0 / l[(s + c) * n + (s + c)];
            for i in (c + 1)..w {
                let mut acc = 0.0;
                for t in c..i {
                    acc += l[(s + i) * n + (s + t)] * yii[t * w + c];
                }
                yii[i * w + c] = -acc / l[(s + i) * n + (s + i)];
            }
        }
        for i in 0..w {
            for j in 0..=i {
                y[(s + i) * n + (s + j)] = yii[i * w + j];
            }
        }
        if s > 0 {
            let mut t = vec![0f64; w * s];
            unsafe {
                pgemm_f64(
                    w,
                    s,
                    s,
                    1.0,
                    l.as_ptr().add(s * n),
                    n as isize,
                    1,
                    y.as_ptr(),
                    n as isize,
                    1,
                    0.0,
                    t.as_mut_ptr(),
                    s as isize,
                    1,
                );
                pgemm_f64(
                    w,
                    w,
                    s,
                    -1.0,
                    yii.as_ptr(),
                    w as isize,
                    1,
                    t.as_ptr(),
                    s as isize,
                    1,
                    0.0,
                    y.as_mut_ptr().add(s * n),
                    n as isize,
                    1,
                );
            }
        }
        s = e;
    }
    y
}

/// Returns the upper-triangular `U` (row-major, f32) with `H^-1 = U^T U` (the factor GPTQ uses).
///
/// This is the single most important kernel in the crate: GPTQ's column-by-column compensation *is* the
/// sequence of Schur-complement updates, and row `j` of `U` is exactly the coefficient vector that update
/// needs. Forming `H^-1` and then Cholesky-ing it would cost a full inverse (and lose accuracy); instead use
/// the flip trick: with `J` the exchange matrix, `J H J = L' L'^T` gives `H = R R^T` for the upper
/// `R = J L' J`, hence `U = R^-1 = J L'^-1 J`. One Cholesky + one triangular inverse.
///
/// `h.reverse()` on the flat row-major buffer is the flip: reversing all elements maps element `(i, j)` to
/// `(n-1-i, n-1-j)`, i.e. conjugates by `J` on both axes — no transpose copy needed. The final scatter undoes
/// the flip while writing only the upper triangle (`for j in i..n`) and downcasting f64 -> f32; keeping f64 up
/// to this point is deliberate, since `U_jj` appears as a divisor in the engine's error propagation and its
/// accuracy directly bounds the reported loss.
pub fn upper_chol_of_inverse(mut h: Vec<f64>, n: usize) -> Result<Vec<f32>> {
    assert_eq!(h.len(), n * n);
    h.reverse(); // flips both axes of the row-major matrix
    cholesky_lower(&mut h, n)?;
    let y = tri_inverse_lower(&h, n);
    drop(h);
    let mut u = vec![0f32; n * n];
    u.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in i..n {
            row[j] = y[(n - 1 - i) * n + (n - 1 - j)] as f32;
        }
    });
    Ok(u)
}

/// `H += X^T Y` where `x`, `y` are `t x n` row-major token matrices (Gram / cross-covariance).
///
/// The Hessian is accumulated as a sum over token chunks, each evaluated by one GEMM into an f32 scratch and
/// then widened into the f64 accumulator. Chunking at 2048 tokens bounds the scratch to `n x n` regardless of
/// how many calibration tokens the user feeds in, and keeping the *running* sum in f64 avoids the catastrophic
/// cancellation you would get from accumulating millions of O(1) terms in f32 — that noise is what would make
/// a Hessian look non-positive-definite.
///
/// The transposed read of `X` (`lda = 1`, `ldb = n`) is exactly why the kernels take explicit strides: `X^T Y`
/// needs no copy of `X`. With `x == y` this computes the symmetric Hessian `X X^T` used by the quantizer; with
/// different `x`/`y` it computes the asymmetric cross-covariance `M = X X~^T`.
pub fn gram_accumulate(x: &[f32], y: &[f32], t: usize, n: usize, h: &mut [f64]) {
    assert_eq!(x.len(), t * n);
    assert_eq!(y.len(), t * n);
    assert_eq!(h.len(), n * n);
    const TC: usize = 2048;
    let mut tmp = vec![0f32; n * n];
    let mut t0 = 0;
    while t0 < t {
        let tc = TC.min(t - t0);
        unsafe {
            pgemm_f32(
                n,
                tc,
                n,
                1.0,
                x.as_ptr().add(t0 * n),
                1,
                n as isize,
                y.as_ptr().add(t0 * n),
                n as isize,
                1,
                0.0,
                tmp.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        h.par_iter_mut().zip(tmp.par_iter()).for_each(|(a, b)| *a += *b as f64);
        t0 += tc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// A random well-conditioned SPD matrix (`M M^T + I`) used by every kernel test here.
    fn rand_spd(n: usize, seed: u64) -> Vec<f64> {
        let mut r = Rng::new(seed);
        let m: Vec<f64> = (0..n * (n + 8)).map(|_| r.normal() as f64).collect();
        let mut h = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..(n + 8) {
                    s += m[i * (n + 8) + k] * m[j * (n + 8) + k];
                }
                h[i * n + j] = s;
            }
        }
        for i in 0..n {
            h[i * n + i] += 1.0;
        }
        h
    }

    /// Check `L L^T = A` element by element. `n = 150` deliberately crosses the `NB = 96` block boundary so
    /// the panel solve and the blocked trailing GEMM are exercised, not just the unblocked first block.
    #[test]
    fn cholesky_matches_definition() {
        let n = 150; // crosses the NB=96 block boundary
        let h = rand_spd(n, 1);
        let mut l = h.clone();
        cholesky_lower(&mut l, n).unwrap();
        let mut maxerr = 0f64;
        for i in 0..n {
            for j in 0..=i {
                let mut s = 0.0;
                for k in 0..=j {
                    s += l[i * n + k] * l[j * n + k];
                }
                maxerr = maxerr.max((s - h[i * n + j]).abs() / h[i * n + j].abs().max(1.0));
            }
        }
        assert!(maxerr < 1e-10, "chol err {maxerr}");
    }

    /// The defining property of the GPTQ factor: `U^T U H = I`. Tolerance is loose (5e-4) because `U` is
    /// stored as f32 while `H` stays f64 — that rounding, not the algorithm, sets the floor. Also asserts the
    /// structural contract the engine relies on: strictly lower entries are exactly zero and every diagonal
    /// entry is positive (it is used as a divisor, so a non-positive one would flip compensation signs).
    #[test]
    fn upper_factor_of_inverse() {
        let n = 200;
        let h = rand_spd(n, 2);
        let u = upper_chol_of_inverse(h.clone(), n).unwrap();
        // check U^T U H = I
        let mut utu = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += u[k * n + i] as f64 * u[k * n + j] as f64;
                }
                utu[i * n + j] = s;
            }
        }
        let mut maxerr = 0f64;
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += utu[i * n + k] * h[k * n + j];
                }
                let t = if i == j { 1.0 } else { 0.0 };
                maxerr = maxerr.max((s - t).abs());
            }
        }
        assert!(maxerr < 5e-4, "U^T U H != I (err {maxerr})");
        for i in 0..n {
            for j in 0..i {
                assert_eq!(u[i * n + j], 0.0);
            }
            assert!(u[i * n + i] > 0.0);
        }
    }

    /// The parallel GEMM must agree with the single-thread one. Two shapes are tested so both split paths run:
    /// `m >= n` (row split) and `n > m` (column split). Run inside an explicit 4-thread pool because a test
    /// binary may otherwise be scheduled on a single thread, which would silently make the parallel path a
    /// pass-through and test nothing.
    #[test]
    fn parallel_gemm_matches_serial() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        pool.install(|| {
            let (m, k, n) = (97, 130, 111);
            let mut r = Rng::new(3);
            let a: Vec<f32> = (0..m * k).map(|_| r.normal()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| r.normal()).collect();
            let mut c1 = vec![0f32; m * n];
            let mut c2 = vec![0f32; m * n];
            unsafe {
                sgemm(m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), n as isize, 1, 0.0, c1.as_mut_ptr(), n as isize, 1);
                pgemm_f32(m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), n as isize, 1, 0.0, c2.as_mut_ptr(), n as isize, 1);
            }
            for i in 0..m * n {
                assert!((c1[i] - c2[i]).abs() < 1e-3);
            }
            // wide case (n > m) exercises the column split
            let (m, k, n) = (20, 300, 400);
            let a: Vec<f32> = (0..m * k).map(|_| r.normal()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| r.normal()).collect();
            let mut c1 = vec![0f32; m * n];
            let mut c2 = vec![0f32; m * n];
            unsafe {
                sgemm(m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), n as isize, 1, 0.0, c1.as_mut_ptr(), n as isize, 1);
                pgemm_f32(m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), n as isize, 1, 0.0, c2.as_mut_ptr(), n as isize, 1);
            }
            for i in 0..m * n {
                assert!((c1[i] - c2[i]).abs() < 1e-3);
            }
        });
    }
}
