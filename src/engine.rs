//! CDCL-GPTQ engine: the GPTQ pass, the exact conflict signal and the monotone repair.
//!
//! # Objective
//!
//! Layer-wise proxy loss (Hessian metric, `H = X X^T` accumulated over calibration tokens,
//! `E = W - W_hat`, one row-vector `e_r` per output neuron):
//!
//! ```text
//!     L(W_hat) = sum_rows  e_r H e_r^T          (rows are independent)
//! ```
//!
//! This is the standard GPTQ objective. It is a quadratic form in the weight error, positive
//! semidefinite because `H` is a Gram matrix, and it decouples across rows of `W`: row `r` only ever
//! interacts with `H`, never with another row. That is why [`process_chunk`] can hand disjoint row
//! ranges to different rayon threads and why the per-layer statistics are plain sums.
//!
//! # GPTQ pass (exact decomposition)
//!
//! With `H^-1 = U^T U` (`U` upper, see [`linalg::upper_chol_of_inverse`]) the loss decomposes
//! *exactly* as
//!
//! ```text
//!     L = sum_j eps_j^2 ,      eps_j = (w_cur_j - q_j) / U_jj
//! ```
//!
//! where `w_cur_j` is the column of `W` at the moment column `j` is quantized (already compensated by
//! all previous columns) and `q_j` its dequantized value. Consequences that this module relies on:
//!
//! * processing columns left to right and compensating with row `j` of `U` *is* the sequence of exact
//!   Schur-complement updates of the remaining sub-problem — no full `H^-1` is ever formed;
//! * each `eps_j^2` is non-negative, available at no extra cost inside the pass, and additive to the
//!   total loss. It is therefore a sound, free "conflict" signal: a sub-block whose `eps_j^2` sum is
//!   large is where the quantizer did most damage. No separate relative-error metric is needed (the
//!   specification's `E_res`, which was noisy and computed out-of-band, is replaced by this).
//! * the tracked total can be compared against an independently evaluated `tr(E H E^T)`
//!   ([`proxy_loss`]) — that comparison is a unit test, not a debug aid.
//!
//! # Conflict repair (monotone by construction)
//!
//! For the sub-blocks with the largest loss contribution the engine repairs two ways, alternating
//! until the improvement per pass falls below `Config::tol`:
//!
//! * **Coordinate descent on the integer codes.** The exact gradient of the loss w.r.t. the weights of
//!   a column block is `G = E H[:, c0..c1]`, maintained incrementally. Holding every other coordinate
//!   fixed, the loss along one column is `H_jj/2 * (w_hat_j - target)^2`-like and its unconstrained
//!   minimiser is `w_hat_j + g_j / H_jj`; restricting to the grid means snapping that value to
//!   `round((target)/scale) + zero`, clamped to `[0, maxq]`. The move is therefore the *exact* best
//!   grid point for that coordinate, and its loss delta is computed in closed form
//!   (`2*de*g_j + de^2*H_jj`) before applying it, so a worsening move is simply never taken.
//! * **Closed-form scale refit.** Writing `u = q - z` for the integer offsets of a group, the loss as
//!   a function of that group's scale `s` is quadratic: `dL/ds = -2 g.u`, `d2L/ds2 = 2 u H_CC u^T`, so
//!   `delta_s = (g.u)/(u H_CC u^T)` is exact. The new scale is rounded to f16 (the storage format) and
//!   applied only if it strictly decreases the loss.
//!
//! Because each move is a minimisation of the *same* global quadratic along one coordinate, the loss
//! sequence is non-increasing and no weight can be left in floating point — which removes the
//! retry/rollback machinery (and its "sub-block left unquantized" bug) from the specification.
//!
//! # Asymmetric target
//!
//! If the input to this layer in the *already quantized* model is `X~` rather than `X`, then
//!
//! ```text
//!     ||W X - W_hat X~||^2 = const + (W* - W_hat) H~ (W* - W_hat)^T ,   W* = W M H~^-1 ,  M = X X~^T
//! ```
//!
//! so the problem becomes the ordinary symmetric one on a compensated target `W*`. The engine applies
//! `W* = ((W M) U^T) U` with three GEMMs, reusing the already computed factor (see
//! [`quantize_layer`]). Note that the calibration scripts in this repository currently capture only
//! `H` (symmetric calibration, `X~ = X`), so `M` is exercised by tests and the benchmark, not yet by
//! the end-to-end model pipeline.

use crate::linalg::*;
use crate::quant::*;
use anyhow::{bail, Result};
use rayon::prelude::*;
use std::time::Instant;

/// Which algorithm to run. Both share the *identical* grid, ordering, damping and block structure, so
/// a run with `Method::Gptq` is the honest A/B baseline of a `Method::Cdcl` run: same bits, same
/// metadata size, same first pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Plain blocked GPTQ (reference algorithm). Phase 2 is skipped entirely.
    Gptq,
    /// GPTQ pass + conflict-driven exact coordinate-descent repair and closed-form scale refit.
    Cdcl,
}

/// Everything that defines a quantization run. `Config` is printed with `{:?}` into `report.json`, so
/// a run directory is self-describing: [`crate::pipeline`] stores the debug string and
/// `scripts/summarize.py` parses it back.
#[derive(Clone, Debug)]
pub struct Config {
    /// Target code width for ordinary groups (2..=8). Codes are stored as u8 either way; `bits` only
    /// changes the grid (`maxq = 2^bits - 1`) and hence the *nominal* bits/weight.
    pub bits: u32,
    /// Group size along the input dimension: one (f16 scale, zero point) pair per row per group.
    /// `0` means a single group per row (per-row symmetric quantization). Smaller groups cost metadata
    /// bits but fit local weight distributions much better — at 2 bit-widths this matters more than the
    /// repair does (see README, "The 3-bit question").
    pub group: usize,
    /// Lazy-update block size of the GPTQ pass: columns are compensated inside the block column by
    /// column and beyond the block with one GEMM. `block` must divide `group` or be a multiple of it,
    /// so that a group's grid is always computed from weights that are final for their group prefix.
    pub block: usize,
    /// Relative damping: `H += damp * mean(diag(H)) * I`. Needed because calibration Hessians are often
    /// rank-deficient (thousands of columns, fewer effective directions), which would make the Cholesky
    /// fail or produce enormous `U_jj` blow-ups.
    pub damp: f64,
    /// Act-order: quantize columns in descending `H_jj`, i.e. the most-activated input channels first,
    /// so their error can still be compensated by the remaining columns. Off = natural order.
    pub act_order: bool,
    /// Percentage of input columns kept at `skeleton_bits` instead of `bits`, rounded to whole groups and
    /// chosen by OBS criticality `H_jj * ||W_:j||^2`. 0 disables the skeleton. It raises bits/weight, so
    /// it is reported in the stats rather than hidden.
    pub skeleton_pct: f64,
    /// Code width of the protected skeleton groups (2..=8).
    pub skeleton_bits: u32,
    /// Grid selection: `false` = min/max affine grid; `true` = clip-ratio search minimising quantization
    /// MSE over the group (24 steps, 1% .. 48% clipping). Costs time in phase 1 but usually dominates
    /// every other knob at low bit-widths. Applies to GPTQ runs too, so comparisons stay fair.
    pub mse_grid: bool,
    pub method: Method,
    /// Fraction of sub-blocks (highest loss first) repaired by coordinate descent, in (0, 1]. `1.0` = all.
    pub refine_frac: f64,
    /// Columns per repair sub-block: the granularity of conflict selection and of the gradient/gradient
    /// update inside phase 2. Smaller = finer targeting, more GEMM overhead.
    pub sub_block: usize,
    /// Coordinate-descent sweeps inside one sub-block before moving on (a sweep stops early if no code
    /// changed). `1` is usually enough per pass; the outer `passes` loop does the rest.
    pub cd_sweeps: usize,
    /// Maximum number of passes over the selected sub-blocks (refit + descent alternating).
    pub passes: usize,
    /// Stop when a pass improves the loss by less than this fraction of the current total loss.
    pub tol: f64,
    /// Closed-form Hessian-optimal refit of every (row, group) scale, still stored as f16 (so no extra
    /// bits). Cheap per group and worth a large part of the gain; disabled automatically when the group
    /// is larger than 1024 columns to bound scratch memory.
    pub refit_scales: bool,
    /// Rows handed to one rayon task (0 = automatic: about `d_out / n_threads`, clamped to 32..=256).
    /// Also controls peak per-chunk memory, since each chunk keeps its own `w`/`wq`/`e` copies.
    pub row_chunk: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            bits: 3,
            group: 128,
            block: 128,
            damp: 0.01,
            act_order: true,
            skeleton_pct: 0.0,
            skeleton_bits: 8,
            mse_grid: false,
            method: Method::Cdcl,
            refine_frac: 1.0,
            sub_block: 64,
            cd_sweeps: 1,
            passes: 10,
            tol: 1e-3,
            refit_scales: true,
            row_chunk: 0,
        }
    }
}

/// Per-layer measurements. `loss_gptq` and `loss_final` are the *same* proxy loss
/// `tr(E H_damped E^T)` evaluated at two points of the same run, which is what makes the reported gain
/// meaningful: it is not a comparison against another program, it is GPTQ-before vs CDCL-after inside a
/// single quantization, with identical grid, ordering and damping.
#[derive(Clone, Debug, Default)]
pub struct LayerStats {
    /// Proxy loss `sum e H e^T` (damped H) after the plain GPTQ pass: what GPTQ alone would give.
    pub loss_gptq: f64,
    /// Proxy loss after conflict repair (equals `loss_gptq` for `Method::Gptq`). Guaranteed `<=` by monotonicity.
    pub loss_final: f64,
    /// Sub-blocks the layer was split into for conflict selection (`ceil(d_in / sub_block)`).
    pub subblocks_total: usize,
    /// Sub-blocks actually selected for repair (`ceil(refine_frac * subblocks_total)`).
    pub subblocks_refined: usize,
    /// Integer code changes accepted by coordinate descent plus scale refits accepted: a proxy for the
    /// repair work actually done (zero means phase 2 found nothing to improve).
    pub code_updates: usize,
    /// Groups promoted to `skeleton_bits` (0 when the skeleton is off).
    pub skeleton_groups: usize,
    /// Nominal code bits/weight averaged over groups (mixes `bits` and `skeleton_bits`).
    pub bits_codes: f64,
    /// Effective bits/weight including per-(row, group) metadata: 24 bits per group per row
    /// (f16 scale + u8 zero point), i.e. `bits_codes + 24 / group_size`. This is the honest number to
    /// compare against fp16 and against other quantizers; it is what the README tables report.
    pub bits_total: f64,
    /// Ordering + damping + Cholesky + (optional) asymmetric-target build. Amortised across layers that
    /// share an input tensor thanks to [`PrepCache`].
    pub secs_prepare: f64,
    /// Phase 1 + phase 2 wall time for this layer's rows (parallel over row chunks).
    pub secs_quant: f64,
}

/// Hessian-derived data shared by all layers fed by the same activations and ordering.
///
/// Keyed in [`PrepCache`] by the Hessian identity (`hkey`) plus the permutation: q/k/v (and gate/up)
/// receive literally the same input tensor, so they share one damped copy of `H` and one Cholesky as long
/// as their column order also matches — which it does whenever act-order is used, since the permutation
/// then depends only on `H`.
pub struct Prepared {
    /// Column permutation actually used (skeleton groups first, then act-order / natural order).
    pub perm: Vec<u32>,
    /// Damped, permuted Hessian (f32), `n x n`, in the same order as `perm`. Used by phase 2, whose
    /// gradient and diagonal entries `H_jj` must be expressed in the permuted basis.
    pub h32: Vec<f32>,
    /// Upper factor with `H^-1 = U^T U`, `n x n`, in the permuted basis. Used by phase 1.
    pub u: Vec<f32>,
    /// Damping actually added to the diagonal (`damp * mean(diag H)`); re-applied to `M` so that
    /// `M = H` reproduces the symmetric target exactly.
    pub damp_val: f64,
}

/// One-entry memo for [`Prepared`]: `(hessian key, prepared data)`. Only one entry is kept because the
/// quantization job order groups same-input layers together (see [`crate::pipeline`]), and holding several
/// `n x n` factors at once would multiply peak memory (`2 * 4 * d_in^2` bytes per entry).
pub type PrepCache = Option<(String, Prepared)>;

/// The group size actually used: `group == 0`, or a group larger than the layer, means "one scale per row".
#[inline]
pub fn effective_group(cfg: &Config, n: usize) -> usize {
    if cfg.group == 0 || cfg.group >= n {
        n
    } else {
        cfg.group
    }
}

/// Reject configurations that would silently produce wrong results rather than fail loudly.
///
/// The group/block divisibility rule is the important one: a group's grid is computed at its first column
/// from the weights current at that moment, and that is only consistent with the blocked lazy update if
/// one of the two sizes divides the other.
fn validate(cfg: &Config, n: usize) -> Result<()> {
    if !(2..=8).contains(&cfg.bits) || !(2..=8).contains(&cfg.skeleton_bits) {
        bail!("bits and skeleton_bits must be in 2..=8");
    }
    if cfg.block == 0 || cfg.sub_block == 0 {
        bail!("block and sub_block must be > 0");
    }
    let g = effective_group(cfg, n);
    if g < n && cfg.block % g != 0 && g % cfg.block != 0 {
        bail!("group size {g} must divide the block size {} (or be a multiple of it)", cfg.block);
    }
    if cfg.skeleton_pct > 0.0 && g >= n {
        bail!("the skeleton needs a finite group size");
    }
    if !(0.0..=1.0).contains(&cfg.refine_frac) {
        bail!("refine_frac must be in [0, 1]");
    }
    Ok(())
}

/// Column order: protected skeleton groups first (by structural criticality `H_jj * ||W_:j||^2`),
/// then the remaining columns by descending `H_jj` (act-order) or natural order.
///
/// Selecting whole *groups* for the skeleton (never scattered single columns) keeps each group's
/// scale/zero-point contiguous, which both the grid logic and the packer assume. The criticality is an
/// OBS-style saliency: `H_jj` measures how much a perturbation of input channel `j` costs on the
/// calibration distribution, `||W_:j||^2` how much weight mass that channel carries. Ties are broken by
/// ascending index so the permutation is deterministic across runs and threads.
///
/// Returns `(perm, nskel)` where `nskel` counts the groups promoted to `skeleton_bits`; at least one
/// non-skeleton group is always left, so `gbits` is never degenerate.
fn compute_perm(hdiag: &[f64], w: &[f32], d_out: usize, n: usize, g: usize, cfg: &Config) -> (Vec<u32>, usize) {
    let mut nskel = 0usize;
    let mut perm: Vec<u32> = Vec::with_capacity(n);
    let mut in_skel = vec![false; n];
    if cfg.skeleton_pct > 0.0 && g < n {
        let full_groups = n / g;
        nskel = ((cfg.skeleton_pct / 100.0 * n as f64 / g as f64).round() as usize).min(full_groups.saturating_sub(1));
        if nskel > 0 {
            let mut colsq = vec![0f64; n];
            for r in 0..d_out {
                let row = &w[r * n..(r + 1) * n];
                for j in 0..n {
                    colsq[j] += (row[j] as f64) * (row[j] as f64);
                }
            }
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| {
                let ca = hdiag[a] * colsq[a];
                let cb = hdiag[b] * colsq[b];
                cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
            });
            for &j in idx.iter().take(nskel * g) {
                in_skel[j] = true;
                perm.push(j as u32);
            }
        }
    }
    let mut rest: Vec<usize> = (0..n).filter(|&j| !in_skel[j]).collect();
    if cfg.act_order {
        rest.sort_by(|&a, &b| hdiag[b].partial_cmp(&hdiag[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    }
    perm.extend(rest.into_iter().map(|j| j as u32));
    (perm, nskel)
}

/// Damp, permute and factorise one Hessian (the expensive part of preparation: O(n^3) in f64).
///
/// Steps, in the order that matters for correctness:
/// 1. permute `H` into the quantization order (`H'_ij = H_{perm_i, perm_j}`), in f64;
/// 2. patch *dead* columns — input channels that never activate on the calibration set have a zero
///    diagonal and make the Cholesky fail. Following GPTQ we give them a unit diagonal instead (their
///    weights are then irrelevant to the objective, and [`quantize_layer`] zeroes them);
/// 3. add relative damping `damp * mean(diag H)` on the diagonal — relative, so it is invariant to the
///    arbitrary positive scaling of `H` produced by token counting;
/// 4. factorise with [`upper_chol_of_inverse`] (f64) and downcast the permuted `H` to f32 for phase 2.
///
/// Doing steps 1-3 in f64 and only then converting keeps the factorisation well conditioned: calibration
/// Hessians of real transformers are extremely ill-conditioned, and a damped Cholesky of an f32 copy can
/// fail outright.
fn build_prepared(h: &[f32], n: usize, perm: Vec<u32>, cfg: &Config) -> Result<Prepared> {
    let mut hp = vec![0f64; n * n];
    hp.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let pi = perm[i] as usize;
        for j in 0..n {
            row[j] = h[pi * n + perm[j] as usize] as f64;
        }
    });
    // dead columns (zero activations): unit diagonal, as in GPTQ
    for i in 0..n {
        if hp[i * n + i] == 0.0 {
            hp[i * n + i] = 1.0;
        }
    }
    let mean_diag = (0..n).map(|i| hp[i * n + i]).sum::<f64>() / n as f64;
    let damp_val = cfg.damp * mean_diag;
    for i in 0..n {
        hp[i * n + i] += damp_val;
    }
    let h32: Vec<f32> = hp.par_iter().map(|&x| x as f32).collect();
    let u = upper_chol_of_inverse(hp, n)?;
    Ok(Prepared { perm, h32, u, damp_val })
}

/// Per-chunk result of one row chunk of a layer (rows are independent, so these sum exactly).
struct ChunkStats {
    loss_gptq: f64,
    loss_final: f64,
    sb_total: usize,
    sb_refined: usize,
    updates: usize,
}

/// Quantize one row chunk of a layer: phase 1 (blocked GPTQ) then phase 2 (conflict repair).
///
/// All indices are in the *permuted* column order: `wt`, `codes`, `u` and `h` share that order, while
/// `scales`/`zeros` are indexed by `(row, group)` with groups contiguous along the permuted axis. The
/// permutation itself is not needed here — it is stored in [`QLayer::perm`] and applied at pack/dequant
/// time — which is why phase 1 and phase 2 can work entirely on contiguous slices.
///
/// * `wt`   : `rows x n` input weights (already zeroed for dead columns, permuted, target-adjusted)
/// * `u`    : upper factor of `H^-1` (`H^-1 = U^T U`), row-major `n x n`
/// * `h`    : damped permuted Hessian, row-major `n x n` (the metric phase 2 optimizes)
/// * `gbits`: code width per group (`bits`, or `skeleton_bits` for protected groups)
/// * `codes`, `scales`, `zeros`: outputs written in place; on return they describe the final model.
#[allow(clippy::too_many_arguments)]
fn process_chunk(
    wt: &[f32],
    rows: usize,
    n: usize,
    ng: usize,
    g: usize,
    u: &[f32],
    h: &[f32],
    gbits: &[u8],
    cfg: &Config,
    codes: &mut [u8],
    scales: &mut [f32],
    zeros: &mut [f32],
) -> ChunkStats {
    let bsz = cfg.block;
    let mut w = wt.to_vec();
    let mut wq = vec![0f32; rows * n];
    let mut colloss = vec![0f64; n];
    let mut w1 = vec![0f32; rows * bsz.min(n)];
    let mut err1 = vec![0f32; rows * bsz.min(n)];
    let mut tmp = vec![0f32; g.min(n)];

    // ---------------- Phase 1: blocked GPTQ with lazy batch updates ----------------
    //
    // Columns are processed left to right in blocks of `bsz`. Inside a block the compensation is applied
    // column by column against the in-block slice of `U` (rank-1, into the contiguous copy `w1`); the
    // effect on *every* column past the block is deferred and applied once as a GEMM
    //     w[:, i2..] -= err1 * U[i1..i2, i2..]
    // which is algebraically identical to doing `cnt` individual rank-1 updates but turns them into one
    // BLAS-style call. This is the standard "lazy batch" trick and it is exact, not an approximation:
    // `U`'s row `j` already encodes the Schur complement of columns 0..j.
    let mut i1 = 0;
    while i1 < n {
        let i2 = (i1 + bsz).min(n);
        let cnt = i2 - i1;
        for r in 0..rows {
            w1[r * cnt..(r + 1) * cnt].copy_from_slice(&w[r * n + i1..r * n + i2]);
        }
        for i in 0..cnt {
            let j = i1 + i;
            // A group's grid is fixed at its first column, from the weights *current at that moment*
            // (already compensated by all preceding columns). `tmp` gathers the group slice from either
            // the in-block copy `w1` or the full-width `w`, depending on where each column lives; this is
            // only well defined when group and block divide each other, which `validate` enforces.
            if j % g == 0 {
                let gi = j / g;
                let bits = gbits[gi] as u32;
                let gend = (j + g).min(n);
                for r in 0..rows {
                    let t = &mut tmp[..gend - j];
                    for (k, col) in (j..gend).enumerate() {
                        t[k] = if col < i2 { w1[r * cnt + (col - i1)] } else { w[r * n + col] };
                    }
                    let (s, z) = find_params(t, bits, cfg.mse_grid);
                    scales[r * ng + gi] = s;
                    zeros[r * ng + gi] = z;
                }
            }
            let gi = j / g;
            let maxq = ((1u32 << gbits[gi]) - 1) as f32;
            // `inv_d = 1 / U_jj`: dividing the quantization error by the pivot turns it into eps_j, whose
            // square is exactly this column's contribution to the total loss (see module docs). Storing
            // `eps`, not the raw error, is what makes the lazy GEMM correct as well.
            let inv_d = 1.0 / u[j * n + j];
            let urow = &u[j * n + j + 1..j * n + i2];
            let mut cl = 0f64;
            for r in 0..rows {
                let wv = w1[r * cnt + i];
                let (c, dq) = quant_code(wv, scales[r * ng + gi], zeros[r * ng + gi], maxq);
                codes[r * n + j] = c;
                wq[r * n + j] = dq;
                let e = (wv - dq) * inv_d;
                cl += (e * e) as f64;
                err1[r * cnt + i] = e;
                if !urow.is_empty() {
                    // Exact Schur-complement compensation of the not-yet-quantized columns inside the block.
                    let row = &mut w1[r * cnt + i + 1..(r + 1) * cnt];
                    for (x, &uu) in row.iter_mut().zip(urow) {
                        *x -= e * uu;
                    }
                }
            }
            colloss[j] = cl;
        }
    // Deferred compensation for all columns past the block: one GEMM instead of `cnt` rank-1 updates.
    // `w += -err1 * U[i1..i2, i2..]` (beta = 1.0 accumulates into the full-width buffer).
    if i2 < n {
            unsafe {
                sgemm(
                    rows,
                    cnt,
                    n - i2,
                    -1.0,
                    err1.as_ptr(),
                    cnt as isize,
                    1,
                    u.as_ptr().add(i1 * n + i2),
                    n as isize,
                    1,
                    1.0,
                    w.as_mut_ptr().add(i2),
                    n as isize,
                    1,
                );
            }
        }
        i1 = i2;
    }
    drop(w1);
    drop(err1);
    drop(w);

    // Sum of the exact per-column terms: equals tr(E H_damped E^T) up to f32 rounding (unit-tested).
    let loss_gptq: f64 = colloss.iter().sum();
    let sbw = cfg.sub_block;
    let n_sb = (n + sbw - 1) / sbw;
    // `loss_delta` is <= 0 and accumulates the *closed-form* delta of every accepted repair move, so
    // `loss_final = loss_gptq + loss_delta` is bookkeeping, never a re-evaluation.
    let mut loss_delta = 0f64;
    let mut refined = 0usize;
    let mut updates = 0usize;

    // ---------------- Phase 2: conflict detection + exact repair ----------------
    let do_cd = cfg.refine_frac > 0.0 && cfg.cd_sweeps > 0 && cfg.passes > 0;
    // The refit needs a `g x g` Hessian sub-block and two `rows x g` scratch buffers per chunk, so it is
    // skipped for very large groups (per-row quantization) to bound memory.
    let do_refit = cfg.refit_scales && g <= 1024;
    if cfg.method == Method::Cdcl && (do_cd || do_refit) {
        // Conflict = sub-block with the largest exact loss contribution in the GPTQ pass. Because the
        // per-column terms are non-negative and additive, "largest sum of eps_j^2" is exactly "the part of
        // the layer where quantization did most damage" — no separate metric, no extra pass over the data.
        let mut sel: Vec<usize> = Vec::new();
        if do_cd {
            let mut sbl: Vec<(f64, usize)> = (0..n_sb)
                .map(|s| {
                    let c0 = s * sbw;
                    let c1 = ((s + 1) * sbw).min(n);
                    (colloss[c0..c1].iter().sum::<f64>(), s)
                })
                .collect();
            sbl.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
            let k = ((cfg.refine_frac * n_sb as f64).ceil() as usize).clamp(1, n_sb);
            sel = sbl[..k].iter().map(|x| x.1).collect();
            sel.sort_unstable();
            refined = sel.len();
        }

        // Scratch for phase 2. `e` is the full weight-error matrix: its gradient against the *real*
        // objective, `G = E H`, is what both repair moves consume, so keeping it exactly up to date after
        // every accepted move is what makes the closed-form loss deltas exact (and the monotonicity real).
        let mut e = vec![0f32; rows * n];
        for i in 0..rows * n {
            e[i] = wt[i] - wq[i];
        }
        let mut gs = vec![0f32; rows * sbw.max(g.min(1024))]; // block gradient G = E H[:, c0..c1]
        let mut hss = vec![0f32; sbw * sbw]; // dense copy of the sub-block Hessian (cache-friendly, gives H_jj)
        let gcap = g.min(1024);
        let mut uu = vec![0f32; if do_refit { rows * gcap } else { 0 }]; // integer offsets u = q - z
        let mut hu = vec![0f32; if do_refit { rows * gcap } else { 0 }]; // u H_CC
        let mut hcc = vec![0f32; if do_refit { gcap * gcap } else { 0 }]; // dense group sub-block of H

        // Hessian-optimal scale refit. With `u = q - z` fixed, the loss restricted to one (row, group) as a
        // function of that group's scale s is quadratic:
        //     dL/ds = -2 g.u        ,   d2L/ds2 = 2 u H_CC u^T
        // so the exact stationary point is `delta_s = (g.u)/(u H_CC u^T)` and the resulting loss change is
        // `-2 d*num + d^2*den` — computed *before* touching anything, so a move that would not help (e.g.
        // after rounding the scale to f16) is simply rejected. Zero extra bits: the new scale is stored as
        // f16 exactly like GPTQ's min/max one.
        let mut refit_all = |codes: &[u8],
                             scales: &mut [f32],
                             zeros: &[f32],
                             wq: &mut [f32],
                             e: &mut [f32],
                             gs: &mut [f32],
                             loss_delta: &mut f64|
         -> usize {
            let mut changed = 0usize;
            for gi in 0..ng {
                let c0 = gi * g;
                let c1 = (c0 + g).min(n);
                let wd = c1 - c0;
                unsafe {
                    sgemm(rows, n, wd, 1.0, e.as_ptr(), n as isize, 1, h.as_ptr().add(c0), n as isize, 1, 0.0, gs.as_mut_ptr(), wd as isize, 1);
                }
                for a in 0..wd {
                    hcc[a * wd..(a + 1) * wd].copy_from_slice(&h[(c0 + a) * n + c0..(c0 + a) * n + c1]);
                }
                for r in 0..rows {
                    let z = zeros[r * ng + gi];
                    for k in 0..wd {
                        uu[r * wd + k] = codes[r * n + c0 + k] as f32 - z;
                    }
                }
                unsafe {
                    sgemm(rows, wd, wd, 1.0, uu.as_ptr(), wd as isize, 1, hcc.as_ptr(), wd as isize, 1, 0.0, hu.as_mut_ptr(), wd as isize, 1);
                }
                for r in 0..rows {
                    let ur = &uu[r * wd..(r + 1) * wd];
                    let num: f32 = ur.iter().zip(&gs[r * wd..(r + 1) * wd]).map(|(a, b)| a * b).sum();
                    let den: f32 = ur.iter().zip(&hu[r * wd..(r + 1) * wd]).map(|(a, b)| a * b).sum();
                    if !(den > 1e-30) {
                        continue; // degenerate group (all codes equal to the zero point): nothing to scale
                    }
                    let s_old = scales[r * ng + gi];
                    // Round to f16 first: the storage format is f16, so the accepted value must be a legal
                    // one, and `F16_MIN_NORMAL` keeps the scale normal (never zero/subnormal -> no NaN codes).
                    let s_new = round_f16(s_old + num / den).max(F16_MIN_NORMAL);
                    let d = s_new - s_old;
                    let dl = (-2.0 * d * num + d * d * den) as f64;
                    if dl < 0.0 {
                        scales[r * ng + gi] = s_new;
                        for k in 0..wd {
                            let j = c0 + k;
                            let ndq = s_new * ur[k];
                            e[r * n + j] += wq[r * n + j] - ndq; // keep the error matrix exact
                            wq[r * n + j] = ndq;
                        }
                        *loss_delta += dl;
                        changed += 1;
                    }
                }
            }
            changed
        };

        // Alternating passes. Codes and scales are coupled: better codes make the old scale suboptimal and
        // a better scale makes new codes affordable, so each pass feeds the next. The synthetic benchmark
        // shows why neither move alone is enough (each ~-1%, alternated ~-10%).
        let mut prev_delta = 0f64;
        for _pass in 0..cfg.passes.max(1) {
            let mut pass_updates = 0usize;
            if do_refit {
                pass_updates += refit_all(codes, scales, zeros, &mut wq, &mut e, &mut gs, &mut loss_delta);
            }
            if do_cd {
                for &s in &sel {
                    let c0 = s * sbw;
                    let c1 = ((s + 1) * sbw).min(n);
                    let wd = c1 - c0;
                    // Exact gradient of the block: G = E H[:, c0..c1]. Recomputed from `e` at the start of a
                    // sub-block (one GEMM), then maintained incrementally per accepted coordinate move.
                    unsafe {
                        sgemm(rows, n, wd, 1.0, e.as_ptr(), n as isize, 1, h.as_ptr().add(c0), n as isize, 1, 0.0, gs.as_mut_ptr(), wd as isize, 1);
                    }
                    for a in 0..wd {
                        hss[a * wd..(a + 1) * wd].copy_from_slice(&h[(c0 + a) * n + c0..(c0 + a) * n + c1]);
                    }
                    for r in 0..rows {
                        let gsr = &mut gs[r * wd..(r + 1) * wd];
                        for _sw in 0..cfg.cd_sweeps {
                            let mut changed = false;
                            for jj in 0..wd {
                                let j = c0 + jj;
                                let gi = j / g;
                                let maxq = ((1u32 << gbits[gi]) - 1) as f32;
                                let sc = scales[r * ng + gi];
                                let z = zeros[r * ng + gi];
                                let hjj = hss[jj * wd + jj];
                                let cur = wq[r * n + j];
                                // 1-D exact minimiser over the grid: nearest point to w_hat + g/H_jj.
                                // Holding every other coordinate fixed, dL/dw_hat_j = -g_j and the curvature
                                // along that coordinate is H_jj, so the unconstrained optimum of the *current*
                                // dequantized value is `cur + g_j/H_jj`; snapping to the affine grid (and
                                // clamping to the legal code range) gives the exact best grid point.
                                let target = cur + gsr[jj] / hjj;
                                let q = ((target / sc).round() + z).clamp(0.0, maxq);
                                let newdq = sc * (q - z);
                                let qc = q as u8;
                                if qc != codes[r * n + j] {
                                    // Closed-form loss change of this single coordinate move:
                                    //   dL = 2*de*g_j + de^2*H_jj ,  de = e_old - e_new.
                                    // It is negative by construction (the move minimises that quadratic), and
                                    // accumulating it keeps `loss_final` exact without re-evaluating tr(EHE^T).
                                    let de = cur - newdq; // change of e_j = w_j - w_hat_j
                                    loss_delta += (2.0 * de * gsr[jj] + de * de * hjj) as f64;
                                    // Rank-1 gradient update: moving column j by de changes G by de * H[j, block].
                                    let hrow = &hss[jj * wd..(jj + 1) * wd];
                                    for (gk, &hk) in gsr.iter_mut().zip(hrow) {
                                        *gk += de * hk;
                                    }
                                    wq[r * n + j] = newdq;
                                    codes[r * n + j] = qc;
                                    e[r * n + j] += de;
                                    pass_updates += 1;
                                    changed = true;
                                }
                            }
                            if !changed {
                                break; // this sub-block is at a coordinate-wise fixed point
                            }
                        }
                    }
                }
            }
            updates += pass_updates;
            let gain = prev_delta - loss_delta;
            prev_delta = loss_delta;
            if pass_updates == 0 || gain < cfg.tol * (loss_gptq + loss_delta) {
                break; // converged: the last pass improved the loss by less than `tol` (relative)
            }
        }
        // Final refit so the scales match the final codes (the descent left them stale).
        if do_refit {
            updates += refit_all(codes, scales, zeros, &mut wq, &mut e, &mut gs, &mut loss_delta);
        }
    }

    ChunkStats { loss_gptq, loss_final: loss_gptq + loss_delta, sb_total: n_sb, sb_refined: refined, updates }
}

/// Quantize one linear layer end to end (preparation + phase 1 + phase 2) and return the quantized
/// representation together with the statistics that make the run auditable.
///
/// * `w`    : `d_out x d_in` weights (row-major, original column order)
/// * `h`    : `d_in x d_in` calibration Hessian `X X^T` (any positive scale: the objective is a ratio and
///            the damping is relative, so an overall factor cancels out)
/// * `m`    : optional cross-covariance `M = X X~^T` for asymmetric calibration (`None` = symmetric)
/// * `hkey` : identity of the Hessian; layers sharing it (q/k/v, gate/up) also share the damped copy and
///            the Cholesky, as long as the permutation matches. Pass the *owner* module name.
/// * `cache`: single-entry memo owned by the caller (see [`PrepCache`])
///
/// Dead input channels (`H_jj == 0`, never activated during calibration) are zeroed in the weights before
/// quantization: they contribute nothing to the objective, and zeroing them keeps their codes/scales clean
/// instead of letting an arbitrary scale/zero pair describe a don't-care value.
pub fn quantize_layer(
    w: &[f32],
    d_out: usize,
    d_in: usize,
    h: &[f32],
    m: Option<&[f32]>,
    hkey: &str,
    cfg: &Config,
    cache: &mut PrepCache,
) -> Result<(QLayer, LayerStats)> {
    let n = d_in;
    assert_eq!(w.len(), d_out * n);
    assert_eq!(h.len(), n * n);
    validate(cfg, n)?;
    let g = effective_group(cfg, n);
    let ng = (n + g - 1) / g;
    let t0 = Instant::now();

    let hdiag: Vec<f64> = (0..n).map(|j| h[j * n + j] as f64).collect();
    let mut w0 = w.to_vec();
    for j in 0..n {
        if hdiag[j] == 0.0 {
            for r in 0..d_out {
                w0[r * n + j] = 0.0;
            }
        }
    }
    let (perm, nskel) = compute_perm(&hdiag, &w0, d_out, n, g, cfg);
    let reuse = matches!(cache, Some((k, p)) if k == hkey && p.perm == perm);
    if !reuse {
        *cache = None; // free the previous factorisation before allocating the next
        *cache = Some((hkey.to_string(), build_prepared(h, n, perm.clone(), cfg)?));
    }
    let prep = &cache.as_ref().unwrap().1;

    // permuted weights
    let mut wp = vec![0f32; d_out * n];
    wp.par_chunks_mut(n).zip(w0.par_chunks(n)).for_each(|(dst, src)| {
        for j in 0..n {
            dst[j] = src[prep.perm[j] as usize];
        }
    });
    drop(w0);

    // Asymmetric calibration target: W* = W M H~^-1 = ((W M) U^T) U, done as three GEMMs that reuse the
    // factor already computed for the metric (H~^-1 = U^T U). The same damping is added to M so that the
    // identity `M = H  =>  W* = W` holds exactly (unit-tested); without it the ridge would break that check.
    if let Some(mm) = m {
        assert_eq!(mm.len(), n * n);
        let mut mp = vec![0f32; n * n];
        mp.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let pi = prep.perm[i] as usize;
            for j in 0..n {
                row[j] = mm[pi * n + prep.perm[j] as usize];
            }
            row[i] += prep.damp_val as f32; // same ridge as H -> M = H gives W* = W
        });
        let mut t1 = vec![0f32; d_out * n];
        let mut t2 = vec![0f32; d_out * n];
        unsafe {
            pgemm_f32(d_out, n, n, 1.0, wp.as_ptr(), n as isize, 1, mp.as_ptr(), n as isize, 1, 0.0, t1.as_mut_ptr(), n as isize, 1);
            // U^T is the same buffer read with a transposed leading dimension (lda = n instead of 1).
            pgemm_f32(d_out, n, n, 1.0, t1.as_ptr(), n as isize, 1, prep.u.as_ptr(), 1, n as isize, 0.0, t2.as_mut_ptr(), n as isize, 1);
            pgemm_f32(d_out, n, n, 1.0, t2.as_ptr(), n as isize, 1, prep.u.as_ptr(), n as isize, 1, 0.0, wp.as_mut_ptr(), n as isize, 1);
        }
    }
    let secs_prepare = t0.elapsed().as_secs_f64();

    // Per-group code width: the first `nskel` groups in the *permuted* order are the protected skeleton
    // (compute_perm placed them there on purpose), everything else uses `bits`.
    let gbits: Vec<u8> = (0..ng).map(|gi| if gi < nskel { cfg.skeleton_bits as u8 } else { cfg.bits as u8 }).collect();
    let mut codes = vec![0u8; d_out * n];
    let mut scales = vec![0f32; d_out * ng];
    let mut zeros = vec![0f32; d_out * ng];

    // Row chunking: rows are independent, so each rayon task runs its own phase 1 + phase 2 over a
    // contiguous row range. The chunk size trades parallelism and cache locality against the per-chunk
    // scratch memory of phase 2 (`w`, `wq`, `e` copies plus the gradient buffers).
    let nt = rayon::current_num_threads();
    let rc = if cfg.row_chunk > 0 { cfg.row_chunk } else { ((d_out + nt - 1) / nt).clamp(32, 256) };
    let rc = rc.min(d_out).max(1);

    let t1 = Instant::now();
    let stats: Vec<ChunkStats> = wp
        .par_chunks(rc * n)
        .zip(codes.par_chunks_mut(rc * n))
        .zip(scales.par_chunks_mut(rc * ng))
        .zip(zeros.par_chunks_mut(rc * ng))
        .map(|(((wt, c), s), z)| {
            let rows = wt.len() / n;
            process_chunk(wt, rows, n, ng, g, &prep.u, &prep.h32, &gbits, cfg, c, s, z)
        })
        .collect();
    let secs_quant = t1.elapsed().as_secs_f64();

    let ql = QLayer { d_out, d_in: n, group: g, perm: prep.perm.clone(), gbits, codes, scales, zeros };
    let st = LayerStats {
        loss_gptq: stats.iter().map(|s| s.loss_gptq).sum(),
        loss_final: stats.iter().map(|s| s.loss_final).sum(),
        subblocks_total: stats.iter().map(|s| s.sb_total).sum(),
        subblocks_refined: stats.iter().map(|s| s.sb_refined).sum(),
        code_updates: stats.iter().map(|s| s.updates).sum(),
        skeleton_groups: nskel,
        bits_codes: ql.bits_codes(),
        bits_total: ql.bits_total(),
        secs_prepare,
        secs_quant,
    };
    Ok((ql, st))
}

/// Direct evaluation of `sum_rows e H e^T` (f32 GEMM, f64 reduction). `w_t`, `w_hat`, `h` must share
/// the same column order.
///
/// This is the *independent* computation of the proxy loss, used to audit the incremental bookkeeping of
/// [`process_chunk`]: the tracked value in [`LayerStats`] must match this one to within f32 rounding
/// (~1e-3 relative), otherwise the closed-form deltas would be drifting and the reported gain would be
/// fiction. It is also what `cdclq bench` reports as the held-out error metric.
pub fn proxy_loss(w_t: &[f32], w_hat: &[f32], rows: usize, n: usize, h: &[f32]) -> f64 {
    let e: Vec<f32> = w_t.iter().zip(w_hat).map(|(a, b)| a - b).collect();
    let mut eh = vec![0f32; rows * n];
    unsafe {
        pgemm_f32(rows, n, n, 1.0, e.as_ptr(), n as isize, 1, h.as_ptr(), n as isize, 1, 0.0, eh.as_mut_ptr(), n as isize, 1);
    }
    e.par_iter().zip(eh.par_iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum()
}

/// Round-to-nearest baseline with the same grids (no error compensation, no reordering).
///
/// Included so that `cdclq bench` can quote RTN next to GPTQ and CDCL under identical grid/damping/group
/// settings: it isolates how much of the quality comes from error compensation rather than from the grid.
/// Typically 8-10x worse than GPTQ at 3 bit, which is the whole point of the GPTQ family.
pub fn rtn_layer(w: &[f32], d_out: usize, n: usize, cfg: &Config) -> Vec<f32> {
    let g = effective_group(cfg, n);
    let maxq = ((1u32 << cfg.bits) - 1) as f32;
    let mut out = vec![0f32; d_out * n];
    out.par_chunks_mut(n).zip(w.par_chunks(n)).for_each(|(dst, src)| {
        let mut j = 0;
        while j < n {
            let e = (j + g).min(n);
            let (s, z) = find_params(&src[j..e], cfg.bits, cfg.mse_grid);
            for k in j..e {
                dst[k] = quant_code(src[k], s, z, maxq).1;
            }
            j = e;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// Correlated activations with channel outliers -> a non-trivial, ill-conditioned Hessian.
    ///
    /// The generator is deliberately *not* i.i.d.: real transformer activations are low-rank-ish (rank
    /// `n/4` here), have per-channel log-normal scaling and additive noise. An identity Hessian would make
    /// GPTQ, CDCL and RTN nearly indistinguishable, so every quantitative test needs this structure.
    fn problem(d_out: usize, n: usize, tokens: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut r = Rng::new(seed);
        let rank = n / 4;
        let a: Vec<f32> = (0..rank * n).map(|_| r.normal() / (rank as f32).sqrt()).collect();
        let z: Vec<f32> = (0..tokens * rank).map(|_| r.normal()).collect();
        let mut x = vec![0f32; tokens * n];
        unsafe {
            sgemm(tokens, rank, n, 1.0, z.as_ptr(), rank as isize, 1, a.as_ptr(), n as isize, 1, 0.0, x.as_mut_ptr(), n as isize, 1);
        }
        let s: Vec<f32> = (0..n).map(|_| (0.5 * r.normal()).exp()).collect();
        for t in 0..tokens {
            for j in 0..n {
                x[t * n + j] = (x[t * n + j] + 0.3 * r.normal()) * s[j];
            }
        }
        let w: Vec<f32> = (0..d_out * n).map(|_| r.normal() * 0.05).collect();
        let mut h = vec![0f64; n * n];
        gram_accumulate(&x, &x, tokens, n, &mut h);
        (w, h.iter().map(|&v| v as f32).collect())
    }

    /// Re-damp a raw Hessian the same way `build_prepared` does, so a test can evaluate `proxy_loss` on the
    /// exact matrix the engine optimized.
    fn damped(h: &[f32], n: usize, damp: f64) -> Vec<f32> {
        let mean = (0..n).map(|i| h[i * n + i] as f64).sum::<f64>() / n as f64;
        let mut d = h.to_vec();
        for i in 0..n {
            d[i * n + i] += (damp * mean) as f32;
        }
        d
    }

    /// Small, fast settings shared by the engine tests (group | block as required by `validate`).
    fn cfg_small() -> Config {
        Config { bits: 3, group: 32, block: 64, sub_block: 16, act_order: false, row_chunk: 8, ..Default::default() }
    }

    /// The exactness claim of the whole design: the sum of `eps_j^2` accumulated during the GPTQ pass must
    /// equal an independently evaluated `tr(E H E^T)`. If this fails, the conflict signal and every reported
    /// gain are meaningless.
    #[test]
    fn gptq_loss_decomposition_is_exact() {
        let (d_out, n) = (24, 96);
        let (w, h) = problem(d_out, n, 400, 11);
        let mut cfg = cfg_small();
        cfg.method = Method::Gptq;
        let mut cache = None;
        let (ql, st) = quantize_layer(&w, d_out, n, &h, None, "t", &cfg, &mut cache).unwrap();
        let hd = damped(&h, n, cfg.damp);
        let direct = proxy_loss(&w, &ql.dequantize(), d_out, n, &hd);
        let rel = (direct - st.loss_gptq).abs() / direct;
        assert!(rel < 2e-3, "decomposition mismatch: direct {direct} vs tracked {} (rel {rel})", st.loss_gptq);
    }

    /// Three guarantees at once: phase 1 of a CDCL run is bit-identical to a GPTQ run (so the A/B is fair),
    /// the repair never increases the loss when measured *directly*, and the tracked `loss_final` matches
    /// that direct measurement. It must also actually improve, otherwise "monotone" would be trivially true.
    #[test]
    fn repair_is_monotone_and_bookkeeping_exact() {
        let (d_out, n) = (24, 96);
        let (w, h) = problem(d_out, n, 400, 12);
        let hd = |c: &Config| damped(&h, n, c.damp);
        let mut cfg = cfg_small();
        cfg.method = Method::Gptq;
        let mut cache = None;
        let (q0, s0) = quantize_layer(&w, d_out, n, &h, None, "t", &cfg, &mut cache).unwrap();
        let l0 = proxy_loss(&w, &q0.dequantize(), d_out, n, &hd(&cfg));
        for frac in [0.25, 1.0] {
            let mut c2 = cfg.clone();
            c2.method = Method::Cdcl;
            c2.refine_frac = frac;
            c2.passes = 2;
            let mut cache = None;
            let (q1, s1) = quantize_layer(&w, d_out, n, &h, None, "t", &c2, &mut cache).unwrap();
            let l1 = proxy_loss(&w, &q1.dequantize(), d_out, n, &hd(&c2));
            assert!((s1.loss_gptq - s0.loss_gptq).abs() / s0.loss_gptq < 1e-6, "pass 1 must be identical to GPTQ");
            assert!(l1 <= l0 * (1.0 + 1e-6), "repair increased the loss: {l1} > {l0}");
            assert!(s1.loss_final <= s1.loss_gptq);
            let rel = (l1 - s1.loss_final).abs() / l1;
            assert!(rel < 5e-3, "tracked final loss {} vs direct {l1}", s1.loss_final);
            assert!(l1 < l0, "no improvement at all for frac {frac}");
        }
    }

    /// The permutation must be a genuine bijection, the skeleton must occupy whole leading groups with the
    /// promoted bit-width, and those columns must actually reconstruct much better than the 3-bit ones —
    /// i.e. the criticality ranking is not just a plausible-looking heuristic.
    #[test]
    fn skeleton_and_act_order_permutation() {
        let (d_out, n) = (16, 128);
        let (w, h) = problem(d_out, n, 300, 13);
        let mut cfg = cfg_small();
        cfg.act_order = true;
        cfg.skeleton_pct = 25.0;
        let mut cache = None;
        let (ql, st) = quantize_layer(&w, d_out, n, &h, None, "t", &cfg, &mut cache).unwrap();
        let mut seen = vec![false; n];
        for &p in &ql.perm {
            assert!(!seen[p as usize]);
            seen[p as usize] = true;
        }
        assert_eq!(st.skeleton_groups, 1);
        assert_eq!(ql.gbits[0], 8);
        assert!(ql.gbits[1..].iter().all(|&b| b == 3));
        // skeleton columns reconstruct far better than 3-bit columns
        let wd = ql.dequantize();
        let first: Vec<usize> = ql.perm[..32].iter().map(|&p| p as usize).collect();
        let mut e8 = 0f64;
        let mut e3 = 0f64;
        for r in 0..d_out {
            for j in 0..n {
                let e = (w[r * n + j] - wd[r * n + j]) as f64;
                if first.contains(&j) {
                    e8 += e * e;
                } else {
                    e3 += e * e;
                }
            }
        }
        assert!(e8 / 32.0 < e3 / (n as f64 - 32.0) * 0.2);
    }

    /// Sanity check on the asymmetric path: passing `M = H` must reproduce the symmetric target (the damping
    /// is applied consistently to both), so the codes may only differ where f32 rounding flips a tie. This
    /// catches sign/transpose errors in the three-GEMM `W M U^T U` chain.
    #[test]
    fn asymmetric_target_reduces_to_symmetric_when_m_equals_h() {
        let (d_out, n) = (16, 64);
        let (w, h) = problem(d_out, n, 300, 14);
        let mut cfg = cfg_small();
        cfg.method = Method::Gptq;
        let mut c1 = None;
        let (a, sa) = quantize_layer(&w, d_out, n, &h, None, "t", &cfg, &mut c1).unwrap();
        let mut c2 = None;
        let (b, sb) = quantize_layer(&w, d_out, n, &h, Some(&h), "t", &cfg, &mut c2).unwrap();
        let same = a.codes.iter().zip(&b.codes).filter(|(x, y)| x == y).count() as f64 / a.codes.len() as f64;
        assert!(same > 0.97, "codes agree only {same}");
        assert!((sa.loss_gptq - sb.loss_gptq).abs() / sa.loss_gptq < 0.05);
    }
}
