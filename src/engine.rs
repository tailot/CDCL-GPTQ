//! CDCL-GPTQ engine.
//!
//! Layer-wise objective (Hessian metric, `H = X X^T`, `E = W - W_hat`):
//!
//! ```text
//!     L(W_hat) = sum_rows  e_r H e_r^T          (rows are independent)
//! ```
//!
//! **GPTQ pass.** With `H^-1 = U^T U` (U upper), the loss decomposes *exactly* as
//! `L = sum_j eps_j^2`, `eps_j = (w_cur_j - q_j) / U_jj`. The per-column terms are free and
//! non-negative: they are the conflict signal (the spec's `E_res`, computed without extra passes).
//!
//! **Conflict repair.** For the sub-blocks with the largest loss (the "conflicts") the engine
//! backjumps: it recomputes the exact gradient `g = E H` on the block and runs coordinate descent
//! on the integer grid. For one coordinate the minimiser over the grid is the grid point nearest to
//! `w_hat_j + g_j / H_jj`, so every update is exact and the loss never increases (monotone).
//!
//! **Asymmetric target.** If the layer input in the already-quantized model is `X~` instead of `X`,
//! `||W X - W_hat X~||^2 = const + (W* - W_hat) H~ (..)^T` with `W* = W M H~^-1`, `M = X X~^T`.
//! The quantization problem is then the standard one on the compensated target `W*`.

use crate::linalg::*;
use crate::quant::*;
use anyhow::{bail, Result};
use rayon::prelude::*;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Plain blocked GPTQ (reference algorithm, same grid / ordering / damping).
    Gptq,
    /// GPTQ + conflict-driven exact coordinate-descent repair.
    Cdcl,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub bits: u32,
    /// Group size along the input dimension; 0 = one group per row.
    pub group: usize,
    /// Lazy-update block size of the GPTQ pass.
    pub block: usize,
    pub damp: f64,
    pub act_order: bool,
    /// Percentage of input columns kept at `skeleton_bits` (rounded to whole groups).
    pub skeleton_pct: f64,
    pub skeleton_bits: u32,
    pub mse_grid: bool,
    pub method: Method,
    /// Fraction of sub-blocks (highest loss first) repaired by coordinate descent, in (0, 1].
    pub refine_frac: f64,
    /// Columns per repair sub-block.
    pub sub_block: usize,
    /// Coordinate-descent sweeps inside a sub-block.
    pub cd_sweeps: usize,
    /// Passes over the selected sub-blocks.
    pub passes: usize,
    /// Stop the repair passes when a pass improves the loss by less than this fraction.
    pub tol: f64,
    /// Closed-form Hessian-optimal refit of every (row, group) scale (f16, same bits as GPTQ).
    pub refit_scales: bool,
    /// Rows per parallel task (0 = automatic).
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

#[derive(Clone, Debug, Default)]
pub struct LayerStats {
    /// Proxy loss `sum e H e^T` (damped H) after the plain GPTQ pass: what GPTQ alone would give.
    pub loss_gptq: f64,
    /// Proxy loss after conflict repair (equals `loss_gptq` for `Method::Gptq`).
    pub loss_final: f64,
    pub subblocks_total: usize,
    pub subblocks_refined: usize,
    pub code_updates: usize,
    pub skeleton_groups: usize,
    pub bits_codes: f64,
    pub bits_total: f64,
    pub secs_prepare: f64,
    pub secs_quant: f64,
}

/// Hessian-derived data shared by all layers fed by the same activations and ordering.
pub struct Prepared {
    pub perm: Vec<u32>,
    /// Damped, permuted Hessian (f32), `n x n`.
    pub h32: Vec<f32>,
    /// Upper factor with `H^-1 = U^T U`, `n x n`.
    pub u: Vec<f32>,
    pub damp_val: f64,
}

pub type PrepCache = Option<(String, Prepared)>;

#[inline]
pub fn effective_group(cfg: &Config, n: usize) -> usize {
    if cfg.group == 0 || cfg.group >= n {
        n
    } else {
        cfg.group
    }
}

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

struct ChunkStats {
    loss_gptq: f64,
    loss_final: f64,
    sb_total: usize,
    sb_refined: usize,
    updates: usize,
}

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
    let mut i1 = 0;
    while i1 < n {
        let i2 = (i1 + bsz).min(n);
        let cnt = i2 - i1;
        for r in 0..rows {
            w1[r * cnt..(r + 1) * cnt].copy_from_slice(&w[r * n + i1..r * n + i2]);
        }
        for i in 0..cnt {
            let j = i1 + i;
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
                    let row = &mut w1[r * cnt + i + 1..(r + 1) * cnt];
                    for (x, &uu) in row.iter_mut().zip(urow) {
                        *x -= e * uu;
                    }
                }
            }
            colloss[j] = cl;
        }
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

    let loss_gptq: f64 = colloss.iter().sum();
    let sbw = cfg.sub_block;
    let n_sb = (n + sbw - 1) / sbw;
    let mut loss_delta = 0f64;
    let mut refined = 0usize;
    let mut updates = 0usize;

    // ---------------- Phase 2: conflict detection + exact repair ----------------
    let do_cd = cfg.refine_frac > 0.0 && cfg.cd_sweeps > 0 && cfg.passes > 0;
    let do_refit = cfg.refit_scales && g <= 1024;
    if cfg.method == Method::Cdcl && (do_cd || do_refit) {
        // Conflict = sub-block with the largest exact loss contribution in the GPTQ pass.
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

        let mut e = vec![0f32; rows * n];
        for i in 0..rows * n {
            e[i] = wt[i] - wq[i];
        }
        let mut gs = vec![0f32; rows * sbw.max(g.min(1024))];
        let mut hss = vec![0f32; sbw * sbw];
        let gcap = g.min(1024);
        let mut uu = vec![0f32; if do_refit { rows * gcap } else { 0 }];
        let mut hu = vec![0f32; if do_refit { rows * gcap } else { 0 }];
        let mut hcc = vec![0f32; if do_refit { gcap * gcap } else { 0 }];

        // Hessian-optimal scale refit: with u = q - z, the loss as a function of the scale s is
        // quadratic; delta_s = (g.u) / (u H_CC u^T), g = e H restricted to the group (exact).
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
                        continue;
                    }
                    let s_old = scales[r * ng + gi];
                    let s_new = round_f16(s_old + num / den).max(F16_MIN_NORMAL);
                    let d = s_new - s_old;
                    let dl = (-2.0 * d * num + d * d * den) as f64;
                    if dl < 0.0 {
                        scales[r * ng + gi] = s_new;
                        for k in 0..wd {
                            let j = c0 + k;
                            let ndq = s_new * ur[k];
                            e[r * n + j] += wq[r * n + j] - ndq;
                            wq[r * n + j] = ndq;
                        }
                        *loss_delta += dl;
                        changed += 1;
                    }
                }
            }
            changed
        };

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
                    // exact gradient of the block: G = E H[:, c0..c1]
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
                                // 1-D exact minimiser over the grid: nearest point to w_hat + g/H_jj
                                let target = cur + gsr[jj] / hjj;
                                let q = ((target / sc).round() + z).clamp(0.0, maxq);
                                let newdq = sc * (q - z);
                                let qc = q as u8;
                                if qc != codes[r * n + j] {
                                    let de = cur - newdq; // change of e_j = w_j - w_hat_j
                                    loss_delta += (2.0 * de * gsr[jj] + de * de * hjj) as f64;
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
                                break;
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
        // final refit so the scales match the final codes
        if do_refit {
            updates += refit_all(codes, scales, zeros, &mut wq, &mut e, &mut gs, &mut loss_delta);
        }
    }

    ChunkStats { loss_gptq, loss_final: loss_gptq + loss_delta, sb_total: n_sb, sb_refined: refined, updates }
}

/// Quantize one linear layer.
///
/// * `w`    : `d_out x d_in` weights (row-major, original column order)
/// * `h`    : `d_in x d_in` calibration Hessian `X X^T` (any positive scale)
/// * `m`    : optional cross-covariance `X X~^T` for asymmetric calibration
/// * `hkey` : identity of the Hessian, used to share the Cholesky work between layers (q/k/v ...)
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

    // asymmetric calibration: W* = W M H~^-1 = ((W M) U^T) U
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
            pgemm_f32(d_out, n, n, 1.0, t1.as_ptr(), n as isize, 1, prep.u.as_ptr(), 1, n as isize, 0.0, t2.as_mut_ptr(), n as isize, 1);
            pgemm_f32(d_out, n, n, 1.0, t2.as_ptr(), n as isize, 1, prep.u.as_ptr(), n as isize, 1, 0.0, wp.as_mut_ptr(), n as isize, 1);
        }
    }
    let secs_prepare = t0.elapsed().as_secs_f64();

    let gbits: Vec<u8> = (0..ng).map(|gi| if gi < nskel { cfg.skeleton_bits as u8 } else { cfg.bits as u8 }).collect();
    let mut codes = vec![0u8; d_out * n];
    let mut scales = vec![0f32; d_out * ng];
    let mut zeros = vec![0f32; d_out * ng];

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
pub fn proxy_loss(w_t: &[f32], w_hat: &[f32], rows: usize, n: usize, h: &[f32]) -> f64 {
    let e: Vec<f32> = w_t.iter().zip(w_hat).map(|(a, b)| a - b).collect();
    let mut eh = vec![0f32; rows * n];
    unsafe {
        pgemm_f32(rows, n, n, 1.0, e.as_ptr(), n as isize, 1, h.as_ptr(), n as isize, 1, 0.0, eh.as_mut_ptr(), n as isize, 1);
    }
    e.par_iter().zip(eh.par_iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum()
}

/// Round-to-nearest baseline with the same grids (no error compensation, no reordering).
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

    /// Correlated activations with channel outliers -> a non-trivial Hessian.
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

    fn damped(h: &[f32], n: usize, damp: f64) -> Vec<f32> {
        let mean = (0..n).map(|i| h[i * n + i] as f64).sum::<f64>() / n as f64;
        let mut d = h.to_vec();
        for i in 0..n {
            d[i * n + i] += (damp * mean) as f32;
        }
        d
    }

    fn cfg_small() -> Config {
        Config { bits: 3, group: 32, block: 64, sub_block: 16, act_order: false, row_chunk: 8, ..Default::default() }
    }

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
