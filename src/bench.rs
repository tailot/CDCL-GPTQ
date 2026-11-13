//! Self-contained benchmark on synthetic layers (correlated activations with channel outliers and massive tokens).
//! Every method gets the same weights, Hessian, grid, ordering and damping, and is evaluated on the calibration tokens
//! **and** on held-out tokens (relative output error).
//!
//! Why a synthetic benchmark at all, when the README also reports real-model numbers: it isolates the algorithm from
//! any single architecture, runs in seconds instead of minutes, and — most importantly — makes both metrics available
//! at once. On a real model only perplexity is cheap enough to measure repeatedly; here the proxy loss (what the
//! optimizer minimizes) and the held-out output error (what actually matters) can be compared directly, which is how we
//! know the proxy is a *faithful* objective rather than something being over-fitted. The synthetic gains do not transfer
//! one-to-one to real models, so neither number should be used to predict the other; they are reported side by side for
//! exactly that reason.

use crate::engine::*;
use crate::linalg::*;
use crate::rng::Rng;
use anyhow::Result;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

pub struct BenchArgs {
    pub rows: usize,
    pub cols: usize,
    /// Calibration tokens. The Hessian rank is capped by this, so it must stay well above `cols` or the problem becomes
    /// singular and damping stops being a mild regularizer.
    pub tokens: usize,
    /// Held-out tokens, never seen by any method; the "held-out err" column comes from these.
    pub test_tokens: usize,
    pub seed: u64,
    /// If > 0: also run the asymmetric (quantized-input) experiment with this relative input noise.
    pub asym_noise: f32,
    pub md: Option<PathBuf>,
    pub dump: Option<PathBuf>,
    pub cfg: Config,
}

/// Activation generator mimicking transformer activations: a low-rank shared component (attention/MLP outputs are not
/// full-rank), per-channel log-normal scaling (the origin of "outlier channels" that break naive 4-bit quantization),
/// rare 20-60x outlier channels, additive noise so the Hessian is full rank, and ~2% "massive" tokens whose norm is 4x
/// typical (the behaviour real LLMs show on BOS/rare tokens, which dominates a least-squares objective).
struct ActGen {
    n: usize,
    rank: usize,
    a: Vec<f32>,
    scale: Vec<f32>,
}

impl ActGen {
    fn new(rng: &mut Rng, n: usize) -> Self {
        let rank = (n / 8).max(8);
        // The `1/sqrt(rank)` factor keeps the product's variance O(1) as the rank grows.
        let a: Vec<f32> = (0..rank * n).map(|_| rng.normal() / (rank as f32).sqrt()).collect();
        let mut scale: Vec<f32> = (0..n).map(|_| (0.6 * rng.normal()).exp()).collect();
        for s in scale.iter_mut() {
            if rng.uniform() < 0.005 {
                *s *= 20.0 + 40.0 * rng.uniform();
            }
        }
        ActGen { n, rank, a, scale }
    }

    /// Draw `tokens` activation rows (`t x n`). One GEMM for the low-rank part, then per-token scaling in a scalar loop
    /// because the outlier/massive decisions are sequential RNG draws and must stay reproducible.
    fn sample(&self, rng: &mut Rng, tokens: usize) -> Vec<f32> {
        let (n, rank) = (self.n, self.rank);
        let z: Vec<f32> = (0..tokens * rank).map(|_| rng.normal()).collect();
        let mut x = vec![0f32; tokens * n];
        unsafe {
            pgemm_f32(tokens, rank, n, 1.0, z.as_ptr(), rank as isize, 1, self.a.as_ptr(), n as isize, 1, 0.0, x.as_mut_ptr(), n as isize, 1);
        }
        for t in 0..tokens {
            let massive = if rng.uniform() < 0.02 { 4.0 } else { 1.0 };
            let row = &mut x[t * n..(t + 1) * n];
            for j in 0..n {
                row[j] = (row[j] + 0.3 * rng.normal()) * self.scale[j] * massive;
            }
        }
        x
    }
}

/// Synthetic weight matrix: per-row log-normal scale (real layers differ a lot row by row) plus rare 8x outliers.
/// The generator is shared by every method, which is what makes the A/B comparison exact rather than statistical.
fn gen_weights(rng: &mut Rng, rows: usize, n: usize) -> Vec<f32> {
    let mut w = vec![0f32; rows * n];
    for r in 0..rows {
        let sigma = 0.02 * (0.3 * rng.normal()).exp();
        for j in 0..n {
            let mut v = rng.normal() * sigma;
            if rng.uniform() < 0.001 {
                v *= 8.0;
            }
            w[r * n + j] = v;
        }
    }
    w
}

/// `||W x - W_hat x~||^2 / ||W x||^2` over `t` tokens (`x`, `xt`: `t x n`).
///
/// The model-level metric: relative output error of the layer, normalized so it is comparable across layers and bit
/// widths. Passing `x == xt` measures pure weight error; passing a noisy `xt` measures the *combined* effect of weight
/// quantization and quantized inputs, which is what an asymmetric-calibration run is supposed to help with. Note this is
/// computed on **held-out** tokens in the tables below: the proxy loss uses calibration data, so if the two disagree the
/// objective is being over-fitted, and that disagreement is itself a result worth reporting.
fn rel_err(w: &[f32], wq: &[f32], x: &[f32], xt: &[f32], rows: usize, n: usize, t: usize) -> f64 {
    let mut o1 = vec![0f32; rows * t];
    let mut o2 = vec![0f32; rows * t];
    unsafe {
        pgemm_f32(rows, n, t, 1.0, w.as_ptr(), n as isize, 1, x.as_ptr(), 1, n as isize, 0.0, o1.as_mut_ptr(), t as isize, 1);
        pgemm_f32(rows, n, t, 1.0, wq.as_ptr(), n as isize, 1, xt.as_ptr(), 1, n as isize, 0.0, o2.as_mut_ptr(), t as isize, 1);
    }
    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in o1.iter().zip(&o2) {
        let d = (*a - *b) as f64;
        num += d * d;
        den += (*a as f64) * (*a as f64);
    }
    num / den
}

/// One row of the report. `loss` is the proxy objective on calibration data, `calib`/`test` are relative output errors on
/// calibration and held-out tokens, and `bpw` is measured (not nominal) bits per weight.
struct Row {
    name: String,
    loss: f64,
    calib: f64,
    test: f64,
    secs: f64,
    bpw: f64,
}

/// Render a markdown table with the comparison against `rows[base]` (always plain GPTQ) on the held-out column — the one
/// number that must not be gamed by calibration. Also printed to stdout so a run can be pasted into the README verbatim.
fn table(title: &str, rows: &[Row], base: usize) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "\n### {title}\n");
    let _ = writeln!(s, "| method | proxy loss | calib err | held-out err | vs GPTQ (held-out) | bits/weight | time (s) |");
    let _ = writeln!(s, "|---|---:|---:|---:|---:|---:|---:|");
    for (i, r) in rows.iter().enumerate() {
        let vs = if i == base || rows[base].test == 0.0 { "-".to_string() } else { format!("{:+.1}%", 100.0 * (r.test / rows[base].test - 1.0)) };
        let _ = writeln!(
            s,
            "| {} | {} | {:.4e} | {:.4e} | {} | {:.3} | {:.2} |",
            r.name,
            if r.loss > 0.0 { format!("{:.4e}", r.loss) } else { "-".into() },
            r.calib,
            r.test,
            vs,
            r.bpw,
            r.secs
        );
    }
    s
}

/// Run the whole benchmark and return the markdown report (also printed to stdout by `main`).
///
/// The comparison is designed so that no method gets an advantage it should not have: one weight matrix, one calibration
/// Hessian, one test set, identical grid/ordering/damping through cloned configs. Each variant changes exactly one knob,
/// which turns the table into an ablation rather than a leaderboard — you can see how much of the gain comes from the
/// scale refit alone, from the coordinate descent alone, and from combining them (the two are complementary: each is
/// around -1% separately and around -10% alternated).
///
/// `xcal` is deliberately capped at 4096 tokens for the *calibration* error column: it keeps the calibration and held-out
/// columns on comparable footing instead of letting the training error look artificially good.
pub fn run(a: &BenchArgs) -> Result<String> {
    let (rows, n) = (a.rows, a.cols);
    let mut rng = Rng::new(a.seed);
    let gen = ActGen::new(&mut rng, n);
    let w = gen_weights(&mut rng, rows, n);
    let x = gen.sample(&mut rng, a.tokens);
    let xte = gen.sample(&mut rng, a.test_tokens);
    let mut h64 = vec![0f64; n * n];
    gram_accumulate(&x, &x, a.tokens, n, &mut h64);
    let h: Vec<f32> = h64.iter().map(|&v| v as f32).collect();
    let ncal = a.tokens.min(4096);
    let xcal = &x[..ncal * n];

    let mut report = String::new();
    let _ = writeln!(
        report,
        "## Synthetic layer {rows}x{n}, {} calibration tokens, {} held-out tokens, {} threads\n\nbits={} group={} damp={} act_order={} mse_grid={} sub_block={} cd_sweeps={}",
        a.tokens,
        a.test_tokens,
        rayon::current_num_threads(),
        a.cfg.bits,
        a.cfg.group,
        a.cfg.damp,
        a.cfg.act_order,
        a.cfg.mse_grid,
        a.cfg.sub_block,
        a.cfg.cd_sweeps
    );

    let eval = |wq: &[f32]| -> (f64, f64) {
        (rel_err(&w, wq, xcal, xcal, rows, n, ncal), rel_err(&w, wq, &xte, &xte, rows, n, a.test_tokens))
    };

    let run_one = |name: &str, cfg: &Config, hh: &[f32], m: Option<&[f32]>, eval: &dyn Fn(&[f32]) -> (f64, f64)| -> Result<Row> {
        let mut cache = None;
        let t0 = Instant::now();
        let (ql, st) = quantize_layer(&w, rows, n, hh, m, "bench", cfg, &mut cache)?;
        let secs = t0.elapsed().as_secs_f64();
        let (c, t) = eval(&ql.dequantize());
        Ok(Row { name: name.to_string(), loss: st.loss_final, calib: c, test: t, secs, bpw: st.bits_total })
    };

    // ---------------- main comparison at equal bits ----------------
    // `base` is the user's config with the skeleton disabled: everything in this table must be equal-bits, and mixed
    // precision would make "bits/weight" differ between rows and invalidate the comparison.
    let mut base = a.cfg.clone();
    base.skeleton_pct = 0.0;
    let mut gptq = base.clone();
    gptq.method = Method::Gptq;
    // RTN gets act-order off because it ignores the Hessian entirely: permuting columns would only change which group a
    // weight lands in, and leaving it on would suggest the ordering was helping RTN when it cannot be.
    let mut rtn_cfg = base.clone();
    rtn_cfg.act_order = false;

    let mut out: Vec<Row> = Vec::new();
    {
        let t0 = Instant::now();
        let wq = rtn_layer(&w, rows, n, &rtn_cfg);
        let secs = t0.elapsed().as_secs_f64();
        let (c, t) = eval(&wq);
        let g = effective_group(&rtn_cfg, n) as f64;
        // loss 0.0 marks "no proxy loss reported": RTN never optimizes the objective, so a loss column for it would be
        // misleading (the table renders 0 as "-").
        out.push(Row { name: "RTN (no compensation)".into(), loss: 0.0, calib: c, test: t, secs, bpw: rtn_cfg.bits as f64 + 24.0 / g });
    }
    out.push(run_one("GPTQ", &gptq, &h, None, &eval)?);
    // (name, repair fraction, sweeps, passes, scale refit)
    // The two single-mechanism rows are the ablations; `tol = 0.0` in all variants forces a fixed number of passes so the
    // cost is comparable instead of depending on when each variant happens to converge.
    let variants: [(&str, f64, usize, usize, bool); 4] = [
        ("ablation: scale refit only (3 passes)", 0.0, 1, 3, true),
        ("ablation: coordinate descent only (4 passes)", 1.0, 1, 4, false),
        ("CDCL-GPTQ fast (50% sub-blocks, 2 passes)", 0.5, 1, 2, true),
        ("CDCL-GPTQ balanced (all sub-blocks, 4 passes)", 1.0, 1, 4, true),
    ];
    for (name, frac, sweeps, passes, refit) in variants {
        let mut c = base.clone();
        c.method = Method::Cdcl;
        c.refine_frac = frac;
        c.cd_sweeps = sweeps;
        c.passes = passes;
        c.refit_scales = refit;
        c.tol = 0.0;
        out.push(run_one(name, &c, &h, None, &eval)?);
    }
    {
        let mut c = base.clone();
        c.method = Method::Cdcl;
        out.push(run_one(&format!("CDCL-GPTQ default (up to {} passes, tol {})", c.passes, c.tol), &c, &h, None, &eval)?);
    }
    report.push_str(&table("Equal-bits comparison (same grid, ordering, damping)", &out, 1));

    // ---------------- skeleton (mixed precision) ----------------
    // Reported separately, and never merged into the equal-bits table: promoting columns to 8 bits raises bits/weight, so
    // its "gain" is bought with size. The reference row is the same GPTQ run without a skeleton, which lets the reader
    // judge the trade rather than being shown an isolated improvement.
    if a.cfg.skeleton_pct > 0.0 {
        let mut g2 = gptq.clone();
        g2.skeleton_pct = a.cfg.skeleton_pct;
        let mut c2 = base.clone();
        c2.skeleton_pct = a.cfg.skeleton_pct;
        c2.method = Method::Cdcl;
        let mut sk: Vec<Row> = Vec::new();
        sk.push(run_one("GPTQ + skeleton", &g2, &h, None, &eval)?);
        sk.push(run_one("CDCL-GPTQ + skeleton", &c2, &h, None, &eval)?);
        let mut sk_row = out[1].name.clone();
        sk_row.push_str(" (no skeleton, for reference)");
        let mut rows_ref = vec![Row { name: sk_row, loss: out[1].loss, calib: out[1].calib, test: out[1].test, secs: out[1].secs, bpw: out[1].bpw }];
        rows_ref.extend(sk);
        report.push_str(&table(
            &format!("Protected skeleton ({}% of input columns at {} bits) - note the higher bits/weight", a.cfg.skeleton_pct, a.cfg.skeleton_bits),
            &rows_ref,
            1,
        ));
    }

    // ---------------- asymmetric calibration (quantized inputs) ----------------
    // Stand-in for a quantized-input pipeline (W4A8, AWQ-style activation schemes): the activations the layer will really
    // see are `x~ = x + noise`, not `x`. The metric becomes `||W x - W_hat x~||`, and the first row — full-precision
    // weights evaluated on noisy inputs — is the floor no quantizer can beat: if a method goes below it, the experiment is
    // wrong rather than clever.
    //
    // Noise is scaled per channel by that channel's RMS, so quiet channels are not swamped and loud ones are not
    // unrealistically punished; `--asym-noise` is therefore "fraction of each channel's own typical magnitude".
    // Three variants separate the design choices: optimize against H~ only (symmetric target), against the asymmetric
    // target `W* = W M H~^-1`, and that same target with the full CDCL repair.
    if a.asym_noise > 0.0 {
        let mut rms = vec![0f32; n];
        for t in 0..a.tokens {
            for j in 0..n {
                rms[j] += x[t * n + j] * x[t * n + j];
            }
        }
        for v in rms.iter_mut() {
            *v = (*v / a.tokens as f32).sqrt();
        }
        let noisy = |src: &[f32], t: usize, rng: &mut Rng| -> Vec<f32> {
            let mut o = src.to_vec();
            for i in 0..t {
                for j in 0..n {
                    o[i * n + j] += a.asym_noise * rms[j] * rng.normal();
                }
            }
            o
        };
        let xt = noisy(&x, a.tokens, &mut rng);
        let xte_t = noisy(&xte, a.test_tokens, &mut rng);
        let mut ht64 = vec![0f64; n * n];
        let mut m64 = vec![0f64; n * n];
        gram_accumulate(&xt, &xt, a.tokens, n, &mut ht64);
        gram_accumulate(&x, &xt, a.tokens, n, &mut m64);
        let ht: Vec<f32> = ht64.iter().map(|&v| v as f32).collect();
        let mm: Vec<f32> = m64.iter().map(|&v| v as f32).collect();
        let xtcal = &xt[..ncal * n];
        let eval_a = |wq: &[f32]| -> (f64, f64) {
            (rel_err(&w, wq, xcal, xtcal, rows, n, ncal), rel_err(&w, wq, &xte, &xte_t, rows, n, a.test_tokens))
        };
        let floor = eval_a(&w);
        let mut ar: Vec<Row> = Vec::new();
        ar.push(Row { name: "FP weights on noisy inputs (floor)".into(), loss: 0.0, calib: floor.0, test: floor.1, secs: 0.0, bpw: 16.0 });
        ar.push(run_one("GPTQ, symmetric target (H~ only)", &gptq, &ht, None, &eval_a)?);
        ar.push(run_one("GPTQ, asymmetric target (W* = W M H~^-1)", &gptq, &ht, Some(&mm), &eval_a)?);
        let mut c = base.clone();
        c.method = Method::Cdcl;
        ar.push(run_one("CDCL-GPTQ, asymmetric target", &c, &ht, Some(&mm), &eval_a)?);
        report.push_str(&table(
            &format!("Quantized-input calibration (relative input noise {}): error is ||W x - W_hat x~||", a.asym_noise),
            &ar,
            1,
        ));
    }

    // ---------------- dump for the numpy reference cross-check ----------------
    // Write a *plain GPTQ, no act-order* case to disk so scripts/ref_gptq_check.py can recompute `tr(E H E^T)` from the
    // same codes/scales/zeros with PyTorch. A second, independent implementation agreeing on the loss is what rules out a
    // bug in the incremental bookkeeping (a self-consistent Rust-only check could not). act-order is forced off because the
    // reference script works in original column order.
    if let Some(dir) = &a.dump {
        std::fs::create_dir_all(dir)?;
        let mut c = base.clone();
        c.method = Method::Gptq;
        c.act_order = false;
        let mut cache = None;
        let (ql, st) = quantize_layer(&w, rows, n, &h, None, "dump", &c, &mut cache)?;
        let wr = |name: &str, bytes: Vec<u8>| std::fs::write(dir.join(name), bytes);
        wr("W.f32", w.iter().flat_map(|v| v.to_le_bytes()).collect())?;
        wr("H.f32", h.iter().flat_map(|v| v.to_le_bytes()).collect())?;
        wr("codes.u8", ql.codes.clone())?;
        wr("scales.f32", ql.scales.iter().flat_map(|v| v.to_le_bytes()).collect())?;
        wr("zeros.f32", ql.zeros.iter().flat_map(|v| v.to_le_bytes()).collect())?;
        wr(
            "meta.json",
            format!(
                "{{\"rows\":{rows},\"cols\":{n},\"bits\":{},\"group\":{},\"damp\":{},\"block\":{},\"loss\":{}}}",
                c.bits, c.group, c.damp, c.block, st.loss_gptq
            )
            .into_bytes(),
        )?;
    }

    if let Some(p) = &a.md {
        std::fs::write(p, &report)?;
    }
    Ok(report)
}
