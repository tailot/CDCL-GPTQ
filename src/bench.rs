//! Self-contained benchmark on synthetic layers (correlated activations with channel outliers and
//! massive tokens). Every method gets the same weights, Hessian, grid, ordering and damping, and is
//! evaluated on the calibration tokens AND on held-out tokens (relative output error).

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
    pub tokens: usize,
    pub test_tokens: usize,
    pub seed: u64,
    /// If > 0: also run the asymmetric (quantized-input) experiment with this relative input noise.
    pub asym_noise: f32,
    pub md: Option<PathBuf>,
    pub dump: Option<PathBuf>,
    pub cfg: Config,
}

struct ActGen {
    n: usize,
    rank: usize,
    a: Vec<f32>,
    scale: Vec<f32>,
}

impl ActGen {
    fn new(rng: &mut Rng, n: usize) -> Self {
        let rank = (n / 8).max(8);
        let a: Vec<f32> = (0..rank * n).map(|_| rng.normal() / (rank as f32).sqrt()).collect();
        let mut scale: Vec<f32> = (0..n).map(|_| (0.6 * rng.normal()).exp()).collect();
        for s in scale.iter_mut() {
            if rng.uniform() < 0.005 {
                *s *= 20.0 + 40.0 * rng.uniform();
            }
        }
        ActGen { n, rank, a, scale }
    }

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

struct Row {
    name: String,
    loss: f64,
    calib: f64,
    test: f64,
    secs: f64,
    bpw: f64,
}

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
    let mut base = a.cfg.clone();
    base.skeleton_pct = 0.0;
    let mut gptq = base.clone();
    gptq.method = Method::Gptq;
    let mut rtn_cfg = base.clone();
    rtn_cfg.act_order = false;

    let mut out: Vec<Row> = Vec::new();
    {
        let t0 = Instant::now();
        let wq = rtn_layer(&w, rows, n, &rtn_cfg);
        let secs = t0.elapsed().as_secs_f64();
        let (c, t) = eval(&wq);
        let g = effective_group(&rtn_cfg, n) as f64;
        out.push(Row { name: "RTN (no compensation)".into(), loss: 0.0, calib: c, test: t, secs, bpw: rtn_cfg.bits as f64 + 24.0 / g });
    }
    out.push(run_one("GPTQ", &gptq, &h, None, &eval)?);
    // (name, repair fraction, sweeps, passes, scale refit)
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
