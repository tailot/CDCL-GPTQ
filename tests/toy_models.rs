//! End-to-end suite over real Hugging Face checkpoints, using the tiny-random Llama family.
//!
//! Why these models: they are ordinary `LlamaForCausalLM` checkpoints — same tensor names, same attention/MLP layout,
//! same alias graph (q/k/v share an input, gate/up share another), three different storage dtypes — but 4 MB to 68 MB
//! instead of gigabytes, so the whole pipeline runs in seconds and can run on a CI runner. They cover what the unit tests
//! inside `engine`/`quant`/`io` cannot: shard walking, name matching, alias resolution, pass-through of uncalibrated
//! tensors, dtype preservation, and the packed-file round-trip on real safetensors files.
//!
//! What they do *not* cover, and no assertion here claims otherwise: accuracy on a trained model. These weights are
//! randomly initialized, so a perplexity number would be meaningless — the quality measurements live in `runs/` and were
//! taken on TinyLlama-1.1B with real WikiText-2 Hessians. The Hessians here are synthetic too (see
//! `scripts/gen_toy_hessians.py`), built to have realistic *structure* (low rank plus ridge, log-normal channel scaling)
//! rather than realistic content. So every assertion below is about the machinery: exact inverses, invariants preserved,
//! no tensor lost or duplicated, and the repair phase never increasing the loss.
//!
//! Fixtures are committed (`tests/fixtures/<name>/hess.safetensors`); the checkpoints live under `models/`, which is
//! git-ignored. Run `python scripts/make_fixtures.py` to (re)generate both. When they are absent every test skips with a
//! message instead of failing, so a fresh clone still runs the unit tests cleanly.

use anyhow::{bail, Result};
use cdclq::engine::{self, Config, Method, PrepCache};
use cdclq::io::{self, HessianStore};
use cdclq::pipeline::{self, QuantOpts};
use cdclq::quant::{self, QLayer};
use half::f16;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// One model under test plus the settings that fit it.
///
/// `group`/`block` are per-model rather than global because `validate` requires one to divide the other, and these
/// checkpoints have input widths of 8, 16, 64, 172 and 256: a fixed group of 128 would collapse to "one scale per row"
/// everywhere (`effective_group`) and the suite would silently stop testing grouped quantization at all.
struct Model {
    /// Directory under `models/` holding `model.safetensors`.
    dir: &'static str,
    /// HF id, used in messages only.
    id: &'static str,
    /// Directory under `tests/fixtures/` holding `hess.safetensors`.
    fixture: &'static str,
    bits: u32,
    group: usize,
    block: usize,
}

const MODELS: &[Model] = &[
    // hidden 16, f16. Every input width divides the group, so this model covers clean grouping and the smallest layers
    // the crate has to handle (8x16 weights), where row-stride off-by-ones are most visible.
    Model { dir: "tiny-random-Llama-3", id: "llamafactory/tiny-random-Llama-3", fixture: "tiny-random-Llama-3", bits: 4, group: 16, block: 16 },
    // hidden 8 with GQA over 16 heads: q_proj is 512x8 and o_proj 8x512. bf16 storage, and the only fixture where a group
    // equal to d_in (8) coexists with a layer 64 wide, so both "one scale per row" and real grouping happen in one run.
    Model { dir: "llama-3", id: "tiny-random/llama-3", fixture: "llama-3", bits: 3, group: 8, block: 8 },
    // hidden 256, f32, intermediate 172 -> a ragged last group (172 = 5*32 + 12) and GQA with k/v at 128 rows. The closest
    // of the three to a real Llama shape, so it also carries the bit-width sweep.
    Model { dir: "tiny-random-llama2", id: "Intel/tiny-random-llama2", fixture: "tiny-random-llama2", bits: 3, group: 32, block: 32 },
];

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is the crate root at compile time, so these paths do not depend on where cargo places the test
    // binary — integration tests run from an unpredictable target subdirectory.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Checkpoint + Hessian fixture for `m`, or `None` when they are absent (the state of a fresh clone, since `models/` is
/// git-ignored). Callers skip rather than fail so the unit tests stay runnable without a download step.
fn inputs(m: &Model) -> Option<(PathBuf, PathBuf)> {
    let w = repo_root().join("models").join(m.dir).join("model.safetensors");
    let h = repo_root().join("tests").join("fixtures").join(m.fixture).join("hess.safetensors");
    if w.is_file() && h.is_file() {
        Some((w, h))
    } else {
        None
    }
}

fn skip(m: &Model, what: &str) {
    eprintln!(
        "SKIP {what}: {} — expected models/{}/model.safetensors and tests/fixtures/{}/hess.safetensors. \
         Generate both with `python scripts/make_fixtures.py`.",
        m.id, m.dir, m.fixture
    );
}

/// Base config for a model: small blocks so the blocked GPTQ pass *and* the repair sub-blocks are both exercised on
/// layers only a few dozen columns wide (a 128-column block would never split a 64-wide layer at all).
fn cfg(m: &Model) -> Config {
    Config {
        bits: m.bits,
        group: m.group,
        block: m.block,
        sub_block: (m.block / 2).max(1),
        damp: 0.01,
        row_chunk: 32,
        ..Default::default()
    }
}

/// Every tensor of a safetensors file as name -> (dtype, shape, raw bytes), plus the file's string metadata. Taking a
/// snapshot rather than streaming keeps the comparisons below independent of tensor ordering inside the file.
struct Snapshot {
    tensors: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
    metadata: HashMap<String, String>,
}

fn snapshot(path: &Path) -> Result<Snapshot> {
    let mm = io::mmap_file(path)?;
    let st = SafeTensors::deserialize(&mm)?;
    let (_, md) = SafeTensors::read_metadata(&mm)?;
    let mut tensors = HashMap::new();
    for name in st.names() {
        let t = st.tensor(name)?;
        tensors.insert(name.clone(), (t.dtype(), t.shape().to_vec(), t.data().to_vec()));
    }
    // `metadata()` hands back a reference to the header's Option<HashMap>; clone through it. A file with no header
    // strings yields an empty map.
    Ok(Snapshot { tensors, metadata: md.metadata().clone().unwrap_or_default() })
}

/// A tensor decoded to f32, so files in different dtypes can be compared numerically rather than byte-wise.
fn f32_of(snap: &Snapshot, name: &str) -> Result<Vec<f32>> {
    let Some((dt, sh, bytes)) = snap.tensors.get(name) else { bail!("missing tensor {name}") };
    let view = TensorView::new(*dt, sh.clone(), bytes)?;
    Ok(io::tensor_to_f32(&view)?)
}

/// Byte-for-byte equality of a pass-through tensor across two files. Names the first offender: "N tensors differ" is not
/// actionable when N is 40 and the cause is one wrong stride.
fn assert_identical(a: &Snapshot, b: &Snapshot, name: &str) -> Result<()> {
    let (x, y) = match (a.tensors.get(name), b.tensors.get(name)) {
        (Some(x), Some(y)) => (x, y),
        _ => bail!("{name}: present in only one of the two files"),
    };
    if x.0 != y.0 || x.1 != y.1 {
        bail!("{name}: dtype/shape changed {:?} {:?} -> {:?} {:?}", x.0, x.1, y.0, y.1);
    }
    if x.2 != y.2 {
        bail!("{name}: pass-through tensor bytes differ (it should never have been touched)");
    }
    Ok(())
}

/// Scratch output directory for one test of one model, cleared first so repeated runs never compare against stale files
/// and nothing is ever written into the source tree.
fn fresh_dir(name: &str) -> Result<PathBuf> {
    let out = repo_root().join("target").join("cdclq-test").join(name);
    if out.exists() {
        std::fs::remove_dir_all(&out)?;
    }
    Ok(out)
}

/// The f32 grid the packed file really describes: `.scales` is stored as f16 and `.qzeros` as u8, so `QLayer::dequantize`
/// on a layer read back from disk uses those quantized parameters. Reconstructing them here gives the *same* f32 values,
/// which is what lets phase 1 be compared across two independent runs without going through a file.
fn grid_of(ql: &QLayer) -> (Vec<f32>, Vec<f32>) {
    let sc: Vec<f32> = ql.scales.iter().map(|&s| f16::from_f32(s).to_f32()).collect();
    let zr: Vec<f32> = ql.zeros.iter().map(|&z| z as u8 as f32).collect();
    (sc, zr)
}

/// Snap every code of a layer back onto its own group's affine grid, `s * (code - z)`.
///
/// Two invariants make this a no-op for codes phase 1 chose and an undo for codes the descent re-pointed, and both are
/// exactly what the container guarantees:
///
/// * the grid is untouched by phase 2 unless `--refit-scales` accepted a move — the coordinate descent reads
///   `scales`/`zeros` but only ever writes `codes` — so in an unrefit run the stored pair *is* the phase-1 pair;
/// * every code lies on its own grid, because both the GPTQ pass and the CDCL move (engine.rs line 631) snap to a legal
///   point of `s*z + s*(q-z)` with the same clamping.
///
/// The phase-1 equality test itself compares two runs directly instead of going through this, which is the stronger
/// statement; it stays here because it is also what an inference kernel has to rely on when it re-reads a packed layer.
fn snap_to_grid(ql: &QLayer) -> Vec<u8> {
    let ng = ql.ngroups();
    let mut codes = ql.codes.clone();
    for r in 0..ql.d_out {
        for j in 0..ql.d_in {
            let gi = j / ql.group;
            let s = ql.scales[r * ng + gi];
            let z = ql.zeros[r * ng + gi];
            let maxq = ((1u32 << ql.gbits[gi]) - 1) as f32;
            codes[r * ql.d_in + j] = quant::quant_code(s * (ql.codes[r * ql.d_in + j] as f32 - z), s, z, maxq).0;
        }
    }
    codes
}

/// Re-evaluate the layer's objective exactly the way phase 2 scores it: `H` damped and permuted, weights compared in
/// permuted order. The engine keeps this matrix (`Prepared::h32`) private, so the test rebuilds it — same dead-column
/// zeroing, same permutation from `compute_perm`, same `damp * mean(diag H)` ridge, same f32 widening — and then checks
/// that what the engine reported matches what this independent evaluation says. If any of those steps drifted, the loss
/// audit below fails loudly rather than passing vacuously.
fn damped_permuted_h(h: &[f32], n: usize, perm: &[u32], damp: f64) -> (Vec<f32>, f64) {
    let mut hp = vec![0f64; n * n];
    for i in 0..n {
        let pi = perm[i] as usize;
        for j in 0..n {
            hp[i * n + j] = h[pi * n + perm[j] as usize] as f64;
        }
    }
    // Dead columns (never activated during calibration) get a unit diagonal, exactly as the engine does: without it the
    // Cholesky would fail and the objective would not be defined for those channels.
    for i in 0..n {
        if hp[i * n + i] == 0.0 {
            hp[i * n + i] = 1.0;
        }
    }
    let mean_diag = (0..n).map(|i| hp[i * n + i]).sum::<f64>() / n as f64;
    let dv = damp * mean_diag;
    for i in 0..n {
        hp[i * n + i] += dv;
    }
    (hp.iter().map(|&x| x as f32).collect(), dv)
}

/// The engine's dead-column rule applied to the weights, so the comparison uses the same `w0` phase 1 saw. `h` is the raw
/// (unpermuted) Hessian: only its diagonal matters here, and dead channels are those with `H_jj == 0`.
fn zero_dead_columns(w: &[f32], h: &[f32], d_out: usize, n: usize) -> Vec<f32> {
    let mut w0 = w.to_vec();
    for j in 0..n {
        if h[j * n + j] == 0.0 {
            for r in 0..d_out {
                w0[r * n + j] = 0.0;
            }
        }
    }
    w0
}

/// Group count of a packed layer, read from the container itself (`.gbits` length) rather than recomputed from config:
/// the point is to check the writer's bookkeeping against the file, so deriving it from the same config would be circular.
fn ng_of_layer(st: &SafeTensors, name: &str) -> Result<usize> {
    let t = st.tensor(&format!("{name}.gbits"))?;
    Ok(t.shape()[0])
}

/// The core end-to-end test, per model: quantize a real checkpoint, then prove the two outputs agree with each other and
/// that nothing outside the quantized layers moved. This is where container-level regressions show up — no unit test
/// touches a real safetensors file from end to end.
#[test]
fn toy_checkpoint_quantize_and_roundtrip() -> Result<()> {
    for m in MODELS {
        let (w, h) = match inputs(m) {
            Some(p) => p,
            None => {
                skip(m, "quantize + round-trip");
                continue;
            }
        };
        let out = fresh_dir(m.dir)?;
        let c = cfg(m);
        let rep = pipeline::quantize_model(&QuantOpts {
            weights: vec![w.clone()],
            hessians: vec![h.clone()],
            out_dir: out.clone(),
            cfg: c.clone(),
            only: Vec::new(),
            skip: Vec::new(),
            emit_packed: true,
            emit_dequant: true,
        })?;
        if rep.layers.is_empty() {
            bail!("{}: no layer quantized — the fixture names do not match the checkpoint", m.id);
        }

        let orig = snapshot(&w)?;
        let deq = snapshot(&out.join("dequant").join("model.safetensors"))?;
        let pk_path = out.join("packed").join("model.safetensors");
        let pk = snapshot(&pk_path)?;

        // (1) The fake-quant model must be a drop-in replacement: same tensor set, shapes and dtypes. Anything else breaks
        // an unmodified inference stack loading it, which is the whole reason `dequant/` exists.
        if orig.tensors.len() != deq.tensors.len() {
            bail!("{}: dequant has {} tensors, input had {}", m.id, deq.tensors.len(), orig.tensors.len());
        }
        for name in orig.tensors.keys() {
            let (dt0, sh0, _) = &orig.tensors[name];
            let (dt1, sh1, _) = &deq.tensors[name];
            if sh0 != sh1 || dt0 != dt1 {
                bail!("{}: {name} became {dt1:?} {sh1:?} in the fake-quant output", m.id);
            }
        }

        // (2) Each reported layer must exist in `packed/` with all six companion tensors, and must unpack through the
        // independent reader path to exactly what the writer put in `dequant/`. Bit-exact: any disagreement means pack +
        // unpack + the permutation scatter are not an inverse pair.
        let pk_mm = io::mmap_file(&pk_path)?;
        let pk_st = SafeTensors::deserialize(&pk_mm)?;
        for l in &rep.layers {
            for suf in ["qweight", "scales", "qzeros", "perm", "gbits", "gsize"] {
                if !pk.tensors.contains_key(&format!("{}.{suf}", l.name)) {
                    bail!("{}: packed tensor {}.{} missing", m.id, l.name, suf);
                }
            }
            let ql = io::read_packed_layer(&pk_st, &l.name)?;
            if ql.d_out != l.d_out || ql.d_in != l.d_in {
                bail!("{}: {} unpacked as {}x{}, report says {}x{}", m.id, l.name, ql.d_out, ql.d_in, l.d_out, l.d_in);
            }
            let via_reader = ql.dequantize();
            let on_disk = f32_of(&deq, &format!("{}.weight", l.name))?;
            // The fake-quant file stores `ql.dequantize()` rounded once to the checkpoint dtype, while `via_reader` is
            // still f32. So the reference here is that same rounding: exact equality of the *rounded* values, which is
            // what a kernel reading the packed file would reproduce.
            let dt = deq.tensors[&format!("{}.weight", l.name)].0;
            let expect = io::f32_to_bytes(dt, &via_reader)?;
            if expect != deq.tensors[&format!("{}.weight", l.name)].2 {
                let mut worst = String::new();
                for (a, b) in via_reader.iter().zip(&on_disk) {
                    if (*a as f64 - *b as f64).abs() > 0.0 {
                        worst = format!("{a} vs {b}");
                        break;
                    }
                }
                bail!("{}: {} pack/unpack does not reproduce the fake-quant file (e.g. {worst})", m.id, l.name);
            }

            // (3) Every code must fit the width its own group declares. A code of 9 in a 3-bit group decodes outside the
            // scale range — precisely what an LSB-first packing bug produces.
            for (r, row) in ql.codes.chunks(ql.d_in).enumerate() {
                for (j, &code) in row.iter().enumerate() {
                    let b = ql.gbits[j / ql.group] as u32;
                    if code as u32 >= 1u32 << b {
                        bail!("{}: {} code {code} out of range for a {b}-bit group at r{r} c{j}", m.id, l.name);
                    }
                }
            }

            // (4) The reported loss must be reproducible from the packed file alone. This is the strongest check in the
            // suite: it ties the engine's incremental bookkeeping to an evaluation of tr(E H E^T) built out of codes,
            // f16 scales and u8 zero points read back off disk, with the damped permuted Hessian rebuilt independently.
            let store = HessianStore::open(&vec![h.clone()])?;
            let owner = store.resolve(&l.name).ok_or_else(|| anyhow::anyhow!("fixture lost {}", l.name))?;
            let (hv, hn) = store.get_h(&owner)?;
            if hn != l.d_in {
                bail!("{}: Hessian is {hn}x{hn} but {} has d_in {}", m.id, l.name, l.d_in);
            }
            let wt = f32_of(&orig, &format!("{}.weight", l.name))?;
            let w0 = zero_dead_columns(&wt, &hv, l.d_out, l.d_in);
            let (hd, _) = damped_permuted_h(&hv, l.d_in, &ql.perm, c.damp);
            let (sc, zr) = grid_of(&ql);
            let ng = ql.ngroups();
            // tr(E H E^T) row by row, in permuted order: E is built from the codes/f16 scales/u8 zeros on disk and H is
            // the independently rebuilt damped matrix. O(d_out * d_in^2), which is trivial at these sizes.
            let mut loss = 0f64;
            for r in 0..l.d_out {
                let e: Vec<f32> = (0..l.d_in)
                    .map(|j| {
                        let gi = j / ql.group;
                        w0[r * l.d_in + ql.perm[j] as usize] - sc[r * ng + gi] * (ql.codes[r * l.d_in + j] as f32 - zr[r * ng + gi])
                    })
                    .collect();
                for j in 0..l.d_in {
                    let mut ehj = 0f64;
                    for k in 0..l.d_in {
                        ehj += (e[k] as f64) * (hd[k * l.d_in + j] as f64);
                    }
                    loss += (e[j] as f64) * ehj;
                }
            }
            if loss <= 0.0 {
                bail!("{}: {} independently evaluated loss is {loss}", m.id, l.name);
            }
            let rel = (loss - l.loss_final).abs() / loss;
            if rel > 5e-3 {
                bail!("{}: {} tracked loss {} != loss rebuilt from the packed file {loss} (rel {rel:.2e})", m.id, l.name, l.loss_final);
            }
            if l.loss_final > l.loss_gptq * (1.0 + 1e-9) {
                bail!("{}: {} repair increased the loss: {} -> {}", m.id, l.name, l.loss_gptq, l.loss_final);
            }
        }

        // (5) Pass-through must be exact. Embeddings and lm_head are the largest tensors here (128k x 16 vocab), so this
        // is also what keeps the reported compression ratio honest instead of pretending they were quantized.
        let quantized: HashSet<String> = rep.layers.iter().map(|l| format!("{}.weight", l.name)).collect();
        let mut untouched = 0usize;
        for name in orig.tensors.keys() {
            if !quantized.contains(name) {
                assert_identical(&orig, &deq, name)?;
                untouched += 1;
            }
        }
        // The fixture deliberately leaves lm_head and embed_tokens uncalibrated, so at least those two must be copied.
        if untouched < 2 {
            bail!("{}: expected embeddings and lm_head among the pass-through tensors, found {untouched}", m.id);
        }

        // (6) The packed header is what `cdclq dequant` uses to recover the original dtype without being told.
        if pk.metadata.get("quantization").map(String::as_str) != Some("cdclq-v1") {
            bail!("{}: packed header lacks quantization=cdclq-v1", m.id);
        }
        if !pk.metadata.contains_key("orig_dtype") {
            bail!("{}: packed header lacks orig_dtype", m.id);
        }

        // (7) The `cdclq dequant` path itself: expanding the packed file must reproduce `dequant/` exactly, tensor for
        // tensor, with no companion tensors left behind and nothing lost.
        let rt = out.join("roundtrip").join("model.safetensors");
        let n = pipeline::dequantize_packed(&pk_path, &rt, None)?;
        if n != rep.layers.len() {
            bail!("{}: dequantize_packed expanded {n} layers, report has {}", m.id, rep.layers.len());
        }
        let rts = snapshot(&rt)?;
        for name in orig.tensors.keys() {
            if !rts.tensors.contains_key(name) {
                bail!("{}: round-trip output lost {name}", m.id);
            }
            let (dt0, sh0, _) = &orig.tensors[name];
            let (dt1, sh1, _) = &rts.tensors[name];
            if dt0 != dt1 || sh0 != sh1 {
                bail!("{}: round-trip changed {name} to {dt1:?} {sh1:?}", m.id);
            }
            if f32_of(&deq, name)? != f32_of(&rts, name)? {
                bail!("{}: round-trip values differ from dequant/ for {name}", m.id);
            }
        }

        // (8) The size claim, measured on real files rather than computed from `--bits`: if the packed codes were not much
        // smaller than the weights they replace, the premise of the crate would be wrong.
        let orig_bytes: usize = quantized.iter().map(|n| orig.tensors[n].2.len()).sum();
        let qw_bytes: usize = pk.tensors.iter().filter(|(n, _)| n.ends_with(".qweight")).map(|(_, v)| v.2.len()).sum();
        if qw_bytes * 2 >= orig_bytes {
            bail!("{}: packed qweights are only {:.2}x smaller than the original weights", m.id, orig_bytes as f64 / qw_bytes as f64);
        }
        eprintln!(
            "OK {}: {} layers quantized, {} copied through, mean bpw {:.2}, repair gain {:+.2}%, qweight {:.1}x smaller",
            m.id,
            rep.layers.len(),
            untouched,
            rep.layers.iter().map(|l| l.bits_total).sum::<f64>() / rep.layers.len() as f64,
            -rep.total_gain_pct,
            orig_bytes as f64 / qw_bytes as f64,
        );
        std::fs::remove_dir_all(&out)?;
    }
    Ok(())
}

/// GPTQ and CDCL must agree bit-for-bit after phase 1 on real checkpoints, and the repair must then lower the loss. This
/// is the fairness claim behind every gain percentage in the README: if a CDCL run's first pass differed from a GPTQ run,
/// part of the reported improvement would be an artifact of two different algorithms rather than of the repair.
///
/// The grid comparison is on *values* the reader would see (f16 scale, u8 zero) rather than raw f32 fields, because phase 1
/// keeps full-precision scales in memory while the packed file rounds them — two runs can differ in the low bits of a scale
/// without ever changing a code.
#[test]
fn gptq_and_cdcl_agree_on_phase_one_of_real_checkpoints() -> Result<()> {
    for m in MODELS {
        let (w, h) = match inputs(m) {
            Some(p) => p,
            None => {
                skip(m, "GPTQ/CDCL phase-1 equality");
                continue;
            }
        };
        // Both runs go through the library engine on the same tensors, so file I/O cannot confound the comparison.
        let store = HessianStore::open(&vec![h.clone()])?;
        let mm = io::mmap_file(&w)?;
        let st = SafeTensors::deserialize(&mm)?;
        // A GPTQ run is phase 1 only: no descent, no refit. That is what `--method gptq` means in this crate.
        let mut gptq_only = cfg(m);
        gptq_only.method = Method::Gptq;
        gptq_only.refine_frac = 0.0;
        gptq_only.refit_scales = false;
        // A CDCL run with the repair switched off must be indistinguishable from a GPTQ run, because `--method` gates only
        // the phase-2 branch of `process_chunk`. That is the fairness claim behind every gain percentage in the README: if
        // a CDCL run's first pass differed from a GPTQ run, part of the reported improvement would be an artifact of two
        // different algorithms rather than of the repair. Scale refitting is off on this side so the two runs share one
        // quantization grid — with `refit_all` re-solving every (row, group) scale in closed form (engine.rs line 572),
        // "same phase 1" would not even be a well posed statement. Note what is *not* touched otherwise: `group`, `block`,
        // `damp` and `mse_grid` stay as configured, because phase 1 does not depend on `cfg.method` at all.
        let cdcl_phase1 = {
            let mut c = cfg(m);
            c.method = Method::Cdcl;
            c.refine_frac = 0.0;
            c.refit_scales = false;
            c
        };
        // The full CDCL configuration, phase-2 knobs at their defaults except refit (see above), used further down to show
        // the repair really does lower the loss on real shapes.
        let cdcl = {
            let mut c = cfg(m);
            c.method = Method::Cdcl;
            c.refit_scales = false;
            c
        };

        let mut checked = 0usize;
        let mut improved = 0usize;
        let mut names: Vec<String> = st.names().into_iter().cloned().collect();
        names.sort();
        for name in &names {
            let Some(layer) = name.strip_suffix(".weight") else { continue };
            let t = st.tensor(name)?;
            if t.shape().len() != 2 {
                continue;
            }
            let Some(owner) = store.resolve(layer) else { continue };
            let (d_out, d_in) = (t.shape()[0], t.shape()[1]);
            let wt = io::tensor_to_f32(&t)?;
            let (hv, _) = store.get_h(&owner)?;

            let mut c1: PrepCache = None;
            let (a, sa) = engine::quantize_layer(&wt, d_out, d_in, &hv, None, &owner, &gptq_only, &mut c1)?;
            // The CDCL twin, with the repair switched off: same code path as GPTQ up to the end of phase 1.
            let mut c3: PrepCache = None;
            let (b, sb) = engine::quantize_layer(&wt, d_out, d_in, &hv, None, &owner, &cdcl_phase1, &mut c3)?;
            let ng = a.ngroups();

            // Same column order first — comparing grids and codes is meaningless across different permutations.
            if a.perm != b.perm {
                bail!("{}: {layer} phase-1 permutation differs between methods", m.id);
            }
            // The grid must be identical, including the part of it the refit could have moved. This is the load-bearing
            // half of the fairness claim: `--refit-scales` changes what a CDCL layer *means*, so if it were active in
            // phase-1 comparisons the two runs would not share an objective at all.
            let (sa_g, za_g) = grid_of(&a);
            let (sb_g, zb_g) = grid_of(&b);
            if sa_g != sb_g || za_g != zb_g {
                let d = sa_g.iter().zip(&sb_g).filter(|(x, y)| x != y).count() + za_g.iter().zip(&zb_g).filter(|(x, y)| x != y).count();
                bail!("{}: {layer} phase-1 grids differ in {d} (row, group) entries", m.id);
            }
            // Codes: with the repair switched off on both sides the two runs must be identical everywhere. Any difference
            // means the GPTQ and CDCL code paths have drifted, which would silently invalidate every gain percentage in the
            // README — those numbers compare one run's phase 1 against its own phase 2, so phase 1 has to be shared.
            if a.codes != b.codes {
                let d = a.codes.iter().zip(&b.codes).filter(|(x, y)| x != y).count();
                // Name the first disagreement in original-column terms: `perm` maps permuted position -> input channel,
                // so a failure here points straight at the channel whose grid was not reproducible.
                let bad = (0..a.codes.len()).find(|&i| a.codes[i] != b.codes[i]);
                let spot = bad
                    .map(|i| {
                        let (r, j) = (i / d_in, i % d_in);
                        let gi = j / a.group;
                        format!("row {r}, input channel {} (grid scale {}, zero {}, width {} bits)", a.perm[j], a.scales[r * ng + gi], a.zeros[r * ng + gi], a.gbits[gi])
                    })
                    .unwrap_or_default();
                bail!(
                    "{}: {layer} phase-1 codes differ in {d} of {} positions (first at {spot}: GPTQ {}, CDCL {})",
                    m.id, a.codes.len(),
                    bad.map(|i| a.codes[i]).unwrap_or(0),
                    bad.map(|i| b.codes[i]).unwrap_or(0),
                );
            }
            // And every code must sit on its own grid: snapping to it is a no-op. This is what an inference kernel relies
            // on when it re-reads the layer, and it holds for both methods.
            if snap_to_grid(&a) != a.codes || snap_to_grid(&b) != b.codes {
                bail!("{}: {layer} a code is not a legal point of its own group's grid", m.id);
            }
            // And the tracked GPTQ loss must agree to within float noise.
            if (sa.loss_gptq - sb.loss_gptq).abs() > 1e-9 * sa.loss_gptq.max(1.0) {
                bail!("{}: {layer} phase-1 loss differs: {} vs {}", m.id, sa.loss_gptq, sb.loss_gptq);
            }
            // With the repair off, CDCL must report exactly GPTQ's loss — `loss_final == loss_gptq` on both sides.
            if (sb.loss_final - sb.loss_gptq).abs() > 1e-12 * sb.loss_gptq.max(1.0) {
                bail!("{}: {layer} CDCL without refinement still moved the loss: {} -> {}", m.id, sb.loss_gptq, sb.loss_final);
            }
            checked += 1;

            // Now switch phase 2 on for real, on the same tensors. The claim this pins down is the one the README makes:
            // the repair strictly lowers the proxy loss on real shapes, and it does so by changing codes (not scales — refit
            // is off here), which is why `code_updates` must be non-zero when the loss moved.
            let mut c4: PrepCache = None;
            let (f, sf) = engine::quantize_layer(&wt, d_out, d_in, &hv, None, &owner, &cdcl, &mut c4)?;
            // Phase 1 is untouched by the repair: same grid, same ordering, same starting codes. If any of these moved,
            // `refine_frac` leaked into phase 1 and the reported gains would not be comparable to GPTQ at all.
            if f.perm != a.perm || grid_of(&f) != (sa_g.clone(), za_g.clone()) {
                bail!("{}: {layer} enabling the repair changed phase 1", m.id);
            }
            // Monotonicity, the property the README states outright: the descent never accepts a move that raises `tr(EHE^T)`.
            if sf.loss_final > sa.loss_gptq * (1.0 + 1e-9) {
                bail!("{}: {layer} repair made the loss worse: {} -> {}", m.id, sa.loss_gptq, sf.loss_final);
            }
            // The codes phase 2 *started from* must be GPTQ's codes. They cannot be read off `f.codes` — the descent
            // re-points codes, and that is the point — so the only place they are still visible is the unrefit run with
            // refinement switched off, which is exactly what `b` is. Asserting `a.codes == b.codes` above already pins this
            // down; what is worth checking on the repaired layer itself is the weaker but load-bearing invariant: every
            // code phase 2 produced is still a legal point of the *same* grid, i.e. the repair never invented a value that
            // the container could not represent.
            let p1_of_f = snap_to_grid(&f);
            if p1_of_f != f.codes {
                let d = p1_of_f.iter().zip(&f.codes).filter(|(x, y)| x != y).count();
                bail!("{}: {layer} phase 2 left {d} code(s) off their own grid", m.id);
            }
            if sf.loss_final < sa.loss_gptq {
                improved += 1;
                if sf.code_updates == 0 {
                    bail!("{}: {layer} loss improved but no code or scale was updated (bookkeeping bug)", m.id);
                }
            }
        }
        if checked == 0 {
            bail!("{}: no quantizable layer found in the checkpoint", m.id);
        }
        if improved == 0 {
            bail!("{}: the repair improved none of the {checked} layers — phase 2 is untested on this model", m.id);
        }
        eprintln!("OK {}: phase-1 equality on {checked} layers, repair improved {improved}", m.id);
    }
    Ok(())
}

/// act-order + skeleton on real checkpoints: the permutation must be a bijection, promoted groups must be contiguous at
/// the front of the permuted order with `skeleton_bits`, and — the part that is easy to get wrong — the `.perm` tensor on
/// disk must carry exactly that permutation, because an inference kernel reads it from there.
#[test]
fn act_order_and_skeleton_survive_packing() -> Result<()> {
    for m in MODELS {
        let (w, h) = match inputs(m) {
            Some(p) => p,
            None => {
                skip(m, "act-order + skeleton");
                continue;
            }
        };
        let mut c = cfg(m);
        // The skeleton needs a group strictly smaller than the layer, so pick a group every layer of this model can
        // actually carry: `effective_group` collapses `group >= d_in` to one scale per row, and `validate` rejects the
        // skeleton in that case instead of silently ignoring it. 4 is below the narrowest input width anywhere here (8
        // for the GQA model). It does not divide each fixture's block size, so the block has to come along: `validate`
        // requires one of the two to divide the other, and a matching block also means no group ever straddles a GPTQ
        // block boundary.
        c.group = 4;
        c.block = 4;
        c.sub_block = 2;
        c.act_order = true;
        // 25% of columns promoted. With group == block this rounds to whole groups, and at least one ordinary group must
        // survive — compute_perm guarantees that, and the assertion below is what pins it down.
        c.skeleton_pct = 25.0;
        let out = fresh_dir(&format!("{}-skel", m.dir))?;
        let rep = pipeline::quantize_model(&QuantOpts {
            weights: vec![w.clone()],
            hessians: vec![h.clone()],
            out_dir: out.clone(),
            cfg: c.clone(),
            only: Vec::new(),
            skip: Vec::new(),
            emit_packed: true,
            emit_dequant: false,
        })?;
        if rep.layers.is_empty() {
            bail!("{}: nothing quantized with act-order + skeleton", m.id);
        }
        let pk_path = out.join("packed").join("model.safetensors");
        let pk_mm = io::mmap_file(&pk_path)?;
        let pk_st = SafeTensors::deserialize(&pk_mm)?;
        // Layers whose input width is a single group cannot carry a skeleton (`validate` rejects them); the suite must
        // still find promoted groups on the layers that can.
        let mut eligible = 0usize;
        let mut promoted = 0usize;
        for l in &rep.layers {
            let ql = io::read_packed_layer(&pk_st, &l.name)?;
            if ql.group >= l.d_in {
                // One scale per row: the skeleton is off here by design, so gbits must be uniform.
                if !ql.gbits.iter().all(|&b| b == c.bits as u8) || l.skeleton_groups != 0 {
                    bail!("{}: {} has one group per row but a non-uniform gbits table", m.id, l.name);
                }
                continue;
            }
            eligible += 1;

            let mut seen = vec![false; l.d_in];
            for &p in &ql.perm {
                if p as usize >= l.d_in || seen[p as usize] {
                    bail!("{}: {} perm is not a bijection (value {p} with d_in {})", m.id, l.name, l.d_in);
                }
                seen[p as usize] = true;
            }
            let ng = ql.gbits.len();
            if l.skeleton_groups == 0 || l.skeleton_groups >= ng {
                bail!("{}: {} has {ng} groups but skeleton_groups {}", m.id, l.name, l.skeleton_groups);
            }
            if !ql.gbits[..l.skeleton_groups].iter().all(|&b| b == c.skeleton_bits as u8) {
                bail!("{}: {} leading groups are not all at skeleton_bits {}", m.id, l.name, c.skeleton_bits);
            }
            if !ql.gbits[l.skeleton_groups..].iter().all(|&b| b == c.bits as u8) {
                bail!("{}: {} trailing groups are not all at bits {}", m.id, l.name, c.bits);
            }
            // Mixed widths inside one row must still decode to in-range codes: this is the case a fixed-stride packer
            // gets wrong, and skeleton promotion is what creates it. The group of a *code* follows the permutation —
            // `perm` maps permuted position -> input channel, and grid/packing both work in permuted order, so code at
            // flat index r*d_in+j belongs to group `j / ql.group`, not to the group of its original channel. (Indexing by
            // `ngroups()` instead silently reads past a short tail group; indexing by `cfg.group` would be wrong on any
            // layer where `effective_group` widened it.)
            for (r, row) in ql.codes.chunks(ql.d_in).enumerate() {
                for (j, &code) in row.iter().enumerate() {
                    let b = ql.gbits[j / ql.group] as u32;
                    if code as u32 >= 1u32 << b {
                        bail!("{}: {} code {code} out of range at permuted r{r} c{j}", m.id, l.name);
                    }
                }
            }
            // The promoted columns must really be the ones compute_perm would pick — and its rule is *per-column*
            // criticality, not per-group: engine.rs sorts individual columns by `H_jj * ||W_:j||^2`, takes the top
            // `nskel*g` of them, and only then marks those columns as protected. Whole groups are what keeps a scale
            // contiguous in the ideal case, but nothing forces the selected set to be group-aligned (down_proj at 172
            // columns with group 4 has a short tail group), so the check below mirrors the column sort exactly: same
            // saliency, same tie-break (ascending index), then the tail by descending `H_jj` under act-order. Re-derived
            // here rather than called because `compute_perm` is private — that is what makes it a check and not a
            // tautology. Keep in sync with engine.rs.
            let store = HessianStore::open(&vec![h.clone()])?;
            let owner = store.resolve(&l.name).unwrap();
            let (hv, _) = store.get_h(&owner)?;
            let wt = f32_of(&snapshot(&w)?, &format!("{}.weight", l.name))?;
            let w0 = zero_dead_columns(&wt, &hv, l.d_out, l.d_in);
            let hdiag: Vec<f64> = (0..l.d_in).map(|j| hv[j * l.d_in + j] as f64).collect();
            let mut colsq = vec![0f64; l.d_in];
            for r in 0..l.d_out {
                for j in 0..l.d_in {
                    colsq[j] += (w0[r * l.d_in + j] as f64).powi(2);
                }
            }
            let gs = c.group;
            // Per-column saliency, in the engine's order: descending `H_jj * ||W_:j||^2`, ties ascending index. The first
            // `nskel*g` entries of this list are the protected columns; everything after them is the act-order tail.
            let mut idx: Vec<usize> = (0..ql.d_in).collect();
            idx.sort_by(|&a, &b| {
                let (sa, sb) = (hdiag[a] * colsq[a], hdiag[b] * colsq[b]);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
            });
            let mut want_perm: Vec<u32> = idx.iter().take(l.skeleton_groups * gs).map(|&j| j as u32).collect();
            let in_skel: HashSet<usize> = idx.iter().take(l.skeleton_groups * gs).copied().collect();
            let mut rest: Vec<usize> = (0..ql.d_in).filter(|&j| !in_skel.contains(&j)).collect();
            if c.act_order {
                rest.sort_by(|&a, &b| hdiag[b].partial_cmp(&hdiag[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
            }
            want_perm.extend(rest.into_iter().map(|j| j as u32));
            if want_perm != ql.perm {
                let bad = (0..ql.d_in).find(|&j| want_perm[j] != ql.perm[j]);
                bail!(
                    "{}: {} permutation does not follow the documented rule (first mismatch at permuted position {:?}: want {}, got {})",
                    m.id, l.name, bad,
                    bad.map(|j| want_perm[j]).unwrap_or(0),
                    bad.map(|j| ql.perm[j]).unwrap_or(0),
                );
            }
            promoted += 1;

            // act-order must be a real reordering: the fixture Hessians have a log-normal diagonal, so sorting by H_jj has
            // to move columns. An identity permutation would mean act-order was silently ignored.
            if ql.perm.iter().enumerate().all(|(j, &p)| p as usize == j) {
                bail!("{}: {} act-order produced the identity permutation", m.id, l.name);
            }
        }
        if promoted == 0 {
            bail!("{}: no layer of this model carries a skeleton ({} eligible)", m.id, eligible);
        }
        // `--no-dequant` must not write a fake-quant file (the flag is user-facing).
        if out.join("dequant").exists() {
            bail!("{}: --no-dequant still wrote a dequant/ directory", m.id);
        }
        eprintln!("OK {}: act-order + skeleton packed on {promoted}/{} layers", m.id, rep.layers.len());
        std::fs::remove_dir_all(&out)?;
    }
    Ok(())
}

/// The filters, plus the failure mode that matters most in practice: a Hessian file from a different architecture must
/// make the run fail. `main` turns an empty layer list into an error, so this pins the pipeline behaviour it relies on —
/// and a width mismatch must be a hard error naming the disagreement, not a silent skip.
#[test]
fn filters_and_unmatched_hessians() -> Result<()> {
    for m in MODELS {
        let (w, h) = match inputs(m) {
            Some(p) => p,
            None => {
                skip(m, "filters");
                continue;
            }
        };

        // `--only` restricts the job list; everything else must stay byte-identical to the input.
        let out = fresh_dir(&format!("{}-only", m.dir))?;
        let rep = pipeline::quantize_model(&QuantOpts {
            weights: vec![w.clone()],
            hessians: vec![h.clone()],
            out_dir: out.clone(),
            cfg: cfg(m),
            only: vec!["down_proj".into()],
            skip: Vec::new(),
            emit_packed: true,
            emit_dequant: true,
        })?;
        if rep.layers.is_empty() {
            bail!("{}: --only down_proj quantized nothing", m.id);
        }
        if !rep.layers.iter().all(|l| l.name.contains("down_proj")) {
            let leaked: Vec<&str> = rep.layers.iter().filter(|l| !l.name.contains("down_proj")).map(|l| l.name.as_str()).collect();
            bail!("{}: --only leaked {leaked:?}", m.id);
        }
        let orig = snapshot(&w)?;
        let deq = snapshot(&out.join("dequant").join("model.safetensors"))?;
        for name in orig.tensors.keys() {
            if !name.contains("down_proj") {
                assert_identical(&orig, &deq, name)?;
            }
        }

        // `--skip` must remove exactly the layers it names and leave the rest quantized. The expected set is derived, not
        // counted from the first run: a layer survives `--skip mlp` only if the checkpoint has its tensor *and* the
        // fixture resolves a Hessian for it (directly or through an alias). That makes this a real statement about the
        // pipeline — in particular that q/k/v and gate/up keep being quantized from one shared Hessian even though only
        // `k_proj` and `gate_proj` appear in the fixture.
        let rep2 = pipeline::quantize_model(&QuantOpts {
            weights: vec![w.clone()],
            hessians: vec![h.clone()],
            out_dir: out.join("skipped"),
            cfg: cfg(m),
            only: Vec::new(),
            skip: vec!["mlp".into()],
            emit_packed: true,
            emit_dequant: true,
        })?;
        let got: HashSet<String> = rep2.layers.iter().map(|l| l.name.clone()).collect();
        if got.iter().any(|n| n.contains("mlp")) {
            bail!("{}: --skip mlp still quantized an MLP layer", m.id);
        }
        let store = HessianStore::open(&vec![h.clone()])?;
        // The report names layers without the `.weight` suffix (the packed container appends its own suffixes), so the
        // expected set is built in that same form — otherwise this comparison fails on a naming convention instead of on
        // a filtering bug.
        let want: HashSet<String> = orig
            .tensors
            .keys()
            .filter(|n| !n.contains("mlp"))
            .filter_map(|n| n.strip_suffix(".weight"))
            .filter(|l| store.resolve(l).is_some())
            .map(str::to_string)
            .collect();
        if want.is_empty() {
            bail!("{}: no non-MLP layer is calibrated by the fixture, so --skip tests nothing", m.id);
        }
        if got != want {
            let missing: Vec<&String> = want.iter().filter(|n| !got.contains(*n)).collect();
            let extra: Vec<&String> = got.iter().filter(|n| !want.contains(*n)).collect();
            bail!("{}: --skip mlp gave the wrong layer set (missing {missing:?}, unexpected {extra:?})", m.id);
        }

        // A Hessian file matching nothing yields an empty report (which `main` turns into a hard error).
        let bogus = out.join("bogus.safetensors");
        io::write_tensors(
            &bogus,
            &[io::OutTensor { name: "not.a.layer.H".into(), dtype: Dtype::F32, shape: vec![4, 4], bytes: vec![0u8; 64] }],
            None,
        )?;
        let empty = pipeline::quantize_model(&QuantOpts {
            weights: vec![w.clone()],
            hessians: vec![bogus],
            out_dir: out.join("empty"),
            cfg: cfg(m),
            only: Vec::new(),
            skip: Vec::new(),
            emit_packed: true,
            emit_dequant: true,
        })?;
        if !empty.layers.is_empty() {
            bail!("{}: an unrelated Hessian file quantized {} layers", m.id, empty.layers.len());
        }

        // A Hessian whose width disagrees with the layer must abort naming both numbers. This is the classic "captured
        // from a different model" mistake; every number downstream would be fiction.
        let mismatched = out.join("mismatch.safetensors");
        io::write_tensors(
            &mismatched,
            &[io::OutTensor { name: "model.layers.0.mlp.down_proj.H".into(), dtype: Dtype::F32, shape: vec![7, 7], bytes: vec![1u8; 4 * 49] }],
            None,
        )?;
        match pipeline::quantize_model(&QuantOpts {
            weights: vec![w.clone()],
            hessians: vec![mismatched],
            out_dir: out.join("mismatch"),
            cfg: cfg(m),
            only: Vec::new(),
            skip: Vec::new(),
            emit_packed: true,
            emit_dequant: true,
        }) {
            Ok(r) => bail!("{}: a 7x7 Hessian for down_proj was accepted ({} layers)", m.id, r.layers.len()),
            Err(e) => {
                let s = e.to_string();
                if !s.contains("Hessian") || !s.contains("down_proj") {
                    bail!("{}: width-mismatch error is not informative: {s}", m.id);
                }
            }
        }
        eprintln!("OK {}: --only/--skip honoured; unmatched and mismatched Hessians rejected", m.id);
        std::fs::remove_dir_all(&out)?;
    }
    Ok(())
}

/// The three checkpoints cover f16, bf16 and f32 storage, which is the point: widening to f32 must be lossless (the
/// quantizer has to see exactly the weights the checkpoint had) and the fake-quant output must keep the input dtype.
#[test]
fn dtypes_are_preserved_and_widening_is_lossless() -> Result<()> {
    for m in MODELS {
        let (w, h) = match inputs(m) {
            Some(p) => p,
            None => {
                skip(m, "dtype coverage");
                continue;
            }
        };
        let orig = snapshot(&w)?;
        let want = match m.dir {
            "tiny-random-Llama-3" => Dtype::F16,
            "llama-3" => Dtype::BF16,
            _ => Dtype::F32,
        };
        for (name, (dt, sh, bytes)) in &orig.tensors {
            if *dt != want {
                bail!("{}: {name} is {dt:?}, expected {want:?}", m.id);
            }
            // f16/bf16/f32 -> f32 -> back must return the original bytes. That is what "the quantizer saw the real
            // weights" means at this layer: a widening that lost bits would inject spurious error into every gain number.
            let view = TensorView::new(*dt, sh.clone(), bytes)?;
            let wide = io::tensor_to_f32(&view)?;
            if wide.len() != sh.iter().product::<usize>() {
                bail!("{}: {name} widened to {} elements for shape {sh:?}", m.id, wide.len());
            }
            if io::f32_to_bytes(*dt, &wide)? != *bytes {
                bail!("{}: {name} f32 widening round-trip is not lossless", m.id);
            }
        }

        // And the fake-quant output keeps that dtype per tensor, so it really is a drop-in checkpoint.
        let out = fresh_dir(&format!("{}-dtype", m.dir))?;
        pipeline::quantize_model(&QuantOpts {
            weights: vec![w],
            hessians: vec![h],
            out_dir: out.clone(),
            cfg: cfg(m),
            only: Vec::new(),
            skip: Vec::new(),
            emit_packed: false,
            emit_dequant: true,
        })?;
        let deq = snapshot(&out.join("dequant").join("model.safetensors"))?;
        for (name, (dt, _, _)) in &orig.tensors {
            if deq.tensors[name].0 != *dt {
                bail!("{}: fake-quant output changed {name} from {dt:?} to {:?}", m.id, deq.tensors[name].0);
            }
        }
        eprintln!("OK {}: dtype {want:?} preserved, widening lossless over {} tensors", m.id, orig.tensors.len());
        std::fs::remove_dir_all(&out)?;
    }
    Ok(())
}

/// The explicit `--dtype` override on the packed reader, and the recorded-header default. Values must not move beyond the
/// target dtype's own rounding: an override changes the container precision, never the numbers.
#[test]
fn dequant_dtype_override() -> Result<()> {
    for m in MODELS {
        let (w, h) = match inputs(m) {
            Some(p) => p,
            None => {
                skip(m, "dtype override");
                continue;
            }
        };
        let out = fresh_dir(&format!("{}-ovr", m.dir))?;
        pipeline::quantize_model(&QuantOpts {
            weights: vec![w],
            hessians: vec![h],
            out_dir: out.clone(),
            cfg: cfg(m),
            only: Vec::new(),
            skip: Vec::new(),
            emit_packed: true,
            emit_dequant: false,
        })?;
        let packed = out.join("packed").join("model.safetensors");

        // Only the quantized layers are affected by the override: pass-through tensors keep their own dtype, which is
        // what a reader expects from a file that still contains embeddings and norms.
        let f32_out = out.join("as_f32.safetensors");
        pipeline::dequantize_packed(&packed, &f32_out, Some(Dtype::F32))?;
        let fs = snapshot(&f32_out)?;
        let native = out.join("native.safetensors");
        pipeline::dequantize_packed(&packed, &native, None)?;
        let ns = snapshot(&native)?;

        let pk = snapshot(&packed)?;
        let quantized: HashSet<String> = pk.tensors.keys().filter_map(|n| n.strip_suffix(".qweight").map(|l| format!("{l}.weight"))).collect();
        if quantized.is_empty() {
            bail!("{}: packed file has no qweight tensors", m.id);
        }
        let mut native_dt = None;
        for name in &quantized {
            if fs.tensors[name].0 != Dtype::F32 {
                bail!("{}: --dtype F32 ignored for quantized layer {name} (got {:?})", m.id, fs.tensors[name].0);
            }
            native_dt = Some(ns.tensors[name].0);
            let a = f32_of(&fs, name)?;
            let b = f32_of(&ns, name)?;
            if a.len() != b.len() {
                bail!("{}: override round-trip shape mismatch on {name}", m.id);
            }
            // Tolerance is the native dtype's own relative precision (f16 ~5e-4, bf16 ~4e-3). Anything larger means the
            // reader recomputed values instead of re-storing them.
            let tol = match ns.tensors[name].0 {
                Dtype::F16 => 2e-3,
                Dtype::BF16 => 2e-2,
                _ => 0.0,
            };
            for (x, y) in a.iter().zip(&b) {
                let d = (*x as f64 - *y as f64).abs();
                if d > tol * x.abs() as f64 + 1e-9 {
                    bail!("{}: {name} value moved because of the dtype override: {x} vs {y}", m.id);
                }
            }
        }
        // Pass-through tensors are untouched by the override, in both files.
        for name in fs.tensors.keys() {
            if !quantized.contains(name) && fs.tensors[name].0 != ns.tensors[name].0 {
                bail!("{}: pass-through tensor {name} changed dtype under --dtype", m.id);
            }
        }
        eprintln!("OK {}: dtype override works (native {:?}, {} quantized layers)", m.id, native_dt, quantized.len());
        std::fs::remove_dir_all(&out)?;
    }
    Ok(())
}

/// Every bit width the crate claims to support, on a real checkpoint. Not about quality — about `gbits` staying
/// consistent and about the reported bits/weight matching what the packed file actually occupies. The second point matters
/// because the README quotes measured bpw: if it were computed from `--bits` instead, the two would drift apart.
#[test]
fn all_bit_widths_pack_consistently() -> Result<()> {
    let m = &MODELS[2];
    let (w, h) = match inputs(m) {
        Some(p) => p,
        None => {
            skip(m, "bit-width sweep");
            return Ok(());
        }
    };
    for bits in 2..=8u32 {
        let mut c = cfg(m);
        c.bits = bits;
        let out = fresh_dir(&format!("{}-b{bits}", m.dir))?;
        let rep = pipeline::quantize_model(&QuantOpts {
            weights: vec![w.clone()],
            hessians: vec![h.clone()],
            out_dir: out.clone(),
            cfg: c.clone(),
            only: Vec::new(),
            skip: Vec::new(),
            emit_packed: true,
            emit_dequant: true,
        })?;
        if rep.layers.is_empty() {
            bail!("{}: the {bits}-bit run quantized nothing", m.id);
        }
        let pk_path = out.join("packed").join("model.safetensors");
        let pk = snapshot(&pk_path)?;
        let pk_mm = io::mmap_file(&pk_path)?;
        let pk_st = SafeTensors::deserialize(&pk_mm)?;
        for l in &rep.layers {
            let ql = io::read_packed_layer(&pk_st, &l.name)?;
            if !ql.gbits.iter().all(|&b| b == bits as u8) {
                bail!("{}: the {bits}-bit run gave {} mixed gbits {:?}", m.id, l.name, ql.gbits);
            }
            let qw = &pk.tensors[&format!("{}.qweight", l.name)].2;
            let sc = &pk.tensors[&format!("{}.scales", l.name)].2;
            let zr = &pk.tensors[&format!("{}.qzeros", l.name)].2;
            // Codes: bits actually occupied by the bit-stream vs the report's average. The row stride is rounded up to a
            // whole byte, so the real cost can be marginally *higher* than `bits` for narrow ragged layers — hence 0.05
            // rather than exact equality. Anything larger means the packer and the report disagree about the format.
            let measured = (qw.len() * 8) as f64 / (l.d_out * l.d_in) as f64;
            if (measured - l.bits_codes).abs() > 0.05 {
                bail!("{}: the {bits}-bit layer {} occupies {measured:.3} bpw of codes, report says {:.3}", m.id, l.name, l.bits_codes);
            }
            // Metadata: one f16 scale plus one u8 zero point per (row, group) = 24 bits spread over the group. The
            // expected cost is computed from the *actual* group count of this layer, not from `--group`: a ragged last
            // group (down_proj at 172 columns with group 32 has six groups, the last only 12 wide) amortizes those 24
            // bits over fewer columns, so the real cost sits above the nominal `24 / group`. That gap is a documented
            // approximation in `bits_total`, not an error, so this assertion checks the file against the geometry and
            // leaves the estimate alone.
            let meta = ((sc.len() + zr.len()) * 8) as f64 / (l.d_out * l.d_in) as f64;
            let ng = ng_of_layer(&pk_st, &l.name)?;
            let want = 24.0 * ng as f64 / l.d_in as f64;
            if (meta - want).abs() > 1e-3 {
                bail!(
                    "{}: the {bits}-bit layer {} costs {meta:.4} bpw of metadata, {want:.4} expected from {ng} groups over {} columns",
                    m.id, l.name, l.d_in
                );
            }
        }
        std::fs::remove_dir_all(&out)?;
    }
    eprintln!("OK {}: bits 2..=8 pack consistently and bpw is measured, not nominal", m.id);
    Ok(())
}
