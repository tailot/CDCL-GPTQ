//! Model-level orchestration: walk safetensors shards, quantize every linear layer that has a calibration
//! Hessian, emit a packed model and/or a dequantized ("fake-quant") model.
//!
//! The whole crate's policy lives here, and it is deliberately simple and conservative:
//!
//! * A tensor is quantized only if it is 2-D, named `*.weight`, passes the `--only`/`--skip` filters **and** a
//!   matching Hessian was captured for it. Anything else — embeddings, `lm_head`, LayerNorm/RMSNorm gains, biases,
//!   rotary buffers — is copied through byte for byte. That is why the reported compression ratio is bounded by the
//!   embedding share of the parameter count rather than being inflated by pretending those tensors were quantized.
//! * Both outputs are produced in the same pass over the shard, so a run can be evaluated two ways (packed size and
//!   fake-quant accuracy) without ever re-quantizing — important because a real layer takes seconds, not microseconds.
//! * Everything is streamed shard by shard with memory-mapped inputs, so a 70B model needs roughly one shard of RAM
//!   plus the Hessians, not the whole checkpoint.

use crate::engine::{quantize_layer, Config, PrepCache};
use crate::io::*;
use anyhow::{anyhow, bail, Result};
use safetensors::{Dtype, SafeTensors};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

/// Everything the user chose on the command line that affects the output.
pub struct QuantOpts {
    pub weights: Vec<PathBuf>,
    pub hessians: Vec<PathBuf>,
    pub out_dir: PathBuf,
    pub cfg: Config,
    /// If non-empty, only layers whose name contains one of these substrings are quantized.
    pub only: Vec<String>,
    /// Layers whose name contains one of these substrings are left untouched.
    pub skip: Vec<String>,
    /// Write `<out_dir>/packed/`: the bit-packed container this crate defines.
    pub emit_packed: bool,
    /// Write `<out_dir>/dequant/`: same shapes and dtypes as the input, values snapped to the quantization grid.
    /// This is what `scripts/eval_ppl.py` loads, and what makes the accuracy claims falsifiable.
    pub emit_dequant: bool,
}

/// Per-layer audit record. Written to `report.json` so a gain claim can be re-checked layer by layer instead of
/// being a single number in a terminal; `loss_gptq` / `loss_final` are the proxy losses *before* and *after* the
/// repair phase, both computed incrementally inside the engine.
#[derive(Serialize, Clone)]
pub struct LayerReport {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    pub loss_gptq: f64,
    pub loss_final: f64,
    pub gain_pct: f64,
    pub subblocks_refined: usize,
    pub subblocks_total: usize,
    pub code_updates: usize,
    pub skeleton_groups: usize,
    pub bits_codes: f64,
    pub bits_total: f64,
    pub secs_prepare: f64,
    pub secs_quant: f64,
}

/// Aggregate report for one `quantize` invocation (`config` is the debug-formatted [`Config`], so a report file is
/// self-describing and two runs can be diffed).
#[derive(Serialize)]
pub struct Report {
    pub config: String,
    pub layers: Vec<LayerReport>,
    pub total_loss_gptq: f64,
    pub total_loss_final: f64,
    pub total_gain_pct: f64,
    pub total_secs: f64,
}

/// Quantize every eligible layer of every shard and write the requested outputs.
///
/// Note what is *not* parallel here: layers are processed sequentially while the engine parallelizes across rows
/// inside each layer. Two rayon pools competing over the same cores would only thrash, and one 2048-wide layer
/// already saturates the machine on a desktop CPU.
pub fn quantize_model(o: &QuantOpts) -> Result<Report> {
    let t_all = Instant::now();
    let store = HessianStore::open(&o.hessians)?;
    let mut layers: Vec<LayerReport> = Vec::new();

    for shard in &o.weights {
        let mm = mmap_file(shard)?;
        let st = SafeTensors::deserialize(&mm).map_err(|e| anyhow!("{}: {e:?}", shard.display()))?;
        let mut names: Vec<String> = st.names().into_iter().cloned().collect();
        names.sort();

        // Layer jobs: 2-D `.weight` tensors that have a Hessian. The third element is the *owner* of that Hessian,
        // which is what lets q/k/v (or gate/up) share one factorisation instead of doing it three times.
        let mut jobs: Vec<(String, String, String)> = Vec::new(); // (tensor name, layer, hessian owner)
        for name in &names {
            let Some(layer) = name.strip_suffix(".weight") else { continue };
            let t = st.tensor(name).map_err(|e| anyhow!("{name}: {e:?}"))?;
            if t.shape().len() != 2 {
                continue;
            }
            if !o.only.is_empty() && !o.only.iter().any(|s| layer.contains(s.as_str())) {
                continue;
            }
            if o.skip.iter().any(|s| layer.contains(s.as_str())) {
                continue;
            }
            if let Some(owner) = store.resolve(layer) {
                jobs.push((name.clone(), layer.to_string(), owner));
            }
        }
        // Layers sharing a Hessian are processed back to back, so the single-entry PrepCache in `cache` hits: one
        // damping + Cholesky + permutation per owner instead of per layer. The engine additionally verifies that the
        // permutation matches before reusing anything.
        jobs.sort_by(|a, b| a.2.cmp(&b.2).then(a.1.cmp(&b.1)));

        let mut deq_out: Vec<OutTensor> = Vec::new();
        let mut pk_out: Vec<OutTensor> = Vec::new();
        let mut done: HashSet<String> = HashSet::new();
        let mut cache: PrepCache = None;
        let mut orig_dtype: Option<Dtype> = None;
        let njobs = jobs.len();

        for (ji, (name, layer, owner)) in jobs.iter().enumerate() {
            let t = st.tensor(name).map_err(|e| anyhow!("{name}: {e:?}"))?;
            let (d_out, d_in) = (t.shape()[0], t.shape()[1]);
            let w = tensor_to_f32(&t)?;
            let (h, hn) = store.get_h(owner)?;
            if hn != d_in {
                bail!("{layer}: Hessian is {hn}x{hn} but the weight has {d_in} input columns");
            }
            let m = store.get_m(owner)?;
            let (ql, s) = quantize_layer(&w, d_out, d_in, &h, m.as_deref(), owner, &o.cfg, &mut cache)?;
            drop(h);
            let gain = if s.loss_gptq > 0.0 { 100.0 * (s.loss_gptq - s.loss_final) / s.loss_gptq } else { 0.0 };
            eprintln!(
                "[{:>3}/{:<3}] {layer} [{d_out}x{d_in}] loss {:.4e} -> {:.4e} ({:+.1}%) | {:.2} bpw | prep {:.1}s quant {:.1}s",
                ji + 1,
                njobs,
                s.loss_gptq,
                s.loss_final,
                -gain,
                s.bits_total,
                s.secs_prepare,
                s.secs_quant
            );
            layers.push(LayerReport {
                name: layer.clone(),
                d_out,
                d_in,
                loss_gptq: s.loss_gptq,
                loss_final: s.loss_final,
                gain_pct: gain,
                subblocks_refined: s.subblocks_refined,
                subblocks_total: s.subblocks_total,
                code_updates: s.code_updates,
                skeleton_groups: s.skeleton_groups,
                bits_codes: s.bits_codes,
                bits_total: s.bits_total,
                secs_prepare: s.secs_prepare,
                secs_quant: s.secs_quant,
            });
            orig_dtype.get_or_insert(t.dtype());
            if o.emit_dequant {
                // Keep the original dtype (f16/bf16/f32): the fake-quant model must be a drop-in replacement for
                // the checkpoint, loadable by an unmodified inference stack.
                deq_out.push(OutTensor {
                    name: name.clone(),
                    dtype: t.dtype(),
                    shape: vec![d_out, d_in],
                    bytes: f32_to_bytes(t.dtype(), &ql.dequantize())?,
                });
            }
            if o.emit_packed {
                pk_out.extend(packed_tensors(layer, &ql));
            }
            done.insert(name.clone());
        }

        // Pass-through for every tensor that was not quantized (embeddings, lm_head, norms, rotary caches ...).
        // `done` is the single source of truth for "already handled", so nothing can be emitted twice and nothing
        // can silently disappear from the output model.
        for name in &names {
            if done.contains(name) {
                continue;
            }
            let t = st.tensor(name).map_err(|e| anyhow!("{name}: {e:?}"))?;
            let mk = || OutTensor { name: name.clone(), dtype: t.dtype(), shape: t.shape().to_vec(), bytes: t.data().to_vec() };
            if o.emit_dequant {
                deq_out.push(mk());
            }
            if o.emit_packed {
                pk_out.push(mk());
            }
        }

        let fname = shard.file_name().ok_or_else(|| anyhow!("bad shard path"))?;
        // `format: pt` keeps the output loadable by HF tooling that inspects the safetensors header.
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("format".into(), "pt".into());
        if o.emit_dequant && !deq_out.is_empty() {
            write_tensors(&o.out_dir.join("dequant").join(fname), &deq_out, Some(meta.clone()))?;
        }
        if o.emit_packed && !pk_out.is_empty() {
            // The packed header records the original dtype so `cdclq dequant` can restore it without being told,
            // and a format tag so a reader can tell a cdclq shard from an ordinary checkpoint.
            meta.insert("quantization".into(), "cdclq-v1".into());
            meta.insert("orig_dtype".into(), dtype_name(orig_dtype.unwrap_or(Dtype::F16)).into());
            meta.insert("bits".into(), o.cfg.bits.to_string());
            write_tensors(&o.out_dir.join("packed").join(fname), &pk_out, Some(meta))?;
        }
    }

    // Aggregate over layers. The proxy loss is additive across layers by construction (each layer has its own H and
    // its own error matrix), so a plain sum is the correct model-level objective — this is also the number `bench`
    // compares between methods.
    let tg: f64 = layers.iter().map(|l| l.loss_gptq).sum();
    let tf: f64 = layers.iter().map(|l| l.loss_final).sum();
    let report = Report {
        config: format!("{:?}", o.cfg),
        total_gain_pct: if tg > 0.0 { 100.0 * (tg - tf) / tg } else { 0.0 },
        total_loss_gptq: tg,
        total_loss_final: tf,
        total_secs: t_all.elapsed().as_secs_f64(),
        layers,
    };
    std::fs::create_dir_all(&o.out_dir)?;
    std::fs::write(o.out_dir.join("report.json"), serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

/// Convert a packed model shard back into a dequantized safetensors file (original column order).
///
/// This is the reader half of the format and the basis of the correctness audit: for every `X.qweight` in the input
/// it rebuilds a [`QLayer`] from the six companion tensors and writes `X.weight` with the original shape and dtype.
/// The acceptance test is that this output matches, tensor by tensor, what `QLayer::dequantize()` produced during
/// quantization — i.e. packing and unpacking are exact inverses (`max |dequant - roundtrip| = 0`), so any accuracy
/// loss in the pipeline comes from quantization itself and not from the container.
///
/// The dtype resolution order is: explicit `--dtype`, then the `orig_dtype` recorded in the packed header, then f16.
/// Returns how many layers were expanded, which the caller prints as a sanity check against the expected layer count.
pub fn dequantize_packed(input: &PathBuf, output: &PathBuf, dtype: Option<Dtype>) -> Result<usize> {
    let mm = mmap_file(input)?;
    let st = SafeTensors::deserialize(&mm).map_err(|e| anyhow!("{}: {e:?}", input.display()))?;
    let (_, md) = SafeTensors::read_metadata(&mm).map_err(|e| anyhow!("{e:?}"))?;
    let meta_dtype = md.metadata().as_ref().and_then(|m| m.get("orig_dtype")).and_then(|s| dtype_from_name(s));
    let dt = dtype.or(meta_dtype).unwrap_or(Dtype::F16);

    let names: Vec<String> = st.names().into_iter().cloned().collect();
    let layers: Vec<String> = names.iter().filter_map(|n| n.strip_suffix(".qweight").map(String::from)).collect();
    // Every companion tensor of a layer we expanded must be dropped, otherwise the output would contain both the
    // packed and the dequantized representation of the same layer. Suffix-matching is enough because the writer in
    // `packed_tensors` emits exactly these six names per layer.
    let packed_suffixes = ["qweight", "scales", "qzeros", "perm", "gbits", "gsize"];
    let mut out: Vec<OutTensor> = Vec::new();
    for l in &layers {
        let ql = read_packed_layer(&st, l)?;
        out.push(OutTensor {
            name: format!("{l}.weight"),
            dtype: dt,
            shape: vec![ql.d_out, ql.d_in],
            bytes: f32_to_bytes(dt, &ql.dequantize())?,
        });
    }
    let lset: HashSet<&String> = layers.iter().collect();
    for n in &names {
        let is_packed = packed_suffixes.iter().any(|s| n.strip_suffix(&format!(".{s}")).map_or(false, |l| lset.contains(&l.to_string())));
        if is_packed {
            continue;
        }
        let t = st.tensor(n).map_err(|e| anyhow!("{n}: {e:?}"))?;
        out.push(OutTensor { name: n.clone(), dtype: t.dtype(), shape: t.shape().to_vec(), bytes: t.data().to_vec() });
    }
    let mut meta = HashMap::new();
    meta.insert("format".to_string(), "pt".to_string());
    write_tensors(output, &out, Some(meta))?;
    Ok(layers.len())
}
