//! Model-level orchestration: walk safetensors shards, quantize every linear layer that has a
//! calibration Hessian, emit a packed model and/or a dequantized ("fake-quant") model.

use crate::engine::{quantize_layer, Config, PrepCache};
use crate::io::*;
use anyhow::{anyhow, bail, Result};
use safetensors::{Dtype, SafeTensors};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

pub struct QuantOpts {
    pub weights: Vec<PathBuf>,
    pub hessians: Vec<PathBuf>,
    pub out_dir: PathBuf,
    pub cfg: Config,
    /// If non-empty, only layers whose name contains one of these substrings are quantized.
    pub only: Vec<String>,
    /// Layers whose name contains one of these substrings are left untouched.
    pub skip: Vec<String>,
    pub emit_packed: bool,
    pub emit_dequant: bool,
}

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

#[derive(Serialize)]
pub struct Report {
    pub config: String,
    pub layers: Vec<LayerReport>,
    pub total_loss_gptq: f64,
    pub total_loss_final: f64,
    pub total_gain_pct: f64,
    pub total_secs: f64,
}

pub fn quantize_model(o: &QuantOpts) -> Result<Report> {
    let t_all = Instant::now();
    let store = HessianStore::open(&o.hessians)?;
    let mut layers: Vec<LayerReport> = Vec::new();

    for shard in &o.weights {
        let mm = mmap_file(shard)?;
        let st = SafeTensors::deserialize(&mm).map_err(|e| anyhow!("{}: {e:?}", shard.display()))?;
        let mut names: Vec<String> = st.names().into_iter().cloned().collect();
        names.sort();

        // layer jobs: 2-D `.weight` tensors with a Hessian
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
        // layers sharing a Hessian are processed back to back -> one factorisation for q/k/v, gate/up
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

        // pass-through for every tensor that was not quantized
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
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("format".into(), "pt".into());
        if o.emit_dequant && !deq_out.is_empty() {
            write_tensors(&o.out_dir.join("dequant").join(fname), &deq_out, Some(meta.clone()))?;
        }
        if o.emit_packed && !pk_out.is_empty() {
            meta.insert("quantization".into(), "cdclq-v1".into());
            meta.insert("orig_dtype".into(), dtype_name(orig_dtype.unwrap_or(Dtype::F16)).into());
            meta.insert("bits".into(), o.cfg.bits.to_string());
            write_tensors(&o.out_dir.join("packed").join(fname), &pk_out, Some(meta))?;
        }
    }

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
pub fn dequantize_packed(input: &PathBuf, output: &PathBuf, dtype: Option<Dtype>) -> Result<usize> {
    let mm = mmap_file(input)?;
    let st = SafeTensors::deserialize(&mm).map_err(|e| anyhow!("{}: {e:?}", input.display()))?;
    let (_, md) = SafeTensors::read_metadata(&mm).map_err(|e| anyhow!("{e:?}"))?;
    let meta_dtype = md.metadata().as_ref().and_then(|m| m.get("orig_dtype")).and_then(|s| dtype_from_name(s));
    let dt = dtype.or(meta_dtype).unwrap_or(Dtype::F16);

    let names: Vec<String> = st.names().into_iter().cloned().collect();
    let layers: Vec<String> = names.iter().filter_map(|n| n.strip_suffix(".qweight").map(String::from)).collect();
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
