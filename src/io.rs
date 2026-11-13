//! safetensors I/O: dtype conversion, Hessian store (with aliases), packed-layer (de)serialisation.

use crate::quant::QLayer;
use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

pub fn mmap_file(p: &Path) -> Result<Mmap> {
    let f = File::open(p).with_context(|| format!("cannot open {}", p.display()))?;
    // SAFETY: the file is only read; callers must not truncate it while the map is alive.
    unsafe { Mmap::map(&f) }.with_context(|| format!("cannot mmap {}", p.display()))
}

pub fn tensor_to_f32(t: &TensorView) -> Result<Vec<f32>> {
    let d = t.data();
    Ok(match t.dtype() {
        Dtype::F32 => d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        Dtype::F16 => d.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        Dtype::BF16 => d.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        other => bail!("unsupported weight dtype {other:?} (need F32/F16/BF16)"),
    })
}

pub fn f32_to_bytes(dtype: Dtype, v: &[f32]) -> Result<Vec<u8>> {
    Ok(match dtype {
        Dtype::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Dtype::F16 => v.iter().flat_map(|&x| f16::from_f32(x).to_le_bytes()).collect(),
        Dtype::BF16 => v.iter().flat_map(|&x| bf16::from_f32(x).to_le_bytes()).collect(),
        other => bail!("unsupported output dtype {other:?}"),
    })
}

pub fn dtype_name(d: Dtype) -> &'static str {
    match d {
        Dtype::F32 => "F32",
        Dtype::F16 => "F16",
        Dtype::BF16 => "BF16",
        _ => "OTHER",
    }
}

pub fn dtype_from_name(s: &str) -> Option<Dtype> {
    match s.to_ascii_uppercase().as_str() {
        "F32" | "FLOAT32" => Some(Dtype::F32),
        "F16" | "FLOAT16" | "HALF" => Some(Dtype::F16),
        "BF16" | "BFLOAT16" => Some(Dtype::BF16),
        _ => None,
    }
}

pub struct OutTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

pub fn write_tensors(path: &Path, tensors: &[OutTensor], metadata: Option<HashMap<String, String>>) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut views: Vec<(String, TensorView)> = Vec::with_capacity(tensors.len());
    for t in tensors {
        let v = TensorView::new(t.dtype, t.shape.clone(), &t.bytes)
            .map_err(|e| anyhow!("tensor {}: {e:?}", t.name))?;
        views.push((t.name.clone(), v));
    }
    safetensors::serialize_to_file(views, &metadata, path).map_err(|e| anyhow!("writing {}: {e:?}", path.display()))?;
    Ok(())
}

/// Calibration Hessians: `<layer>.H` (f32 `n x n`), optional `<layer>.M` (cross-covariance),
/// plus an `aliases` metadata entry (JSON `{layer: owner_layer}`) for layers that share inputs.
pub struct HessianStore {
    maps: Vec<Mmap>,
    index: HashMap<String, usize>,
    aliases: HashMap<String, String>,
}

impl HessianStore {
    pub fn open(paths: &[PathBuf]) -> Result<Self> {
        let mut maps = Vec::new();
        let mut index = HashMap::new();
        let mut aliases = HashMap::new();
        for (fi, p) in paths.iter().enumerate() {
            let mm = mmap_file(p)?;
            {
                let st = SafeTensors::deserialize(&mm).map_err(|e| anyhow!("{}: {e:?}", p.display()))?;
                for name in st.names() {
                    if let Some(layer) = name.strip_suffix(".H") {
                        index.insert(layer.to_string(), fi);
                    }
                }
                let (_, md) = SafeTensors::read_metadata(&mm).map_err(|e| anyhow!("{}: {e:?}", p.display()))?;
                if let Some(meta) = md.metadata() {
                    if let Some(js) = meta.get("aliases") {
                        let a: HashMap<String, String> =
                            serde_json::from_str(js).with_context(|| "invalid 'aliases' metadata")?;
                        aliases.extend(a);
                    }
                }
            }
            maps.push(mm);
        }
        Ok(HessianStore { maps, index, aliases })
    }

    /// Layer whose `.H` tensor serves `layer` (itself or its alias target).
    pub fn resolve(&self, layer: &str) -> Option<String> {
        if self.index.contains_key(layer) {
            return Some(layer.to_string());
        }
        let o = self.aliases.get(layer)?;
        if self.index.contains_key(o.as_str()) {
            Some(o.clone())
        } else {
            None
        }
    }

    fn tensor(&self, owner: &str, suffix: &str) -> Result<Option<(Vec<f32>, usize)>> {
        let Some(&fi) = self.index.get(owner) else { bail!("no Hessian for {owner}") };
        let st = SafeTensors::deserialize(&self.maps[fi]).map_err(|e| anyhow!("{e:?}"))?;
        let key = format!("{owner}.{suffix}");
        let Ok(t) = st.tensor(&key) else { return Ok(None) };
        let sh = t.shape();
        if sh.len() != 2 || sh[0] != sh[1] {
            bail!("{key}: expected a square matrix, got {sh:?}");
        }
        Ok(Some((tensor_to_f32(&t)?, sh[0])))
    }

    pub fn get_h(&self, owner: &str) -> Result<(Vec<f32>, usize)> {
        self.tensor(owner, "H")?.ok_or_else(|| anyhow!("missing {owner}.H"))
    }

    pub fn get_m(&self, owner: &str) -> Result<Option<Vec<f32>>> {
        Ok(self.tensor(owner, "M")?.map(|x| x.0))
    }
}

/// Tensors describing one packed layer.
pub fn packed_tensors(layer: &str, ql: &QLayer) -> Vec<OutTensor> {
    let (qw, rb) = ql.pack();
    let ng = ql.ngroups();
    let scales: Vec<u8> = ql.scales.iter().flat_map(|&s| f16::from_f32(s).to_le_bytes()).collect();
    let zeros: Vec<u8> = ql.zeros.iter().map(|&z| z as u8).collect();
    let perm: Vec<u8> = ql.perm.iter().flat_map(|&p| (p as i32).to_le_bytes()).collect();
    vec![
        OutTensor { name: format!("{layer}.qweight"), dtype: Dtype::U8, shape: vec![ql.d_out, rb], bytes: qw },
        OutTensor { name: format!("{layer}.scales"), dtype: Dtype::F16, shape: vec![ql.d_out, ng], bytes: scales },
        OutTensor { name: format!("{layer}.qzeros"), dtype: Dtype::U8, shape: vec![ql.d_out, ng], bytes: zeros },
        OutTensor { name: format!("{layer}.perm"), dtype: Dtype::I32, shape: vec![ql.d_in], bytes: perm },
        OutTensor { name: format!("{layer}.gbits"), dtype: Dtype::U8, shape: vec![ng], bytes: ql.gbits.clone() },
        OutTensor {
            name: format!("{layer}.gsize"),
            dtype: Dtype::I32,
            shape: vec![1],
            bytes: (ql.group as i32).to_le_bytes().to_vec(),
        },
    ]
}

/// Rebuild a `QLayer` from the tensors written by [`packed_tensors`].
pub fn read_packed_layer(st: &SafeTensors, layer: &str) -> Result<QLayer> {
    let get = |suffix: &str| -> Result<TensorView> {
        st.tensor(&format!("{layer}.{suffix}")).map_err(|e| anyhow!("{layer}.{suffix}: {e:?}"))
    };
    let qw = get("qweight")?;
    let sc = get("scales")?;
    let qz = get("qzeros")?;
    let pm = get("perm")?;
    let gb = get("gbits")?;
    let gs = get("gsize")?;
    let d_out = qw.shape()[0];
    let d_in = pm.shape()[0];
    let group = i32::from_le_bytes(gs.data()[..4].try_into().unwrap()) as usize;
    let perm: Vec<u32> = pm.data().chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32).collect();
    let scales: Vec<f32> = sc.data().chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
    let zeros: Vec<f32> = qz.data().iter().map(|&z| z as f32).collect();
    Ok(QLayer::from_packed(d_out, d_in, group, perm, gb.data().to_vec(), qw.data(), scales, zeros))
}
