//! safetensors I/O: dtype conversion, Hessian store (with aliases), packed-layer (de)serialisation.
//!
//! Three concerns live here, all of them deliberately thin so the *policy* stays in `pipeline` and the *math* stays
//! in `engine`:
//!
//! 1. dtype bridging — checkpoints arrive as f16/bf16/f32, the engine works in f32, outputs go back to the original
//!    dtype. Conversion is explicit and little-endian, matching the safetensors wire format.
//! 2. [`HessianStore`] — a memory-mapped index over one or more Hessian files, with alias resolution so layers that
//!    share an input (q/k/v, gate/up) reference one matrix instead of storing it three times.
//! 3. the packed layer format — six tensors per layer, described in [`packed_tensors`].
//!
//! Files are always read through `mmap`: a Hessian for a 8192-wide model is hundreds of MB and a checkpoint shard is
//! gigabytes, and the OS page cache handles eviction far better than an explicit read would.

use crate::quant::QLayer;
use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Open a file as a memory map. Callers must not modify or truncate the file while the map is alive.
pub fn mmap_file(p: &Path) -> Result<Mmap> {
    let f = File::open(p).with_context(|| format!("cannot open {}", p.display()))?;
    // SAFETY: the file is only read; callers must not truncate it while the map is alive.
    unsafe { Mmap::map(&f) }.with_context(|| format!("cannot mmap {}", p.display()))
}

/// Widen any supported tensor dtype to f32 (the engine's working precision).
///
/// Manual little-endian decoding rather than a slice cast, because the data comes from a byte view of an mmap and its
/// alignment is not guaranteed. f16/bf16 -> f32 is lossless in this direction, which matters: the quantizer must see
/// exactly the weights the original model had, or the "gain" numbers would include a spurious conversion error.
pub fn tensor_to_f32(t: &TensorView) -> Result<Vec<f32>> {
    let d = t.data();
    Ok(match t.dtype() {
        Dtype::F32 => d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        Dtype::F16 => d.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        Dtype::BF16 => d.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        other => bail!("unsupported weight dtype {other:?} (need F32/F16/BF16)"),
    })
}

/// Narrow f32 back to `dtype` for output. The dtype is whatever the input tensor had, so a fake-quant checkpoint is a
/// drop-in replacement: same names, shapes and dtypes, only the values are snapped to the quantization grid. Note this
/// rounds *once*, from the exact f32 dequantized value — never from an already-rounded intermediate.
pub fn f32_to_bytes(dtype: Dtype, v: &[f32]) -> Result<Vec<u8>> {
    Ok(match dtype {
        Dtype::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Dtype::F16 => v.iter().flat_map(|&x| f16::from_f32(x).to_le_bytes()).collect(),
        Dtype::BF16 => v.iter().flat_map(|&x| bf16::from_f32(x).to_le_bytes()).collect(),
        other => bail!("unsupported output dtype {other:?}"),
    })
}

/// Stable name for a dtype, written into the packed header (`orig_dtype`).
pub fn dtype_name(d: Dtype) -> &'static str {
    match d {
        Dtype::F32 => "F32",
        Dtype::F16 => "F16",
        Dtype::BF16 => "BF16",
        _ => "OTHER",
    }
}

/// Inverse of [`dtype_name`], also accepting the spellings HF tooling uses, so `cdclq dequant` can recover the
/// original dtype from a packed file without a `--dtype` flag. Unknown names return `None` (the caller falls back).
pub fn dtype_from_name(s: &str) -> Option<Dtype> {
    match s.to_ascii_uppercase().as_str() {
        "F32" | "FLOAT32" => Some(Dtype::F32),
        "F16" | "FLOAT16" | "HALF" => Some(Dtype::F16),
        "BF16" | "BFLOAT16" => Some(Dtype::BF16),
        _ => None,
    }
}

/// One tensor ready to be written out: name, dtype, shape and already-encoded little-endian bytes. The pipeline uses
/// this so it can accumulate a whole shard's output in memory and write the file in a single pass (safetensors needs
/// all tensors up front to lay out the header).
pub struct OutTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// Serialize tensors to a safetensors file, creating parent directories as needed.
///
/// `metadata` is the optional string-to-string header block (`format`, `quantization`, `orig_dtype`, `aliases`). The
/// shape/byte-length agreement is checked by [`TensorView::new`] before anything is written, and the tensor name is
/// put in the error message because a mismatch inside a 200-tensor shard is otherwise impossible to locate.
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

/// Calibration Hessians: `<layer>.H` (f32 `d_in x d_in`), optional `<layer>.M` (cross-covariance),
/// plus an `aliases` metadata entry (JSON `{layer: owner_layer}`) for layers that share inputs.
///
/// The capture script also writes `format`, `generator` and `calib_tokens` — the token count `n`, which bounds
/// `rank(H) <= n` and is otherwise recoverable only from the file name. The loader reads `aliases` and ignores every
/// other key, so the header can grow without breaking older binaries.
///
/// Index of the first non-finite element, if any. Scanned at load time rather than left to the factorization: a NaN or
/// Inf in a captured statistic reaches Cholesky as a non-positive pivot, and the hint there ("increase --damp or check
/// the Hessian") points at a knob that cannot repair a corrupt capture. The cost is one linear pass over the matrix,
/// negligible next to the Cholesky the same layer pays anyway.
fn first_non_finite(v: &[f32]) -> Option<usize> {
    v.iter().position(|&x| !x.is_finite())
}

/// Calibration Hessians: `<layer>.H` (f32 `d_in x d_in`), optional `<layer>.M` (cross-covariance),
/// plus an `aliases` metadata entry (JSON `{layer: owner_layer}`) for layers that share inputs.
///
/// The capture script also writes `format`, `generator` and `calib_tokens` — the token count `n`, which bounds
/// `rank(H) <= n` and is otherwise recoverable only from the file name. The loader reads `aliases` and ignores every
/// other key, so the header can grow without breaking older binaries.
///
/// The store keeps the files mapped and only indexes names at open time, so a Hessian file can be gigabytes without
/// costing RAM up front; matrices are decoded on demand, one layer at a time, by [`HessianStore::get_h`]. Aliases are
/// what makes that cheap for grouped-query attention: `q_proj`, `k_proj` and `v_proj` all consume the same hidden
/// state, so only one of them stores a `d_in x d_in` matrix and the other two resolve to it. Without this, a 70B
/// model's Hessian file would be several times larger than the checkpoint itself.
pub struct HessianStore {
    maps: Vec<Mmap>,
    /// layer name -> index into `maps` (only layers that *own* an `.H`).
    index: HashMap<String, usize>,
    /// dependent layer -> owner layer, from the file's metadata.
    aliases: HashMap<String, String>,
}

impl HessianStore {
    /// Open every Hessian file and build the name index. Later files override earlier ones for the same layer name,
    /// which lets a user pass a small hand-written override after the generated file.
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

    /// Layer whose `.H` tensor serves `layer` (itself or its alias target). Returning the *owner* name is what the
    /// pipeline uses as the PrepCache key, so q/k/v also share the damped copy and the Cholesky factor.
    /// `None` means "no calibration data for this layer", which makes the pipeline leave it untouched rather than
    /// guessing an identity Hessian (that would silently degrade into RTN).
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

    /// Fetch one square matrix for `owner`. A missing optional tensor (`M`) is `Ok(None)`; a present-but-wrong-shaped
    /// one is a hard error, because a non-square Hessian means the capture script and the checkpoint disagree about
    /// the model and every downstream number would be meaningless. Non-finite entries are also a hard error here, named
    /// by tensor and position: they mean a corrupt capture or a diverged forward, and letting them through only moves
    /// the failure into Cholesky with a misleading hint.
    fn tensor(&self, owner: &str, suffix: &str) -> Result<Option<(Vec<f32>, usize)>> {
        let Some(&fi) = self.index.get(owner) else { bail!("no Hessian for {owner}") };
        let st = SafeTensors::deserialize(&self.maps[fi]).map_err(|e| anyhow!("{e:?}"))?;
        let key = format!("{owner}.{suffix}");
        let Ok(t) = st.tensor(&key) else { return Ok(None) };
        let sh = t.shape();
        if sh.len() != 2 || sh[0] != sh[1] {
            bail!("{key}: expected a square matrix, got {sh:?}");
        }
        let data = tensor_to_f32(&t)?;
        if let Some(i) = first_non_finite(&data) {
            let (d, r, c) = (sh[0], i / sh[0], i % sh[0]);
            bail!(
                "{key}: {d} x {d} matrix has a non-finite entry ({}) at [ {r}, {c} ]; \
                 the calibration data or the captured forward is corrupt, refusing to quantize",
                data[i]
            );
        }
        Ok(Some((data, sh[0])))
    }

    /// The Hessian `X X^T` of the layer's input activations, plus its dimension (the caller checks it against the
    /// weight's `d_in` before quantizing).
    pub fn get_h(&self, owner: &str) -> Result<(Vec<f32>, usize)> {
        self.tensor(owner, "H")?.ok_or_else(|| anyhow!("missing {owner}.H"))
    }

    /// The optional asymmetric cross-covariance `M = X X~^T`; `None` selects symmetric calibration.
    pub fn get_m(&self, owner: &str) -> Result<Option<Vec<f32>>> {
        Ok(self.tensor(owner, "M")?.map(|x| x.0))
    }
}

/// Tensors describing one packed layer — the on-disk format, six tensors named `<layer>.<suffix>`:
///
/// * `.qweight` `u8 [d_out, row_bytes]` : LSB-first bit-stream of codes in *permuted* column order. Widths come from
///   `.gbits`, so a row is not byte-aligned per code and 3-bit codes really do cost 3 bits.
/// * `.scales`  `f16 [d_out, ngroups]` : one scale per (row, group). Stored f16 because that is exactly the precision
///   the engine optimized at — no hidden widening.
/// * `.qzeros`  `u8  [d_out, ngroups]` : integer zero point (`0..=maxq`). u8 covers up to 8 bit codes; it costs 8 bits
///   per group whether or not a smaller encoding would fit, which is the honest accounting in `bits_total`.
/// * `.perm`    `i32 [d_in]`           : act-order permutation. Required at inference time: the caller must permute its
///   activations the same way, otherwise the weights are multiplied against the wrong input channels.
/// * `.gbits`   `u8  [ngroups]`        : per-group code width (the skeleton promotion is visible here, not implied by
///   a global `bits`).
/// * `.gsize`   `i32 [1]`              : group size; needed to map a column index to its group.
///
/// `row_bytes` is written as the second dimension of `.qweight`, so a reader never has to recompute it — and if it
/// did recompute it differently, the shape check in [`TensorView::new`] would catch the disagreement at write time.
pub fn packed_tensors(layer: &str, ql: &QLayer) -> Vec<OutTensor> {
    let (qw, rb) = ql.pack();
    let ng = ql.ngroups();
    let scales: Vec<u8> = ql.scales.iter().flat_map(|&s| f16::from_f32(s).to_le_bytes()).collect();
    // Safe because the grid search and the engine clamp every zero point into 0..=maxq with maxq <= 255.
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

/// Rebuild a `QLayer` from the tensors written by [`packed_tensors`]. Exact inverse of the writer: it must agree with
/// it on every encoding choice (endianness, u8 zero points, permuted code order), and this is the function exercised
/// by the `cdclq dequant` audit. A missing companion tensor is an error naming that tensor, since a partially written
/// layer would otherwise decode into plausible-looking garbage.
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
    // `d_in` comes from the permutation length, not from `.gsize`: it is the only tensor that records the column count
    // exactly (the qweight row width is rounded up to whole bytes).
    let d_in = pm.shape()[0];
    let group = i32::from_le_bytes(gs.data()[..4].try_into().unwrap()) as usize;
    let perm: Vec<u32> = pm.data().chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32).collect();
    let scales: Vec<f32> = sc.data().chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
    let zeros: Vec<f32> = qz.data().iter().map(|&z| z as f32).collect();
    Ok(QLayer::from_packed(d_out, d_in, group, perm, gb.data().to_vec(), qw.data(), scales, zeros))
}

#[cfg(test)]
mod tests {
    use super::first_non_finite;

    #[test]
    fn finite_matrices_pass_the_load_scan() {
        assert_eq!(first_non_finite(&[]), None);
        assert_eq!(first_non_finite(&[0.0, -0.0, 1.0, f32::MIN, f32::MAX]), None);
    }

    #[test]
    fn load_scan_reports_the_first_bad_element() {
        assert_eq!(first_non_finite(&[1.0, f32::NAN, f32::INFINITY]), Some(1));
        assert_eq!(first_non_finite(&[1.0, 2.0, f32::NEG_INFINITY]), Some(2));
    }
}
