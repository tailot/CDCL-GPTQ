//! CDCL-GPTQ: a conflict-driven, Hessian-exact post-training quantizer for LLM weights.
//!
//! The crate quantizes the 2-D weight tensors of a transformer (safetensors shards) to grouped
//! integer formats from INT2 to INT8, using calibration Hessians captured once with PyTorch
//! (`scripts/capture_hessians.py`). It is a *weight-only*, *offline* quantizer: it produces either
//! fake-quant weights (same dtype and shape as the original, so any HF evaluation or serving stack
//! can load them) or a bit-packed artifact plus its exact dequantization. There are no inference
//! kernels in this repository — `dequant` output is what you run today.
//!
//! # Per-layer objective
//!
//! For one linear layer `W (d_out x d_in)` with calibration Hessian `H = X X^T` (summed over
//! calibration tokens, any positive scaling), the proxy loss is
//!
//! ```text
//!     L(W_hat) = sum_r  e_r H e_r^T ,   e = W_row - W_hat_row
//! ```
//!
//! which upper-bounds the layer's output error on the calibration distribution. Rows of `W` are
//! statistically independent under this objective, so they can be quantized in parallel chunks; all
//! coupling lives along `d_in`, which is exactly what the Hessian describes.
//!
//! # Pipeline (see [`engine::quantize_layer`])
//!
//! 1. **Column ordering** — protected "skeleton" groups first (ranked by OBS criticality
//!    `H_jj * ||W_:j||^2`), then act-order (descending `H_jj`) or natural order. Ordering whole
//!    groups keeps each group's scale/zero-point contiguous, which the packer relies on.
//! 2. **Factorisation** — damped Hessian, dead columns patched to a unit diagonal, then the *upper*
//!    Cholesky factor `U` of `H^-1`, i.e. `H^-1 = U^T U` ([`linalg::upper_chol_of_inverse`]). This is
//!    the true GPTQ factorisation: processing column `j` and compensating with row `j` of `U` applies
//!    the exact Schur-complement update of the remaining sub-problem. The factor is cached per
//!    distinct input tensor, so q/k/v and gate/up pay one Cholesky each.
//! 3. **GPTQ pass** — blocked over `block` columns with lazy batch updates: within a block the
//!    compensation is applied column by column against the in-block slice of `U`, and the effect on
//!    columns beyond the block is applied as one GEMM ([`linalg::sgemm`]). Rows are parallelised
//!    through rayon.
//! 4. **Conflict detection** — the GPTQ pass yields, for free, the exact decomposition
//!    `L = sum_j eps_j^2` with `eps_j = (w_cur_j - q_j) / U_jj`. Sub-blocks whose `eps_j^2` sum is
//!    largest are the "conflicts"; no extra evaluation pass and no separate error metric is needed.
//! 5. **Conflict repair** — exact coordinate descent over the integer grid, driven by the exact block
//!    gradient `g = E H`, alternated with a closed-form refit of the f16 group scales. Every accepted
//!    move strictly decreases `L`, so the sequence is monotone and no weight can be left unquantized.
//! 6. **Asymmetric calibration** (optional) — if the layer input in the already-quantized model is
//!    `X~` rather than `X`, the objective is rewritten exactly as a standard one on a compensated
//!    target `W* = W M H~^-1`, `M = X X~^T`.
//!
//! # Modules
//!
//! * [`linalg`] — f32/f64 GEMM (parallel, AVX/FMA kernels selected at runtime), blocked Cholesky,
//!   triangular inverse and the `U` factor of `H^-1`.
//! * [`quant`] — symmetric affine grids with f16 scales (the format GPTQ-family quantizers emit),
//!   clip-search parameter selection, LSB-first bit packing and the [`quant::QLayer`] container.
//! * [`engine`] — the algorithm above, its configuration and per-layer statistics.
//! * [`pipeline`] — shard/alias handling, Hessian lookup, `report.json`, packed and dequantized output.
//! * [`io`] — safetensors reading (F32/F16/BF16) and writing, memory-mapped tensors, Hessian store.
//! * [`bench`] — self-contained synthetic benchmark used to quantify the gain without a real model.
//! * [`rng`] — deterministic splitmix64 + Box-Muller generator so benchmarks are reproducible.

pub mod bench;
pub mod engine;
pub mod io;
pub mod linalg;
pub mod pipeline;
pub mod quant;
pub mod rng;
