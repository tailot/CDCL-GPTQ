# CDCL-GPTQ
A weight-only *post-training quantizer* for LLMs (safetensors → grouped INT2–INT8), written in Rust.
It starts from GPTQ (exact Cholesky factorisation, blocked lazy updates) and then **repairs the
conflicts**: it exploits the exact decomposition of the proxy loss to locate the worst sub-blocks and
improves them with two exact, monotone moves — coordinate descent over integer codes plus a
closed-form refit of the group scales — alternated until convergence.

Both moves are *exact* with respect to the layer objective: every accepted update provably lowers
`tr(E H Eᵀ)`, and the loss tracked during quantization is verified by tests against an independent
direct evaluation. Nothing here is a heuristic retry/rollback scheme.

**Measured on a real LLM (TinyLlama-1.1B, WikiText-2 test; identical bits, group size, column order
and damping for both methods):** perplexity reduced by **−9.0% at 3 bits** and **−16.6% at 3 bits
with `--mse-grid`** relative to GPTQ, at *identical* model size (3.19 effective bits/weight → 3.38×
smaller checkpoint), for **~3–4× the wall-clock time** of GPTQ. At 2 bits the difference is
qualitative rather than quantitative: plain GPTQ yields a dead model (ppl 29,000) while CDCL with
`--group 64 --mse-grid` keeps it at 47 ppl. On synthetic layers the gain was −6% … −17%; see
"Results on a real model" below and the "3-bit corner case" section for why the synthetic numbers
must not be extrapolated to 4 bits.

## What was fixed relative to the specification

| # | Specification | Problem | Here |
|---|---|---|---|
| 1 | Pseudocode updating with a full, static `H_inv` | Not GPTQ: no Schur-complement update after each column, so the compensations are wrong | **Upper** Cholesky factor `U` of `H⁻¹` (`H⁻¹ = UᵀU`), computed with the flip trick (1 Cholesky + 1 triangular inverse). Test: `UᵀU·H = I`; codes match an independent float64 numpy GPTQ **exactly (100%)** (`scripts/ref_gptq_check.py`) |
| 2 | After 2 failed retries restore `checkpoint_W` and exit | The sub-block is left **unquantized** (still floating point) | No retry, no rollback: the repair is monotone and can never leave a weight unquantized |
| 3 | "Clause" forcing the opposite rounding on the worst weight | By construction it *worsens* that weight; no guarantee on the total | **Exact** per-coordinate minimisation: the grid point nearest `ŵⱼ + gⱼ/Hⱼⱼ`, with the exact block gradient `g = E H` |
| 4 | Conflict = relative error vs `‖WX‖` | Noisy signal, computed separately, not tied to the actual cost | Exact decomposition `L = Σⱼ εⱼ²`, `εⱼ = (w_cur − q)/Uⱼⱼ` (non-negative, free during the GPTQ pass, verified by a test) |
| 5 | Element-wise `C_ij` with a gradient over structural tokens | Expensive (per-element gradient), unstructured mask (scattered indices) | Criticality per **column group** `H_jj·‖W_:j‖²` (OBS saliency): whole groups kept at 8 bits. Structural tokens are instead folded into the Hessian (`--struct-weight` in the capture script). The element-wise gradient is **not** implemented |
| 6 | Cross-layer propagation `X ← X + βΔY` | The sign pushes the input back to its FP value: the next layer optimizes as if the previous one were perfect, without compensating | Exact **asymmetric calibration**: `W* = W·M·H̃⁻¹` with `M = X X̃ᵀ` (normal equations). Implemented in the engine and validated on synthetic data; the capture script does **not** yet emit `M` |
| 7 | "INT3" with 5% at INT8 | Effective bits were never counted | The report gives both code bits and total bits (f16 scale + 8-bit zero point per row/group); the skeleton raises bits/weight and the benchmark reports it |
| 8 | Groups, scales, damping, batch update `...` | Left unspecified | Implemented and configurable |

## Algorithm (per layer `W ∈ R^{d_out×d_in}`, `H = X Xᵀ`)

The objective is the standard GPTQ proxy loss, which upper-bounds the true output error of the layer
(`E = W − Ŵ`):

```text
    L(Ŵ) = Σ_rows  e_r H e_rᵀ          (rows are independent → embarrassingly parallel)
```

1. **Column order.** Protected "skeleton" groups first, if `--skeleton-pct > 0`, ranked by the OBS
   criticality `H_jj · ‖W_:j‖²`; then act-order (descending `H_jj`) or natural order. Whole groups are
   protected so that a group's scale/zero-point stay contiguous.
2. **Factorisation.** Damping (`--damp × mean(diag H)`), dead columns (zero activations) get a unit
   diagonal, then `U` with `H⁻¹ = UᵀU` in f64 (blocked, parallel). Shared between layers fed by the
   same input tensor (q/k/v, gate/up) whenever the permutation matches — one Cholesky per distinct
   input, not per linear layer.
3. **GPTQ pass** over blocks of `--block` columns with lazy batch updates (one GEMM per block instead
   of one rank-1 update per column). Rows are independent, so they are chunked and run under rayon.
   This pass also emits `εⱼ²` per column at no extra cost.
4. **Conflict selection.** The sub-blocks (`--sub_block` columns wide) with the largest exact loss
   contribution; `--refine-frac` is the fraction kept (default: all of them).
5. **Repair** (`--passes`, early stop when a pass improves the loss by less than `--tol`), alternating
   - **scale refit:** with `u = q − z` the loss is quadratic in the scale, so
     `Δs = (g·u)/(u H_CC uᵀ)` in closed form; the new scale is rounded to f16 and accepted only if it
     strictly decreases the loss;
   - **coordinate descent over the integer codes**, using the exact block gradient: the 1-D minimiser
     on the grid is the point nearest `ŵⱼ + gⱼ/Hⱼⱼ`, and the gradient is updated incrementally.

   Neither move can increase the loss (tested: the tracked loss matches a direct evaluation, and
   `loss_final ≤ loss_gptq`). Every layer also reports `loss_gptq`, i.e. what GPTQ alone would have
   achieved under exactly the same settings — that is the honest A/B baseline inside one run.

## Installation

Rust is required (https://rustup.rs). Dependencies are pinned in `Cargo.toml`/`Cargo.lock` and build
with Rust ≥ 1.75.

```bash
cargo build --release          # binary: target/release/cdclq (Windows: target\release\cdclq.exe)
cargo test --release           # 9 unit tests + 7 end-to-end tests (see "Test suite")
```

On PowerShell, with local-CPU codegen (optional): `$env:RUSTFLAGS="-C target-cpu=native"; cargo build
--release`. The GEMM kernels use AVX/FMA selected at runtime. Release assets for Linux x86_64 are
attached to each `v*` tag on GitHub; the binary there is statically linked and built for
`x86-64-v3`, so it runs on any x86-64 machine with AVX2.

## Usage

```bash
# 1) calibration Hessians (PyTorch + transformers; see the docstring for memory and --only)
python scripts/capture_hessians.py --model ./model --text calib.txt --out hess.safetensors \
       --nsamples 128 --seqlen 2048 [--struct-chars "(){}=<>" --struct-weight 4]
#    What costs GPU memory is the fp32 accumulator: d_in^2 * 4 bytes per *distinct layer input*
#    (21 MB at hidden 2304, 339 MB at down_proj's 9216). --max-hessian-mem 6 (GiB) bounds it and
#    splits the capture into hess.1of2.safetensors, hess.2of2.safetensors, ..., printing the
#    `--hessians a --hessians b` line to paste. A split Hessian is bit-identical to a single-pass one.
#    The header records `format=hessian_v1`, the generator and `calib_tokens` (N, which bounds
#    rank(H) <= N: keep N well above the widest d_in, or the layer is quantized against statistics
#    that leave most input directions unconstrained).

# 2) quantization
cdclq quantize --weights model.safetensors --hessians hess.safetensors --out-dir out \
       --bits 3 --group 128            # defaults: act-order on, full repair

# 3) outputs
#   out/dequant/<shard>.safetensors  dequantized weights, original dtype (for HF / lm-eval harnesses)
#   out/packed/<shard>.safetensors   bit-packed model (qweight, f16 scales, u8 qzeros, perm, gbits, gsize)
#   out/report.json                  GPTQ vs final loss, bits and timings per layer
cdclq dequant --input out/packed/model.safetensors --output rebuilt.safetensors

# 4) evaluation (ppl + top-1 on held-out text, fake-quant weights) and generated text
python scripts/eval_ppl.py --model ./model --text test.txt --dequant out/dequant/model.safetensors \
       --json out.eval.json
python scripts/gen_compare.py --model ./model --dequant out/dequant/model.safetensors --labels cdcl
python scripts/summarize.py            # markdown table from every report.json + *.eval.json under runs/
```

Main options (`cdclq quantize --help`): `--bits`, `--group` (0 = one scale per row), `--damp`,
`--no-act-order`, `--mse-grid`, `--skeleton-pct/--skeleton-bits`, `--method gptq|cdcl` (for A/B
comparison), `--passes`, `--tol`, `--refine-frac`, `--sub-block`, `--cd-sweeps`, `--no-refit-scales`,
`--only/--skip` (substring filters), `--threads`. For multiple shards:
`--weights a.safetensors b.safetensors` (output keeps the input file names). Hessians are looked up
as `<tensor name without .weight>.H`, with shared inputs resolved through the `aliases` metadata entry.

Speed profiles: `--passes 2 --refine-frac 0.5` (fast), `--passes 4` (balanced), defaults (full).

## Results on a real model (TinyLlama-1.1B)

First measurement on a real LLM (machine: 24-core CPU; the RTX 5070 Ti is used only for Hessian
capture and evaluation). Model `TinyLlama/TinyLlama_v1.1` (1.1 B parameters, fp16, 2200 MB),
calibration on WikiText-2 **train** (128 × 2048 tokens), evaluation on WikiText-2 **test**
(334,521 tokens in non-overlapping chunks of 1024) using the fake-quant weights from `out/dequant`.
`--group 128 --damp 0.01`, act-order on; GPTQ and CDCL otherwise identical. Full protocol, generated
text samples and reproduction commands: `runs/RESULTS-tinyllama.md`.

The checkpoint itself is **not in this repository** — it is a 6.3 GB download (weights included) that
CI never needs, since the test suite runs on tiny-random models. To reproduce these numbers, fetch it
first: `huggingface-cli download TinyLlama/TinyLlama_v1.1 --local-dir models/TinyLlama-1.1B`.

| configuration | bits/weight | model (MB) | vs fp16 | time (s) | proxy-loss gain | perplexity | vs fp16 | top-1 % | vs GPTQ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 (reference) | 16 | 2200 | 1.00× | — | — | **8.72** | — | 54.18 | — |
| 4 bit g128 GPTQ | 4.19 | 771 | 2.85× | 78 | — | 9.10 | +4.4% | 53.55 | — |
| 4 bit g128 CDCL | 4.19 | 771 | 2.85× | 235 | −8.3% | **9.06** | +4.0% | 53.62 | −0.4% |
| 3 bit g128 `--mse-grid` GPTQ | 3.19 | 650 | 3.38× | 109 | — | 12.88 | +47.8% | 49.08 | — |
| 3 bit g128 `--mse-grid` CDCL | 3.19 | 650 | 3.38× | 260 | −8.2% | **10.75** | +23.3% | 50.84 | **−16.6%** |
| 3 bit g128 GPTQ | 3.19 | 650 | 3.38× | 55 | — | 13.67 | +56.7% | 47.94 | — |
| 3 bit g128 CDCL | 3.19 | 650 | 3.38× | 234 | −13.9% | **12.44** | +42.6% | 49.04 | **−9.0%** |
| 2 bit g64 `--mse-grid` GPTQ | 2.38 | 552 | 3.99× | 92 | — | 197.9 | +2170% | 22.32 | — |
| 2 bit g64 `--mse-grid` CDCL | 2.38 | 552 | 3.99× | 257 | −10.4% | **47.2** | +442% | 33.49 | **−76.1%** |
| 2 bit g128 GPTQ | 2.19 | 529 | 4.16× | 69 | — | 2.9e+04 | — | 0.34 | — |
| 2 bit g128 CDCL | 2.19 | 529 | 4.16× | 250 | −37.8% | 1671 | — | 7.66 | −94.2% |

`bits/weight` includes the per-row/per-group metadata (f16 scale + 8-bit zero point = 24 bits per
(row, group)); at `--group 128` that is 3.00 code bits + 0.19 metadata bits. The **vs fp16** ratio is
computed on the whole `packed` file, so it includes `embed_tokens`, `lm_head`, norms and biases left
in fp16 (~131 M parameters ≈ 262 MB). On the quantized weights alone the ratio is 16/3.19 = **5.0×**.

Does the model still work? Yes, down to 3 bits: top-1 accuracy 50.84% vs 54.18% fp16, and generated
text is still coherent (sometimes *more* varied than fp16, because vanilla TinyLlama is extremely
repetitive). The `packed → dequant` round-trip is bit-exact: 201 tensors, same dtypes and shapes,
`max |dequant − roundtrip| = 0`.

## The 3-bit question (read this before choosing a bit-width)

**3 bits is where CDCL-GPTQ pays off; 4 bits is not, and 2 bits is a different problem.** Measured on
TinyLlama-1.1B, at constant storage:

| bit-width | GPTQ ppl | CDCL ppl | CDCL gain | verdict |
|---|---:|---:|---:|---|
| 4 bit g128 | 9.10 | 9.06 | **−0.4%** | not worth it: you pay 3× the time for a rounding error on ppl (fp16 is 8.72) |
| 3 bit g128 | 13.67 | 12.44 | **−9.0%** | worth it |
| 3 bit g128 `--mse-grid` | 12.88 | **10.75** | **−16.6%** | best configuration of the whole matrix |
| 2 bit g64 `--mse-grid` | 197.9 | 47.2 | −76.1% | still not usable in practice, but qualitatively different |
| 2 bit g128 | 29056 | 1671 | −94.2% | both methods fail; ignore this row |

Why the gain is small at 4 bits and large at 3 bits:

- **GPTQ is already near the error floor at 4 bits.** fp16 sits at 8.72 ppl, GPTQ at 9.10: only 4.4%
  of headroom remains before the fp16 reference, so even an ideal repairer has almost nothing to win.
  The proxy loss does drop (−8.3%), but that residual no longer dominates the perplexity — quantization
  noise is not yet the bottleneck.
- **At 3 bits quantization noise becomes the dominant error source**, which is exactly the regime the
  repair targets: the proxy loss drops −13.9% and perplexity follows with −9.0%. The synthetic model
  predicted −9.6% at 3 bits, so this is the one point where the synthetic benchmark was quantitatively
  right.
- **`--mse-grid` and the repair are complementary, not redundant.** With MSE-optimal clipping every
  group starts from a better grid, and the code/scale alternation then exploits it further: GPTQ goes
  13.67 → 12.88 (−5.8%) with the grid alone, CDCL goes 12.44 → 10.75 (−13.6%). Combined versus plain
  GPTQ at default settings that is **13.67 → 10.75 = −21.4%** at identical size. In the synthetic
  benchmark the two moves were worth only −1.9% and −0.6% individually, but −9.6% alternated; on the
  real model the interaction is even stronger.
- **The compression ratio itself is capped by non-linear weights, not by the bit-width.** At 3 bits the
  checkpoint only reaches 3.38× because `embed_tokens` + `lm_head` (65.5 M parameters each) stay fp16.
  Going from 4 to 3 bits buys you just 0.9× of overall ratio; changing `--group 128` → `--group 64
  --mse-grid` buys ~600× in ppl at 2 bits, and `--mse-grid` alone buys −5.8% at 3 bits. **Grid and
  group size dominate the repair in the low-bit regime; the repair dominates in the mid-bit regime.**

Cost of the win: one-off quantization time, 234–260 s vs 55–109 s for 969 M quantized parameters on
24 CPU cores. Nothing at inference time — the packed artifact is bit-identical in size to GPTQ's.

Recommendation: use **3 bit + `--mse-grid`** (or `--group 64 --mse-grid` if you can afford the extra
metadata bits); skip CDCL at 4 bit (`--method gptq`) unless the proxy loss itself is what you care
about; treat 2 bit as research, not as a deployable configuration.

## Results (synthetic layer)

Synthetic layer 512×2048, 16384 calibration tokens, 4096 **held-out** tokens, 1 core, `--group 128`.
Correlated activations with channel outliers and "massive" tokens. Error = `‖Wx − Ŵx‖² / ‖Wx‖²` on
unseen tokens. All methods share the weights, Hessian, grid, ordering and damping.

| 3 bit (3.19 bits/weight) | held-out error | vs GPTQ | time |
|---|---:|---:|---:|
| RTN | 5.74e-2 | +826% | 0.01 s |
| **GPTQ** | 6.19e-3 | — | 0.46 s |
| scale refit only | 6.08e-3 | −1.9% | 0.83 s |
| coordinate descent only | 6.15e-3 | −0.6% | 0.87 s |
| CDCL fast (50% sub-blocks, 2 passes) | 5.99e-3 | −3.3% | 0.85 s |
| CDCL balanced (4 passes) | 5.72e-3 | −7.6% | 1.30 s |
| **CDCL default** (up to 10 passes) | **5.60e-3** | **−9.6%** | 2.36 s |

The two moves are worth little on their own; **alternated** they deliver most of the gain — it is the
codes ↔ scales interplay that matters. Coordinate descent alone is not enough, contrary to what the
specification implied.

| Configuration (held-out, vs GPTQ with identical settings) | GPTQ | CDCL default | change |
|---|---:|---:|---:|
| 4 bit, g128 | 1.378e-3 | 1.284e-3 | **−6.9%** |
| 3 bit, g128 | 6.195e-3 | 5.602e-3 | **−9.6%** |
| 3 bit, g128, `--mse-grid` (also for GPTQ) | 4.857e-3 | 4.264e-3 | **−12.2%** |
| 2 bit, g64 | 2.380e-2 | 1.968e-2 | **−17.3%** |

Do not use these to predict real-model behaviour at high bit-widths: the synthetic −6.9% at 4 bits
became −0.4% of perplexity on TinyLlama, because a single synthetic layer has no fp16 floor to hit and
its Hessian is far less ill-conditioned than a trained transformer's.

Other experiments (same layer, 3 bit):

- **5% skeleton at 8 bit (3.5 bits/weight):** GPTQ+skeleton 5.23e-4, CDCL+skeleton 4.33e-4 (−17.2%).
  The huge drop versus "no skeleton" is an artefact of the synthetic outliers and does not transfer to
  real models.
- **Quantized inputs (relative noise 0.08):** symmetric GPTQ 1.330e-2 → GPTQ with asymmetric target
  1.269e-2 (−4.6%) → CDCL with asymmetric target 1.206e-2 (**−9.3%**).
- **Layer 512×4096** (6144 tokens, only 1.5× the dimension: calibration overfits, held-out error ≈2×
  calibration error): CDCL default −6.2% in 10.5 s against 3.1 s for GPTQ.

Reproduction: `cdclq bench --rows 512 --cols 2048 --tokens 16384 --test-tokens 4096 [--bits 4] [--group 64] [--mse-grid] [--skeleton-pct 5] [--asym-noise 0.08]`.

## Checks performed

- 9 unit tests: blocked Cholesky, `UᵀU·H = I`, parallel GEMM == serial GEMM, mixed bit-width
  pack/unpack, exactness of the loss decomposition, monotonicity + exact bookkeeping of the repair,
  skeleton/act-order permutation, asymmetric case `M = H ⇒ W* = W`.
- 7 end-to-end tests over three real tiny-random checkpoints — see [Test suite](#test-suite); both groups
  run on every push in CI.
- Cross-check against an independent float64 numpy GPTQ: **identical codes (100%)**, relative loss
  difference 0.
- End-to-end CLI test on a toy model (q/k/v aliases, 1-D tensors and embeddings passed through):
  names/dtypes/shapes preserved, `packed → dequant` **bit-exact** versus the direct output, reported
  loss equal to an independent numpy computation (relative difference 3e-5).

## Test suite

```
cargo test --release          # 9 unit tests + 7 end-to-end tests over real checkpoints
```

The end-to-end file (`tests/toy_models.rs`) runs the whole pipeline against three ordinary
`LlamaForCausalLM` checkpoints from the tiny-random family — `llamafactory/tiny-random-Llama-3`
(hidden 16, **F16**), `tiny-random/llama-3` (hidden 8, GQA 16/8, **BF16**, so `q_proj` is 512×8 and
`o_proj` 8×512) and `Intel/tiny-random-llama2` (hidden 256, intermediate **172**, GQA 4/2, **F32**).
They are 4–68 MB instead of gigabytes, so the full suite takes a few seconds on a CI runner while still
exercising what the unit tests cannot: shard walking, name matching, the q/k/v and gate/up alias graph,
pass-through of uncalibrated tensors, three storage dtypes, ragged last groups (172 is not a multiple of
any group size) and the packed-file round-trip on real safetensors files.

The Hessians are **synthetic** (`scripts/gen_toy_hessians.py`: low-rank `AᵀA` plus a ridge, log-normal
channel scaling) and the weights are randomly initialized, so these tests say nothing about accuracy —
quality numbers live in `runs/`, measured on TinyLlama-1.1B with real WikiText-2 Hessians. What they do
pin down is the machinery:

* per-layer proxy loss re-evaluated from the packed file (`tr(E H Eᵀ)` rebuilt from `.qweight`,
  `.scales`, `.qzeros` and `.perm`) agrees with the reported loss to 5e-3 relative;
* a CDCL run with `--refine-frac 0` is **bit-identical** to a GPTQ run (same permutation, same grid, same
  codes, same loss) — this is what makes every gain percentage in the tables above a like-for-like
  comparison instead of a difference between two algorithms; enabling the repair then lowers the loss on
  13/14, 14/14 and 14/14 layers without ever moving phase 1;
* act-order + skeleton: the permutation is a bijection, it reproduces the documented selection rule
  (descending `H_jj·‖W_:j‖²`, whole groups first, ties by index), promoted groups carry
  `skeleton_bits` in `.gbits`, and mixed-width rows still decode to legal codes;
* `--only`/`--skip` select exactly the layers a fixture calibrates (aliases included); an unmatched or
  shape-mismatched Hessian is a hard error, not a silent skip;
* dtypes are preserved, F32 widening is lossless, and `bits_total` in the report matches what the packed
  file actually costs (`bits + 24·ngroups/d_in`, not the nominal `24/group`).

Checkpoints live under `models/` (git-ignored) and the Hessians under `tests/fixtures/` (committed, ~3.4 MB
total). After a fresh clone the end-to-end tests **skip with a message** rather than fail; to fetch the
checkpoints and regenerate the fixtures:

```
python scripts/make_fixtures.py     # downloads the three checkpoints and rewrites tests/fixtures/
```

### Continuous integration

`.github/workflows/test.yml` runs the suite on `ubuntu-22.04` for every push to `main` and every pull
request, pinned to Rust 1.100 with `--locked` (the same toolchain `release.yml` ships, so a green test
run and the published binary are built against one dependency graph). The workflow fetches the three
tiny-random checkpoints from Hugging Face — weight tensors only, no tokenizer files — because they are
not in git.

A green `cargo test` alone would not prove much here: every end-to-end test *skips with a message* when
its checkpoint or fixture is missing, so a broken path leaves the job green while validating nothing but
the unit tests. The workflow therefore counts what the suite prints. Each check emits one labelled line
(`OK roundtrip <model>: …`, `PASS filters <model>: …`), and the job fails unless it sees exactly 19 of
them — 6 checks over all three models plus the bit-width sweep, which runs on one model only — and zero
`SKIP` lines. It also asserts that no scratch was left under `target/cdclq-test/`, which is how a test
that bailed out early would show up. The full log, including the engine's per-layer loss table, is
uploaded as the `test-log` artifact.

## Known limitations

- **One real model measured** (TinyLlama-1.1B, 1.1 B); the procedure is in `runs/RESULTS-tinyllama.md`.
  The checkpoint is not in this repository — fetch it with `huggingface-cli download
  TinyLlama/TinyLlama_v1.1 --local-dir models/TinyLlama-1.1B` to rerun those measurements.
  Not measured: 7 B+ models, other families (Mistral/Qwen), downstream benchmarks (lm-eval), and the
  sequential pipeline with quantized inputs. On TinyLlama the gains are real (−9% … −17% at 3 bits) but
  **negligible at 4 bits**; do not use the synthetic numbers as a prediction at high bit-widths.
- `scripts/capture_hessians.py` **runs and works** on TinyLlama (88 Hessians + 66 q/k/v and gate/up
  aliases, `--skip lm_head` by default). It captures the FP-model Hessians (symmetric calibration) and
  does not produce `M`. It needs an environment with `huggingface-hub < 1.0` (transformers 4.52 does not
  work with hub 1.x).
- The engine supports `M`, but a *sequential* pipeline (quantized inputs layer after layer) needs a
  driver that runs the quantized model; not included.
- The `packed` format stores an act-order permutation and 8-bit zero points (24 metadata bits per row
  and group): **there are no inference kernels**. To evaluate or serve the model today you need
  `out/dequant` (fake-quant weights).
- CPU only. Measured on a real model: **3–4× GPTQ** at defaults (234–260 s vs 55–109 s for 969 M
  weights, 24 cores). Peak memory per layer ≈ 24·d_in² bytes (d_in = 4096 → ~0.4 GB; 11008 → ~2.9 GB).
- Hessians are large: `hess-128x2048.safetensors` = 3.7 GB for a 1.1 B model (and ~3.9 GB of fp32
  accumulators during capture). `capture_hessians.py --max-hessian-mem 6` bounds that footprint, splits
  the capture into as many forward passes as fit the budget and writes one file per pass; `--hessians`
  then takes several files. `--only` still works for a manual split.
- Module A (`lllayer`) is untouched by this project.

## Layout

```
src/linalg.rs    parallel GEMM, blocked Cholesky, triangular inverse, U of H⁻¹
src/quant.rs     grids, f16 scales, bit packing, QLayer
src/engine.rs    GPTQ + conflicts + repair + asymmetric target (+ tests)
src/pipeline.rs  shards, aliases, report, packed → dequant
src/io.rs        safetensors (F32/F16/BF16), Hessian store
src/bench.rs     synthetic benchmark
tests/           toy_models.rs (end-to-end) + fixtures/<model>/hess.safetensors (synthetic Hessians)
scripts/         capture_hessians.py, ref_gptq_check.py, eval_ppl.py, gen_compare.py, summarize.py,
                 gen_toy_hessians.py, make_fixtures.py, run_matrix.ps1
runs/            RESULTS-tinyllama.md = protocol, full table, generated text, reproduction commands
                 (models/ holds the local checkpoints and is git-ignored)
```
