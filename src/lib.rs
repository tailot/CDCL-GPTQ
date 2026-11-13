//! CDCL-GPTQ: a conflict-driven, Hessian-exact post-training quantizer.
//!
//! Pipeline per linear layer `W (d_out x d_in)` with calibration Hessian `H = X X^T`:
//!   1. column ordering (protected "skeleton" groups first, then act-order),
//!   2. upper Cholesky factor `U` of `H^-1` (true GPTQ factorisation),
//!   3. blocked GPTQ pass with lazy batch updates (rows are independent -> parallel),
//!   4. conflict detection from the *exact* per-column loss decomposition of the GPTQ pass,
//!   5. conflict repair by exact coordinate descent on the integer grid (monotone),
//!   6. optional asymmetric (cross-layer) calibration target `W* = W M H~^-1`.

pub mod bench;
pub mod engine;
pub mod io;
pub mod linalg;
pub mod pipeline;
pub mod quant;
pub mod rng;
