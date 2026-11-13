use anyhow::{bail, Result};
use cdclq::bench::{self, BenchArgs};
use cdclq::engine::{Config, Method};
use cdclq::io::dtype_from_name;
use cdclq::pipeline::{self, QuantOpts};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "cdclq", version, about = "CDCL-GPTQ: conflict-driven, Hessian-exact post-training quantizer")]
struct Cli {
    /// Worker threads (default: all cores)
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Quantize a safetensors model using calibration Hessians
    Quantize(QuantizeArgs),
    /// Convert a packed model shard back to dequantized weights
    Dequant(DequantArgs),
    /// Synthetic benchmark: GPTQ vs CDCL-GPTQ
    Bench(BenchCli),
}

#[derive(Copy, Clone, ValueEnum)]
enum MethodArg {
    Gptq,
    Cdcl,
}

#[derive(Args, Clone)]
struct QArgs {
    /// Code bits for the dense part
    #[arg(long, default_value_t = 3)]
    bits: u32,
    /// Group size along the input dimension (0 = one group per row)
    #[arg(long, default_value_t = 128)]
    group: usize,
    /// Hessian damping, relative to the mean diagonal
    #[arg(long, default_value_t = 0.01)]
    damp: f64,
    /// Disable activation-order column permutation
    #[arg(long)]
    no_act_order: bool,
    /// Percentage of input columns kept at --skeleton-bits (rounded to whole groups)
    #[arg(long, default_value_t = 0.0)]
    skeleton_pct: f64,
    #[arg(long, default_value_t = 8)]
    skeleton_bits: u32,
    /// Clip-ratio search for the group grids (applies to every method)
    #[arg(long)]
    mse_grid: bool,
    #[arg(long, value_enum, default_value = "cdcl")]
    method: MethodArg,
    /// Fraction of highest-loss sub-blocks repaired by coordinate descent
    #[arg(long, default_value_t = 1.0)]
    refine_frac: f64,
    /// Columns per repair sub-block
    #[arg(long, default_value_t = 64)]
    sub_block: usize,
    /// Coordinate-descent sweeps inside a sub-block
    #[arg(long, default_value_t = 1)]
    cd_sweeps: usize,
    /// Maximum passes (scale refit + coordinate descent) over the selected sub-blocks
    #[arg(long, default_value_t = 10)]
    passes: usize,
    /// Stop early when a pass improves the proxy loss by less than this fraction
    #[arg(long, default_value_t = 1e-3)]
    tol: f64,
    /// Disable the closed-form scale refit
    #[arg(long)]
    no_refit_scales: bool,
    /// Lazy-update block size of the GPTQ pass
    #[arg(long, default_value_t = 128)]
    block: usize,
    /// Rows per parallel task (0 = automatic)
    #[arg(long, default_value_t = 0)]
    row_chunk: usize,
}

impl QArgs {
    fn to_cfg(&self) -> Config {
        Config {
            bits: self.bits,
            group: self.group,
            block: self.block,
            damp: self.damp,
            act_order: !self.no_act_order,
            skeleton_pct: self.skeleton_pct,
            skeleton_bits: self.skeleton_bits,
            mse_grid: self.mse_grid,
            method: match self.method {
                MethodArg::Gptq => Method::Gptq,
                MethodArg::Cdcl => Method::Cdcl,
            },
            refine_frac: self.refine_frac,
            sub_block: self.sub_block,
            cd_sweeps: self.cd_sweeps,
            passes: self.passes,
            tol: self.tol,
            refit_scales: !self.no_refit_scales,
            row_chunk: self.row_chunk,
        }
    }
}

#[derive(Args)]
struct QuantizeArgs {
    /// Model weight shard(s), .safetensors (F32/F16/BF16)
    #[arg(long, num_args = 1.., required = true)]
    weights: Vec<PathBuf>,
    /// Hessian file(s) produced by scripts/capture_hessians.py
    #[arg(long, num_args = 1.., required = true)]
    hessians: Vec<PathBuf>,
    #[arg(long)]
    out_dir: PathBuf,
    /// Only layers whose name contains one of these substrings (comma separated)
    #[arg(long, value_delimiter = ',')]
    only: Vec<String>,
    /// Skip layers whose name contains one of these substrings (comma separated)
    #[arg(long, value_delimiter = ',')]
    skip: Vec<String>,
    /// Do not write the bit-packed model
    #[arg(long)]
    no_packed: bool,
    /// Do not write the dequantized (fake-quant) model
    #[arg(long)]
    no_dequant: bool,
    #[command(flatten)]
    q: QArgs,
}

#[derive(Args)]
struct DequantArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// f16 | bf16 | f32 (default: dtype recorded in the packed file)
    #[arg(long)]
    dtype: Option<String>,
}

#[derive(Args)]
struct BenchCli {
    #[arg(long, default_value_t = 1024)]
    rows: usize,
    #[arg(long, default_value_t = 2048)]
    cols: usize,
    #[arg(long, default_value_t = 16384)]
    tokens: usize,
    #[arg(long, default_value_t = 4096)]
    test_tokens: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Also run the quantized-input experiment with this relative input noise (e.g. 0.05)
    #[arg(long, default_value_t = 0.0)]
    asym_noise: f32,
    /// Write the markdown report here
    #[arg(long)]
    md: Option<PathBuf>,
    /// Dump a small case for scripts/ref_gptq_check.py
    #[arg(long)]
    dump: Option<PathBuf>,
    #[command(flatten)]
    q: QArgs,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(t) = cli.threads {
        rayon::ThreadPoolBuilder::new().num_threads(t).build_global()?;
    }
    match cli.cmd {
        Cmd::Quantize(a) => {
            if a.no_packed && a.no_dequant {
                bail!("nothing to write: both --no-packed and --no-dequant are set");
            }
            let opts = QuantOpts {
                weights: a.weights,
                hessians: a.hessians,
                out_dir: a.out_dir,
                cfg: a.q.to_cfg(),
                only: a.only,
                skip: a.skip,
                emit_packed: !a.no_packed,
                emit_dequant: !a.no_dequant,
            };
            let rep = pipeline::quantize_model(&opts)?;
            eprintln!(
                "\n{} layers | proxy loss {:.4e} (plain GPTQ pass) -> {:.4e} ({:+.2}%) | {:.1}s",
                rep.layers.len(),
                rep.total_loss_gptq,
                rep.total_loss_final,
                -rep.total_gain_pct,
                rep.total_secs
            );
            if rep.layers.is_empty() {
                bail!("no layer was quantized: check that the Hessian names match '<weight name without .weight>.H'");
            }
        }
        Cmd::Dequant(a) => {
            let dt = match &a.dtype {
                Some(s) => Some(dtype_from_name(s).ok_or_else(|| anyhow::anyhow!("unknown dtype {s}"))?),
                None => None,
            };
            let n = pipeline::dequantize_packed(&a.input, &a.output, dt)?;
            eprintln!("dequantized {n} layers -> {}", a.output.display());
        }
        Cmd::Bench(b) => {
            let args = BenchArgs {
                rows: b.rows,
                cols: b.cols,
                tokens: b.tokens,
                test_tokens: b.test_tokens,
                seed: b.seed,
                asym_noise: b.asym_noise,
                md: b.md,
                dump: b.dump,
                cfg: b.q.to_cfg(),
            };
            let rep = bench::run(&args)?;
            println!("{rep}");
        }
    }
    Ok(())
}
