#!/usr/bin/env python3
"""Collect per-layer calibration Hessians for `cdclq quantize`.

For every nn.Linear it accumulates  H = (1/N) * sum_t  w_t * x_t x_t^T  over the calibration tokens
(x_t = the layer input at token t, w_t = optional per-token weight) and writes them to a
safetensors file as `<module name>.H` (float32).  Linear layers that receive the very same input
tensor (q/k/v, gate/up) are stored once and recorded in the `aliases` metadata entry.

    python scripts/capture_hessians.py --model <hf dir or id> --text calib.txt --out hess.safetensors

Structural weighting (e.g. for code / formal text): tokens whose decoded string contains one of
`--struct-chars` get weight `--struct-weight` in every Hessian, so the quantizer protects the
weights that matter for those tokens.

MEMORY, AND WHY THIS SCRIPT SPLITS BY ITSELF
--------------------------------------------
What ends the run is not the forward pass, it is the fp32 accumulator: one `d_in x d_in` matrix per
*distinct layer input*, i.e. d_in^2 * 4 bytes. That is 21 MB at hidden 2304 and 339 MB at d_in 9216
(`down_proj` of Gemma-2-2b), so a model can need ~10 GB of accumulators while its fp16 weights need
5 GB and the forward pass a few more. The usual failure is an OOM minutes into a capture, after
which the recovery is a hand-made `--only` split that quietly produces a partial Hessian file.

`--max-hessian-mem` bounds the accumulator footprint and does that splitting automatically: it
learns which modules share an input, packs the distinct inputs into bins that each fit the budget,
and runs one capture pass per bin, writing one file per bin. The sharing graph is not guessed from
module names: one probe batch -- a forward pass that accumulates nothing -- reveals the *exact*
graph from the identity of the input tensor objects. Packing happens on that graph,
so a q/k/v family is never spread over two files -- that would store the same matrix twice and cost
memory rather than save it.

Two things to keep straight about the budget: it bounds *accumulators only* (model weights,
activations and the temporary `xf` blocks sit on top of it), and each extra pass re-runs the forward
pass over the whole calibration set, so a budget that is too small is not free. Raise it as far as
the GPU allows. With a budget that fits everything, this script behaves exactly as it always did:
one forward pass, one output file named `--out`.

With several bins the files are named `<out stem>.<i>of<n>.safetensors` and the script prints the
ready-made `--hessians a.safetensors --hessians b.safetensors` line. `cdclq quantize` merges the
files and their alias metadata, and a split Hessian is bit-identical to a single-pass one: every
pass walks the same batches in the same order and divides by its own token count.

NOTE: this script is not exercised by the cdclq test-suite (it needs torch + transformers).
It captures the FP-model Hessians (symmetric calibration, X~ = X).
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

GIB = 1 << 30


def accumulator_bytes(d_in: int) -> int:
    """Size of one fp32 `d_in x d_in` accumulator: the only thing `--max-hessian-mem` bounds."""
    return d_in * d_in * 4


def fmt_bytes(n: int) -> str:
    """GiB, MiB or KiB, whichever is the natural unit. Someone tuning a budget that splits a small
    model needs numbers that are not all zeros."""
    if n >= GIB:
        return f"{n / GIB:.2f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MiB"
    return f"{n / (1 << 10):.1f} KiB"


def plan_bins(families: dict, budget: int) -> list:
    """Pack distinct layer inputs into bins of at most `budget` bytes, largest family first.

    `families` maps owner module name -> `d_in`. Only owners cost memory (an aliased module stores
    nothing), so a family weighs exactly its owner's accumulator in the plan. First-fit-decreasing
    is not optimal bin packing, but the families are few and dominated by one or two widths
    (`down_proj` dwarfs everything else), so the result sits at or one bin above optimal -- and it
    always puts the largest family in the first bin, which keeps the plan predictable when a user
    tunes the budget and re-runs.
    """
    bins: list = []
    for owner, d_in in sorted(families.items(), key=lambda kv: (-accumulator_bytes(kv[1]), kv[0])):
        need = accumulator_bytes(d_in)
        if need > budget:
            raise SystemExit(
                f"{owner}: a single Hessian at d_in={d_in} is {fmt_bytes(need)}, above "
                f"--max-hessian-mem {fmt_bytes(budget)}. Raise the budget, or capture this layer "
                f"in its own run with --only."
            )
        for b in bins:
            if b["bytes"] + need <= budget:
                b["bytes"] += need
                b["owners"].append(owner)
                break
        else:
            bins.append({"bytes": need, "owners": [owner]})
    return bins


def out_path_for(out: str, i: int, n: int) -> str:
    """`--out` verbatim in the single-file case, `<stem>.<i>of<n>.safetensors` when split."""
    if n == 1:
        return out
    stem, ext = os.path.splitext(out)
    return f"{stem}.{i}of{n}{ext}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model id or local directory")
    ap.add_argument("--text", required=True, help="UTF-8 text file used as calibration data")
    ap.add_argument("--out", required=True, help="output .safetensors with the Hessians")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default="", help="comma separated substrings: only hook matching modules")
    ap.add_argument("--skip", default="lm_head", help="comma separated substrings of modules to skip")
    ap.add_argument("--struct-chars", default="", help="characters marking structural tokens")
    ap.add_argument("--struct-weight", type=float, default=1.0)
    ap.add_argument(
        "--max-hessian-mem",
        type=float,
        default=6.0,
        metavar="GIB",
        help="GiB of fp32 Hessians held at once; anything above that is captured in several passes, one "
        "output file per pass (default: 6.0). Accumulators only: model weights and activations sit on "
        "top of this, and every extra pass re-runs the forward pass over the calibration data.",
    )
    args = ap.parse_args()

    if args.max_hessian_mem <= 0:
        sys.exit("--max-hessian-mem must be a positive number of GiB")

    only = [s for s in args.only.split(",") if s]
    skip = [s for s in args.skip.split(",") if s]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = model.to(args.device).eval()

    ids = tok(open(args.text, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
    need = args.nsamples * args.seqlen
    if ids.numel() < need:
        sys.exit(f"calibration text has {ids.numel()} tokens, need {need}: use a bigger file or lower --nsamples/--seqlen")
    samples = ids[:need].view(args.nsamples, args.seqlen)

    token_w = None
    if args.struct_chars and args.struct_weight != 1.0:
        vocab = len(tok)
        token_w = torch.ones(vocab)
        for t in range(vocab):
            if any(c in tok.decode([t]) for c in args.struct_chars):
                token_w[t] = args.struct_weight
        token_w = token_w.to(args.device)

    # Matched Linear modules in named_modules order, which is also the order hooks fire in -- and
    # that order is what decides which member of a shared-input family becomes its owner.
    targets: list = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if only and not any(s in name for s in only):
            continue
        if any(s in name for s in skip):
            continue
        targets.append((name, mod))
    if not targets:
        sys.exit("no Linear module matched --only/--skip")

    def capture_batches(state, label):
        with torch.no_grad():
            nb = (args.nsamples + args.batch - 1) // args.batch
            for s in range(0, args.nsamples, args.batch):
                batch = samples[s:s + args.batch].to(args.device)
                state["w"] = token_w[batch].reshape(-1) if token_w is not None else None
                model(input_ids=batch, use_cache=False)
                print(f"{label} batch {s // args.batch + 1}/{nb}", file=sys.stderr, flush=True)

    # ---- probe: learn the sharing graph without allocating a single Hessian --------------
    # The test is the identity of the input tensor object, which does not depend on batch size, so
    # one batch is enough. Nothing is accumulated, and the forward is stateless (use_cache=False,
    # no running statistics), so probing cannot change one bit of the Hessians captured later.
    probe: dict = {}
    probe_state = {"last_x": None, "last_owner": None}

    def make_probe_hook(name):
        def hook(module, inputs):
            x = inputs[0]
            if x is probe_state["last_x"]:
                probe[name] = probe_state["last_owner"]
                return
            probe_state["last_x"], probe_state["last_owner"] = x, name

        return hook

    handles = [mod.register_forward_pre_hook(make_probe_hook(name)) for name, mod in targets]
    with torch.no_grad():
        model(input_ids=samples[0:1].to(args.device), use_cache=False)
    for h in handles:
        h.remove()

    families = {name: mod.in_features for name, mod in targets if name not in probe}
    budget = int(args.max_hessian_mem * GIB)
    bins = plan_bins(families, budget)

    total = sum(accumulator_bytes(d) for d in families.values())
    print(
        f"{len(targets)} Linear modules, {len(families)} distinct layer inputs "
        f"({len(probe)} aliased) = {fmt_bytes(total)} of Hessians; "
        f"budget {fmt_bytes(budget)} -> {len(bins)} pass(es)",
        file=sys.stderr,
    )
    for bi, b in enumerate(bins, 1):
        k = len(b["owners"])
        print(f"  pass {bi}: {k} layer input{'' if k == 1 else 's'}, {fmt_bytes(b['bytes'])}", file=sys.stderr)

    # ---- capture: one forward pass per bin, one file per bin ------------------------------
    # Owners only: the sharing graph is already known, so an aliased module has nothing to compute
    # and no hook at all. `last_x` bookkeeping disappears from the capture path, which also removes
    # the one place where an unlucky tensor-object coincidence could have merged two families.
    written = []
    for bi, b in enumerate(bins, 1):
        owners = set(b["owners"])
        hess, count = {}, {}
        state: dict = {"w": None}

        def make_hook(name):
            def hook(module, inputs):
                x = inputs[0]
                xf = x.reshape(-1, x.shape[-1]).float()
                if state["w"] is not None:
                    xf = xf * state["w"].sqrt().unsqueeze(1)
                if name not in hess:
                    hess[name] = torch.zeros(xf.shape[1], xf.shape[1], dtype=torch.float32, device=xf.device)
                    count[name] = 0
                hess[name].addmm_(xf.t(), xf)
                count[name] += xf.shape[0]

            return hook

        handles = [mod.register_forward_pre_hook(make_hook(name)) for name, mod in targets if name in owners]
        capture_batches(state, f"pass {bi}/{len(bins)}")
        for h in handles:
            h.remove()

        aliases_here = {dep: own for dep, own in probe.items() if own in owners}
        tensors = {f"{n}.H": (hess[n] / count[n]).cpu().contiguous() for n in hess}
        path = out_path_for(args.out, bi, len(bins))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_file(tensors, path, metadata={"aliases": json.dumps(aliases_here)})
        written.append(path)
        n = len(tensors)
        print(
            f"saved {n} Hessian{'s' if n != 1 else ''} ({len(aliases_here)} aliased, "
            f"{fmt_bytes(os.path.getsize(path))}) -> {path}",
            file=sys.stderr,
        )

        # Free the accumulators before the next pass, or splitting never lowers the peak.
        del hess, count, tensors
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if len(written) > 1:
        print("pass everything to one run:", file=sys.stderr)
        print("  " + " ".join(f"--hessians {p}" for p in written), file=sys.stderr)
    print(f"done: {len(written)} Hessian file(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
