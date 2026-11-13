#!/usr/bin/env python3
"""Synthesize per-layer calibration Hessians for the tiny-random HF models used by tests/toy_models.rs.

The real capture path is scripts/capture_hessians.py: it hooks every nn.Linear and accumulates
H = (1/N) sum_t x_t x_t^T from a real forward pass. That is what production runs use, and it is *not*
what this script does -- the models here are randomly initialized, so their activations carry no signal
worth calibrating on, and a suite that depends on them would only be testing noise.

What a test fixture needs instead is a Hessian with realistic *structure*, produced without torch:

    H = A^T A + eps I        (A random `rank x d_in`, rank ~= 2*d_in/3)

which has the two properties the engine's behaviour actually depends on: it is low-rank before the ridge
(so damping and the dead-column patch are exercised rather than a well-conditioned identity), and its
diagonal is strongly non-uniform (log-normal per channel), which is what makes act-order, the skeleton
ranking and conflict selection do anything at all. An identity Hessian would leave GPTQ, CDCL and RTN
indistinguishable and every assertion in the suite would pass vacuously.

Because it reads only config.json plus the tensor shapes from model.safetensors, this works for any Llama
architecture and needs no tokenizer -- so the fixtures can be regenerated on a CI runner.

    python scripts/gen_toy_hessians.py --model llamafactory/tiny-random-Llama-3 \
        --out tests/fixtures/llamafactory-tiny-random-llama-3/hess.safetensors

The `aliases` metadata entry mirrors capture_hessians.py: q/k/v/o and gate/up share inputs in a real
forward pass, so the engine sees a genuine alias graph and PrepCache reuse is covered.
"""
import argparse
import hashlib
import json
import math
import os
import random

from safetensors import safe_open
from safetensors.torch import save_file


def hessian(d_in: int, seed: int):
    """Low-rank + ridge Hessian with log-normal channel scaling (see module docstring)."""
    import torch

    g = random.Random(seed)
    rank = max(2, (2 * d_in) // 3)
    # A is the "activation basis": its rows are directions in input space, so A^T A has rank <= rank < d_in and the
    # ridge below is what makes the result invertible -- exactly the situation a rank-deficient calibration Hessian
    # creates in production. 1/sqrt(rank) keeps the entries O(1) whatever the width, so one damping value behaves the
    # same on a 16-wide toy model and on an 8192-wide real one.
    a = torch.tensor([[g.gauss(0.0, 1.0 / math.sqrt(rank)) for _ in range(d_in)] for _ in range(rank)])
    s = torch.tensor([math.exp(g.gauss(0.0, 0.75)) for _ in range(d_in)])
    a = a * s.unsqueeze(0)
    h = a.t() @ a
    tr = float(torch.diagonal(h).mean())
    return (h + 1e-2 * tr * torch.eye(d_in)).float()


def linear_layers(path: str):
    """2-D `*.weight` tensors of the checkpoint, as `(layer_name, d_out, d_in)`.

    Mirrors what the Rust pipeline accepts: only 2-D `.weight` tensors are candidates, so norms (1-D) and any rotary
    buffer are skipped here for the same reason they are skipped there.
    """
    out = []
    with safe_open(path, framework="pt") as f:
        for name in sorted(f.keys()):
            if not name.endswith(".weight"):
                continue
            sh = f.get_slice(name).get_shape()
            if len(sh) != 2:
                continue
            out.append((name[: -len(".weight")], sh[0], sh[1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face id or local dir containing model.safetensors")
    ap.add_argument("--out", required=True, help="output .safetensors with `<layer>.H` tensors")
    ap.add_argument(
        "--skip",
        default="lm_head,embed_tokens",
        help="comma separated substrings of layers to leave uncalibrated (the pipeline then copies them through)",
    )
    ap.add_argument("--seed", type=int, default=0, help="base seed; the layer name is mixed in, so a fixture is stable")
    args = ap.parse_args()

    if os.path.isdir(args.model):
        path = os.path.join(args.model, "model.safetensors")
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(args.model, "model.safetensors")

    skip = [s for s in args.skip.split(",") if s]
    layers = [(n, o, i) for (n, o, i) in linear_layers(path) if not any(s in n for s in skip)]
    if not layers:
        raise SystemExit(f"no 2-D .weight tensor matched --skip in {path}")

    # Layers that consume the same activation share one H, exactly as in capture_hessians.py. In a Llama block q/k/v/o
    # all read the input_layernorm output and gate/up both read post_attention_layernorm, so they are aliased to one
    # owner each; down_proj reads the MLP activation and owns its own H. Keying on (block, d_in) reproduces that split
    # for Llama-style checkpoints, where o_proj is the only attention tensor whose input width differs from hidden_size
    # in some configs -- which is precisely why the key carries d_in instead of just the block name.
    by_input: dict = {}
    tensors: dict = {}
    aliases: dict = {}
    for name, _d_out, d_in in layers:
        block = name.rsplit(".", 1)[0] if name.endswith("_proj") else name
        owner = by_input.get((block, d_in))
        if owner is None:
            by_input[(block, d_in)] = name
            # Stable per-layer seed: hashing the name (rather than Python's randomized hash) keeps a regenerated
            # fixture byte-identical to the committed one, so `git diff` on tests/fixtures stays meaningful.
            tag = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            tensors[f"{name}.H"] = hessian(d_in, args.seed + tag)
        else:
            aliases[name] = owner

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(tensors, args.out, metadata={"aliases": json.dumps(aliases), "source": args.model})
    print(f"{args.model}: {len(tensors)} Hessians ({len(aliases)} aliased layers) -> {args.out}")


if __name__ == "__main__":
    main()
