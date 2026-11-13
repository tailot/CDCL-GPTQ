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

Memory: one float32 `d_in x d_in` matrix per distinct layer input (a Llama-7B block needs ~0.7 GB).
For large models run it in several passes with `--only` (substring filter) and give all the
resulting files to `cdclq quantize --hessians a.safetensors b.safetensors ...`.

NOTE: this script is not exercised by the cdclq test-suite (it needs torch + transformers).
It captures the FP-model Hessians (symmetric calibration, X~ = X).
"""
import argparse
import json
import sys

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    args = ap.parse_args()

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

    state = {"w": None, "first": True, "last_x": None, "last_owner": None}
    hess, count, aliases = {}, {}, {}

    def make_hook(name):
        def hook(module, inputs):
            if name in aliases:
                return
            x = inputs[0]
            if state["first"]:
                # same tensor object as the previous Linear call -> shares its Hessian
                if x is state["last_x"]:
                    aliases[name] = state["last_owner"]
                    return
                state["last_x"], state["last_owner"] = x, name
            xf = x.reshape(-1, x.shape[-1]).float()
            if state["w"] is not None:
                xf = xf * state["w"].sqrt().unsqueeze(1)
            if name not in hess:
                hess[name] = torch.zeros(xf.shape[1], xf.shape[1], dtype=torch.float32, device=xf.device)
                count[name] = 0
            hess[name].addmm_(xf.t(), xf)
            count[name] += xf.shape[0]

        return hook

    handles = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if only and not any(s in name for s in only):
            continue
        if any(s in name for s in skip):
            continue
        handles.append(mod.register_forward_pre_hook(make_hook(name)))
    if not handles:
        sys.exit("no Linear module matched --only/--skip")

    with torch.no_grad():
        for i in range(0, args.nsamples, args.batch):
            batch = samples[i:i + args.batch].to(args.device)
            state["w"] = token_w[batch].reshape(-1) if token_w is not None else None
            model(input_ids=batch, use_cache=False)
            state["first"] = False
            print(f"batch {i // args.batch + 1}/{(args.nsamples + args.batch - 1) // args.batch}", file=sys.stderr)
    for h in handles:
        h.remove()

    tensors = {f"{n}.H": (hess[n] / count[n]).cpu().contiguous() for n in hess}
    save_file(tensors, args.out, metadata={"aliases": json.dumps(aliases)})
    print(f"saved {len(tensors)} Hessians ({len(aliases)} aliased layers) -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
