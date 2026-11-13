#!/usr/bin/env python3
"""Evaluate a model with (optionally) dequantized/fake-quant weights from `cdclq quantize`.

    python scripts/eval_ppl.py --model <hf dir> --text data/wikitext2-test.txt [--dequant out/dequant/model.safetensors]

Loads the FP model in float16, overwrites every tensor whose name appears in the dequantized
safetensors file, then computes:
  * perplexity on a held-out text (non-overlapping chunks of --seqlen tokens)
  * next-token top-1 accuracy on the same chunks
  * relative L2 error of the replaced weights vs the FP weights
  * a greedy generation sample (unless --no-gen), so you can eyeball that the model still works
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def prep(path: str) -> str:
    lines = [l for l in open(path, encoding="utf-8").read().split("\n")]
    return " \n ".join(l for l in lines if l.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local directory")
    ap.add_argument("--text", required=True, help="held-out text file (perplexity)")
    ap.add_argument("--dequant", default="", help="safetensors (or dir containing it) with replaced weights")
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--json", default="", help="also write the metrics here")
    args = ap.parse_args()

    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt, low_cpu_mem_usage=True)

    rel_err, n_rep, rep_bytes, tot_bytes = None, 0, 0, 0
    if args.dequant:
        path = args.dequant
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in sorted(os.path.join(path, f)) if f.endswith(".safetensors")] if os.path.isdir(path) else [path]
        else:
            files = [path]
        sd = {}
        for f in files:
            sd.update(load_file(f))
        old = dict(model.state_dict())
        num = den = 0.0
        for k, v in sd.items():
            if k not in old:
                continue
            if old[k].shape == v.shape:
                if v.dim() == 2 and v.numel() > 4096:  # only report on the real weight matrices
                    a, b = v.float(), old[k].float()
                    d2 = (a - b).pow(2).sum().item()
                    if d2 == 0.0:
                        continue  # passthrough tensor (embeddings, lm_head)
                    num += d2
                    den += b.pow(2).sum().item()
                    n_rep += 1
                    rep_bytes += v.numel() * v.element_size()
                    tot_bytes += v.numel() * v.element_size()
                old[k] = v
            else:
                print(f"shape mismatch, skipped: {k} {tuple(v.shape)} vs {tuple(old[k].shape)}", file=sys.stderr)
        model.load_state_dict(old, strict=False)
        rel_err = (num / den) ** 0.5 if den > 0 else float("nan")

    model = model.to(args.device).eval()
    ids = tok(prep(args.text), return_tensors="pt").input_ids[0]
    nch = ids.numel() // args.seqlen
    ids = ids[: nch * args.seqlen].view(nch, args.seqlen)
    print(f"perplexity over {nch} x {args.seqlen} = {ids.numel()} tokens", file=sys.stderr)

    tot_nll, tot_tok, hit = 0.0, 0, 0
    with torch.no_grad():
        for i in range(nch):
            x = ids[i : i + 1].to(args.device)
            logits = model(input_ids=x, use_cache=False).logits[0, :-1].float()
            tgt = x[0, 1:]
            tot_nll += F.cross_entropy(logits, tgt, reduction="sum").item()
            tot_tok += tgt.numel()
            hit += (logits.argmax(-1) == tgt).sum().item()
    ppl = math.exp(tot_nll / tot_tok)
    print(f"ppl {ppl:.3f}   top1 {100.0 * hit / tot_tok:.2f}%   tokens {tot_tok}")
    if rel_err is not None:
        print(f"replaced 2-D weight tensors: {n_rep}   relative L2 vs FP: {rel_err:.4e}")

    if not args.no_gen:
        prompts = [
            "The capital of France is",
            "def fibonacci(n):\n    if n <= 1:",
            "Once upon a time,",
        ]
        print("\n--- greedy generation ---")
        outs = {}
        for p in prompts:
            x = tok(p, return_tensors="pt").input_ids.to(args.device)
            y = model.generate(x, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
            s = tok.decode(y[0], skip_special_tokens=True)
            outs[p] = s
            print(f"IN : {p!r}\nOUT: {s!r}\n")
    else:
        outs = {}

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump({"ppl": ppl, "top1": 100.0 * hit / tot_tok, "tokens": tot_tok, "rel_l2_weights": rel_err, "gen": outs}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
