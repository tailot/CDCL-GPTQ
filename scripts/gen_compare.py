#!/usr/bin/env python3
"""Side-by-side generation: FP16 model vs one or more dequantized (fake-quant) models.

    python scripts/gen_compare.py --model models/TinyLlama-1.1B --dequant runs/b3-cdcl/dequant/model.safetensors ...

Same prompts, same fixed sampling seed, so the differences you see come only from the weights.
"""
import argparse
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:",
    "The three main causes of the French Revolution were",
    "Il giorno dopo andammo al mercato e",
]


def load(model_dir, dequant):
    m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    if dequant:
        sd = load_file(dequant)
        cur = dict(m.state_dict())
        for k, v in sd.items():
            if k in cur and cur[k].shape == v.shape:
                cur[k] = v
        m.load_state_dict(cur, strict=False)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dequant", action="append", default=[], help="dequantized safetensors (repeatable)")
    ap.add_argument("--labels", default="")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else []
    tok = AutoTokenizer.from_pretrained(args.model)
    variants = [("fp16", None)] + [(lbl or f"q{i + 1}", d) for i, (lbl, d) in enumerate(zip(labels, args.dequant))]
    for name, dq in variants:
        print(f"\n===== {name}  ({dq or 'pesi originali'}) =====")
        m = load(args.model, dq).to(args.device)
        for p in PROMPTS:
            torch.manual_seed(args.seed)
            x = tok(p, return_tensors="pt").input_ids.to(args.device)
            y = m.generate(x, max_new_tokens=args.max_new, do_sample=args.temperature > 0, temperature=args.temperature,
                           top_p=args.top_p, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
            print(f"IN : {p!r}\nOUT: {tok.decode(y[0], skip_special_tokens=True)}\n")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
