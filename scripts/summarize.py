#!/usr/bin/env python3
"""Tabella finale dei run in runs/ (report.json di cdclq + eval_ppl.py).

    python scripts/summarize.py            # stampa markdown su stdout
    python scripts/summarize.py --md runs/RESULTS-tinyllama.md
"""
import argparse
import glob
import json
import os
import re


def parse_cfg(s):
    g = lambda k, cast=str: cast(re.search(k + r": ([^,}]+)", s).group(1))
    return dict(bits=int(g("bits")), group=int(g("group")), mse=g("mse_grid") == "true",
                method=g("method"), act=g("act_order") == "true", skel=float(g("skeleton_pct")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--orig", default=r"models\TinyLlama-1.1B\model.safetensors")
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    orig = os.path.getsize(args.orig)
    lines = []
    fp = json.load(open(os.path.join(args.root, "fp16.eval.json")))
    base = fp["ppl"]
    hdr = ("configurazione", "bpw", "modello (MB)", "vs fp16", "tempo quant (s)", "gain loss proxy",
           "perplexity", "Δppl vs fp16", "top-1 %", "Δppl vs GPTQ")
    rows = [hdr, tuple(["---"] * 10)]
    rows.append(("**fp16 (riferimento)**", "16.0", f"{orig / 1e6:.0f}", "1.00x", "—", "—",
                 f"{base:.2f}", "—", f"{fp['top1']:.2f}", "—"))

    data = []
    ppl_gptq = {}
    for d in sorted(glob.glob(os.path.join(args.root, "b*"))):
        rp = os.path.join(d, "report.json")
        if not os.path.exists(rp):
            continue
        rep = json.load(open(rp))
        c = parse_cfg(rep["config"])
        ev = os.path.join(args.root, os.path.basename(d) + ".eval.json")
        e = json.load(open(ev)) if os.path.exists(ev) else {}
        npar = sum(l["d_out"] * l["d_in"] for l in rep["layers"])
        bt = sum(l["bits_total"] * l["d_out"] * l["d_in"] for l in rep["layers"]) / npar
        pk = os.path.join(d, "packed", os.path.basename(args.orig))
        pk_b = os.path.getsize(pk)
        tag = f"{c['bits']} bit, g{c['group']}" + (", mse-grid" if c["mse"] else "") + f", {c['method'].lower()}"
        ppl = e.get("ppl")
        row = [tag, f"{bt:.2f}", f"{pk_b / 1e6:.0f}", f"{orig / pk_b:.2f}x", f"{rep['total_secs']:.0f}",
               f"{-rep['total_gain_pct']:+.1f}%", f"{ppl:.2f}" if ppl and ppl < 5000 else f"{ppl:.3g}",
               f"{100.0 * (ppl - base) / base:+.1f}%" if ppl else "—", f"{e.get('top1', 0):.2f}", ""]
        key = (-c["bits"], c["group"], not c["mse"], 0 if c["method"] == "Gptq" else 1)
        data.append((key, row, c, ppl))
        if c["method"] == "Gptq":
            ppl_gptq[(c["bits"], c["group"], c["mse"])] = ppl

    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * 10]
    for r in rows[2:]:
        out.append("| " + " | ".join(r) + " |")
    for key, row, c, ppl in sorted(data, key=lambda x: x[0]):
        if c["method"] != "Gptq":
            gp = ppl_gptq.get((c["bits"], c["group"], c["mse"]))
            if gp and ppl:
                row[9] = f"{100.0 * (ppl - gp) / gp:+.1f}%"
        out.append("| " + " | ".join(row) + " |")
    txt = "\n".join(out)
    print(txt)
    if args.md:
        open(args.md, "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
