#!/usr/bin/env python3
"""Final table over the runs in runs/ (cdclq's report.json + eval_ppl.py output).

    python scripts/summarize.py            # markdown on stdout
    python scripts/summarize.py --md runs/table.md

One row per run directory matching ``runs/b*``. Two sources are joined:

* ``<run>/report.json``  - written by ``cdclq quantize``: config string, wall-clock seconds, per-layer proxy loss and
  the *effective* bits/weight (code bits + the 24 metadata bits amortised over the group size). Recomputed here as a
  parameter-weighted mean rather than trusted from the report.
* ``<run>.eval.json``    - written by ``scripts/eval_ppl.py``: perplexity and top-1 token accuracy on WikiText-2 test.

Columns that need a reference are computed against it in-script: "vs fp16" is the size ratio over the original
safetensors, and the last column compares perplexity with the GPTQ run at the same bits/group/mse-grid, which is the
only apples-to-apples comparison for "did CDCL help". Rows are ordered by (descending bits, group, mse, method) so a
GPTQ row is always immediately above the CDCL row it is compared against.

A run with no eval file still appears, with em dashes in the accuracy columns - quantization and evaluation are
separate steps on purpose, so an incomplete matrix is a normal state, not an error.
"""
import argparse
import glob
import json
import os
import re


def parse_cfg(s):
    """Pull the fields out of report.json's free-form config string ("bits: 3, group: 128, mse_grid: true, ...")."""
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
    hdr = ("configuration", "bpw", "model (MB)", "vs fp16", "quant time (s)", "proxy loss gain",
           "perplexity", "dPPL vs fp16", "top-1 %", "dPPL vs GPTQ")
    rows = [hdr, tuple(["---"] * 10)]
    rows.append(("**fp16 (reference)**", "16.0", f"{orig / 1e6:.0f}", "1.00x", "\u2014", "\u2014",
                 f"{base:.2f}", "\u2014", f"{fp['top1']:.2f}", "\u2014"))

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
        # Weighted mean of bits_total: layer sizes differ (q/k/v are smaller than the MLP projections), so an unweighted
        # mean over layers would misstate the model's real bits/weight.
        npar = sum(l["d_out"] * l["d_in"] for l in rep["layers"])
        bt = sum(l["bits_total"] * l["d_out"] * l["d_in"] for l in rep["layers"]) / npar
        pk = os.path.join(d, "packed", os.path.basename(args.orig))
        pk_b = os.path.getsize(pk)
        tag = f"{c['bits']} bit, g{c['group']}" + (", mse-grid" if c["mse"] else "") + f", {c['method'].lower()}"
        ppl = e.get("ppl")
        # A perplexity above 5000 is already meaningless as a "2 decimal places" number, so switch to 3 significant
        # digits rather than printing 29163.00 - the reader needs the order of magnitude, not the cents.
        row = [tag, f"{bt:.2f}", f"{pk_b / 1e6:.0f}", f"{orig / pk_b:.2f}x", f"{rep['total_secs']:.0f}",
               f"{-rep['total_gain_pct']:+.1f}%", f"{ppl:.2f}" if ppl and ppl < 5000 else f"{ppl:.3g}",
               f"{100.0 * (ppl - base) / base:+.1f}%" if ppl else "\u2014", f"{e.get('top1', 0):.2f}", ""]
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
