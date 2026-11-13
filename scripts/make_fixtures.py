"""One-shot fixture builder for the toy-model test suite: download the tiny-random checkpoints and synthesize their
Hessians. Not part of the Rust suite itself -- see scripts/gen_toy_hessians.py for what the Hessians are and why they are
synthetic rather than captured from a forward pass. The checkpoints stay out of git (`models/` is ignored); the generated
Hessians under `tests/fixtures/` are committed, so a normal clone runs the suite with no download step at all.
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
IDS = ["llamafactory/tiny-random-Llama-3", "tiny-random/llama-3", "Intel/tiny-random-llama2"]

for rid in IDS:
    dst = os.path.join(REPO, "models", rid.split("/")[-1])
    if not os.path.isfile(os.path.join(dst, "model.safetensors")):
        snapshot_download(
            rid,
            local_dir=dst,
            ignore_patterns=["*.bin", "*.png", "*.jpg", "*.msgpack"],
        )
    print("model:", dst)

for rid in IDS:
    dst = os.path.join(REPO, "models", rid.split("/")[-1])
    out = os.path.join(REPO, "tests", "fixtures", rid.split("/")[-1], "hess.safetensors")
    subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "gen_toy_hessians.py"), "--model", dst, "--out", out],
        check=True,
    )
