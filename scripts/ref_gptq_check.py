#!/usr/bin/env python3
"""Cross-check the Rust GPTQ baseline against a plain float64 numpy implementation of the GPTQ
algorithm (per-column updates, no lazy batching).

    cdclq bench --rows 64 --cols 256 --group 64 --dump /tmp/case
    python3 scripts/ref_gptq_check.py /tmp/case

Convention note: group grids are computed at the first column of each group from the *current*
(already compensated) weights, which is what the Rust engine does when group | block.
"""
import json, sys
import numpy as np

d = sys.argv[1]
meta = json.load(open(f"{d}/meta.json"))
rows, n, bits, group, damp = meta["rows"], meta["cols"], meta["bits"], meta["group"], meta["damp"]
W0 = np.fromfile(f"{d}/W.f32", dtype="<f4").reshape(rows, n).astype(np.float64)
H0 = np.fromfile(f"{d}/H.f32", dtype="<f4").reshape(n, n).astype(np.float64)
codes_rs = np.fromfile(f"{d}/codes.u8", dtype=np.uint8).reshape(rows, n)
ng = (n + group - 1) // group
scales_rs = np.fromfile(f"{d}/scales.f32", dtype="<f4").reshape(rows, ng).astype(np.float64)
zeros_rs = np.fromfile(f"{d}/zeros.f32", dtype="<f4").reshape(rows, ng).astype(np.float64)
maxq = 2 ** bits - 1

H = H0.copy()
dead = np.diag(H) == 0
H[dead, dead] = 1.0
W = W0.copy()
W[:, dead] = 0.0
H += damp * np.mean(np.diag(H)) * np.eye(n)
U = np.linalg.cholesky(np.linalg.inv(H)).T  # upper, H^-1 = U^T U


def params(v):
    lo = np.minimum(v.min(axis=1), 0.0)
    hi = np.maximum(v.max(axis=1), 0.0)
    flat = (lo == 0) & (hi == 0)
    lo = np.where(flat, -1.0, lo)
    hi = np.where(flat, 1.0, hi)
    s = ((hi - lo) / maxq).astype(np.float16).astype(np.float64)
    s = np.maximum(s, 6.103515625e-5)
    z = np.clip(np.round(-lo / s), 0, maxq)
    return s, z


Wq = np.zeros_like(W)
codes = np.zeros((rows, n), dtype=np.uint8)
scales = np.zeros((rows, ng))
zeros = np.zeros((rows, ng))
loss = 0.0
for j in range(n):
    if j % group == 0:
        gi = j // group
        s, z = params(W[:, j:j + group])
        scales[:, gi], zeros[:, gi] = s, z
    gi = j // group
    s, z = scales[:, gi], zeros[:, gi]
    q = np.clip(np.round(W[:, j] / s) + z, 0, maxq)
    dq = s * (q - z)
    err = (W[:, j] - dq) / U[j, j]
    loss += float(np.sum(err ** 2))
    codes[:, j] = q.astype(np.uint8)
    Wq[:, j] = dq
    W[:, j:] -= np.outer(err, U[j, j:])


def proxy(Wt, Wh):
    E = Wt - Wh
    return float(np.sum((E @ H) * E))


Wq_rs = scales_rs.repeat(group, axis=1)[:, :n] * (codes_rs - zeros_rs.repeat(group, axis=1)[:, :n])
print(f"rows={rows} cols={n} bits={bits} group={group}")
print(f"codes identical (numpy ref vs Rust): {np.mean(codes == codes_rs) * 100:.3f}%")
print(f"scales identical:                    {np.mean(scales == scales_rs) * 100:.3f}%")
print(f"loss  numpy-reference (tracked):     {loss:.6e}")
print(f"loss  numpy-reference (direct):      {proxy(W0 * (~dead), Wq):.6e}")
print(f"loss  Rust (tracked, meta.json):     {meta['loss']:.6e}")
print(f"loss  Rust codes (direct, float64):  {proxy(W0 * (~dead), Wq_rs):.6e}")
rel = abs(proxy(W0 * (~dead), Wq_rs) - proxy(W0 * (~dead), Wq)) / proxy(W0 * (~dead), Wq)
print(f"relative loss difference Rust vs numpy: {rel:.2e}")
sys.exit(0 if (rel < 5e-3 and np.mean(codes == codes_rs) > 0.995) else 1)
