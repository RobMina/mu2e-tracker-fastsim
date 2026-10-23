"""Summarise f1_diag.sh variants: fraction within 0.1 sigma per parameter, and sliced by active hits and crossings.
python3 scripts/f1_diag_summary.py DIR"""
import json
import os
import sys

import numpy as np

d = sys.argv[1]
KL = ("d0", "phi0", "z0", "theta", "t0")
for v in sorted(os.listdir(d)):
    f = os.path.join(d, v, "pulls.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    g = z["good"]
    row = " ".join(f"{k} {np.mean(np.abs(z[k][g]) < 0.1) * 100:5.1f}" for k in KL)
    print(f"{v:8s} n {g.sum():5d}  {row}")
    for lo, hi in ((0, 16), (16, 24), (24, 1000)):
        m = g & (z["nact"] >= lo) & (z["nact"] < hi)
        print(f"   nact {lo:2d}-{hi:4d} n {m.sum():5d}  " +
              " ".join(f"{k} {np.mean(np.abs(z[k][m]) < 0.1) * 100:5.1f}" for k in ("z0", "theta")) +
              f"  q16/84 z0 {np.quantile(z['z0'][m], .16):+.3f} {np.quantile(z['z0'][m], .84):+.3f}")

# self-consistency: each variant vs base, track by track, in stock-error units (how far the variant moved our fit)
fb = os.path.join(d, "base", "pulls.npz")
if os.path.exists(fb):
    b = np.load(fb)
    for v in sorted(os.listdir(d)):
        f = os.path.join(d, v, "pulls.npz")
        if v == "base" or not os.path.exists(f):
            continue
        z = np.load(f)
        g = z["good"] & b["good"]
        print(f"{v:8s} vs base: " + " ".join(f"{k} {np.mean(np.abs(z[k][g] - b[k][g]) < 0.1) * 100:5.1f}" for k in KL)
              + "   (% of tracks moved by < 0.1 stock sigma)")

# piece choice: the stock parameters sit on one trajectory piece; how do the neighbouring pieces compare
for v in sorted(os.listdir(d)):
    f = os.path.join(d, v, "pulls.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    if "prev_z0" not in z.files:
        continue
    g = z["good"]
    for tag in ("prev", "next"):
        print(f"{v:8s} {tag} piece: " + " ".join(f"{k} {np.mean(np.abs(z[tag + '_' + k][g]) < 0.1) * 100:5.1f}" for k in KL))
    best = {k: np.min(np.abs(np.stack([z[k], z["prev_" + k], z["next_" + k]])), 0) for k in KL}
    print(f"{v:8s} best of 3: " + " ".join(f"{k} {np.mean(best[k][g] < 0.1) * 100:5.1f}" for k in KL))
