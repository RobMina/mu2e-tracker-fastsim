"""Per-crossing material diagnostic from reco-vr2f1 xings.npz: our gas/wall paths vs the stock KalSeed values, and
the (|DOCA|, sigma) that would reproduce the stock paths (inverted numerically). python3 scripts/f1_xing_diag.py DIR"""
import os
import sys

import numpy as np

from trkfastsim.reco import kkmaterial as km

z = np.load(os.path.join(sys.argv[1], "xings.npz"))
ad, sg = np.abs(z["doca"]), np.sqrt(np.maximum(z["dvar"], 0))
dg, dw = z["gas"] - z["kst_gas"], z["wall"] - z["kst_wall"]
print(f"crossings {len(ad)}; with hit {z['hashit'].mean() * 100:.1f} %")
print("sigma(DOCA) q16/50/84 [mm]", np.round(np.quantile(sg, [.16, .5, .84]), 3))
for lo, hi in ((0, .5), (.5, 1.5), (1.5, 2.2), (2.2, 2.6), (2.6, 9)):
    m = (ad >= lo) & (ad < hi)
    if m.sum() == 0:
        continue
    print(f"|doca| {lo:.1f}-{hi:.1f}: n {m.sum():6d}  gas diff q16/50/84 {np.round(np.quantile(dg[m], [.16, .5, .84]), 3)}"
          f"  wall diff q50 {np.quantile(dw[m], .5):+.4f}  sigma q50 {np.quantile(sg[m], .5):.3f}")
for hh in (True, False):
    m = z["hashit"] == hh
    print(f"hashit {hh}: n {m.sum()} gas diff q16/50/84 {np.round(np.quantile(dg[m], [.16, .5, .84]), 3)}")
# the stock DOCA implied by the stock gas path at our sigma (gas path is monotonic in |doca| at fixed sigma)
grid = np.linspace(0, 2.6, 521)
sel = np.arange(len(ad))[:20000]
imp = np.empty(len(sel))
for j, i in enumerate(sel):
    g = np.asarray(km.path_lengths(grid, np.full_like(grid, z["dvar"][i]), np.full_like(grid, z["dirdot"][i]), 1.0)[1])
    imp[j] = grid[np.argmin(np.abs(g - z["kst_gas"][i]))]
d = imp - ad[sel]
ok = ad[sel] < 2.3
print("implied stock |doca| - ours (|doca| < 2.3) q16/50/84 [mm]", np.round(np.quantile(d[ok], [.16, .5, .84]), 3))
