"""F1 throughput benchmark on toy tracks of real size (N hits, M crossings), any JAX backend:
python3 scripts/f1_bench.py --tracks 4096 --hits 64 --xings 96 [--repeat 3]. F1_SEQUENTIAL=1 selects the sequential filter."""
import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from kalman_toy import make_track  # noqa: E402
from trkfastsim.reco import kalman_jax as K, linefit_jax as LJ  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tracks", type=int, default=1024)
ap.add_argument("--hits", type=int, default=64)
ap.add_argument("--xings", type=int, default=96)
ap.add_argument("--repeat", type=int, default=3)
a = ap.parse_args()
rng = np.random.default_rng(5)
N, M = a.hits, a.xings
base = []
for i in range(64):
    xt, h, crs, d = make_track(rng, n=N)
    h = dict(h, valid=np.ones(N, bool))
    extra = M - N
    c = {"mid": np.concatenate([crs["mid"], crs["mid"][:extra] + 1.0]), "wdir": np.concatenate([crs["wdir"], crs["wdir"][:extra]]),
         "valid": np.ones(M, bool), "hit": np.concatenate([np.arange(N), -np.ones(extra, int)])}
    base.append((h, c, xt + rng.normal(0, 1, 5) * np.array([2, 2, .005, .005, 1.])))
idx = np.arange(a.tracks) % len(base)
stack = lambda j: {k: jnp.asarray(np.stack([base[i][j][k] for i in idx])) for k in base[0][j]}
hj, cj = stack(0), stack(1)
x0 = jnp.asarray(np.stack([base[i][2] for i in idx]))
dj = {k: jnp.asarray(d[k]) for k in LJ.DRIFT_KEYS}
ann, stg = K.load_anns(), K.stock_stages()
f = jax.jit(jax.vmap(lambda x, h, c: K.fit_track(x, h, c, dj, stg, ann)[2]))
t = time.time(); g = f(x0, hj, cj); jax.block_until_ready(g); first = time.time() - t
ts = []
for _ in range(a.repeat):
    t = time.time(); g = f(x0, hj, cj); jax.block_until_ready(g); ts.append(time.time() - t)
w = min(ts)
print(f"backend {jax.default_backend()} parallel {K.PARALLEL} tracks {a.tracks} N {N} M {M}: first {first:.1f} s, "
      f"warm {w:.3f} s = {a.tracks / w:.0f} tracks/s, good {float(np.mean(np.asarray(g))):.3f}", flush=True)
