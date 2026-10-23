"""Per-component timing of one F1 iteration on any backend (toy tracks of real size): where the time goes.
python3 scripts/f1_parts.py --tracks 8192 --hits 32 --xings 64"""
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
ap.add_argument("--tracks", type=int, default=8192)
ap.add_argument("--hits", type=int, default=32)
ap.add_argument("--xings", type=int, default=64)
a = ap.parse_args()
rng = np.random.default_rng(5)
T, N, M = a.tracks, a.hits, a.xings
base = []
for i in range(64):
    xt, h, crs, d = make_track(rng, n=N)
    h = dict(h, valid=np.ones(N, bool))
    ex = M - N
    c = {"mid": np.concatenate([crs["mid"], crs["mid"][:ex] + 1.0]), "wdir": np.concatenate([crs["wdir"], crs["wdir"][:ex]]),
         "valid": np.ones(M, bool), "hit": np.concatenate([np.arange(N), -np.ones(ex, int)])}
    base.append((h, c, xt))
idx = np.arange(T) % 64
st = lambda j: {k: jnp.asarray(np.stack([base[i][j][k] for i in idx])) for k in base[0][j]}
hj, cj = st(0), st(1)
x0 = jnp.asarray(np.stack([base[i][2] for i in idx]))
dj = {k: jnp.asarray(d[k]) for k in LJ.DRIFT_KEYS}
ann, stg = K.load_anns(), K.stock_stages()
C0 = jax.vmap(K.seed_covariance)(x0)
xe = jnp.tile(x0[:, None], (1, N + M, 1))
Ce = jnp.tile(C0[:, None], (1, N + M, 1, 1))
hs = {"active": hj["valid"], "lr": hj["lr"], "dvar": jnp.full((T, N), 0.02)}
pr = {"xf": x0, "Cf": C0, "xb": x0, "Cb": C0}
stt = {"active": hj["valid"], "lr": hj["lr"], "frozen": jnp.zeros((T, N), bool), "nulldv": jnp.zeros((T, N), bool)}
sg = {k: v[8] for k, v in stg.items()}


def tm(name, fn, *args):
    g = jax.jit(fn)
    jax.block_until_ready(g(*args))
    t = time.time()
    for _ in range(5):
        r = g(*args)
    jax.block_until_ready(r)
    print(f"{name:28s} {(time.time() - t) / 5 * 1e3:8.1f} ms  ({T} tracks, N {N}, M {M})", flush=True)


tm("hit_rows", jax.vmap(lambda x, h: K.hit_rows(x[:N], h, dj)), xe, hj)
tm("crossing_rows", jax.vmap(lambda x, C, c: K.crossing_rows(x[N:], C[N:], c, 1.0, 1.0)), xe, Ce, cj)
tm("hit_geometry", jax.vmap(lambda x, C, h: K.hit_geometry(x[:N], C[:N], h, dj)), xe, Ce, hj)
tm("update_states", jax.vmap(lambda s, x, C, h: K.update_states(s, x[:N], C[:N], h, dj, sg, ann)[1]), stt, xe, Ce, hj)
tm("iterate_once (all)", jax.vmap(lambda x, C, h, c, s, p: K.iterate_once(x, C, h, c, dj, s, jnp.ones(M, bool), 1.0, 1.0, 1.0, p, 1e6)["chi2"]),
   xe, Ce, hj, cj, hs, pr)
E = N + M
eff = {"xref": xe, "H": jnp.ones((T, E, 3, 5)) * 0.1, "r": jnp.zeros((T, E, 3)), "w": jnp.ones((T, E, 3)),
       "mvar": jnp.ones((T, E, 3)), "D": jnp.ones((T, E, 5, 2)) * 0.01, "sv": jnp.full((T, E), 1e-8)}
W0 = jax.vmap(jnp.linalg.inv)(C0 * 1e6)
tm("filter_smooth (scan)", jax.vmap(lambda e, W: K.filter_smooth(e, W, W @ jnp.zeros(5), W, W @ jnp.zeros(5))["chi2"]), eff, W0)
tm("filter_smooth + smoothed", jax.vmap(lambda e, W: K.smoothed(K.filter_smooth(e, W, jnp.zeros(5), W, jnp.zeros(5)), e)[0]), eff, W0)
tm("batched 5x5 inv [T,E]", lambda C: jnp.linalg.inv(C), Ce)
