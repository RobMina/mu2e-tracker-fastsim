"""JAX twin of combo.panel_hits (CombineStrawHits port): dense per-panel-run tensors, one sequential pass per run.

Hits are packed as [runs, L] (a run = one event's hits in one panel, in collection order; L a bucketed max run
length). The stock greedy loop becomes a fori_loop over the position in the run, vectorised over all runs: the next
unused hit seeds a combo and, if good, absorbs up to MAXNCOMBO-1 later unused good hits passing the straw, time and
wire-distance cuts. Sums are accumulated member by member in collection order, as np.bincount does, so the
outputs are bitwise equal to the numpy port on CPU. Settings are runtime data; L and the run count set the shape.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .combo import MAXNCOMBO, PH_SETTINGS, SEL
from .hitreco import FLAG
from .hitreco_jax import bucket

jax.config.update("jax_enable_x64", True)

F32, F64 = jnp.float32, jnp.float64
SH_COLS = ("straw", "time", "wdist", "uvar", "timevar", "edep", "et0", "et1", "x", "y", "z", "udx", "udy", "flag")


def run_index(sh, settings="mc"):
    """Collection order of the hits and their (event, panel) runs: order, run start, run length."""
    s = PH_SETTINGS[settings]
    n = len(sh["ev"])
    order = np.lexsort((sh["upanel"], sh["ev"])) if s["unsorted"] else np.arange(n)
    ev, up = sh["ev"][order], sh["upanel"][order]
    newrun = np.concatenate([[True], (ev[1:] != ev[:-1]) | (up[1:] != up[:-1])]) if n else np.zeros(0, bool)
    start = np.flatnonzero(newrun)
    length = np.diff(np.append(start, n))
    return order, start, length


def pack_runs(sh, order, start, length, runs, L, R):
    """Runs `runs` (indices into start/length) -> [R, L] tensors, padded rows past len(runs); host glue."""
    nr = len(runs)
    ln = length[runs]
    rr = np.repeat(np.arange(nr), ln)
    pos = np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln)
    src = order[np.repeat(start[runs], ln) + pos]
    t = {"valid": np.zeros((R, L), bool), "src": np.full((R, L), -1, np.int64)}
    t["valid"][rr, pos] = True
    t["src"][rr, pos] = src
    for k in SH_COLS:
        a = np.asarray(sh[k])
        z = np.zeros((R, L), np.int64 if k == "straw" else a.dtype)
        z[rr, pos] = a[src]
        t[k] = z
    return t


OUTS = ("ncombo", "edep", "et0", "et1", "time", "wdist", "x", "y", "z", "uvar", "tdiv_all", "is_combo")


def _group(p, s):
    """Greedy stock loop as a fori_loop over the position in the run, vectorised over runs; carries [R, L] only."""
    R, L = p["valid"].shape
    good = p["valid"] & ((p["flag"] & SEL) == SEL) & ((p["flag"] & (1 << FLAG["dead"])) == 0)
    jpos = jnp.arange(L)[None, :]

    def body(i, state):
        used, owner = state
        col = lambda c: jnp.take(p[c], i, axis=1)[:, None]
        seed = p["valid"][:, i] & ~used[:, i]
        gseed = seed & good[:, i]
        ok = (jpos > i) & ~used & good & gseed[:, None]
        ok &= jnp.abs(p["straw"] - col("straw")) <= s["maxds"]
        ok &= jnp.abs(col("time") - p["time"]) <= s["maxdt"]
        ok &= (jnp.abs(col("wdist") - p["wdist"]) / jnp.sqrt(col("uvar") + p["uvar"])) <= s["maxwdchi"]
        take = ok & (jnp.cumsum(ok, axis=1) <= MAXNCOMBO - 1)
        used = used.at[:, i].set(used[:, i] | seed) | take
        owner = jnp.where(take, i, owner)
        owner = owner.at[:, i].set(jnp.where(gseed, i, owner[:, i]))
        return used, owner

    _, owner = jax.lax.fori_loop(0, L, body, (jnp.zeros((R, L), bool), jnp.full((R, L), -1, jnp.int64)))
    return owner


def _members(owner):
    """Member positions of the combo seeded at each (r, i), in collection order: [R, L, MAXNCOMBO], -1 = none.
    Members of a seed lie at j >= i with owner == i; a stable sort by (owner, j) per run gives them in order."""
    R, L = owner.shape
    jpos = jnp.broadcast_to(jnp.arange(L)[None, :], (R, L))
    key = jnp.where(owner >= 0, owner, L) * L + jpos
    order = jnp.argsort(key, axis=1)                                   # positions grouped by owner, ascending j
    own_sorted = jnp.take_along_axis(owner, order, axis=1)
    start = jax.vmap(lambda o: jnp.searchsorted(o, jnp.arange(L), side="left"))(jnp.where(own_sorted >= 0, own_sorted, L))
    k = jnp.arange(MAXNCOMBO)[None, None, :]
    idx = start[:, :, None] + k                                        # [R, i, k] into the sorted row
    idxc = jnp.minimum(idx, L - 1)
    row = jnp.arange(R)[:, None, None] * L
    pos = order.reshape(-1)[row + idxc]
    ok = (idx < L) & (own_sorted.reshape(-1)[row + idxc] == jnp.arange(L)[None, :, None])
    return jnp.where(ok, pos, -1)


def _outputs(p, owner, s):
    R, L = owner.shape
    mem = _members(owner)                                               # [R, L, K]
    has = mem >= 0
    m = jnp.maximum(mem, 0)
    flat = jnp.arange(R)[:, None, None] * L + m
    g = lambda c: p[c].reshape(-1)[flat]                                 # [R, L, K], no [R, L, L] copy
    ncombo = has.sum(axis=2).astype(jnp.int64)

    def acc(v):
        a = jnp.zeros((R, L), F64)
        for k in range(MAXNCOMBO):
            a = a + jnp.where(has[:, :, k], v[:, :, k], 0.0)
        return a

    wd = g("wdist").astype(F64)
    twt, wwt = 1.0 / g("timevar").astype(F64), 1.0 / g("uvar").astype(F64)
    s_edep, s_et0, s_et1 = acc(g("edep").astype(F64)), acc(g("et0").astype(F64)), acc(g("et1").astype(F64))
    s_tt, s_t = acc(g("time") * twt), acc(twt)
    cx, cy = acc(g("x") - wd * g("udx").astype(F64)), acc(g("y") - wd * g("udy").astype(F64))
    cz = acc(g("z").astype(F64))
    wsum, wacc, wacc2 = acc(wwt), acc(wd * wwt), acc(wd * wd * wwt)
    tdiv = acc(((g("flag") >> FLAG["tdiv"]) & 1).astype(F64))
    nsh = jnp.maximum(ncombo, 1).astype(F64)
    cx, cy, cz = cx / nsh, cy / nsh, cz / nsh
    wdist = wacc / wsum
    multi = ncombo > 1
    uerr = s["uerr"]
    uvar = 1.0 / wsum + uerr ** 2
    ndof2 = (jnp.maximum(ncombo - 1, 1) ** 2).astype(F64)
    uvar = jnp.where(s["checkWres"], jnp.maximum(uvar, jnp.maximum(uerr ** 2, (wacc2 / wsum - wdist ** 2) / ndof2)), uvar)
    out = {"ncombo": ncombo,
           "edep": jnp.where(multi, s_edep / nsh, p["edep"]).astype(F32),
           "et0": jnp.where(multi, s_et0 / nsh, p["et0"]).astype(F32),
           "et1": jnp.where(multi, s_et1 / nsh, p["et1"]).astype(F32),
           "time": jnp.where(multi, s_tt / s_t, p["time"]).astype(F32),
           "wdist": jnp.where(multi, wdist, p["wdist"]).astype(F32),
           "x": jnp.where(multi, cx + wdist * p["udx"].astype(F64), p["x"]).astype(F32),
           "y": jnp.where(multi, cy + wdist * p["udy"].astype(F64), p["y"]).astype(F32),
           "z": jnp.where(multi, cz, p["z"]).astype(F32),
           "uvar": jnp.where(multi, uvar, p["uvar"]).astype(F32),
           "tdiv_all": tdiv == ncombo.astype(F64),
           "is_combo": p["valid"] & (owner == jnp.arange(L)[None, :]),
           "slots": jnp.where(has, g("src"), -1)}
    return out


@jax.jit
def _device(p, s):
    return _outputs(p, _group(p, s), s)


CHUNK = 16384                 # runs per device call (bounded memory; one compiled shape per length bucket)
L_BUCKETS = (16, 32, 64, 128)  # run-length buckets: real runs are 2 hits median, <= 50 seen in 1.6 M digis


def panel_hits(sh, settings="mc", chunk=CHUNK):
    """numpy-in, numpy-out wrapper with the combo.panel_hits columns and constituent lists."""
    s = PH_SETTINGS[settings]
    order, start, length = run_index(sh, settings)
    sd = {"maxds": jnp.int64(s["maxds"]), "maxdt": jnp.float32(s["maxdt"]), "maxwdchi": jnp.float32(s["maxwdchi"]),
          "uerr": jnp.float64(s["uerr"]), "checkWres": jnp.bool_(s["checkWres"])}
    if len(length) and length.max() > L_BUCKETS[-1]:
        raise ValueError(f"panel run of {length.max()} hits exceeds the largest bucket {L_BUCKETS[-1]}")
    bucket_of = np.searchsorted(np.asarray(L_BUCKETS), length)       # smallest bucket >= length
    keys, cols = [], {k: [] for k in OUTS + ("slots",)}
    for b, L in enumerate(L_BUCKETS):
        runs = np.flatnonzero(bucket_of == b)
        R = min(chunk, max(bucket(len(runs), 64), 64))
        for r0 in range(0, len(runs), R):
            rs = runs[r0:r0 + R]
            pk = pack_runs(sh, order, start, length, rs, L, R)
            o = {k: np.asarray(v) for k, v in _device({k: jnp.asarray(v) for k, v in pk.items()}, sd).items()}
            rr, ii = np.nonzero(o["is_combo"][:len(rs)])
            keys.append(start[rs][rr] + ii)                              # collection-order position of the seed
            for k in OUTS:
                cols[k].append(o[k][:len(rs)][rr, ii])
            sl = np.full((len(rr), MAXNCOMBO), -1, np.int64)
            sl[:, :] = o["slots"][:len(rs)][rr, ii, :]
            cols["slots"].append(sl)
    key = np.concatenate(keys) if keys else np.zeros(0, np.int64)
    srt = np.argsort(key, kind="stable")                                 # stock output order = seed order
    out = {k: (np.concatenate(v) if v else np.zeros(0))[srt] for k, v in cols.items()}
    res = {k: out[k] for k in ("ncombo", "edep", "et0", "et1", "time", "wdist", "x", "y", "z", "uvar")}
    slots = out["slots"].reshape(-1, MAXNCOMBO).astype(np.int64)
    first = slots[:, 0]
    res["ev"], res["sid"] = sh["ev"][first], sh["sid"][first]
    r2 = res["x"].astype(np.float64) ** 2 + res["y"].astype(np.float64) ** 2
    nc = res["ncombo"].astype(np.int64)
    flag = np.uint32(1 << FLAG["panelcombo"]) | np.where(out["tdiv_all"].astype(bool), np.uint32(1 << FLAG["tdiv"]), np.uint32(0))
    for bit, ok in ((FLAG["nhitsel"], (nc >= s["minN"]) & (nc <= s["maxN"])),
                    (FLAG["timesel"], (res["time"] >= s["minT"]) & (res["time"] <= s["maxT"])),
                    (FLAG["energysel"], (res["edep"] >= s["minE"]) & (res["edep"] <= s["maxE"])),
                    (FLAG["radsel"], (r2 >= s["minR"] ** 2) & (r2 <= s["maxR"] ** 2))):
        flag = flag | np.where(ok, np.uint32(1 << bit), np.uint32(0))
    res["ncombo"], res["flag"] = nc, flag
    m = slots >= 0
    res["ind_file"] = slots[m]
    res["ind_combo"] = np.repeat(np.arange(len(first)), nc)
    ev0 = np.searchsorted(sh["ev"], sh["ev"], side="left")
    res["ind"] = res["ind_file"] - ev0[res["ind_file"]]
    return res
