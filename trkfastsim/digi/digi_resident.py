"""Device-resident packing for the JAX digitiser: digi_jax.digitize with its host glue on the device.

Opt-in. `digitize(st, nev, c, key, ...)` takes flat device step arrays (the digi_jax column set plus `ev`) and returns
the digi_jax output dict as device arrays. It runs the same kernels with the same keys, the same cluster order and the
same per-call batches as digi_jax.digitize, so for the same inputs and key it gives the same digis bit for bit: every
random draw in the digitiser is keyed by the original step index, the cluster ordinal, the global cluster index or the
straw-instance key, never by a batch position. What moves to the device is the work between the kernels: the step
time window, the cluster expansion and its order, straw-instance grouping, microbunch ghosts, the per-bucket gathers
and sorts, and the final compaction and sort. These run as jitted stages on arrays padded to power-of-two sizes, so a
new chunk recompiles only when a size crosses a power of two; the host reads only counts and the per-batch failure
flags that digi_jax also reads.

The tracker device float must be float64 (devfloat default); in float32 mode use digi_jax.digitize.
"""
from functools import lru_cache, partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import random

from trkfastsim import devfloat as df
from trkfastsim.digi import digi_clusters_jax as DC
from trkfastsim.digi import digi_jax as DJ
from trkfastsim.digi import digitize as reference

jax.config.update("jax_enable_x64", True)

COLS = ("shape", "ion", "eion", "len", "width", "t", "x0", "y0", "z0", "x1", "y1", "z1", "sid", "ev")
CLASSES = (64, DC.MAX_CLUSTER_BUCKET)      # capacity classes of the multi-cluster sampling (draws do not depend on it)
# minion fill: loop bodies per device-loop trip and the searchsorted method (digi_clusters_jax: both leave the draws
# and outputs unchanged; fewer device-loop predicate reads)
MINION_UNROLL, SEARCH_METHOD = 8, "compare_all"
MAX_DRIFT_CLUSTERS = 16_000_000            # no host copies here: the device buffers set the limit
BIG = np.int64(2) ** 62


def _b(n, lo=256):
    """Power-of-two bucket of a size."""
    return max(lo, 1 << int(np.ceil(np.log2(max(int(n), 1)))))


def _padto(a, n, value=0):
    return jnp.pad(a, ((0, n - a.shape[0]),) + ((0, 0),) * (a.ndim - 1), constant_values=value)


def _unique_straw(sid):
    return ((sid >> 10) * reference.NPANELS + ((sid >> 7) & 0x7)) * reference.NSTRAWS + (sid & 0x7F)


def _unique_panel(sid):
    return (sid >> 10) * reference.NPANELS + ((sid >> 7) & 0x7)


# --------------------------------------------------------------------------- stage A: step window

@lru_cache(maxsize=8)
def _window(c):
    allp = jnp.asarray(np.asarray(c.allPlanes, np.int64))
    start = c.digitizationStart - c.timeFromProtonsToDRMarker
    mb = float(c.mbtime)

    @jax.jit
    def run(st, nstep, pbtime, evlen):
        sid, ev = st["sid"], st["ev"]
        valid = jnp.arange(sid.shape[0]) < nstep
        ra = ((sid & 0x7f) >= c.allStraw) & jnp.isin(sid >> 10, allp)
        time = st["t"].astype(jnp.float64) + pbtime[ev]
        if c.onspill:
            time = jnp.fmod(time, mb)
            time = jnp.where(time < 0, time + mb, time)
            end = c.digitizationEnd - c.timeFromProtonsToDRMarker
        else:
            end = evlen[ev]
        inwin = (time > start - c.electronicsTimeDelay - c.steptimebuf) & \
            (time < jnp.maximum(c.mbtime, end) - c.electronicsTimeDelay + c.steptimebuf)
        return time, valid & (st["eion"] > c.minstepE) & (inwin | ra), _unique_straw(sid)
    return run


@partial(jax.jit, static_argnames=("size",))
def _compact(mask, size):
    return jnp.nonzero(mask, size=size, fill_value=0)[0]


# --------------------------------------------------------------------------- stage B: clusters

@lru_cache(maxsize=8)
def _cluster_prep(c):
    @jax.jit
    def run(st, idx, nidx):
        valid = jnp.arange(idx.shape[0]) < nidx
        eion = st["eion"][idx].astype(jnp.float64)
        length = st["len"][idx].astype(jnp.float64)
        single = (st["shape"][idx] == 3) | (length < c.meanFreePath)
        minion = st["ion"][idx] == 0
        p0 = jnp.stack([st[k][idx] for k in ("x0", "y0", "z0")], 1).astype(jnp.float64)
        p1 = jnp.stack([st[k][idx] for k in ("x1", "y1", "z1")], 1).astype(jnp.float64)
        return eion, length, single, minion, p0, p1, valid
    return run


@partial(jax.jit, static_argnames=("cap", "k"))
def _class_slots(key, mm, nmm, idx, eion, nc, minion, p0, p1, sdir, width, eaverage, energies, probabilities, cap, k):
    """DC._sample on the multi steps mm[:nmm] (padded to k); the kept slots with their sort key local * 2^13 + slot."""
    lane = jnp.arange(k) < nmm
    m = jnp.where(lane, mm, 0)
    pos, ne = DC._sample(key, idx[m], eion[m], jnp.where(lane, nc[m], 0), minion[m], p0[m], p1[m], sdir[m], width[m],
                         eaverage, energies, probabilities, cap, MINION_UNROLL, SEARCH_METHOD)
    slot = jnp.arange(cap)[None, :]
    keep = lane[:, None] & (slot < nc[m][:, None])
    skey = jnp.where(keep, m[:, None] * 8192 + slot, BIG)
    return skey.ravel(), pos.reshape(-1, 3), ne.ravel()


@partial(jax.jit, static_argnames=("total_b",))
def _expand(skey, pos, ne, minion, idx, s_rows, n_single, n_multi_slots, total_b):
    """Cluster arrays in digi_clusters_jax.make_clusters order: single steps (in idx order), then the multi-step slots
    in (step, slot) order with each minion slot repeated ne times (ne -> 1), padded to total_b."""
    order = jnp.argsort(skey, stable=True)
    skey, pos, ne = skey[order], pos[order], ne[order]
    local = skey // 8192
    valid = jnp.arange(skey.shape[0]) < n_multi_slots
    isminion = minion[jnp.where(valid, local, 0)]
    repeats = jnp.where(valid, jnp.where(isminion, ne, 1), 0)
    rep = jnp.repeat(jnp.arange(skey.shape[0]), repeats, total_repeat_length=total_b)
    j = jnp.arange(total_b)
    from_single = j < n_single
    sj = jnp.minimum(j, s_rows.shape[0] - 1)
    mj = jnp.clip(j - n_single, 0, total_b - 1)
    r = rep[mj]
    cs = jnp.where(from_single, idx[s_rows[sj]], idx[jnp.where(valid[r], local[r], 0)])
    return cs, r, jnp.where(isminion[r], 1, ne[r]), pos[r], from_single, sj


def make_clusters(key, c, st, idx, nidx):
    """digi_clusters_jax.make_clusters on padded device arrays: (cluster step indices, positions [n, 3], ne, ncl), the
    first ncl entries in the reference order with the reference draws."""
    eion, length, single, minion, p0, p1, valid = _cluster_prep(c)(st, idx, nidx)
    energies = jnp.asarray(c.EIonize, jnp.float64)
    probabilities = jnp.asarray(c.intNProb, jnp.float64)
    nc, single_ne = DC._counts(key, idx, eion, length, single & valid, minion, c.meanFreePath, c.EAverage,
                               np.asarray(c.EIonize, np.float64)[0], c.maxnclu)
    single = single & valid
    multi = valid & ~single & (nc > 0)
    n_single, n_multi = int(jnp.sum(single)), int(jnp.sum(multi))
    s_rows = _compact(single, _b(n_single))
    ib = idx.shape[0]
    if c.randrad:
        sdir = jnp.asarray(c.straw_dir, jnp.float64)[st["_us"][idx]]
        width = st["width"][idx].astype(jnp.float64)
    else:
        sdir, width = jnp.zeros((ib, 3)), jnp.zeros(ib)
    keys, poss, nes = [], [], []
    if n_multi:
        ncmax = int(jnp.max(jnp.where(multi, nc, 0)))
        if ncmax > DC.MAX_CLUSTER_BUCKET:
            raise OverflowError(f"cluster count {ncmax} exceeds MAX_CLUSTER_BUCKET={DC.MAX_CLUSTER_BUCKET}")
        lo = 0
        for cap in CLASSES:
            sel = multi & (nc > lo) & (nc <= cap)
            lo = cap
            nsel = int(jnp.sum(sel))
            if not nsel:
                continue
            cap_used = cap if cap == CLASSES[0] else _b(ncmax, 128)
            lanes = max(1, DC.MAX_SAMPLE_LANES // cap_used)
            mm_all = _compact(sel, _b(nsel))
            for s0 in range(0, nsel, lanes):
                nm = min(lanes, nsel - s0)
                k = _b(nm, 16)
                mm = jax.lax.dynamic_slice(_padto(mm_all, max(mm_all.shape[0], s0 + k)), (s0,), (k,))
                sk, ps, ns_ = _class_slots(key, mm, nm, idx, eion, nc, minion, p0, p1, sdir, width, c.EAverage,
                                           energies, probabilities, cap_used, k)
                keys.append(sk); poss.append(ps); nes.append(ns_)
    if keys:
        n_slots = sum(int(x.shape[0]) for x in keys)
        sb = _b(n_slots)
        skey = _padto(jnp.concatenate(keys), sb, BIG)
        pos = _padto(jnp.concatenate(poss), sb)
        ne = _padto(jnp.concatenate(nes), sb)
        n_multi_slots = int(jnp.sum(skey < BIG))
        reps = jnp.where(minion[jnp.where(skey < BIG, skey // 8192, 0)], ne, 1)
        total = int(jnp.sum(jnp.where(skey < BIG, reps, 0)))
    else:
        skey, pos, ne = jnp.full(256, BIG), jnp.zeros((256, 3)), jnp.zeros(256, jnp.int64)
        n_multi_slots = total = 0
    ncl = n_single + total
    if ncl > MAX_DRIFT_CLUSTERS:
        raise OverflowError(f"electron expansion exceeds MAX_DRIFT_CLUSTERS={MAX_DRIFT_CLUSTERS}")
    tb = _b(ncl)
    cs, r, ne_m, pos_m, from_single, sj = _expand(skey, pos, ne, minion, idx, s_rows, n_single, n_multi_slots, tb)
    pos_out = jnp.where(from_single[:, None], p0[s_rows[sj]], pos_m)
    ne_out = jnp.where(from_single, single_ne[s_rows[sj]], ne_m)
    return cs, pos_out, ne_out, ncl


# --------------------------------------------------------------------------- stage D: straw instances and ghosts

@partial(jax.jit, static_argnames=("ns_b",))
def _instances(ev, sid, cs, ncl, ns_b):
    n = cs.shape[0]
    valid = jnp.arange(n) < ncl
    skey = jnp.where(valid, ev[cs] * 65536 + sid[cs], BIG)
    order = jnp.argsort(skey, stable=True)
    ss = skey[order]
    first = jnp.concatenate([jnp.ones(1, bool), ss[1:] != ss[:-1]]) & (ss < BIG)
    inst = jnp.zeros(n, jnp.int64).at[order].set(jnp.cumsum(first) - 1)
    ukey = jnp.where(jnp.arange(ns_b) < jnp.sum(first), ss[jnp.nonzero(first, size=ns_b, fill_value=0)[0]], 0)
    return jnp.where(valid, inst, ns_b), ukey, jnp.sum(first)


@partial(jax.jit, static_argnames=("g_b", "ns_b"))
def _ghost_tables(gt, ncl, inst, mbtime, buffer, g_b, ns_b):
    """timing.ghost_indices on the 2 ncl cluster ends, then the per-(instance, end) ghost lists (digi_jax)."""
    n2 = gt.shape[0] * 2
    flat = gt.ravel()
    live = jnp.arange(n2) < 2 * ncl
    original = flat.astype(jnp.float32)
    t = original.astype(jnp.float64)
    early = live & (t < buffer)
    late = live & (t > mbtime - buffer)
    ne_, nl_ = jnp.sum(early), jnp.sum(late)
    ei = jnp.nonzero(early, size=n2, fill_value=0)[0]
    li = jnp.nonzero(late, size=n2, fill_value=0)[0]
    j = jnp.arange(g_b)
    n0 = 2 * ncl
    src = jnp.where(j < n0, j, jnp.where(j < n0 + ne_, ei[jnp.clip(j - n0, 0, n2 - 1)],
                                         li[jnp.clip(j - n0 - ne_, 0, n2 - 1)]))
    gtime = jnp.where(j < n0, original[jnp.minimum(j, n2 - 1)],
                      jnp.where(j < n0 + ne_, (t[src] + mbtime).astype(jnp.float32), (t[src] - mbtime).astype(jnp.float32)))
    gvalid = j < n0 + ne_ + nl_
    row = jnp.where(gvalid, 2 * inst[src // 2] + src % 2, 2 * ns_b)
    end_counts = jnp.bincount(row, length=2 * ns_b + 1)[:2 * ns_b].reshape(ns_b, 2)
    order = jnp.argsort(row, stable=True)
    starts = (jnp.cumsum(end_counts.ravel()) - end_counts.ravel()).reshape(ns_b, 2)
    return src, gtime, end_counts, order, starts, n0 + ne_ + nl_


@partial(jax.jit, static_argnames=("ns_b",))
def _offspill_tables(inst, ncl, ns_b):
    counts = jnp.bincount(inst, length=ns_b + 1)[:ns_b]
    order = jnp.argsort(inst, stable=True)
    return counts, order, jnp.cumsum(counts) - counts, ncl


# --------------------------------------------------------------------------- stage E: per-batch packing

@partial(jax.jit, static_argnames=("cap", "batch", "onspill"))
def _pack(which, nb, cap, batch, onspill, tabs, gt, charge, wd, cs, straw):
    """digi_jax's per-batch t, q, w, steps, n (zeros beyond nb) and the straw arguments."""
    lane = jnp.arange(batch) < nb
    w_ = jnp.where(lane, which, 0)
    ar = jnp.arange(cap)[None, :]
    T, Q, W, S, N = [], [], [], [], []
    if not onspill:
        counts, order, starts, ncl = tabs
        gather = starts[w_, None] + ar
        mask = ar < counts[w_, None]
        ci_ = order[jnp.minimum(gather, ncl - 1)]
    for e in (0, 1):
        if onspill:
            src, gtime, end_counts, order, starts, nglen = tabs
            gather = starts[w_, e, None] + ar
            mask = ar < end_counts[w_, e, None]
            gi = order[jnp.minimum(gather, nglen - 1)]
            oo = jnp.argsort(jnp.where(mask, gtime[gi], jnp.inf), axis=1, stable=True)
            gi = jnp.take_along_axis(gi, oo, 1)
            ix = src[gi] // 2
            T.append(gtime[gi].astype(jnp.float64)); N.append(end_counts[w_, e])
        else:
            oo = jnp.argsort(jnp.where(mask, gt[ci_, e], jnp.inf), axis=1, stable=True)
            ix = jnp.take_along_axis(ci_, oo, 1)
            T.append(gt[ix, e].astype(jnp.float64)); N.append(counts[w_])
        Q.append(charge[ix].astype(jnp.float64)); W.append(wd[ix, e].astype(jnp.float64)); S.append(cs[ix])
    z = lambda a: jnp.where(lane.reshape((-1,) + (1,) * (a.ndim - 1)), a, 0)
    t, q, w, s = (z(jnp.stack(x, 1)) for x in (T, Q, W, S))
    n = z(jnp.stack(N, 1)).astype(jnp.int32)
    straw_us, straw_ra, ukey, jitter, lengths = straw
    st_args = (z(straw_us[w_]), z(straw_ra[w_]), z(ukey[w_]), z(jitter[w_]))
    return (t.reshape(2 * batch, cap), q.reshape(2 * batch, cap), w.reshape(2 * batch, cap),
            s.reshape(2 * batch, cap), n.ravel(), *st_args, lane, jnp.where(lane, lengths[w_], 0.0))


@partial(jax.jit, static_argnames=("pair_cap", "batch"), donate_argnames=("buf",))
def _append(buf, offset, result, which, straw_ev, straw_sid, pair_cap, batch):
    """Kept digis of one batch (row-major, as np.nonzero) written into the output buffers at offset."""
    keep = result["keep"].ravel()
    m = batch * pair_cap
    kp = jnp.nonzero(keep, size=m, fill_value=0)[0]
    bi, pi = kp // pair_cap, kp % pair_cap
    out = {k: result[k][bi, pi] for k in ("adc", "tdc", "tot", "pmp", "step", "t_analog")}
    out["ev"] = straw_ev[which[bi]]
    out["sid"] = straw_sid[which[bi]]
    return {k: jax.lax.dynamic_update_slice(buf[k], out[k].astype(buf[k].dtype), (offset,) + (0,) * (out[k].ndim - 1))
            for k in buf}, jnp.sum(keep)


@partial(jax.jit, static_argnames=("size",))
def _final(buf, total, size):
    valid = jnp.arange(buf["ev"].shape[0]) < total
    order = jnp.lexsort((jnp.where(valid, buf["t_analog"][:, 0], jnp.inf), jnp.where(valid, buf["sid"], BIG),
                         jnp.where(valid, buf["ev"], BIG)))
    return {k: v[order[:size]] for k, v in buf.items()}


def _jitter(key, straw_ev, panel, nev, scale):
    """digi_jax._panel_jitter: the same (JITTER_BATCH, NUPANELS) blocks, gathered on the device."""
    nb = (nev + DJ.JITTER_BATCH - 1) // DJ.JITTER_BATCH
    blocks = jnp.stack([DJ._jitter_block(random.fold_in(key, b)) for b in range(nb)])
    e = jnp.clip(straw_ev, 0, nev - 1)
    return blocks[e // DJ.JITTER_BATCH, e % DJ.JITTER_BATCH, panel] * scale


def digitize(st, nev, c, key, stats=None, *, event_timing=None):
    """digi_jax.digitize with device-resident packing (same digis for the same inputs and key). st: flat device step
    arrays (COLS; `ev` the event index in [0, nev)). Returns a dict of device arrays."""
    if df.SINGLE:
        raise NotImplementedError("digi_resident: float64 tracker device float only (use digi_jax.digitize)")
    nstep = int(np.shape(st["ev"])[0])
    sb = _b(nstep)
    s = {k: _padto(jnp.asarray(st[k]), sb) for k in COLS}
    s["sid"], s["ev"] = s["sid"].astype(jnp.int64), s["ev"].astype(jnp.int64)
    eb = _b(nev, 64)
    if event_timing is None:
        if c.onspill:
            raise ValueError("event_timing with pbtime and event_length is required on spill")
        pbtime, evlen = jnp.zeros(eb), jnp.full(eb, float(c.eventWindowLength))
    else:
        pbtime = _padto(jnp.asarray(event_timing["pbtime"], jnp.float64), eb)
        evlen = _padto(jnp.asarray(event_timing["event_length"], jnp.float64), eb, float(c.eventWindowLength))
    time, sel, us = _window(c)(s, nstep, pbtime, evlen)
    s["_us"] = us
    nidx = int(jnp.sum(sel))
    if not nidx:
        return {k: jnp.asarray(v) for k, v in DJ._empty(c).items()}
    idx = _compact(sel, _b(nidx))
    cs, pos, ne, ncl = make_clusters(random.fold_in(key, 0), c, s, idx, nidx)
    if not ncl:
        return {k: jnp.asarray(v) for k, v in DJ._empty(c).items()}
    tb = cs.shape[0]
    ctime = time[cs]
    group = DJ.CHARGE_BLOCK * DJ.CHARGE_BLOCKS
    kern = DJ._charge_blocks_kernel(c)
    gt_l, ch_l, wd_l = [], [], []
    usc = us[cs]
    padg = (-tb) % group
    usc, posg, neg, ctg = _padto(usc, tb + padg), _padto(pos, tb + padg), _padto(ne, tb + padg, 1), _padto(ctime, tb + padg)
    nmask = jnp.arange(tb + padg) < ncl
    usc, posg, neg, ctg = (jnp.where(nmask.reshape((-1,) + (1,) * (a.ndim - 1)), a, v)
                           for a, v in ((usc, 0), (posg, 0.0), (neg, 1), (ctg, 0.0)))
    for first in range(0, ncl, group):
        r = kern(key, jnp.int64(first), *(jax.lax.dynamic_slice_in_dim(a, first, group) for a in (usc, posg, neg, ctg)))
        gt_l.append(r[0]); ch_l.append(r[1]); wd_l.append(r[2])
    gt, charge, wd = (_padto(jnp.concatenate(x), tb) if sum(y.shape[0] for y in x) < tb else
                      jnp.concatenate(x)[:tb] for x in (gt_l, ch_l, wd_l))
    inst, ukey, ns = _instances(s["ev"], s["sid"], cs, ncl, _b(ncl))      # ns <= ncl
    ns = int(ns)
    ns_b = _b(ns)
    ukey = ukey[:ns_b] if ukey.shape[0] >= ns_b else _padto(ukey, ns_b)
    inst = jnp.where(inst < ns, inst, ns_b)
    straw_ev, straw_sid = ukey // 65536, ukey % 65536
    straw_us = _unique_straw(straw_sid)
    allp = jnp.asarray(np.asarray(c.allPlanes, np.int64))
    straw_ra = ((straw_sid & 0x7f) >= c.allStraw) & jnp.isin(straw_sid >> 10, allp)
    jitter = _jitter(random.fold_in(key, 2), straw_ev, _unique_panel(straw_sid), nev, c.eventWindowMarkerROCJitter)
    lengths = evlen[jnp.clip(straw_ev, 0, eb - 1)]
    if c.onspill:
        buffer = (c.nADCSamples - c.nADCPresamples) * c.ADCPeriod
        tabs = _ghost_tables(gt, ncl, inst, float(c.mbtime), float(buffer), _b(4 * tb), ns_b)
        counts = tabs[2].max(axis=1)
    else:
        tabs = _offspill_tables(inst, ncl, ns_b)
        counts = tabs[0]
    counts_h = np.asarray(counts[:ns])
    if counts_h.max() > DJ.MAX_ROW_CLUSTERS:
        raise OverflowError(f"waveform cluster count exceeds MAX_ROW_CLUSTERS={DJ.MAX_ROW_CLUSTERS}; no output returned")
    caps_h = np.left_shift(np.int64(1), np.ceil(np.log2(counts_h)).astype(np.int64))
    straw = (straw_us, straw_ra, ukey, jitter, lengths)
    obuf = _b(4 * ns, 1024)
    buf = None
    offset = 0
    for cap in np.unique(caps_h):
        cap = int(cap)
        bucket_h = np.flatnonzero(caps_h == cap)
        batch = DJ._waveform_batch(cap)
        for b0 in range(0, len(bucket_h), batch):
            wh = bucket_h[b0:b0 + batch]
            nb = len(wh)
            which = jnp.asarray(np.pad(wh, (0, batch - nb)))
            t, q, w, stp, n, su, sr, sk, ji, live, lens = _pack(which, nb, cap, batch, bool(c.onspill), tabs, gt,
                                                                 charge, wd, cs, straw)
            timing_args = {}
            if event_timing is not None:
                lens = jnp.where(live, lens, float(c.eventWindowLength))
                timing_args = {"window_end": lens,
                               "event_end_tdc": jnp.clip(jnp.floor(((lens - c.electronicsTimeDelay)
                                                                    + c.electronicsTimeDelay) / c.TDCLSB),
                                                         0, c.maxTDC).astype(jnp.int64)}
            pair_cap = 4
            while True:
                result = DJ._electronics_kernel(c, cap, pair_cap, batch)(
                    random.fold_in(key, 3), t, q, w, stp, n, su, sr, sk, live, ji, **timing_args)
                if bool(result["sat_overflow"]):
                    raise RuntimeError("JAX ADC saturation time bucket overflow; no output returned")
                if not bool(result["failed"]):
                    break
                if pair_cap >= 2 * cap:
                    raise RuntimeError("JAX threshold search failed to progress; no output returned")
                pair_cap *= 2
            need = offset + batch * pair_cap
            if buf is None or need > buf["ev"].shape[0]:
                size = _b(max(need, obuf))
                tmpl = {"adc": ((c.nADCSamples,), result["adc"].dtype), "tdc": ((2,), result["tdc"].dtype),
                        "tot": ((2,), result["tot"].dtype), "pmp": ((), result["pmp"].dtype),
                        "step": ((2,), result["step"].dtype), "t_analog": ((2,), result["t_analog"].dtype),
                        "ev": ((), jnp.int64), "sid": ((), jnp.int64)}
                new = {k: jnp.zeros((size,) + sh, dt) for k, (sh, dt) in tmpl.items()}
                if buf is not None:
                    new = {k: jax.lax.dynamic_update_slice(new[k], buf[k], (0,) * new[k].ndim) for k in new}
                buf = new
            buf, kept = _append(buf, jnp.int64(offset), result, which, straw_ev, straw_sid, pair_cap, batch)
            offset += int(kept)
            if stats is not None:
                for field, val in (("pairs", result["pairs"]), ("nan_crossing_ends", result["nan_ends"]),
                                   ("saturated_ends", result["saturated_ends"]), ("jax_buckets", 1)):
                    stats[field] = stats.get(field, 0) + int(val)
    if stats is not None:
        for field, val in (("clusters", ncl), ("straws", ns), ("ends", 2 * ns), ("digis", offset)):
            stats[field] = stats.get(field, 0) + int(val)
    if buf is None or not offset:
        return {k: jnp.asarray(v) for k, v in DJ._empty(c).items()}
    out = _final(buf, offset, _b(offset))
    return {k: v[:offset] for k, v in out.items()}
