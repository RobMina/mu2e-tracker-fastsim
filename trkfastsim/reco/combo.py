"""Vectorised numpy port of Offline CombineStrawHits (makePH): greedy per-panel grouping of single-straw hits.

The stock loop takes hits in collection order; a seed absorbs later unused hits of the same contiguous panel run
(|dstraw| <= MaxDS, |dt| <= MaxDt on corrected time, wire-distance pull <= MaxWireDistDiffPull) up to MaxNCombo.
Here one round = one seed per panel run, all runs at once; rounds repeat until every hit is consumed.
"""
import numpy as np

from .hitreco import F32, FLAG

MAXNCOMBO = 8
PH_SETTINGS = {
    "mc":   dict(maxdt=45.0, maxwdchi=5.0, maxds=3, uerr=10.0, minT=0.0, maxT=100.0e3, minE=0.0, maxE=1.0,
                 minR=300.0, maxR=750.0, minN=1, maxN=8, checkWres=True, unsorted=False),
    "data": dict(maxdt=45.0, maxwdchi=5.0, maxds=3, uerr=10.0, minT=0.0, maxT=256000.0, minE=0.0, maxE=1.0,
                 minR=300.0, maxR=750.0, minN=1, maxN=8, checkWres=True, unsorted=True),
}
SEL = (1 << FLAG["energysel"]) | (1 << FLAG["timesel"]) | (1 << FLAG["radsel"])


def panel_hits(sh, settings="mc"):
    """Single-straw hit columns (hitreco.straw_hits) -> panel ComboHit columns, constituents as SH indices."""
    s = PH_SETTINGS[settings]
    n = len(sh["ev"])
    order = np.lexsort((sh["upanel"], sh["ev"])) if s["unsorted"] else np.arange(n)  # lexsort is stable
    ev, up = sh["ev"][order], sh["upanel"][order]
    run = np.concatenate([[0], np.cumsum((ev[1:] != ev[:-1]) | (up[1:] != up[:-1]))]) if n else np.zeros(0, int)
    good = ((sh["flag"][order] & SEL) == SEL) & ((sh["flag"][order] & (1 << FLAG["dead"])) == 0)
    straw, t, wd, wv = sh["straw"][order].astype(np.int64), sh["time"][order], sh["wdist"][order], sh["uvar"][order]
    used = np.zeros(n, bool)
    owner = np.full(n, -1)                          # seed position (in sorted order) of the combo each hit joined
    runstart = np.searchsorted(run, np.arange(run[-1] + 1 if n else 0))
    runend = np.append(runstart[1:], n)
    while not used.all():
        # seed = first unused hit of every run that still has one
        cand = np.flatnonzero(~used)
        seeds = cand[np.concatenate([[True], run[cand[1:]] != run[cand[:-1]]])]
        used[seeds] = True
        gs = seeds[good[seeds]]
        owner[gs] = gs
        if len(gs) == 0:
            continue
        # candidate pairs: later unused good hits in the seed's run (dense per seed, bounded by run length)
        lens = runend[run[gs]] - gs - 1
        rep = np.repeat(gs, lens)
        j = rep + 1 + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
        ok = (~used[j]) & good[j] & (np.abs(straw[j] - straw[rep]) <= s["maxds"])
        ok &= np.abs(t[rep] - t[j]) <= F32(s["maxdt"])
        ok &= F32(np.abs(wd[rep] - wd[j]) / np.sqrt(wv[rep] + wv[j], dtype=F32)) <= F32(s["maxwdchi"])
        cs = np.cumsum(ok)
        f0 = np.minimum(np.cumsum(lens) - lens, max(len(j) - 1, 0))  # first pair of each seed (clamped for empty)
        rank = cs - np.repeat(cs[f0] - ok[f0], lens) if len(j) else cs
        take = ok & (rank <= MAXNCOMBO - 1)
        used[j[take]] = True
        owner[j[take]] = rep[take]
    # assemble combos: grouped by seed, constituents in collection order
    member = np.flatnonzero(owner >= 0)
    grp = np.lexsort((member, owner[member]))
    member = member[grp]
    seed_of = owner[member]
    starts = np.flatnonzero(np.concatenate([[True], seed_of[1:] != seed_of[:-1]]))
    ncombo = np.diff(np.append(starts, len(member)))
    combo_order = np.argsort(seed_of[starts], kind="stable")      # stock output order = seed order
    return _combine(sh, order, member, starts, ncombo, combo_order, s)


def _combine(sh, order, member, starts, ncombo, corder, s):
    """CombineStrawHits::combineHits sums (single-hit combos copy the hit) + the panel-level flags."""
    idx = order[member]                                 # SH collection indices of the constituents
    cid = np.repeat(np.arange(len(starts)), ncombo)
    nsh = ncombo.astype(np.float64)
    g = lambda k: np.bincount(cid, sh[k][idx].astype(np.float64), len(starts))
    twt = 1.0 / sh["timevar"][idx].astype(np.float64)
    wwt = 1.0 / sh["uvar"][idx].astype(np.float64)
    wd = sh["wdist"][idx].astype(np.float64)
    ux, uy = sh["udx"][idx].astype(np.float64), sh["udy"][idx].astype(np.float64)
    cx = np.bincount(cid, sh["x"][idx] - wd * ux, len(starts)) / nsh
    cy = np.bincount(cid, sh["y"][idx] - wd * uy, len(starts)) / nsh
    cz = np.bincount(cid, sh["z"][idx].astype(np.float64), len(starts)) / nsh
    wsum, wacc, wacc2 = np.bincount(cid, wwt), np.bincount(cid, wd * wwt), np.bincount(cid, wd * wd * wwt)
    wdist = wacc / wsum
    first = idx[starts]
    multi = ncombo > 1
    out = {"ev": sh["ev"][first], "sid": sh["sid"][first], "ncombo": ncombo,
           "edep": np.where(multi, g("edep") / nsh, sh["edep"][first]).astype(F32),
           "et0": np.where(multi, g("et0") / nsh, sh["et0"][first]).astype(F32),
           "et1": np.where(multi, g("et1") / nsh, sh["et1"][first]).astype(F32),
           "time": np.where(multi, np.bincount(cid, sh["time"][idx] * twt) / np.bincount(cid, twt), sh["time"][first]).astype(F32),
           "wdist": np.where(multi, wdist, sh["wdist"][first]).astype(F32),
           "x": np.where(multi, cx + wdist * ux[starts], sh["x"][first]).astype(F32),
           "y": np.where(multi, cy + wdist * uy[starts], sh["y"][first]).astype(F32),
           "z": np.where(multi, cz, sh["z"][first]).astype(F32)}
    uvar = 1.0 / wsum + s["uerr"] ** 2
    if s["checkWres"]:
        ndof2 = np.maximum(ncombo - 1, 1) ** 2
        uvar = np.maximum(uvar, np.maximum(s["uerr"] ** 2, (wacc2 / wsum - wdist ** 2) / ndof2))
    out["uvar"] = np.where(multi, uvar, sh["uvar"][first]).astype(F32)
    tdiv = np.bincount(cid, (sh["flag"][idx] >> FLAG["tdiv"]) & 1, len(starts)) == ncombo
    r2 = out["x"].astype(np.float64) ** 2 + out["y"].astype(np.float64) ** 2
    flag = np.uint32(1 << FLAG["panelcombo"]) | np.where(tdiv, np.uint32(1 << FLAG["tdiv"]), np.uint32(0))
    for bit, ok in ((FLAG["nhitsel"], (ncombo >= s["minN"]) & (ncombo <= s["maxN"])),
                    (FLAG["timesel"], (out["time"] >= s["minT"]) & (out["time"] <= s["maxT"])),
                    (FLAG["energysel"], (out["edep"] >= s["minE"]) & (out["edep"] <= s["maxE"])),
                    (FLAG["radsel"], (r2 >= s["minR"] ** 2) & (r2 <= s["maxR"] ** 2))):
        flag = flag | np.where(ok, np.uint32(1 << bit), np.uint32(0))
    out["flag"] = flag
    out["ind"], out["ind_combo"] = idx, cid
    # reorder combos into stock output order (by seed position), keeping the flat constituent list aligned
    inv = np.empty_like(corder)
    inv[corder] = np.arange(len(corder))
    for k in ("ev", "sid", "ncombo", "edep", "et0", "et1", "time", "wdist", "x", "y", "z", "uvar", "flag"):
        out[k] = out[k][corder]
    o = np.argsort(inv[cid], kind="stable")
    out["ind_file"], out["ind_combo"] = idx[o], inv[cid][o]
    ev0 = np.searchsorted(sh["ev"], sh["ev"], side="left")                # first SH of each hit's event (ev sorted)
    out["ind"] = out["ind_file"] - ev0[out["ind_file"]]                    # per-event SH index, as stock ComboHit::_sind
    return out
