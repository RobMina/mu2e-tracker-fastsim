"""Measured-only four-point helix hypotheses scored on distinct finite 3D wires.

This is bounded development pattern recognition, not a drift fit or a tracker
performance result. Shared stereo alternatives propose hypotheses but never
multiply their source-wire votes. One best hypothesis per time group is retained;
multiple-track separation, fit covariance and direction remain outside this API.
"""
import math

import numpy as np

from .helixseed import fit_helix_seed
from .hitreco import unique_straw
from .stereoseed import _columns, _integer


def _first_two(group, count, score, rank):
    order = np.lexsort((rank, score, -count, group))
    g = group[order]
    start = np.maximum.accumulate(np.where(np.r_[True, np.diff(g) != 0], np.arange(len(g)), 0))
    return order[np.arange(len(g)) - start < 2]


def _wire_votes(inlier, cost, sid):
    """One minimum-cost pulse per straw per hypothesis; sid is row-wise sorted."""
    keep = np.zeros(inlier.shape, bool)
    if not keep.size:
        return keep
    first = np.ones(inlier.shape, bool)
    first[:, 1:] = sid[:, 1:] != sid[:, :-1]
    starts = np.flatnonzero(first)
    run = np.cumsum(first.ravel()) - 1
    value = np.where(inlier, cost, np.inf).ravel()
    minimum = np.minimum.reduceat(value, starts)
    winner = np.full(len(starts), value.size, np.int64)
    eligible = np.isfinite(value) & (value == minimum[run])
    np.minimum.at(winner, run[eligible], np.flatnonzero(eligible))
    keep.ravel()[winner[winner < value.size]] = True
    return keep


def _geometry(sh, cond):
    h = _columns(sh, "ev sid x y z uvar", ("ev", "sid"))
    st = _columns(cond["straw"], "midx midy midz dirx diry dirz hlen")
    sid = h["sid"]
    if np.any(h["ev"] < 0):
        raise ValueError("negative event index")
    if np.any((sid < 0) | (sid > 65535) | ((sid & 127) >= 96) | (((sid >> 7) & 7) >= 6)):
        raise ValueError("invalid packed sid")
    us = unique_straw(sid)
    if np.any(us >= len(st["hlen"])) or np.any(h["uvar"] <= 0):
        raise ValueError("invalid referenced geometry or longitudinal variance")
    mid = np.column_stack([st[k][us] for k in ("midx", "midy", "midz")])
    direction = np.column_stack([st[k][us] for k in ("dirx", "diry", "dirz")])
    norm = np.linalg.norm(direction, axis=1)
    if np.any(~np.isfinite(norm) | (norm <= 0) | (st["hlen"][us] <= 0)):
        raise ValueError("invalid wire direction or half-length")
    direction /= norm[:, None]
    position = np.column_stack([h[k] for k in ("x", "y", "z")])
    with np.errstate(over="ignore", invalid="ignore"):
        u = np.sum((position - mid) * direction, axis=1)
        residual = np.linalg.norm(position - mid - u[:, None] * direction, axis=1)
    if np.any(~np.isfinite(u) | ~np.isfinite(residual) | (residual > .01)):
        raise ValueError("measured position inconsistent with wire")
    return h, mid, direction, u, st["hlen"][us]


def associate_wires(points, sh, cond, *, hit_group, ngroups, bz,
                    max_hypotheses=2_000_000, hypothesis_batch=1024,
                    max_score_cells=262144, max_doca=10., max_wire_pull=3., min_hits=8):
    """Enumerate every four-candidate combination, then score unique source wires.

    ``points`` has positions[P,3], group[P], hit_index[P,2] from stereo_points;
    ``hit_group`` maps every original SH row to a group or -1. All ngroups are
    returned, including those without seeds. Complete event keys stay with the
    caller. Four distinct-z candidates must use eight different straw IDs.
    Stereo candidates are canonically sorted; combination ranks use the colex
    combinatorial number system, not a random sampler or truncated scan.

    fit_helix_seed proposes constant-Bz helices. Their local closest approaches
    to ORIGINAL normalized 3D wires must converge inside finite wire lengths,
    with DOCA <=10 mm and longitudinal pull <=3. These fixed engineering seed
    limits are not drift errors or tuned physics selections. Each physical straw
    contributes at most once, choosing its minimum-cost compatible pulse. Rank
    by most supported straws then mean[(DOCA/max_doca)^2+(pull/max_wire_pull)^2].
    This ranking is NOT chi-square or an unbiased momentum estimator. Hypothesis
    defining wires also participate; no independent validation set is claimed.

    At least eight supported straws are required. The winning and second-ranked
    hypotheses are retained (the latter can describe the SAME trajectory/support,
    so it is NOT a second track or calibrated ambiguity probability). Winner
    candidate and source-hit provenance are explicit. No refit is performed.

    Preflight raises if >256 candidates in a group or >max_hypotheses raw
    combinations; never drops a tail. Hypothesis and score matrices are bounded
    by hypothesis_batch/max_score_cells. Incremental reduction keeps at most
    two hypotheses per group in persistent working storage. Loops are over batches, fixed coordinates and outputs,
    never events, hits or combinations. Invalid data/configuration raises.
    """
    from .helixwire import wire_residuals

    ngroups = _integer("ngroups", ngroups, 0)
    max_hypotheses = _integer("max_hypotheses", max_hypotheses, 1)
    hypothesis_batch = _integer("hypothesis_batch", hypothesis_batch, 1)
    max_score_cells = _integer("max_score_cells", max_score_cells, 1)
    min_hits = _integer("min_hits", min_hits, 8)
    for name, value in (("max_doca", max_doca), ("max_wire_pull", max_wire_pull)):
        a = np.asarray(value)
        if a.ndim or a.dtype.kind not in "fiu" or not np.isfinite(a) or a <= 0:
            raise ValueError(f"{name} must be a positive finite scalar")
    seed_empty = fit_helix_seed(np.empty((ngroups, 0, 3)), np.empty((ngroups, 0), bool), bz=bz)
    h, mid, direction, measured_u, hlen = _geometry(sh, cond)
    hg = np.asarray(hit_group)
    if hg.shape != h["sid"].shape or hg.dtype.kind not in "iu" or np.any((hg < -1) | (hg >= ngroups)):
        raise ValueError("invalid hit_group")
    p, g, pair = (np.asarray(points[k]) for k in ("positions", "group", "hit_index"))
    if (p.ndim != 2 or p.shape[1] != 3 or p.dtype.kind not in "fiu" or not np.isfinite(p).all()
            or g.shape != (len(p),) or g.dtype.kind not in "iu" or np.any((g < 0) | (g >= ngroups))
            or pair.shape != (len(p), 2) or pair.dtype.kind not in "iu"
            or np.any((pair < 0) | (pair >= len(hg)))):
        raise ValueError("invalid candidate arrays")
    g, pair = g.astype(np.int64), np.sort(pair.astype(np.int64), axis=1)
    if np.any(pair[:, 0] == pair[:, 1]) or np.any(hg[pair] != g[:, None]):
        raise ValueError("candidate source provenance inconsistent with groups")
    if len(np.unique(pair, axis=0)) != len(pair):
        raise ValueError("duplicate source pair")
    member = np.flatnonzero(hg >= 0)
    member = member[np.lexsort((member, h["sid"][member], hg[member]))]
    hn = np.bincount(hg[member], minlength=ngroups)
    hs = np.r_[0, np.cumsum(hn)]
    if len(member):
        begin = hs[:-1][hn > 0]
        if np.any(np.minimum.reduceat(h["ev"][member], begin) != np.maximum.reduceat(h["ev"][member], begin)):
            raise ValueError("one time group crosses event identities")
    order = np.lexsort((pair[:, 1], pair[:, 0], p[:, 1], p[:, 0], p[:, 2], g))
    p, g, pair = p[order].astype(float), g[order], pair[order]
    n = np.bincount(g, minlength=ngroups)
    if np.any(n > 256):
        raise ValueError("candidate limit exceeded (>256); no candidates truncated")
    choose = np.array([[math.comb(i, k) if i >= k else 0 for i in range(257)] for k in range(5)], np.int64)
    nc = choose[4, n]
    total = int(nc.sum())
    if total > max_hypotheses:
        raise ValueError(f"hypothesis limit exceeded: {total} > {max_hypotheses}; no hypotheses truncated")
    starts, ends = np.r_[0, np.cumsum(n)[:-1]], np.cumsum(nc)
    comb_start = ends - nc
    max_hits = int(hn.max(initial=0))
    if max_hits > max_score_cells:
        raise ValueError("one group exceeds max_score_cells; no hits truncated")
    batch = min(hypothesis_batch, max(1, max_score_cells // max(1, max_hits)))
    diag = {k: np.zeros(ngroups, np.int64) for k in ("independent", "fit_success", "supported")}
    saved = None

    def score(seed, groups):
        width = max_hits
        valid = np.arange(width)[None, :] < hn[groups, None]
        rows = hs[groups, None] + np.arange(width)[None, :]
        idx = member[np.minimum(rows, len(member) - 1)]
        result = wire_residuals(seed, mid[idx], direction[idx], measured_u[idx], h["uvar"][idx], hlen[idx])
        inlier = (valid & result["converged"] & result["inside"]
                  & (result["doca"] <= max_doca) & (np.abs(result["wire_pull"]) <= max_wire_pull))
        cost = (result["doca"] / max_doca)**2 + (result["wire_pull"] / max_wire_pull)**2
        mask = _wire_votes(inlier, cost, h["sid"][idx])
        count = mask.sum(axis=1)
        value = np.sum(np.where(mask, cost, 0.), axis=1) / np.maximum(count, 1)
        return count, value, mask, idx, result

    for offset in range(0, total, batch):
        rank = np.arange(offset, min(offset + batch, total))
        group = np.searchsorted(ends, rank, side="right")
        remainder = rank - comb_start[group]
        local = np.empty((len(rank), 4), np.int64)
        for k in (4, 3, 2, 1):
            ix = np.searchsorted(choose[k], remainder, side="right") - 1
            local[:, k - 1] = ix
            remainder -= choose[k, ix]
        ci = starts[group, None] + local
        source = pair[ci].reshape(-1, 8)
        sid = np.sort(h["sid"][source], axis=1)
        independent = np.all(np.diff(sid, axis=1) != 0, axis=1)
        diag["independent"] += np.bincount(group[independent], minlength=ngroups)
        if not independent.any():
            continue
        ci, group, rank = ci[independent], group[independent], rank[independent]
        seed = fit_helix_seed(p[ci], np.ones(ci.shape, bool), bz=bz)
        ok = seed["success"]
        diag["fit_success"] += np.bincount(group[ok], minlength=ngroups)
        if not ok.any():
            continue
        seed = {k: v[ok] for k, v in seed.items()}
        ci, group, rank = ci[ok], group[ok], rank[ok]
        count, value, _, _, _ = score(seed, group)
        good = count >= min_hits
        diag["supported"] += np.bincount(group[good], minlength=ngroups)
        if not good.any():
            continue
        seed = {k: v[good] for k, v in seed.items()}
        ci, group, rank, count, value = ci[good], group[good], rank[good], count[good], value[good]
        top = _first_two(group, count, value, rank)
        new = dict(group=group[top], rank=rank[top], candidate=order[ci[top]], count=count[top], score=value[top])
        new.update({"seed_" + k: v[top] for k, v in seed.items()})
        merged = new if saved is None else {k: np.concatenate((saved[k], new[k])) for k in new}
        keep = _first_two(merged["group"], merged["count"], merged["score"], merged["rank"])
        saved = {k: v[keep] for k, v in merged.items()}
    out = dict(seed=seed_empty, group_hypotheses=nc, **{"group_" + k: v for k, v in diag.items()},
               group_support=np.zeros(ngroups, np.int64), group_score=np.full(ngroups, np.nan),
               group_candidate=np.full((ngroups, 4), -1, np.int64),
               runner_support=np.zeros(ngroups, np.int64), runner_score=np.full(ngroups, np.nan),
               runner_candidate=np.full((ngroups, 4), -1, np.int64), runner_p_abs=np.full(ngroups, np.nan),
               source_used=np.zeros(len(hg), bool), source_doca=np.full(len(hg), np.nan),
               source_wire_pull=np.full(len(hg), np.nan), source_poca_z=np.full(len(hg), np.nan))
    if saved is None:
        return out
    top = _first_two(saved["group"], saved["count"], saved["score"], saved["rank"])
    first = np.r_[True, np.diff(saved["group"][top]) != 0]
    win, runner = top[first], top[~first]
    wg, rg = saved["group"][win], saved["group"][runner]
    for k in seed_empty:
        out["seed"][k][wg] = saved["seed_" + k][win]
    for key, field in (("support", "count"), ("score", "score"), ("candidate", "candidate")):
        out["group_" + key][wg] = saved[field][win]
        out["runner_" + key][rg] = saved[field][runner]
    out["runner_p_abs"][rg] = saved["seed_p_abs"][runner]
    for offset in range(0, len(wg), batch):
        groups = wg[offset:offset + batch]
        seed = {k: v[groups] for k, v in out["seed"].items()}
        _, _, mask, idx, result = score(seed, groups)
        out["source_used"][idx[mask]] = True
        for key, field in (("doca", "doca"), ("wire_pull", "wire_pull"), ("poca_z", "z")):
            out["source_" + key][idx[mask]] = result[field][mask]
    return out
