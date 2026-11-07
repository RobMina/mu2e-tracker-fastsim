"""Measured-only development diagnostics, NOT tracking, KinKal or efficiency.

Run: python -m trkfastsim.validation.run1a_seed --hits numpy.npz
     --conditions conditions.json --bz 1.0117 --outdir NEW
Fixed engineering association: stereo defaults (20 ns adjacent time gap,
100 ns group span; local wire intersections, not true 3D track points).
Use the original normalized 3D wire directions, including alignment tilts;
wire parameters/lengths/pulls are in actual along-wire mm, z and dz are evaluated
at the projected intersection. Local track slope and drift radius remain ignored.
Retain all original time groups; reject a WHOLE group only for an axial or
numerically singular XY projection (unit XY norm <=sqrt(float64 eps)) of a good
hit. Never flatten wire geometry. Also reject groups with shared source hits or repeated
candidate z: do not choose a preferred alternative or jitter z. At least four
independent points are required; at most 256 are supported (reject, not truncate).
Stereo enumerates at most 2 million pairs (raises rather than truncating).
Fit in <=1024-group chunks, <=262144 padded points. These are engineering limits,
not tuned momentum/residual selections. Even unambiguous groups can mix tracks.
No truth, stock membership, momentum or fit-quality selection is consumed.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from trkfastsim.reco.helixseed import fit_helix_seed
from trkfastsim.reco.hitreco import FLAG, load_conditions, unique_straw
from trkfastsim.reco.stereoseed import MIN_XY_NORM, stereo_points

COLS = "ev sid x y z time uvar timevar flag".split()
REASONS = {"unsupported_geometry": 1, "wide_time": 2, "fewer_than_four": 4,
           "shared_source_hits": 8, "repeated_z": 16, "over_256_points": 32}


def validate(keys, sh):
    keys = np.asarray(keys)
    if (keys.ndim != 2 or keys.shape[1] != 3 or keys.dtype.kind not in "iu"
            or np.any(keys < 0) or np.any(keys > np.iinfo(np.uint32).max)
            or len(np.unique(keys, axis=0)) != len(keys)):
        raise ValueError("invalid or duplicate full (run,subrun,event) keys")
    if any(k not in sh for k in COLS):
        raise ValueError("missing required s_* column")
    n = np.asarray(sh["ev"]).size
    for k, a in sh.items():
        if a.shape != (n,) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
            raise ValueError(f"invalid/nonfinite s_{k}")
        if k in ("ev", "sid", "flag") and a.dtype.kind not in "iu":
            raise ValueError(f"noninteger s_{k}")
    ev, sid, flag = (sh[k] for k in ("ev", "sid", "flag"))
    if np.any(ev < 0) or np.any(ev >= len(keys)):
        raise ValueError("s_ev outside complete event census")
    if np.any((sid < 0) | (sid > 65535) | ((sid & 127) >= 96) | (((sid >> 7) & 7) >= 6)):
        raise ValueError("invalid packed sid")
    if np.any((flag < 0) | (flag > 2**32 - 1) | (sh["uvar"] <= 0) | (sh["timevar"] < 0)):
        raise ValueError("invalid flag or measurement variance")


def geometry_supported(sh, cond):
    """Validate wire measurements; only axial/singular XY projections unsupported."""
    st = cond["straw"]
    names = "midx midy midz dirx diry dirz hlen".split()
    a = {k: np.asarray(st[k]) for k in names}
    for k, v in a.items():
        if v.ndim != 1 or v.shape != a["hlen"].shape or v.dtype.kind not in "fiu" or not np.isfinite(v).all():
            raise ValueError(f"invalid geometry {k}")
    us = unique_straw(sh["sid"].astype(np.int64))
    if np.any(us >= len(a["hlen"])):
        raise ValueError("sid outside straw geometry")
    d = np.column_stack([a[k][us] for k in ("dirx", "diry", "dirz")])
    norm = np.linalg.norm(d, axis=1)
    if np.any(~np.isfinite(norm) | (norm <= 0) | (a["hlen"][us] <= 0)):
        raise ValueError("invalid referenced direction/half-length")
    d = d / norm[:, None]
    delta = np.column_stack([sh[k] - a["mid" + k][us] for k in ("x", "y", "z")])
    residual = np.linalg.norm(delta - np.sum(delta * d, axis=1)[:, None] * d, axis=1)
    if np.any(~np.isfinite(residual) | (residual > .01)):
        raise ValueError("measured xyz inconsistent with wire")
    return np.linalg.norm(d[:, :2], axis=1) > MIN_XY_NORM


def associate(points, group_npoints, nhits, initial_bits):
    """Reject alternatives at group level; return reasons and used-candidate mask."""
    g, z, pair = points["group"], points["positions"][:, 2], points["hit_index"]
    shared, repeated = np.zeros(len(group_npoints), bool), np.zeros(len(group_npoints), bool)
    if len(g):
        degree = np.bincount(pair.ravel(), minlength=nhits)
        np.logical_or.at(shared, g, np.any(degree[pair] > 1, axis=1))
        order = np.lexsort((z, g))
        dup = (np.diff(g[order]) == 0) & (np.diff(z[order]) == 0)
        repeated[g[order[1:][dup]]] = True
    bits = initial_bits.copy()
    evaluated = (initial_bits & REASONS["unsupported_geometry"]) == 0
    bits |= np.where(evaluated & (group_npoints < 4), REASONS["fewer_than_four"], 0)
    bits |= np.where(shared, REASONS["shared_source_hits"], 0)
    bits |= np.where(repeated, REASONS["repeated_z"], 0)
    bits |= np.where(group_npoints > 256, REASONS["over_256_points"], 0)
    return bits, bits[g] == 0


def fit_groups(points, counts, bits, bz):
    """All groups survive as seed lanes, including rejected/empty groups."""
    ng = len(counts)
    seeds = fit_helix_seed(np.empty((ng, 0, 3)), np.empty((ng, 0), bool), bz=bz)
    groups = np.flatnonzero(bits == 0)
    use = np.flatnonzero(bits[points["group"]] == 0)
    use = use[np.lexsort((points["positions"][use, 2], points["group"][use]))]
    ends = np.r_[0, np.cumsum(counts[groups])]
    for start in range(0, len(groups), 1024):
        stop = min(start + 1024, len(groups))
        ids, n = groups[start:stop], counts[groups[start:stop]]
        row = np.repeat(np.arange(len(ids)), n)
        col = np.arange(n.sum()) - np.repeat(np.r_[0, np.cumsum(n)[:-1]], n)
        pos = np.zeros((len(ids), int(n.max()), 3))
        mask = np.zeros(pos.shape[:2], bool)
        pos[row, col] = points["positions"][use[ends[start]:ends[stop]]]
        mask[row, col] = True
        result = fit_helix_seed(pos, mask, bz=bz)
        for k in seeds:
            seeds[k][ids] = result[k]
    return seeds


def run(keys, sh, cond, *, bz=1.0117):
    keys, sh = np.asarray(keys), {k: np.asarray(v) for k, v in sh.items()}
    validate(keys, sh)
    sh.update({k: sh[k].astype(np.int64) for k in ("ev", "sid", "flag")})
    ne, nh = len(keys), len(sh["ev"])
    supported = geometry_supported(sh, cond)
    badflags = (1 << 6) | (1 << FLAG["dead"]) | (1 << FLAG["noisy"])
    good = ((sh["flag"] & badflags) == 0) & ((sh["flag"] & (1 << FLAG["tdiv"])) != 0)
    member = np.flatnonzero(good)
    member = member[np.lexsort((sh["sid"][member], sh["time"][member], sh["ev"][member]))]
    ev, t = sh["ev"][member], sh["time"][member].astype(float)
    breaks = np.r_[True, (np.diff(ev) != 0) | (np.diff(t) > 20.)][:len(member)]
    start = np.flatnonzero(breaks)
    mg = np.cumsum(breaks) - 1
    n = np.diff(np.r_[start, len(member)]) if len(start) else np.empty(0, int)
    span, gev = t[start + n - 1] - t[start], ev[start]
    hg = np.full(nh, -1, int)
    hg[member] = mg
    unsupported = np.zeros(len(n), bool)
    np.logical_or.at(unsupported, mg, ~supported[member])
    source = member[~unsupported[mg]]  # whole groups, never a geometry-pruned subset
    out = stereo_points({k: sh[k][source] for k in COLS}, cond, nevents=ne)
    kernel_diag = out["diagnostics"]
    group_map = hg[source[out["member"]]][np.r_[True, np.diff(out["member_group"]) != 0][:len(out["member"])]]
    points = {k: out[k] for k in ("positions", "ev", "group", "hit_index", "wire_pull", "dt")}
    points["group"] = group_map[points["group"]]
    points["hit_index"] = source[points["hit_index"]]
    counts = np.bincount(points["group"], minlength=len(n))
    npairs = np.zeros(len(n), int)
    npairs[group_map] = out["group_npairs"]
    bits, used = associate(points, counts, nh, unsupported.astype(int) | np.where(span > 100., 2, 0))
    seeds = fit_groups(points, counts, bits, bz)
    payload = dict(evkey=keys, source_hit_ev=sh["ev"], source_hit_sid=sh["sid"], hit_group=hg,
                   hit_geometry_supported=supported, member=member, member_group=mg,
                   group_ev=gev, group_n=n, group_span=span, group_npoints=counts,
                   group_npairs=npairs, group_candidates_evaluated=~unsupported,
                   group_reject_bits=bits, candidate_used=used)
    payload.update({"candidate_" + k: v for k, v in points.items()})
    payload.update({"seed_" + k: v for k, v in seeds.items()})
    for name, index in (("nhits", sh["ev"]), ("ngood", ev), ("ngroups", gev),
                        ("npoints", points["ev"]), ("nseeds", gev[seeds["success"]])):
        payload["event_" + name] = np.bincount(index, minlength=ne)
    report = dict(events=ne, unique_full_event_keys=True, finite_input=True, hits=nh,
                  zero_hit_events=int((payload["event_nhits"] == 0).sum()), good_hits=int(good.sum()),
                  rejected_flag_hits=int((~good).sum()), unsupported_geometry_hits=int((~supported).sum()),
                  groups=len(n), groups_with_candidates=int((counts > 0).sum()), candidates=len(used),
                  groups_candidates_evaluated=int((~unsupported).sum()),
                  groups_ambiguity_unevaluated=int(unsupported.sum()),
                  group_size_histogram={str(k): int(v) for k, v in zip(*np.unique(n, return_counts=True))},
                  candidate_count_histogram_evaluated={str(k): int(v) for k, v in
                      zip(*np.unique(counts[~unsupported], return_counts=True))},
                  rejected_ambiguous_groups=int(((bits & (8 | 16)) != 0).sum()),
                  group_rejection_counts={k: int(((bits & v) != 0).sum()) for k, v in REASONS.items()},
                  attempted_groups=int((bits == 0).sum()), successful_seeds=int(seeds["success"].sum()),
                  events_with_candidates=int((payload["event_npoints"] > 0).sum()),
                  events_with_seeds=int((payload["event_nseeds"] > 0).sum()),
                  stereo_supported_groups_only=kernel_diag, group_reject_bit_definitions=REASONS,
                  bz_tesla=float(bz), engineering_policy=__doc__,
                  caveat="No tracking efficiency or momentum resolution claim; rejection bits overlap. "
                         "Geometry-rejected groups have unevaluated candidate ambiguity, not zero physical ambiguity. "
                         "Failed/rejected seed estimates are NaN (condition may be inf) in NPZ, never JSON.")
    return report, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--bz", type=float, default=1.0117)
    parser.add_argument("--outdir", type=Path, required=True, help="new directory, never overwritten")
    args = parser.parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    begin = time.perf_counter()
    with np.load(args.hits, allow_pickle=False) as saved:
        keys = saved["evkey"]
        sh = {k[2:]: saved[k] for k in saved.files if k.startswith("s_")}
    cond = load_conditions(args.conditions)
    loaded = time.perf_counter()
    report, payload = run(keys, sh, cond, bz=args.bz)
    report.update(hits_file=str(args.hits.resolve()), conditions_file=str(args.conditions.resolve()),
                  load_seconds=loaded - begin, diagnostics_and_seed_seconds=time.perf_counter() - loaded)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.outdir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.outdir / "diagnostics.npz", **payload)
    (args.outdir / "report.json").write_text(text)
    print(text, end="")
    return report


if __name__ == "__main__":
    main()
