"""Bounded measured-hit stereo candidates; no track finding, truth, or drift fit."""
import numpy as np

from .hitreco import FLAG, unique_straw

_BAD = (1 << 6) | (1 << FLAG["dead"]) | (1 << FLAG["noisy"])
# Numerical rank guard on the XY projection of a normalized 3D direction.
MIN_XY_NORM = np.sqrt(np.finfo(np.float64).eps)


def _columns(table, names, integers=()):
    out, shape = {}, None
    for name in names.split():
        try:
            a = np.asarray(table[name])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"missing column {name}") from exc
        kind = "iu" if name in integers else "fiu"
        if a.ndim != 1 or a.dtype.kind not in kind or (shape is not None and a.shape != shape):
            raise ValueError(f"invalid shape/type for {name}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"nonfinite {name}")
        shape = a.shape
        out[name] = a.astype(np.int64 if name in integers else np.float64)
    return out


def _integer(name, value, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not minimum <= value <= 2**63 - 1:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def stereo_points(sh, cond, *, nevents, time_gap=20., max_time_span=100.,
                  min_dz=1e-3, max_dz=50., min_sin=0.25, wire_nsigma=3.,
                  time_nsigma=3., max_dt=20., max_pairs=2_000_000, pair_batch=65_536):
    """Intersect projected wire lines in mm; times are ns and variances mm²/ns².

    ``sh`` is single-hit columns ev,sid,x,y,z,time,uvar,timevar,flag. ``ev``
    indexes the COMPLETE full-(run,subrun,event) census, not the event number;
    caller supplies ``nevents`` including empty events. Input need not be sorted.
    Reject background/dead/noisy flags and missing tdiv (truncated longitudinal
    measurements); no radial/energy selection or external track membership.

    Good hits are grouped by event and adjacent measured-time gaps <=20 ns;
    groups spanning >100 ns or with <2 hits produce no pairs, but remain in
    membership/count outputs. This is explicit single-link clustering (chains
    can bridge tracks), not time-of-flight or multi-track association. Pair time
    differences must ALSO be <=20 ns and <=3*sqrt(timevar_i+timevar_j).

    Default local-z window is 0.001 < |dz| <=50 mm at the PROJECTED intersection,
    and |sin(projected wire angle)|>=0.25. These are fixed engineering seed
    limits, NOT tuned/validated physics cuts. Directions/midpoints/half-lengths
    use cond['straw'] via unique_straw(sid). Tilted 3D wires are supported;
    axial/numerically singular XY projections (unit XY norm <=sqrt(float64 eps))
    raise. Measured xyz must lie on its reconstructed wire within 0.01 mm.
    Solve mid_i.xy + ui*dir_i.xy = mid_j.xy + uj*dir_j.xy using normalized 3D
    directions, NOT unit XY vectors: ui/uj are actual along-wire mm. Require
    BOTH points inside finite half-lengths and within 3*sqrt(uvar) of measured
    3D wire distance (projection of measured xyz-midpoint). Evaluate the two
    3D wire points at ui/uj; their z difference supplies dz and their mean is
    the seed position. Measured hit z is not substituted for intersection z.

    IMPORTANT: different-z wires do not actually intersect in 3D. This ignores
    local track slope and drift radius; errors grow with dz and inverse stereo
    angle. No covariance, helix fit, charge, or momentum is inferred. Cosmics
    with large transverse slopes can be biased/rejected. All surviving pairs
    are alternatives, often sharing hits and repeating z; they are NOT a set
    of independent points ready to pass wholesale to fit_helix_seed. Same-z
    pairs are rejected; duplicate candidate z is retained, never jittered or
    arbitrarily resolved. A subsequent association and wire-residual fit is
    still required.

    Returns positions[P,3], ev[P], group[P], hit_index[P,2] (original flat row
    indices), wire_pull[P,2], dt[P]; group_ev/group_n/group_span/group_npairs/
    group_npoints; member/member_group (time-sorted original indices), hit_group
    (-1 for flag rejection); event_nhits/event_ngood/event_ngroups/event_npoints
    of length nevents, and diagnostics. Pair rejection counts are exclusive in
    the documented code order; their sum plus accepted equals pairs_considered.

    Enumerates ALL unordered pairs within eligible groups, including rejected
    same-z/parallel pairs. Preflight raises ValueError if total exceeds max_pairs
    (2 million by default); NEVER truncates. Temporary pair arrays have at most
    pair_batch rows (65536 by default), using vectorized triangular-row ranks.
    Output is O(accepted pairs), bounded by max_pairs; membership is O(hits).
    Loops are over fixed columns/cuts and pair batches, never events or hits.
    Malformed/nonfinite measurements and referenced geometry raise ValueError.
    """
    nevents = _integer("nevents", nevents, 0)
    max_pairs = _integer("max_pairs", max_pairs, 1)
    pair_batch = _integer("pair_batch", pair_batch, 1)
    for name, value in (("time_gap", time_gap), ("max_time_span", max_time_span),
                        ("min_dz", min_dz), ("max_dz", max_dz), ("min_sin", min_sin),
                        ("wire_nsigma", wire_nsigma), ("time_nsigma", time_nsigma), ("max_dt", max_dt)):
        a = np.asarray(value)
        if a.ndim != 0 or a.dtype.kind not in "fiu" or not np.isfinite(a) or a <= 0:
            raise ValueError(f"{name} must be a positive finite scalar")
    if min_sin > 1 or min_dz >= max_dz:
        raise ValueError("require min_sin <=1 and min_dz < max_dz")
    h = _columns(sh, "ev sid x y z time uvar timevar flag", ("ev", "sid", "flag"))
    try:
        st = _columns(cond["straw"], "midx midy midz dirx diry dirz hlen")
    except (KeyError, TypeError) as exc:
        raise ValueError("missing straw geometry") from exc
    ev, sid, flag = h["ev"], h["sid"], h["flag"]
    if np.any((ev < 0) | (ev >= nevents)):
        raise ValueError("ev outside full event census")
    if np.any((sid < 0) | (sid > 65535) | ((sid & 127) >= 96) | (((sid >> 7) & 7) >= 6)):
        raise ValueError("invalid packed sid")
    us = unique_straw(sid)
    if np.any(us >= len(st["hlen"])):
        raise ValueError("sid outside straw geometry")
    if np.any((flag < 0) | (flag > 2**32 - 1) | (h["uvar"] <= 0) | (h["timevar"] < 0)):
        raise ValueError("invalid flag or measurement variance")
    pos = np.stack([h[k] for k in ("x", "y", "z")], axis=1)
    mid = np.stack([st[k][us] for k in ("midx", "midy", "midz")], axis=1)
    direction = np.stack([st[k][us] for k in ("dirx", "diry", "dirz")], axis=1)
    norm = np.linalg.norm(direction, axis=1)
    if np.any((norm <= 0) | ~np.isfinite(norm) | (st["hlen"][us] <= 0)):
        raise ValueError("invalid referenced wire direction/half-length")
    direction /= norm[:, None]
    d = direction[:, :2]  # preserve 3D normalization: solve parameters in wire mm
    xy_norm = np.linalg.norm(d, axis=1)
    if np.any(xy_norm <= MIN_XY_NORM):
        raise ValueError("axial or ill-conditioned XY wire projection")
    measured_u = np.sum((pos - mid) * direction, axis=1)
    residual = np.linalg.norm(pos - mid - measured_u[:, None] * direction, axis=1)
    if np.any(~np.isfinite(residual) | (residual > 0.01)):
        raise ValueError("measured xyz inconsistent with reconstructed wire")
    good = ((flag & _BAD) == 0) & ((flag & (1 << FLAG["tdiv"])) != 0)
    member = np.flatnonzero(good)
    member = member[np.lexsort((sid[member], h["time"][member], ev[member]))]
    e, t = ev[member], h["time"][member]
    breaks = np.r_[True, (np.diff(e) != 0) | (np.diff(t) > time_gap)][:len(member)]
    start = np.flatnonzero(breaks)
    n = np.diff(np.r_[start, len(member)]) if len(start) else np.empty(0, np.int64)
    group = np.cumsum(breaks) - 1
    span = t[start + n - 1] - t[start]
    eligible = (n >= 2) & (span <= max_time_span)
    npairs = np.where(eligible, n * (n - 1) // 2, 0)
    total = int(npairs.sum())
    if total > max_pairs:
        raise ValueError(f"pair limit exceeded: {total} > max_pairs={max_pairs}; no pairs truncated")
    # One triangular row per member; search cumulative row ends in each batch.
    rows = np.where(eligible[group], n[group] - 1 - (np.arange(len(member)) - start[group]), 0)
    row_end = np.cumsum(rows)
    row_start = row_end - rows
    counts = dict(input_hits=len(ev), good_hits=len(member), rejected_flags=int((~good).sum()),
                  groups=len(n), singleton_groups=int((n < 2).sum()),
                  wide_time_groups=int((span > max_time_span).sum()), pairs_considered=total)
    # Undefined/ill-conditioned intersections must not be assigned a z rejection.
    names = ("parallel", "nonfinite", "same_z", "far_z", "time", "wire_length", "wire_distance")
    counts.update({"rejected_" + key: 0 for key in names})
    parts = {key: [] for key in ("positions", "ev", "group", "hit_index", "wire_pull", "dt")}
    for offset in range(0, total, pair_batch):
        rank = np.arange(offset, min(offset + pair_batch, total))
        a = np.searchsorted(row_end, rank, side="right")
        b = a + 1 + rank - row_start[a]
        i, j = member[a], member[b]
        cross = d[i, 0] * d[j, 1] - d[i, 1] * d[j, 0]
        delta = mid[j, :2] - mid[i, :2]
        sine = np.abs(cross) / (xy_norm[i] * xy_norm[j])
        parallel = sine < min_sin
        denom = np.where(~parallel, cross, 1.)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ui = (delta[:, 0] * d[j, 1] - delta[:, 1] * d[j, 0]) / denom
            uj = (delta[:, 0] * d[i, 1] - delta[:, 1] * d[i, 0]) / denom
            pi = mid[i] + ui[:, None] * direction[i]
            pj = mid[j] + uj[:, None] * direction[j]
            pull = np.stack(((ui - measured_u[i]) / np.sqrt(h["uvar"][i]),
                             (uj - measured_u[j]) / np.sqrt(h["uvar"][j])), axis=1)
            dz, dt = np.abs(pi[:, 2] - pj[:, 2]), h["time"][i] - h["time"][j]
            xyz = pi / 2 + pj / 2
            cuts = (parallel,
                    ~np.isfinite(pi).all(axis=1) | ~np.isfinite(pj).all(axis=1)
                    | ~np.isfinite(pull).all(axis=1) | ~np.isfinite(dz),
                    dz <= min_dz, dz > max_dz,
                    (np.abs(dt) > max_dt) | (np.abs(dt) > time_nsigma * np.hypot(np.sqrt(h["timevar"][i]), np.sqrt(h["timevar"][j]))),
                    (np.abs(ui) > st["hlen"][us[i]]) | (np.abs(uj) > st["hlen"][us[j]]),
                    np.any(np.abs(pull) > wire_nsigma, axis=1))
        keep = np.ones(len(rank), bool)
        for name, reject in zip(names, cuts):
            counts["rejected_" + name] += int(np.count_nonzero(keep & reject))
            keep &= ~reject
        if not np.any(keep):
            continue
        values = (xyz, ev[i], group[a], np.column_stack((i, j)), pull, dt)
        for key, value in zip(parts, values):
            parts[key].append(value[keep])
    out = {}
    for key, chunks in parts.items():
        width = {"positions": 3, "hit_index": 2, "wire_pull": 2}.get(key)
        dtype = np.int64 if key in ("ev", "group", "hit_index") else np.float64
        out[key] = np.concatenate(chunks) if chunks else np.empty((0, width) if width else (0,), dtype=dtype)
    hit_group = np.full(len(ev), -1, np.int64)
    hit_group[member] = group
    counts["accepted"] = len(out["ev"])
    out.update(nevents=nevents, member=member, member_group=group, hit_group=hit_group,
               group_ev=e[start], group_n=n, group_span=span, group_npairs=npairs,
               group_npoints=np.bincount(out["group"], minlength=len(n)), diagnostics=counts,
               event_nhits=np.bincount(ev, minlength=nevents),
               event_ngood=np.bincount(e, minlength=nevents),
               event_ngroups=np.bincount(e[start], minlength=nevents),
               event_npoints=np.bincount(out["ev"], minlength=nevents))
    return out
