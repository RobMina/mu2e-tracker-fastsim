"""Wire-only refinement of SAVED measured seeds; not a physical drift/CH fit.

The original association membership is fixed. No candidate search, truth,
stock-track membership, drift-radius assignment, hit dropping or hit additions.
Full event/group/source-hit censuses and numerical failure lanes are retained.
The objective uses geometric wire distance and measured longitudinal position;
it cannot establish momentum resolution, tracking efficiency or calibrated chi2.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from trkfastsim.reco.hitreco import load_conditions
from trkfastsim.reco.helixwire import wire_residuals
from trkfastsim.reco.seedassociation import _geometry
from trkfastsim.reco.stereoseed import _integer
from .run1a_seed import validate


def prepare(keys, sh, cond, saved):
    """Validate saved group/seed identities and fixed, unique-straw support."""
    keys, sh = np.asarray(keys), {k: np.asarray(v) for k, v in sh.items()}
    validate(keys, sh)
    sh.update({k: sh[k].astype(np.int64) for k in ("ev", "sid", "flag")})
    for name, actual in (("evkey", keys), ("source_hit_ev", sh["ev"]), ("source_hit_sid", sh["sid"])):
        if name not in saved or not np.array_equal(saved[name], actual):
            raise ValueError(f"saved {name} differs from input hit census")
    required = ("hit_group", "group_ev", "association_source_used", "association_group_support",
                "association_source_doca", "association_source_wire_pull", "association_source_poca_z")
    if any(k not in saved for k in required):
        raise ValueError("missing saved association columns")
    hg, gev = np.asarray(saved["hit_group"]), np.asarray(saved["group_ev"])
    used, count = np.asarray(saved["association_source_used"]), np.asarray(saved["association_group_support"])
    nh, ng = len(sh["ev"]), len(gev)
    if (gev.ndim != 1 or gev.dtype.kind not in "iu" or np.any((gev < 0) | (gev >= len(keys)))
            or hg.shape != (nh,) or hg.dtype.kind not in "iu" or np.any((hg < -1) | (hg >= ng))
            or used.shape != (nh,) or used.dtype.kind != "b"
            or count.shape != (ng,) or count.dtype.kind not in "iu" or np.any(count < 0)):
        raise ValueError("invalid saved membership or support columns")
    hg, gev = hg.astype(np.int64), gev.astype(np.int64)
    grouped = hg >= 0
    if np.any(sh["ev"][grouped] != gev[hg[grouped]]) or np.any(used & ~grouped):
        raise ValueError("saved hit/group/event provenance mismatch")
    selected = np.flatnonzero(used)
    identities = np.column_stack((hg[selected], sh["sid"][selected]))
    if (len(np.unique(identities, axis=0)) != len(selected)
            or not np.array_equal(np.bincount(hg[selected], minlength=ng), count)):
        raise ValueError("saved distinct-straw support recount failed")
    needed = ("success", "center_xy", "radius", "pitch", "phase_ref", "z_ref")
    if "associated_seed_local_parameters" in saved:
        raise ValueError("alternate saved seed representation is not supported")
    if any("associated_seed_" + k not in saved for k in needed):
        raise ValueError("missing saved seed")
    seed = {k: np.asarray(saved["associated_seed_" + k]) for k in needed}
    count = count.astype(np.int64)
    for k in needed:
        a = seed[k]
        shape = (ng, 2) if k == "center_xy" else (ng,)
        if a.shape != shape or a.dtype.kind not in ("b" if k == "success" else "fiu"):
            raise ValueError(f"invalid saved seed {k}")
    if not np.array_equal(seed["success"], count > 0) or np.any((count > 0) & (count < 8)):
        raise ValueError("seed/support census mismatch")
    for k in needed[1:]:
        if not np.isfinite(seed[k][seed["success"]]).all():
            raise ValueError("nonfinite successful saved seed")
    if np.any(seed["success"] & ((seed["radius"] <= 0) | (seed["pitch"] == 0))):
        raise ValueError("nonphysical successful saved seed")
    for name in required[4:]:
        a = np.asarray(saved[name])
        if a.shape != (nh,) or a.dtype.kind not in "fiu" or not np.isfinite(a[selected]).all():
            raise ValueError(f"invalid saved residual {name}")
    h, mid, direction, u, hlen = _geometry(sh, cond)
    selected = selected[np.lexsort((selected, sh["sid"][selected], hg[selected]))]
    return dict(keys=keys, sh=sh, hg=hg, gev=gev, count=count, selected=selected, seed=seed,
                mid=mid, direction=direction, u=u, uvar=h["uvar"], hlen=hlen)


def _quantiles(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return dict(zip(("min", "median", "p90", "max"), map(float, np.quantile(a, [0, .5, .9, 1])))) if len(a) else {}


def run(keys, sh, cond, saved, *, bz=1.0117, group_batch=128, max_cells=65536, **fit_settings):
    """Fit selected wires in bounded chunks; never rerun the association search."""
    from trkfastsim.reco.helixfit import fit_wire_helix
    group_batch = _integer("group_batch", group_batch, 1)
    max_cells = _integer("max_cells", max_cells, 1)
    data = prepare(keys, sh, cond, saved)
    sh = data["sh"]
    count, selected, seed = data["count"], data["selected"], data["seed"]
    ng, nh, ne = len(count), len(data["hg"]), len(data["keys"])
    width = int(count.max(initial=0))
    if width > 256 or width > max_cells:
        raise ValueError("wire-fit row/cell limit exceeded; no support truncated")
    batch = min(group_batch, max(1, max_cells // max(1, width)))
    starts = np.r_[0, np.cumsum(count)]
    payload = {"evkey": data["keys"], "source_hit_ev": sh["ev"], "source_hit_sid": sh["sid"],
               "hit_group": data["hg"], "group_ev": data["gev"], "source_selected": saved["association_source_used"],
               "group_support": count}
    grouped_results = {}
    source_results = {}
    for begin in range(0, max(1, ng), batch):
        groups = np.arange(begin, min(begin + batch, ng))
        valid = np.arange(width)[None, :] < count[groups, None]
        idx = np.zeros(valid.shape, int)
        ranks = starts[groups, None] + np.arange(width)[None, :]
        if len(selected):
            idx[valid] = selected[ranks[valid]]
        mids = np.zeros(valid.shape + (3,))
        dirs = np.zeros_like(mids)
        dirs[..., 0] = 1.
        u, var, hl = np.zeros(valid.shape), np.ones(valid.shape), np.ones(valid.shape)
        mids[valid], dirs[valid] = data["mid"][idx[valid]], data["direction"][idx[valid]]
        u[valid], var[valid], hl[valid] = data["u"][idx[valid]], data["uvar"][idx[valid]], data["hlen"][idx[valid]]
        current_seed = {k: v[groups] for k, v in seed.items()}
        before = wire_residuals(current_seed, mids, dirs, u, var, hl)
        if np.any(valid & ~(before["valid"] & before["converged"] & before["inside"])):
            raise ValueError("saved support no longer has valid converged finite-wire POCA")
        for field, name in (("doca", "doca"), ("wire_pull", "wire_pull"), ("z", "poca_z")):
            if not np.allclose(before[field][valid], saved["association_source_" + name][idx[valid]], rtol=1e-10, atol=1e-8):
                raise ValueError(f"saved support {name} differs; measurements/conditions/seed changed")
        result = fit_wire_helix(current_seed, mids, dirs, u, var, hl, valid, bz=bz, **fit_settings)
        for k, a in result.items():
            a = np.asarray(a)
            if k in ("residuals", "doca", "wire_u", "wire_pull", "poca_z", "poca_valid", "poca_converged", "poca_inside", "poca_boundary", "poca_iterations"):
                if a.shape[:2] != valid.shape:
                    raise RuntimeError(f"unexpected per-hit fit output {k}")
                shape = (nh,) + a.shape[2:]
                if k not in source_results:
                    source_results[k] = np.full(shape, np.nan if a.dtype.kind == "f" else 0, dtype=a.dtype)
                source_results[k][idx[valid]] = a[valid]
            else:
                if a.shape[:1] != groups.shape:
                    raise RuntimeError(f"unexpected group fit output {k}")
                grouped_results.setdefault(k, []).append(a)
        for field in ("doca", "wire_pull"):
            key = "initial_" + field
            source_results.setdefault(key, np.full(nh, np.nan))[idx[valid]] = before[field][valid]
    combined = {k: np.concatenate(v) for k, v in grouped_results.items()}
    payload.update({"fit_" + k: v for k, v in combined.items()})
    payload.update({"source_" + k: v for k, v in source_results.items()})
    success = combined["success"]
    payload["event_nfits"] = np.bincount(data["gev"][success], minlength=ne)
    report = dict(events=ne, hits=nh, groups=ng, zero_hit_events=int((np.bincount(sh["ev"], minlength=ne) == 0).sum()),
                  input_seed_groups=int(seed["success"].sum()), fixed_selected_hits=len(selected),
                  successful_fits=int(success.sum()), events_with_fit=int((payload["event_nfits"] > 0).sum()),
                  status_counts={str(k): int(v) for k, v in zip(*np.unique(combined["status"], return_counts=True))},
                  bz_tesla=float(bz), group_batch=group_batch, max_cells=max_cells,
                  fit_settings={k: np.asarray(v).item() for k, v in fit_settings.items()},
                  selected_seed_residuals_replayed=True,
                  original_association_inputs_hash_bound=False,
                  provenance_caveat="Saved association has no originating input hashes: replay checks selected "
                                    "seed residuals and identities, not equality of every original time/condition. "
                                    "CLI hashes identify this invocation only.", caveat=__doc__)
    for name in ("initial_objective", "final_objective", "condition", "iterations", "gradient_norm"):
        if name in combined:
            report[name + "_successful"] = _quantiles(combined[name][success])
    good_hits = np.zeros(nh, bool)
    good_hits[selected] = success[data["hg"][selected]]
    for name in ("initial_doca", "doca", "initial_wire_pull", "wire_pull"):
        if name in source_results:
            report[name + "_successful_hits"] = _quantiles(np.abs(source_results[name][good_hits]))
    objective_valid = combined["objective_valid"]
    if np.any(objective_valid & (combined["final_objective"] > combined["initial_objective"] + 1e-10)):
        raise RuntimeError("wire fit increased fixed-membership objective")
    report["objective_decreased_groups"] = int(np.count_nonzero(objective_valid & (combined["final_objective"] < combined["initial_objective"])))
    report["physical_identifiable_fits"] = int(combined["physical_identifiable"].sum())
    report["successful_fit_hits"] = int(good_hits.sum())
    return report, payload


def _sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--association", type=Path, required=True, help="saved diagnostics.npz")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bz", type=float, default=1.0117)
    parser.add_argument("--group-batch", type=int, default=128)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--compare-saved", type=Path)
    args = parser.parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    start = time.perf_counter()
    with np.load(args.hits, allow_pickle=False) as data:
        keys = data["evkey"]
        sh = {k[2:]: data[k] for k in data.files if k.startswith("s_")}
    with np.load(args.association, allow_pickle=False) as data:
        saved = dict(data)
    cond = load_conditions(args.conditions)
    loaded = time.perf_counter()
    report, payload = run(keys, sh, cond, saved, bz=args.bz, group_batch=args.group_batch, max_iterations=args.max_iterations)
    report.update(load_seconds=loaded-start, fit_seconds=time.perf_counter()-loaded,
                  inputs={name: dict(path=str(path.resolve()), sha256=_sha256(path)) for name, path in
                          (("hits", args.hits), ("conditions", args.conditions), ("association", args.association))})
    if args.compare_saved:
        with np.load(args.compare_saved, allow_pickle=False) as previous:
            if set(previous.files) != set(payload):
                raise ValueError("saved fit schema differs")
            different = [k for k, v in payload.items() if not np.array_equal(previous[k], v, equal_nan=v.dtype.kind in "fc")]
        report["saved_comparison"] = dict(path=str(args.compare_saved.resolve()), exact=not different, different_arrays=different)
        if different:
            raise ValueError(f"saved fit differs: {different}")
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.outdir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.outdir / "diagnostics.npz", **payload)
    (args.outdir / "report.json").write_text(text)
    print(text, end="")
    return report


if __name__ == "__main__":
    main()
