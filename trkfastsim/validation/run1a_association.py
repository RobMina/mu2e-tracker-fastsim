"""Development-only stereo association scored on distinct original finite wires.

No truth/stock membership, drift radius, direction, material or covariance is
used. A geometrically supported seed is not yet a reconstructed CentralHelix
track. Use only the development block until the algorithm and cuts are frozen.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from trkfastsim.reco.hitreco import load_conditions
from trkfastsim.reco.seedassociation import associate_wires
from .run1a_seed import run as candidate_run


def run(keys, sh, cond, *, bz=1.0117, **settings):
    control, payload = candidate_run(keys, sh, cond, bz=bz)
    points = {k: payload["candidate_" + k] for k in ("positions", "group", "hit_index")}
    result = associate_wires(points, sh, cond, hit_group=payload["hit_group"],
                             ngroups=control["groups"], bz=bz, **settings)
    settings = {k: np.asarray(v).item() for k, v in settings.items()}
    seed = result.pop("seed")
    payload.update({"association_" + k: v for k, v in result.items()})
    payload.update({"associated_seed_" + k: v for k, v in seed.items()})
    payload["event_associated_seeds"] = np.bincount(payload["group_ev"][seed["success"]], minlength=len(keys))
    selected = result["source_used"]
    group = payload["hit_group"][selected]
    identities = np.column_stack((group, np.asarray(sh["sid"])[selected]))
    if (len(np.unique(identities, axis=0)) != len(identities)
            or not np.array_equal(np.bincount(group, minlength=control["groups"]), result["group_support"])):
        raise RuntimeError("distinct-wire support/provenance audit failed")
    if not np.array_equal(seed["success"], result["group_support"] > 0):
        raise RuntimeError("seed/support census audit failed")
    def quantiles(a):
        return dict(zip(("min", "median", "p90", "max"), map(float, np.quantile(a, [0, .5, .9, 1])))) if len(a) else {}
    report = dict(events=len(keys), hits=control["hits"], zero_hit_events=control["zero_hit_events"],
                  groups=control["groups"], candidates=control["candidates"],
                  raw_hypotheses=int(result["group_hypotheses"].sum()),
                  independent_hypotheses=int(result["group_independent"].sum()),
                  fitted_hypotheses=int(result["group_fit_success"].sum()),
                  supported_hypotheses=int(result["group_supported"].sum()),
                  groups_with_fitted_hypotheses=int((result["group_fit_success"] > 0).sum()),
                  successful_seeds=int(seed["success"].sum()),
                  events_with_seeds=int((payload["event_associated_seeds"] > 0).sum()),
                  supported_source_hits=int(result["source_used"].sum()),
                  groups_with_runner=int((result["runner_support"] > 0).sum()),
                  groups_without_independent_quartet=int(((payload["group_npoints"] >= 4) & (result["group_independent"] == 0)).sum()),
                  groups_fit_but_insufficient_support=int(((result["group_fit_success"] > 0) & ~seed["success"]).sum()),
                  runner_equal_support_groups=int(((result["runner_support"] > 0) & (result["runner_support"] == result["group_support"])).sum()),
                  unique_straw_support_audit=True,
                  supported_wire_doca_mm=quantiles(result["source_doca"][selected]),
                  supported_abs_wire_pull=quantiles(np.abs(result["source_wire_pull"][selected])),
                  seed_condition_diagnostic=quantiles(seed["condition"][seed["success"]]),
                  support_histogram={str(k): int(v) for k, v in
                                     zip(*np.unique(result["group_support"], return_counts=True))},
                  bz_tesla=float(bz), settings=settings, unambiguous_control=control,
                  engineering_policy=associate_wires.__doc__, caveat=__doc__)
    return report, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bz", type=float, default=1.0117)
    parser.add_argument("--max-hypotheses", type=int, default=2_000_000)
    parser.add_argument("--hypothesis-batch", type=int, default=1024)
    parser.add_argument("--compare-saved", type=Path, help="audit all NPZ arrays against a previous same-input run")
    args = parser.parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    begin = time.perf_counter()
    with np.load(args.hits, allow_pickle=False) as saved:
        keys = saved["evkey"]
        sh = {k[2:]: saved[k] for k in saved.files if k.startswith("s_")}
    cond = load_conditions(args.conditions)
    loaded = time.perf_counter()
    report, payload = run(keys, sh, cond, bz=args.bz, max_hypotheses=args.max_hypotheses,
                           hypothesis_batch=args.hypothesis_batch)
    report.update(hits_file=str(args.hits.resolve()), conditions_file=str(args.conditions.resolve()),
                  load_seconds=loaded - begin, association_seconds=time.perf_counter() - loaded)
    if args.compare_saved:
        with np.load(args.compare_saved, allow_pickle=False) as previous:
            if set(previous.files) != set(payload):
                raise RuntimeError("saved comparison schema differs")
            different = [k for k, v in payload.items() if not np.array_equal(previous[k], v, equal_nan=True)]
        report["saved_comparison"] = dict(path=str(args.compare_saved.resolve()),
                                           exact=not different, different_arrays=different)
        if different:
            raise RuntimeError(f"saved comparison differs: {different}")
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.outdir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.outdir / "diagnostics.npz", **payload)
    (args.outdir / "report.json").write_text(text)
    print(text, end="")
    return report


if __name__ == "__main__":
    main()
