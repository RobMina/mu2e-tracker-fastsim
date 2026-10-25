"""Compare saved slab arms and audit highest-cycle reference event coverage without resimulation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import uproot

from .. import tables as T
from .slabcheck import FIT_MIN_PEXIT, _TAG


def _key(row):
    return row["material"], float(row["g_cm2"]), int(row["p_MeV"])


def _index(rows):
    out = {_key(row): row for row in rows}
    if len(out) != len(rows):
        raise ValueError("duplicate slab point")
    return out


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(before, after):
    """Require identical saved G4 observables; report unweighted point summaries, not significance."""
    b, a = _index(before["rows"]), _index(after["rows"])
    if b.keys() != a.keys():
        raise ValueError("slab arms have different point coverage")
    for key in a:
        for field in ("n_g4", "surv_g4", "g4"):
            if not np.array_equal(a[key][field], b[key][field]):
                raise ValueError(f"different G4 reference at {key}: {field}")
    summary = []
    for material in sorted({key[0] for key in a}):
        keys = [key for key in sorted(a) if key[0] == material]
        selected = [key for key in keys if min(a[key]["p_exit_frac"], b[key]["p_exit_frac"]) >= FIT_MIN_PEXIT]
        row = {"material": material, "n_points": len(keys), "n_fit_domain": len(selected),
               "n_outside_fit_domain": len(keys) - len(selected), "arms": {}}
        for name, source in (("before", b), ("after", a)):
            row["arms"][name] = {}
            for arm in ("fast", "fast_cal"):
                if not selected or not all(arm + "_over_g4" in source[key] for key in selected):
                    continue
                ratio = np.array([source[key][arm + "_over_g4"] for key in selected])
                if not np.isfinite(ratio).all():
                    raise ValueError("nonfinite slab ratio")
                row["arms"][name][arm] = {
                    "mean_width_ratio": float(ratio[:, 0].mean()),
                    "max_abs_width_deviation": float(np.abs(ratio[:, 0] - 1).max()),
                    "n_width_within_3pct": int((np.abs(ratio[:, 0] - 1) < .03).sum()),
                    "mean_q90_ratio": float(ratio[:, 1].mean()),
                    "mean_q99_ratio": float(ratio[:, 2].mean()),
                }
        summary.append(row)
    return {"n_matched_points": len(a), "fit_min_exit_fraction": FIT_MIN_PEXIT,
            "interpretation": "unweighted summaries of slab points; no paired-error or significance claim",
            "baseline_settings": before.get("settings"), "candidate_settings": after.get("settings"),
            "summary": summary}


def audit_references(inputs, rows):
    """Count actual primary-muon events from each highest-cycle particle tree, not job metadata."""
    points = _index(rows)
    records, seen = [], set()
    for path in T.expand(inputs):
        match = _TAG.search(Path(path).name)
        if match is None:
            raise ValueError(f"unrecognized slab filename: {path}")
        key = (Path(path).parent.name, float(match.group(1)), int(match.group(2)))
        if key in seen:
            raise ValueError(f"duplicate reference point: {key}")
        seen.add(key)
        with uproot.open(path) as root:
            tree = root["dump/particles"]
            names = [name for name in root.keys(recursive=True, cycle=True) if name.split(";")[0] == "dump/particles"]
            cycle = max(int(name.rsplit(";", 1)[1]) for name in names)
            df = tree.arrays(T.KEY + ["isprim", "pdg", "epz"], library="pd")
            primary = df[(df.isprim == 1) & (df.pdg.abs() == 13)]
            events = len(primary.drop_duplicates(T.KEY))
            if events != len(primary):
                raise ValueError(f"reference has multiple primary muons per event: {path}")
            forward = int((primary.epz > 0).sum())
            if key in points:
                row = points[key]
                if events != row["n_g4"] or not np.isclose(forward / max(events, 1), row["surv_g4"], rtol=0, atol=1e-14):
                    raise ValueError(f"reference entries disagree with saved result: {path}")
            elif forward >= 1000:
                raise ValueError(f"unexplained missing slab point: {key}")
            records.append({"path": str(Path(path).resolve()), "particle_tree_cycle": cycle,
                            "particle_rows": int(tree.num_entries), "primary_muon_events": events,
                            "forward_primary_muons": forward, "included": key in points,
                            "size_bytes": Path(path).stat().st_size, "mtime_ns": Path(path).stat().st_mtime_ns})
    if not points.keys() <= seen:
        raise ValueError("reference inventory does not cover all saved points")
    return {"n_files": len(records), "primary_muon_events": sum(r["primary_muon_events"] for r in records),
            "files": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--references", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tracker-ref", required=True)
    parser.add_argument("--core-ref", required=True)
    args = parser.parse_args(argv)
    with open(args.before) as stream:
        before = json.load(stream)
    with open(args.after) as stream:
        after = json.load(stream)
    result = compare(before, after)
    result["references"] = audit_references(args.references, after["rows"])
    result["provenance"] = {"before": str(Path(args.before).resolve()), "before_sha256": _digest(args.before),
                            "after": str(Path(args.after).resolve()), "after_sha256": _digest(args.after),
                            "candidate_tracker_ref": args.tracker_ref, "candidate_core_ref": args.core_ref,
                            "baseline_provenance": "historical saved output; settings absent unless recorded in baseline_settings"}
    with open(args.out, "x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"matched {result['n_matched_points']} points; {result['references']['n_files']} reference files; "
          f"{result['references']['primary_muon_events']} actual primary-muon events")
    for row in result["summary"]:
        print(row["material"], "fit-domain points", row["n_fit_domain"])
        for name, arms in row["arms"].items():
            for arm, values in arms.items():
                print(name, arm, json.dumps(values, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
