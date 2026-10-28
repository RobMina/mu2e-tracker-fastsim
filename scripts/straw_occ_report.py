"""Summarize saved secondary-census and path-A comparisons with evidence hashes.

Run with LABEL=JSON arguments; use --out FILE to retain the compact report.
"""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(path):
    data = Path(path).read_bytes()
    d = json.loads(data)
    if 'all_defaults_passed' in d:
        files = d['files']
        numeric = ('drawn', 'placed', 'reflected', 'off_lattice', 'in_muon_straw', 'merged_same_particle')
        placement = {arm: {k: sum(f[arm][k] if arm == 'placement' else f['default']['placement']['original'][k]
                                  for f in files) for k in numeric}
                     for arm in ('placement', 'default')}
        return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'status': d['status'],
                'all_defaults_passed': d['all_defaults_passed'], 'files': len(files),
                'file_indices': [f['file_index'] for f in files], 'elapsed_s': d['elapsed_s'],
                'candidate_counts': {k: sum(f['candidate_counts'][k] for f in files)
                                     for k in ('primary_steps', 'instraw_steps', 'offaxis_steps')},
                'placement': placement,
                'all_preserved': all(v['passed'] for f in files for v in f['preserved'].values())}
    out = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "paired": d["paired"]}
    if "tv1b" in d:
        out.update({k: d[k] for k in ("selected_events", "totals", "tv1b")})
        out["eion_per_straw"] = {k: v for k, v in d["eion_per_straw"].items() if k != "bins" and not k.startswith("event_clustered")}
        for k, v in d["eion_per_straw"].items():
            if k.startswith("event_clustered"):
                out["eion_per_straw"][k] = {f: v.get(f) for f in ("available", "max_abs_dev", "max_abs_dev_minus_2sigma",
                        "covariance_rank", "unavailable_reason", "difference", "difference_error", "ratio_minus_1", "ratio_error")}

        out["breakdowns"] = {k: d["breakdowns"][k] for k in ("location", "plane_distance")}
        occupancy = d["occupancy_steps"]
    else:
        out.update({k: d[k] for k in ("selected_events", "selected_in_ntuple", "tracks", "yield_per_s",
                                       "yield_factorised", "thresholds")})
        out["clustered_shapes"] = {k: {f: v.get(f) for f in ("available", "max_abs_dev", "max_abs_dev_minus_2sigma",
                                                         "covariance_rank", "unavailable_reason", "flow_fast", "flow_ref")}
                                   for k, v in d.items() if "event_clustered" in k}
        for key in ("yield_paired", "paired_exposure", "width_jackknife_groups", "livetime_s", "digis_per_event", "nactive_mean"):
            if key in d:
                out[key] = d[key]
        if "yield_paired" in d:
            paired = d["yield_paired"]
            out["yield_attribution_counts"] = {
                "total_track_difference": paired["tracks"][0] - paired["tracks"][1],
                "selection_only_track_difference": paired["a_selection_only"]["tracks_fast_only"] - paired["a_selection_only"]["tracks_g4_only"],
                "both_selected_track_difference": paired["b_both_selected_content"]["tracks"][0] - paired["b_both_selected_content"]["tracks"][1],
                "diagnostic_bin_variable": "max good StrawGasStep count for one primary muon; not unique straw count",
                "interpretation": "selection/content split is algebraic attribution, not causal identification of transport or secondary errors"}
        out["widths"] = {k: v for k, v in d.items() if k.startswith("resid_")}
        occupancy = {k: v for k, v in d.items() if k.startswith("panel_occupancy") and isinstance(v, dict)}
    out["occupancy"] = {k: {f: v[f] for f in ("chi2ndf_poisson", "chi2ndf_clustered", "chi2ndf_covariance",
                                                    "covariance_rank", "max_abs_pull", "panels_beyond_5sigma") if f in v}
                        for k, v in occupancy.items()}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="LABEL=JSON")
    p.add_argument("--out")
    args = p.parse_args()
    report = {label: summarize(path) for label, path in (s.split("=", 1) for s in args.inputs)}
    root = Path(__file__).resolve().parents[1]
    report["analysis_source_sha256"] = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                                        for f in sorted((root / "trkfastsim" / "analyses").glob("*.py"))}
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "x") as f:
            f.write(text + "\n")
    for label, d in report.items():
        if label == "analysis_source_sha256":
            continue
        print(f"=== {label}: {d['path']}")
        if 'all_defaults_passed' in d:
            print(json.dumps(d, indent=2))
            continue
        print("selected", d["selected_events"])
        if "tv1b" in d:
            for k, row in d["tv1b"].items():
                print(f"{k}: {100 * row['ratio_minus_1']:+.3f}% +- {100 * row['err']:.3f}%")
            print("eion per straw", json.dumps(d["eion_per_straw"]))
            for name, rows in d["breakdowns"].items():
                print(name, [(r["bin"], round(100 * r["ratio_minus_1"], 2), round(100 * r["err"], 2)) for r in rows])
        else:
            print("yield", json.dumps(d["yield_per_s"]))
            print("factorization", json.dumps(d["yield_factorised"]))
            print("thresholds", json.dumps(d["thresholds"]))
        print("occupancy", json.dumps(d["occupancy"]))


if __name__ == "__main__":
    main()
