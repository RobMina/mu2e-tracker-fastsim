"""Compact package-1 closure metrics and exact saved-baseline comparison."""
import argparse
import json
from pathlib import Path

import numpy as np

from trkfastsim.reco.nativetime import digest


def compare_arrays(left, right):
    with np.load(left) as a, np.load(right) as b:
        if set(a.files) != set(b.files):
            return {"same_fields": False}
        changed = [k for k in a.files if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape
                   or not np.array_equal(a[k], b[k], equal_nan=True)]
    return {"same_fields": True, "changed_fields": changed, "arrays_equal": not changed,
            "byte_identical": digest(left) == digest(right)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report = json.loads(a.report.read_text())
    out = {"report": str(a.report), "report_sha256": digest(a.report), "arms": {}, "cross": report.get("cross", {})}
    for name, arm in report["arms"].items():
        dest = dict(configuration=arm["configuration"], n_tracks=arm["n_tracks"])
        if "yield" in arm:
            dest["yield"] = arm["yield"]
            n = arm["yield"]["n"]
            rate = arm["yield"]["acceptance_agreement"]
            dest["agreement_binomial_se"] = float(np.sqrt(rate * (1 - rate) / n))
        if "fidelity" in arm:
            dest["fidelity"] = arm["fidelity"]
        dest["numerical_validity"] = arm.get("numerical_validity")
        reference_file = Path(arm["dir"]) / "reference_search.npz"
        if reference_file.exists():
            with np.load(reference_file) as flags, np.load(Path(arm["dir"]) / "ids.npz") as ids:
                if not all(np.array_equal(flags[k], ids[k]) for k in ("subrun", "event")):
                    raise ValueError("reference flag identities differ from arm")
                unsupported = flags["unsupported"]
                masks = {"all": np.ones(len(unsupported), bool), "accepted": ids["good"],
                         "gains": ids["good"] & ~ids["stockok"], "losses": ~ids["good"] & ids["stockok"]}
                dest["unsupported_populations"] = {k: {"n": int(m.sum()), "unsupported": int((m & unsupported).sum())}
                                                    for k, m in masks.items()}
                dest["unsupported_scope"] = "any valid saved candidate after accepted iteration, including not-yet-constructed extension candidates"
        base_name = "own_stock" if name.startswith("own_") else "fixed_stock"
        if name != base_name and base_name in report["arms"]:
            base = Path(report["arms"][base_name]["dir"])
            candidate = Path(arm["dir"])
            files = ["ids.npz", "validity.npz", "xings.npz"]
            if name.startswith("own_"):
                files.append("yield.npz")
            dest["vs_preserved_baseline"] = {f: compare_arrays(base / f, candidate / f) for f in files}
        out["arms"][name] = dest
    encoded = json.dumps(out, indent=2, allow_nan=False) + "\n"
    with a.out.open("x") as stream:
        stream.write(encoded)
    compact = {"report": out["report"], "arms": {}, "cross": out["cross"]}
    for name, arm in out["arms"].items():
        row = {"n_tracks": arm["n_tracks"], "configuration": arm["configuration"]}
        if "yield" in arm:
            row.update(yield_result=arm["yield"], agreement_binomial_se=arm["agreement_binomial_se"])
        if "fidelity" in arm:
            row["fidelity"] = {key: {k: value for k, value in metric.items()
                                      if k in ("n", "n_finite", "frac_within_0p1", "binomial_se")}
                               for key, metric in arm["fidelity"].items() if isinstance(metric, dict)}
        row["unsupported_populations"] = arm.get("unsupported_populations")
        row["vs_preserved_baseline"] = arm.get("vs_preserved_baseline")
        compact["arms"][name] = row
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
