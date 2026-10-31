"""Saved-only native root/owner/plane evidence audit, not literal GDML or G4 physics certification."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest
from trkfastsim import exactplanar, nativeenvelope

_DECODE = np.frompyfunc(lambda n, d: Fraction(int(n), int(d)), 2, 1)


def audit_roots(roots, footprint):
    required = {"lane", "source", *[f"q{i}_{name}" for i in (0, 1) for name in
                                   ("numerator", "denominator", "lower_mm", "upper_mm")]}
    if not required <= set(roots) or len(roots) != len(footprint):
        raise ValueError("complete native roots/possible-footprint rows required")
    if not np.array_equal(roots[["lane", "source"]].to_numpy(), footprint[["lane", "source"]].to_numpy()):
        raise ValueError("native root/footprint owner ordering differs")
    values, widths = {}, []
    for i in (0, 1):
        stem = f"q{i}_"
        numerator, denominator = roots[stem + "numerator"].astype(str), roots[stem + "denominator"].astype(str)
        if not numerator.str.fullmatch(r"-?[0-9]+").all() or not denominator.str.fullmatch(r"[1-9][0-9]*").all():
            raise ValueError("canonical rational endpoint integer text required")
        q = _DECODE(numerator.to_numpy(), denominator.to_numpy())
        values[i] = q
        lower, upper = roots[stem + "lower_mm"].to_numpy(float), roots[stem + "upper_mm"].to_numpy(float)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower > upper).any():
            raise ValueError("finite ordered native directed brackets required")
        if not np.array_equal(lower, exactplanar.export(q, False)) or not np.array_equal(upper, exactplanar.export(q, True)):
            raise ValueError("native rational root disagrees with directed binary64 bracket")
        if (upper - lower > 1e-10).any():
            raise ValueError("native root projection exceeds original endpoint budget")
        widths.append(upper - lower)
    if (values[1] <= values[0]).any():
        raise ValueError("native root intervals must be exactly positive")
    if not np.array_equal(footprint.s0_mm.to_numpy(), roots.q0_lower_mm.to_numpy(float)) or not np.array_equal(
            footprint.s1_mm.to_numpy(), roots.q1_upper_mm.to_numpy(float)):
        raise ValueError("native parent footprint is not complete outward root enclosure")
    if len(footprint) and not (footprint.ancestors_verified.all() and footprint.leaf_verified.all()):
        raise ValueError("native possible-footprint hierarchy unverified")
    sizes = roots.groupby("lane").size()
    if int((sizes * sizes).sum()) > 2000000:
        raise ValueError("native exact occupancy audit pair cap")
    owners = roots[["lane", "source"]].assign(q0=values[0], q1=values[1])
    pairs = owners.merge(owners, on="lane", suffixes=("_a", "_b"))
    pairs = pairs[pairs.source_a < pairs.source_b]
    if (np.minimum(pairs.q1_a, pairs.q1_b) > np.maximum(pairs.q0_a, pairs.q0_b)).any():
        raise ValueError("positive native physical leaf overlap")
    return dict(root_rows=len(roots), sources=int(roots.source.nunique()),
        max_projection_width_mm=float(np.concatenate(widths).max(initial=0.)), exact_positive_leaf_overlaps=0)


def run(directory, out):
    directory, out = Path(directory), Path(out)
    receipt = json.loads((directory / "complete.json").read_text())
    if receipt.get("envelope_policy") != nativeenvelope.POLICY:
        raise ValueError("explicit native geometry producer required")
    inputs = [p for p in directory.rglob("*") if p.is_file()]
    hashes = {str(p): digest(p) for p in inputs}
    summaries = []
    for query in sorted(directory.glob("navigation_*")):
        status = json.loads((query / "receipt.json").read_text())
        roots_path, footprint_path = query / "result.native_root_endpoints.csv", query / "result.native_possible_footprint.csv"
        if not roots_path.is_file():
            if status["status"] == "complete":
                raise ValueError("completed native query lacks exact root evidence")
            continue
        roots = pd.read_csv(roots_path, dtype={f"q{i}_{k}": str for i in (0, 1) for k in ("numerator", "denominator")}, float_precision="round_trip")
        footprint = pd.read_csv(footprint_path, float_precision="round_trip")
        result = audit_roots(roots, footprint)
        if not status["envelope_summary"]["all_pairs_admitted"] and status["status"] == "complete":
            raise ValueError("completed native query advanced refused pair")
        summaries.append(dict(query=query.name, status=status["status"], **result))
    result = dict(scope="saved native exact-root/topology/footprint contracts only; no G4 or world ownership claim",
        producer_status=receipt["status"], queries=summaries, input_sha256=hashes,
        source_sha256={str(Path(__file__)): digest(__file__), str(Path(exactplanar.__file__)): digest(exactplanar.__file__)},
        admitted_for_physics=False)
    if hashes != {str(p): digest(p) for p in inputs}:
        raise ValueError("native saved evidence changed during audit")
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.directory, args.out)
