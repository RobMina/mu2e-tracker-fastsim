"""Quantify retained paired-rate/churn evidence without reading or regenerating events."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.transport_capture_audit import digest


def summarize(report):
    result = dict(scope="historical saved paired counts; not current ordered-transport closure", arms={})
    for name, arm in report["arms"].items():
        counts = arm["any_muon"]["all"]
        ref, only_fast, only_g4 = counts["ref"], counts["arm_only"], counts["ref_only"]
        churn = only_fast + only_g4
        diff = only_fast - only_g4
        sigma = np.sqrt(churn)
        replay = arm["replay"]
        if replay["g4_only"]["total"] != only_g4 or replay["fast_only"]["total"] != only_fast:
            raise ValueError("replay and paired counts disagree")
        conditional = only_fast / churn
        z = 1.959963984540054
        denom = 1 + z * z / churn
        center = (conditional + z * z / (2 * churn)) / denom
        half = z * np.sqrt(conditional * (1 - conditional) / churn + z * z / (4 * churn * churn)) / denom
        fates = replay["g4_only"]["fates"]
        result["arms"][name] = dict(generated=arm["gen_events"], ref=ref, fast=counts["arm"],
            diff=diff, paired_sigma_events=float(sigma), relative_percent=100 * diff / ref,
            paired_error_percent=100 * sigma / ref, normal_95_percent=[100 * (diff - z * sigma) / ref,
                                                                      100 * (diff + z * sigma) / ref],
            discordant=churn, fast_wins_among_discordant=conditional,
            fast_wins_wilson95=[center - half, center + half],
            discordant_normal_z=float(diff / sigma),
            g4_only_fates=fates, fast_only_fates=replay["fast_only"]["fates"],
            explicit_fast_interior_stops_in_g4_only=int(fates.get("few_stopped_interior", 0)),
            interpretation="fates diagnose unmatched acceptance; no same-draw causal allocation of net residual")
    return result


def run(source, out):
    source, out = Path(source), Path(out)
    before = digest(source)
    result = summarize(json.loads(source.read_text()))
    result["input_sha256"] = before
    if digest(source) != before:
        raise ValueError("input changed")
    with out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args.source, args.out)
