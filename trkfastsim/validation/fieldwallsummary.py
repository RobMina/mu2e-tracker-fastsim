"""Compact presentation of the saved complete-wall development pilot; no metric recalculation."""
import argparse
import json
from pathlib import Path

from trkfastsim.validation import fieldpathaudit as pa


def _scaled(values, scale=1.):
    return [None if x is None else x*scale for x in values]


def compact(report):
    if report.get("status") != "OK" or not report.get("baseline_check",{}).get("unchanged"):
        raise ValueError("comparison is not closed with unchanged legacy metrics")
    def mean(record,scale=1.):
        return _scaled([record["mean"],record["sem"]],scale)
    def momentum(record):
        core=record["core"]
        return dict(events=core["events"],median_pp=_scaled([core["difference"][0],core["paired_error"][0]],100.),
                    width_pp=_scaled([core["difference"][1],core["paired_error"][1]],100.),
                    candidate_relative_core_percent=_scaled(core["candidate"],100.),
                    reference_relative_core_percent=_scaled(core["reference"],100.),
                    tails_pp={cut:mean(r["difference"],100.) for cut,r in record["descriptive_tails"].items()})
    def sample(block):
        out={}
        for pair,data in block["comparisons"].items():
            eff=data["efficiency"]
            out[pair]=dict(good=[eff["candidate_good"],eff["reference_good"]],gained=eff["gained"],lost=eff["lost"],
                           efficiency_pp=mean(eff["difference"],100.),
                           active_hits=mean(data["active_hits_both_good"]["difference"]),
                           distinct_active_straws=mean(data["distinct_active_straws_both_good"]["difference"]),
                           momentum=momentum(data["momentum"]))
        return dict(events=block["denominator_events"],comparisons=out,
                    common_three_way_diagnostic={pair:momentum(data) for pair,data in
                        block["three_way_common_diagnostic"]["comparisons"].items()})
    return dict(status="OK",scope=report["scope"],units="pp means percentage points; hit/straw differences are counts; [estimate,error]",
                whole=sample(report["whole"]),
                bins=[dict(p_lo=b["reference_first_straw_p_lo"],p_hi=b["reference_first_straw_p_hi"],**sample(b)) for b in report["bins"]],
                ntuple_exposure=report["ntuple_exposure"],entry_exposure=report["entry_exposure"],
                unchanged_legacy_quantiles=report["baseline_check"]["legacy_fieldquantiles"])


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report")
    ap.add_argument("--verify-inputs",action="store_true")
    a=ap.parse_args(argv)
    report=json.loads(Path(a.report).read_text())
    if a.verify_inputs:
        for name,r in report["inputs"].items():
            if r["sha256_before"] != r["sha256_after"] or pa.sha256(r["path"]) != r["sha256_before"]:
                raise ValueError("closed comparison input changed: "+name)
    print(json.dumps(compact(report),indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
