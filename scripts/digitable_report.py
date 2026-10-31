"""Verify completed art digi-replay sequences without intersecting event populations."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from trkfastsim import digitable


PAYLOAD_BRANCHES = {"run", "subrun", "event", "pbt", "single_hits", "panel_hits", "stereo_hits", "flagged_hits",
                    "straw_hits", "hit_intensity", "cluster_intensity", "time_clusters", "line_seeds", "kal_seeds",
                    "kal_line_assns", "background_clusters", "background_hits", "used_pbt", "flag_ph_active",
                    "backgrounds_active"}
PAYLOAD_SCALARS = set(digitable.KEYS) | {"flag_ph_active", "backgrounds_active"}


def validate_payload(tree):
    """Configured inactive producers are explicit, never silently missing products."""
    counters = {"n" + key for key in PAYLOAD_BRANCHES - PAYLOAD_SCALARS}
    branches = set(tree.keys())
    if not PAYLOAD_BRANCHES <= branches or branches - PAYLOAD_BRANCHES - counters:
        raise ValueError("incomplete persistent payload branch inventory")
    arrays = tree.arrays(sorted(PAYLOAD_BRANCHES), library="ak")
    active, backgrounds = arrays.flag_ph_active, arrays.backgrounds_active
    if bool(ak.any((active != 0) & (active != 1))) or bool(ak.any((backgrounds != 0) & (backgrounds != 1))):
        raise ValueError("invalid persistent payload activity markers")
    if bool(ak.any(backgrounds > active)):
        raise ValueError("background payload without active flagPH")
    for key in PAYLOAD_BRANCHES - PAYLOAD_SCALARS:
        expected = active if key in {"flagged_hits", "cluster_intensity"} else backgrounds if key in {
            "background_clusters", "background_hits"} else 1
        if bool(ak.any((ak.num(arrays[key]) > 0) != expected)):
            raise ValueError("missing active or nonempty inactive persistent payload: " + key)
    keys = np.column_stack([ak.to_numpy(arrays[key]) for key in digitable.KEYS])
    if len(np.unique(keys, axis=0)) != tree.num_entries:
        raise ValueError("duplicate persistent payload event identities")


def compare_dump(reference, candidate, *, payload=False):
    """Exact exported reco values and optional complete persistent ROOT byte payloads."""
    result = {"equal": True, "trees": {}, "scope": "all exported StrawHitTableDumper branches at highest cycle"}
    if payload:
        result["scope"] += "; complete persistent reco ROOT-streamed bytes, identical labels"
    names = ("trkdump/events", "trkdump/conditions") + (("payload/events",) if payload else ())
    with uproot.open(reference) as left, uproot.open(candidate) as right:
        for name in names:
            a, b = left[name], right[name]
            if name == "payload/events":
                for root, tree in ((left, a), (right, b)):
                    validate_payload(tree)
                    for key in digitable.KEYS:
                        if not ak.almost_equal(tree[key].array(), root["trkdump/events"][key].array(),
                                               rtol=0, atol=0, dtype_exact=True):
                            raise ValueError("persistent payload event population differs from reco dump")
            item = {"reference_entries": a.num_entries, "candidate_entries": b.num_entries,
                    "same_branches": set(a.keys()) == set(b.keys()), "different": []}
            equal = item["same_branches"] and a.num_entries == b.num_entries
            for branch in sorted(set(a.keys()) & set(b.keys())):
                aa, bb = a[branch].array(), b[branch].array()
                if not ak.almost_equal(aa, bb, rtol=0, atol=0, dtype_exact=True):
                    item["different"].append(branch)
                    equal = False
            item["equal"] = equal
            result["equal"] &= equal
            result["trees"][name] = item
    return result


def census(path):
    with uproot.open(path) as root:
        tree = root["trkdump/events"]
        a = tree.arrays(["run", "subrun", "event", "d_sid", "d_nadc", "kl_status"], library="ak")
        keys = np.column_stack([ak.to_numpy(a[key]) for key in digitable.KEYS])
        if len(keys) != tree.num_entries or len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError("duplicate/incomplete event census")
        return {"events": len(keys), "digis": int(ak.sum(ak.num(a.d_sid))),
                "empty_events": int(ak.sum(ak.num(a.d_sid) == 0)), "adc_samples": int(ak.sum(a.d_nadc)),
                "stored_stock_fits": int(ak.sum(ak.num(a.kl_status))),
                "cycle": tree.object_path}


def report(directory, expected_events, *, payload=False):
    directory = Path(directory)
    if not (directory / "sequence.done").is_file():
        raise ValueError("art sequence did not finish")
    a, sa = digitable.read(directory / "digis.root")
    b, sb = digitable.read(directory / "replay.root")
    digi = digitable.compare(a, b)
    reco = compare_dump(directory / "original_reco.root", directory / "replay_reco.root", payload=payload)
    original, replayed = (census(directory / name) for name in ("original_reco.root", "replay_reco.root"))
    counts = [sa["events"], sb["events"], original["events"], replayed["events"]]
    if payload:
        counts += [reco["trees"]["payload/events"][key] for key in ("reference_entries", "candidate_entries")]
    complete = all(n == expected_events for n in counts)
    result = {"directory": str(directory), "expected_events": expected_events, "complete": complete,
              "digis": digi, "exported_reco": reco, "original_census": original, "replayed_census": replayed,
              "equal": complete and digi["equal"] and reco["equal"],
              "full_persistent_reco_product_equality": (bool(complete and reco["trees"]["payload/events"]["equal"])
                                                        if payload else "not tested by reduced dumper")}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--payload", action="store_true", help="require complete ROOT-streamed persistent reco branches")
    args = parser.parse_args(argv)
    result = report(args.directory, args.expected_events, payload=args.payload)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
