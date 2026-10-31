"""Compare completed bounded replays on exactly the same full muon identities."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.transport_capture_audit import digest


def compare(a, b):
    if a["status"] != "bounded_transport_complete" or b["status"] != "bounded_transport_complete":
        raise ValueError("both bounded transports must complete")
    if a["identities"] != b["identities"] or a["input_sha256"] != b["input_sha256"]:
        raise ValueError("paired probes need identical ordered identities and inputs")
    if a["requested_length_mm"] != b["requested_length_mm"]:
        raise ValueError("requested extent differs")
    out = dict(scope="same-identity numerical/model sensitivity; not statistical or rigorous error bounds",
               identities=a["identities"], lanes=len(a["identities"]), deltas={})
    for name in ("final_ke_MeV", "path_mm", "boundary_relabel_mm"):
        delta = np.asarray(b[name]) - np.asarray(a[name])
        out["deltas"][name] = dict(values=delta.tolist(), max_abs=float(np.max(np.abs(delta))))
    out["stopped_changed"] = int(np.count_nonzero(np.asarray(a["stopped"]) != np.asarray(b["stopped"])))
    out["entry_count_change"] = b["entries"] - a["entries"]
    out["settings"] = [{k: r.get(k) for k in ("mode", "scatter", "max_step_mm", "boundary_resolution_mm")}
                       for r in (a, b)]
    if a["scatter"] or b["scatter"]:
        out["rng_warning"] = "different slice/boundary counts can realign shared RNG; paired values are not a strong-error estimate"
    return out


def run(first, second, out):
    paths = [Path(first), Path(second)]
    hashes = {str(p): digest(p) for p in paths}
    result = compare(*[json.loads(p.read_text()) for p in paths])
    result["input_sha256"] = hashes
    if hashes != {str(p): digest(p) for p in paths}:
        raise ValueError("probe report changed")
    with Path(out).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.first, args.second, args.out)
