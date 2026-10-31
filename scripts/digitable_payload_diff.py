"""Locate exact streamed-payload byte differences without changing the equality gate."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from digitable_report import PAYLOAD_BRANCHES, PAYLOAD_SCALARS, validate_payload


def differences(reference, candidate):
    result = {}
    with uproot.open(reference) as left, uproot.open(candidate) as right:
        a, b = left["payload/events"], right["payload/events"]
        validate_payload(a)
        validate_payload(b)
        keys = a.arrays(["run", "subrun", "event"], library="np")
        for key in ("run", "subrun", "event"):
            if not np.array_equal(keys[key], b[key].array(library="np")):
                raise ValueError("different payload event identities")
        for name in sorted(PAYLOAD_BRANCHES - PAYLOAD_SCALARS):
            aa, bb = a[name].array(), b[name].array()
            lengths_equal = ak.num(aa) == ak.num(bb)
            item = {"equal_lengths": bool(ak.all(lengths_equal)), "events": len(aa)}
            if item["equal_lengths"]:
                bad = aa != bb
                event_rows = ak.to_numpy(ak.local_index(aa, axis=0)[ak.any(bad, axis=1)])
                offsets = ak.to_numpy(ak.flatten(ak.local_index(aa)[bad]))
                rows = ak.to_numpy(ak.flatten(ak.broadcast_arrays(ak.local_index(aa, axis=0), aa)[0][bad]))
                sample = slice(0, 40)
                item.update(different_events=len(event_rows), different_bytes=len(offsets),
                            event_ids=np.column_stack([keys[key][event_rows] for key in ("run", "subrun", "event")]).tolist(),
                            sample_rows=rows[sample].tolist(), sample_offsets=offsets[sample].tolist(),
                            sample_reference=ak.to_numpy(ak.flatten(aa[bad]))[sample].tolist(),
                            sample_candidate=ak.to_numpy(ak.flatten(bb[bad]))[sample].tolist())
            result[name] = item
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    result = differences(args.reference, args.candidate)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
