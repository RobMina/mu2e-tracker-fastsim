"""Exact persisted digi-table comparison, including empty events and truth indices."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


def compare(a, b, differences=False):
    with uproot.open(a) as left, uproot.open(b) as right:
        x = left["trkdump/events"].arrays()
        y = right["trkdump/events"].arrays()
    if set(ak.fields(x)) != set(ak.fields(y)) or len(x) != len(y):
        raise ValueError("persisted schema or event count differs")
    different = [k for k in ak.fields(x) if not bool(ak.all(x[k] == y[k]))]
    detail = {}
    if different:
        for name in different:
            xx, yy = ak.to_numpy(ak.flatten(x[name], axis=None)), ak.to_numpy(ak.flatten(y[name], axis=None))
            if xx.shape != yy.shape:
                raise ValueError(f"different column populations: {name}")
            mismatch = xx != yy
            delta = xx.astype(np.int64) - yy.astype(np.int64)
            detail[name] = {"different": int(mismatch.sum()), "total": len(xx),
                            "max_abs": int(np.max(np.abs(delta), initial=0))}
        if not differences:
            raise ValueError(f"persisted columns differ: {detail}")
    return {"left": str(a), "right": str(b), "events": len(x),
            "digis": int(ak.sum(ak.num(x.d_sid))), "identical": not different, "differences": detail}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--json", type=Path)
    p.add_argument("--differences", action="store_true", help="diagnostic counts, never an identity pass")
    args = p.parse_args(argv)
    if args.left.is_dir() and args.right.is_dir():
        a = sorted(args.left.glob("*.root"))
        b = sorted(args.right.glob("*.root"))
        if not a or [f.name for f in a] != [f.name for f in b]:
            raise ValueError("empty or mismatched file populations")
    else:
        a, b = [args.left], [args.right]
    rows = [compare(x, y, args.differences) for x, y in zip(a, b)]
    text = json.dumps({"files": rows, "events": sum(r["events"] for r in rows),
                       "digis": sum(r["digis"] for r in rows)}, indent=2) + "\n"
    if args.json:
        args.json.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
