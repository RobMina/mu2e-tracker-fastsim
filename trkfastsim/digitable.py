"""Lossless, detector-only StrawDigi table admission, conversion and equality audit.

No truth is manufactured. Legacy tables without digi flags require an explicit
zero-flag assumption and cannot establish an original-product round trip.
"""
import argparse
import hashlib
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

KEYS = ("run", "subrun", "event")
LIMITS = {"d_sid": 65535, "d_tdc0": 2**32 - 1, "d_tdc1": 2**32 - 1,
          "d_tot0": 65535, "d_tot1": 65535, "d_pmp": 65535,
          "d_flags": 255, "d_nadc": 2**32 - 1, "d_adc": 65535}


def _integers(array, limit, name):
    values = ak.to_numpy(array)
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{name}: missing integer")
    values = np.asarray(values)
    if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values > limit):
        raise ValueError(f"{name}: expected integers in [0, {limit}]")
    return values.astype(np.uint32)


def validate(columns, *, assume_zero_flags=False):
    """Normalize complete event rows with vectorized checks; retain digi/ADC order."""
    cols = dict(columns)
    missing = set(KEYS) | set(LIMITS)
    missing -= cols.keys()
    assumed = "d_flags" in missing and assume_zero_flags
    if assumed:
        missing.remove("d_flags")
        cols["d_flags"] = ak.zeros_like(cols["d_sid"]) if "d_sid" in cols else ak.Array([])
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    out = {key: _integers(cols[key], 2**32 - 1, key) for key in KEYS}
    if any(out[key].ndim != 1 for key in KEYS):
        raise ValueError("event identities must be scalar columns")
    n = len(out["run"])
    if any(len(out[key]) != n for key in KEYS):
        raise ValueError("unequal event column lengths")
    keys = np.column_stack([out[key] for key in KEYS])
    if len(np.unique(keys, axis=0)) != n:
        raise ValueError("duplicate run/subrun/event identity")
    if np.any(out["run"] == 0) or np.any(out["event"] == 0):
        raise ValueError("art run/event identities must be positive")
    counts = {}
    flat = {}
    for name, limit in LIMITS.items():
        arr = ak.Array(cols[name])
        if arr.ndim != 2 or len(arr) != n:
            raise ValueError(f"{name}: expected one integer list per event")
        count = ak.to_numpy(ak.num(arr, axis=1))
        if np.any(np.ma.getmaskarray(count)):
            raise ValueError(f"{name}: missing event list")
        values = _integers(ak.flatten(arr, axis=1), limit, name)
        flat[name], counts[name] = values, count
        out[name] = ak.unflatten(values, count)
    nd = counts["d_sid"]
    for name in LIMITS:
        if name != "d_adc" and not np.array_equal(counts[name], nd):
            raise ValueError(f"{name}: count does not match digis per event")
    sid = flat["d_sid"]
    if np.any((sid >> 10) >= 36) or np.any(((sid >> 7) & 7) >= 6) or np.any((sid & 127) >= 96):
        raise ValueError("d_sid: invalid packed StrawId")
    expected = ak.to_numpy(ak.sum(out["d_nadc"], axis=1))
    if not np.array_equal(expected, counts["d_adc"]):
        raise ValueError("d_nadc: summed waveform lengths do not match d_adc per event")
    summary = dict(events=n, digis=int(nd.sum()), adc_samples=int(counts["d_adc"].sum()),
                   empty_digi_events=int(np.count_nonzero(nd == 0)),
                   assumed_zero_flags=assumed, flags_present_in_input=not assumed)
    return out, summary


def read(path, tree="trkdigi/events", *, assume_zero_flags=False):
    """Read the highest ROOT cycle, rejecting absent/unsupported tables."""
    with uproot.open(path) as root:
        obj = root[tree]
        if obj.classname != "TTree":
            raise ValueError("expected a TTree, not an RNTuple or another ROOT object")
        names = [name for name in (*KEYS, *LIMITS) if name in obj.keys()]
        arrays = obj.arrays(names, library="ak", how=dict)
        count = obj.num_entries
    cols, summary = validate(arrays, assume_zero_flags=assume_zero_flags)
    if summary["events"] != count:
        raise ValueError("incomplete highest-cycle event population")
    return cols, summary


def write(path, columns):
    """Write canonical unsigned counted arrays; refuse accidental overwrite."""
    cols, summary = validate(columns)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.create(path) as root:
        root.mktree("trkdigi/events", {name: (value.type.content if isinstance(value, ak.Array) else value.dtype)
                                     for name, value in cols.items()})
        root["trkdigi/events"].extend(cols)
    return summary


def compare(a, b):
    """Exact full-population equality, allowing only event-row reordering."""
    a, sa = validate(a)
    b, sb = validate(b)
    def ordered(cols):
        order = np.lexsort(tuple(cols[key] for key in reversed(KEYS)))
        return {key: value[order] for key, value in cols.items()}
    a, b = ordered(a), ordered(b)
    equal = {name: bool(ak.almost_equal(a[name], b[name], rtol=0, atol=0, dtype_exact=True)) for name in a}
    return {"equal": all(equal.values()), "columns_equal": equal, "reference": sa, "candidate": sb,
            "scope": "all unique events and every ordered digi field/ADC sample; no truth/reco-product claim"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert")
    convert.add_argument("input", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--tree", default="trkdump/events")
    convert.add_argument("--assume-zero-flags", action="store_true")
    audit = sub.add_parser("compare")
    audit.add_argument("reference", type=Path)
    audit.add_argument("candidate", type=Path)
    audit.add_argument("--tree", default="trkdigi/events")
    for cmd in (convert, audit):
        cmd.add_argument("--report", type=Path, required=True)
    args = p.parse_args(argv)
    if args.report.exists() or (args.command == "convert" and args.output.exists()):
        p.error("output/report already exists")
    if args.command == "convert":
        cols, summary = read(args.input, args.tree, assume_zero_flags=args.assume_zero_flags)
        write(args.output, cols)
        persisted, receipt = read(args.output)
        report = {"input": str(args.input), "input_sha256": sha256(args.input), "input_tree": args.tree,
                  "output": str(args.output), "output_sha256": sha256(args.output),
                  "admission": summary, "persisted": receipt, "roundtrip": compare(cols, persisted)}
        ok = report["roundtrip"]["equal"]
    else:
        a, _ = read(args.reference, args.tree)
        b, _ = read(args.candidate, args.tree)
        report = compare(a, b)
        report.update(reference_file=str(args.reference), candidate_file=str(args.candidate),
                      reference_sha256=sha256(args.reference), candidate_sha256=sha256(args.candidate))
        ok = report["equal"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
