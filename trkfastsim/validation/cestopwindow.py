"""Audit exact art FileIndex identities for disjoint bounded upstream stop-event windows.

This reads metadata, not art physics. Disjoint event records do not establish independent stop-library uncertainty.
Custom-streamer byte reversal is an explicit option and is never guessed from a plausible event count.
"""
import argparse
import hashlib
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from trkfastsim.validation import cecontract

KEY = cecontract.KEY


def ordered_keys(run, subrun, event, entry, expected, byteswap=False):
    """Require integer event metadata and a complete physical Events-entry permutation."""
    arrays = [np.asarray(v) for v in (run, subrun, event, entry)]
    if any(v.ndim != 1 or v.dtype.kind not in "iu" for v in arrays):
        raise ValueError("FileIndex metadata must be one-dimensional integers")
    if len({len(v) for v in arrays}) != 1:
        raise ValueError("FileIndex metadata lengths differ")
    if byteswap:
        arrays = [v.byteswap() for v in arrays]
    run, subrun, event, entry = arrays
    sentinel = np.iinfo(np.uint32).max
    valid = (event > 0) & (event != sentinel) & (subrun != sentinel)
    if ((run[valid] <= 0) | (run[valid] >= sentinel) | (subrun[valid] < 0)
            | (subrun[valid] >= sentinel) | (event[valid] >= sentinel)).any():
        raise ValueError("invalid native art identity bounds in FileIndex event row")
    entries = entry[valid]
    if len(entries) != expected or not np.array_equal(np.sort(entries), np.arange(expected)):
        raise ValueError("FileIndex does not cover each physical Events entry exactly once")
    d = pd.DataFrame(dict(zip(KEY, (run[valid], subrun[valid], event[valid]))))
    d = cecontract.integer_keys(d, "stop-window metadata")
    cecontract.unique(d, "stop-window metadata")
    return d.iloc[np.argsort(entries)].reset_index(drop=True)


def load(path, expected, byteswap=False):
    """Read the highest tree cycles and retain the declared metadata byte order."""
    with uproot.open(path) as root:
        n = root["Events"].num_entries
        if n != expected:
            raise ValueError(f"finished art has {n} Events, expected {expected}")
        a = root["FileIndex"]["Element"].array(library="ak")
        keys = ordered_keys(ak.to_numpy(a.eventID.subRun_.run_.run_), ak.to_numpy(a.eventID.subRun_.subRun_),
                            ak.to_numpy(a.eventID.event_), ak.to_numpy(a.entry), n, byteswap)
    return keys


def disjoint(old, new):
    """Reject any exact shared upstream event identity, without proximity or timing matching."""
    old = cecontract.unique(old, "previous stop window")
    new = cecontract.unique(new, "new stop window")
    if len(old.merge(new, on=KEY, how="inner", validate="one_to_one")):
        raise ValueError("new stop window overlaps previous upstream event identities")
    return dict(previous_events=len(old), new_events=len(new), shared_event_identities=0,
                previous_keys_sha256=cecontract.fingerprint(old), new_keys_sha256=cecontract.fingerprint(new),
                scope="disjoint upstream event records from one shared stop library; not POT normalization")


def checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--previous", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--expected-events", type=int, required=True)
    ap.add_argument("--fileindex-byteswap", action="store_true")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    if a.expected_events <= 0:
        ap.error("expected-events must be positive")
    old, new = (load(p, a.expected_events, a.fileindex_byteswap) for p in (a.previous, a.new))
    report = disjoint(old, new)
    report.update(previous_art_sha256=checksum(a.previous), new_art_sha256=checksum(a.new),
                  explicit_uproot_metadata_byteswap=a.fileindex_byteswap)
    out = Path(a.outdir)
    out.mkdir(exist_ok=False)
    old.to_csv(out / "previous.csv.gz", index=False)
    new.to_csv(out / "new.csv.gz", index=False)
    (out / "cestopwindow.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
