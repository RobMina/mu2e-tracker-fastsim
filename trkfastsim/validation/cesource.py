"""Inspect and audit persisted mixer EventID metadata against a declared sequential stop window.

This reads identity metadata only, not art physics. A shared upstream library remains shared even with zero
record overlap, and exact source identities do not establish generated-primary or detector-response completeness.
"""
import argparse
import json
from pathlib import Path
import re

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from trkfastsim.validation import cecontract

KEY = cecontract.KEY
SUFFIX = "_TargetStopResampler__CaloLibraryCE."


def branch(root):
    names = [k for k in root["Events"].keys() if k.endswith(SUFFIX) and "EventIDs" in k]
    if len(names) != 1:
        raise ValueError("exactly one persisted TargetStopResampler EventIDs branch required")
    return names[0]


def inspect(path):
    """Expose fixed identity-record fields before implementing any custom-streamer assumptions."""
    with uproot.open(path) as root:
        name = branch(root)
        b = root["Events"][name]
        report = dict(branch=name, typename=b.typename, interpretation=str(b.interpretation),
                      children={c.name: dict(typename=c.typename, interpretation=str(c.interpretation))
                                for c in b.branches}, raw_branch_names=root["Events"].keys(recursive=False),
                      events=int(root["Events"].num_entries),
                      scope="identity metadata shape inspection only; decoding failures are not certified source identities")
        try:
            a = b.array(entry_stop=2, library="ak")
            report.update(type=str(a.type), fields=a.fields, preview=str(a), decoded=True)
        except (ValueError, TypeError, NotImplementedError) as exc:
            report.update(decoded=False, error=str(exc))
        return report


def read_log(text):
    """Pair serial EventLister records with exactly one following persisted source EventID."""
    current = re.compile(r"^Event:\s+(\d+)\s+run:\s*(\d+)\s+subRun:\s*(\d+)\s+event:\s*(\d+)\s*$", re.M)
    source = re.compile(r"^EventIDSequence InputTag: label = 'TargetStopResampler', instance = '', "
                        r"process = 'CaloLibraryCE' :\s*\( run:\s*(\d+) subRun:\s*(\d+) event:\s*(\d+) \)\s*$", re.M)
    headers, records = list(current.finditer(text)), list(source.finditer(text))
    count = len(headers)
    if not count or count != len(records) or text.count("EventIDSequence InputTag:") != count:
        raise ValueError("one exact current-event and persisted mixer source record required")
    start = np.fromiter((m.start() for m in headers), dtype=np.int64)
    end = np.r_[start[1:], len(text)]
    pos = np.fromiter((m.start() for m in records), dtype=np.int64)
    if not ((pos > start) & (pos < end)).all():
        raise ValueError("source EventID outside its serial current-event block")
    rows = np.asarray([m.groups() for m in headers], dtype=np.uint64)
    if not np.array_equal(rows[:, 0], np.arange(1, count + 1)):
        raise ValueError("EventLister sequence is incomplete")
    generated = pd.DataFrame(rows[:, 1:], columns=KEY)
    upstream = pd.DataFrame([m.groups() for m in records], columns=KEY).astype(np.uint64)
    cecontract.unique(generated, "source-read generated events")
    cecontract.unique(upstream, "persisted source occurrences")
    return generated, upstream


def compare(generated, source, window):
    """Compare the full source set and declared sequential physical order without inferred shifts."""
    generated = cecontract.integer_keys(generated, "generated metadata")
    source = cecontract.integer_keys(source, "source metadata")
    window = cecontract.integer_keys(window, "stop window")
    cecontract.unique(generated, "generated metadata")
    cecontract.unique(source, "source metadata")
    cecontract.unique(window, "stop window")
    if not (len(generated) == len(source) == len(window)):
        raise ValueError("mixer/generated/source-window event counts differ")
    pd.testing.assert_frame_equal(source[KEY].reset_index(drop=True), window[KEY].reset_index(drop=True),
                                  check_exact=True)
    return dict(events=len(source), exact_sequential_order=True, source_window_complete=True,
                source_keys_sha256=cecontract.fingerprint(source),
                generated_keys_sha256=cecontract.fingerprint(generated),
                scope="exact source occurrence/order only; generated particles and physics still require audit")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-art")
    mode.add_argument("--source-log")
    ap.add_argument("--window-art")
    ap.add_argument("--expected-events", type=int)
    ap.add_argument("--fileindex-byteswap", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.inspect_art:
        report = inspect(a.inspect_art)
    else:
        from trkfastsim.validation import cestopwindow
        if not a.window_art or a.expected_events is None or a.expected_events <= 0:
            ap.error("source-log requires window-art and positive expected-events")
        generated, source = read_log(Path(a.source_log).read_text())
        window = cestopwindow.load(a.window_art, a.expected_events, a.fileindex_byteswap)
        report = compare(generated, source, window)
        report["input_sha256"] = {"source_log": cestopwindow.checksum(a.source_log),
                                  "window_art": cestopwindow.checksum(a.window_art)}
    path = Path(a.out)
    with path.open("x") as f:
        f.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
