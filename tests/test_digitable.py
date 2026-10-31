"""Lossless digi bridge admission and full-population audits."""
import copy
import json

import awkward as ak
import numpy as np
import pytest
import uproot

from trkfastsim import digitable as dt


def columns():
    counts = [2, 0, 1]
    cols = {"run": np.array([1400, 1400, 1400], np.uint32),
            "subrun": np.array([1001, 1001, 1001], np.uint32),
            "event": np.array([7, 8, 2**31 + 1], np.uint32)}
    fields = {"d_sid": [3, 3, (35 << 10) | (5 << 7) | 95],
              "d_tdc0": [2**32 - 1, 0, 42], "d_tdc1": [0, 30, 99],
              "d_tot0": [0, 15, 65535], "d_tot1": [15, 0, 65535],
              "d_pmp": [0, 65535, 800], "d_flags": [255, 128, 7], "d_nadc": [3, 0, 2]}
    cols.update({name: ak.unflatten(np.array(vals, np.uint32), counts) for name, vals in fields.items()})
    cols["d_adc"] = ak.unflatten(np.array([0, 65535, 100, 7, 8], np.uint32), [3, 0, 2])
    return cols


def test_canonical_roundtrip_preserves_full_width_flags_order_and_empty(tmp_path):
    source = columns()
    path = tmp_path / "table.root"
    summary = dt.write(path, source)
    read, receipt = dt.read(path)
    assert summary == receipt
    assert summary == {"events": 3, "digis": 3, "adc_samples": 5, "empty_digi_events": 1,
                       "assumed_zero_flags": False, "flags_present_in_input": True}
    assert dt.compare(source, read)["equal"]
    assert int(ak.max(read["d_tdc0"])) == 2**32 - 1
    with pytest.raises(FileExistsError):
        dt.write(path, source)


def test_compare_allows_only_event_reordering_not_digi_reordering():
    source = columns()
    shuffled = {key: value[[2, 0, 1]] for key, value in source.items()}
    assert dt.compare(source, shuffled)["equal"]
    different = copy.copy(source)
    different["d_tdc1"] = ak.Array([[30, 0], [], [99]])
    assert not dt.compare(source, different)["equal"]
    dropped_empty = {key: value[[0, 2]] for key, value in source.items()}
    assert not dt.compare(source, dropped_empty)["equal"]


@pytest.mark.parametrize("name,value,match", [
    ("d_tdc0", [[-1, 2], [], [3]], "expected integers"),
    ("d_tdc0", [[2**32, 2], [], [3]], "expected integers"),
    ("d_pmp", [[65536, 2], [], [3]], "expected integers"),
    ("d_flags", [[256, 0], [], [0]], "expected integers"),
    ("d_tot0", [[1.2, 2.0], [], [3.0]], "expected integers"),
    ("d_sid", [[65535, 1], [], [3]], "invalid packed"),
    ("d_sid", [[96, 1], [], [3]], "invalid packed"),
    ("d_sid", [[6 << 7, 1], [], [3]], "invalid packed"),
    ("d_tot1", [[1], [], [3]], "count does not match"),
    ("d_nadc", [[2, 0], [], [3]], "summed waveform"),
    ("d_nadc", [[4, 0], [], [1]], "summed waveform"),
    ("d_flags", [[None, 0], [], [0]], "missing integer"),
    ("d_flags", [[0, 0], None, [0]], "missing event list"),
])
def test_reject_malformed_fields(name, value, match):
    cols = columns()
    cols[name] = ak.Array(value)
    with pytest.raises(ValueError, match=match):
        dt.validate(cols)


@pytest.mark.parametrize("key,value,match", [
    ("event", [7, 7, 9], "duplicate"), ("run", [0, 1, 2], "positive"),
    ("event", [7, 0, 9], "positive"), ("subrun", [1, -1, 2], "expected integers"),
    ("event", [7, 2**32, 9], "expected integers"),
])
def test_reject_bad_keys(key, value, match):
    cols = columns()
    cols[key] = np.array(value)
    with pytest.raises(ValueError, match=match):
        dt.validate(cols)


def test_legacy_flags_need_explicit_labeled_assumption():
    cols = columns()
    del cols["d_flags"]
    with pytest.raises(ValueError, match="missing required.*d_flags"):
        dt.validate(cols)
    normal, summary = dt.validate(cols, assume_zero_flags=True)
    assert summary["assumed_zero_flags"] and not summary["flags_present_in_input"]
    assert ak.all(normal["d_flags"] == 0)


def test_zero_event_file(tmp_path):
    cols = {key: value[:0] for key, value in columns().items()}
    dt.write(tmp_path / "empty.root", cols)
    normal, summary = dt.read(tmp_path / "empty.root")
    assert summary["events"] == summary["digis"] == summary["adc_samples"] == 0
    assert dt.compare(normal, cols)["equal"]


def test_signed_legacy_counted_arrays_and_conversion_receipt(tmp_path):
    cols = columns()
    cols["d_tdc0"] = ak.Array([[1, 2], [], [3]])
    del cols["d_flags"]
    path = tmp_path / "legacy.root"
    with uproot.recreate(path) as root:
        schema = {key: ("var * int32" if key.startswith("d_") else "uint32") for key in cols}
        root.mktree("trkdump/events", schema).extend(cols)
    out, report = tmp_path / "new.root", tmp_path / "receipt.json"
    assert dt.main(["convert", str(path), str(out), "--report", str(report), "--assume-zero-flags"]) == 0
    receipt = json.loads(report.read_text())
    assert receipt["roundtrip"]["equal"]
    assert not receipt["admission"]["flags_present_in_input"]
    assert receipt["persisted"]["events"] == 3
    assert receipt["input_sha256"] == dt.sha256(path)
    compare = tmp_path / "compare.json"
    assert dt.main(["compare", str(out), str(out), "--report", str(compare)]) == 0
    with pytest.raises(SystemExit):
        dt.main(["convert", str(path), str(out), "--report", str(report)])


def test_reader_uses_highest_cycle(tmp_path):
    path = tmp_path / "cycles.root"
    cols = columns()
    dt.write(path, cols)
    with uproot.update(path) as root:
        root.mktree("trkdigi/events", {key: ("var * uint32" if key.startswith("d_") else "uint32") for key in cols}).extend(cols)
    assert dt.read(path)[1]["events"] == 3
