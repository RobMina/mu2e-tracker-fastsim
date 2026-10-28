"""Exact generator-content audits preserve full particle/event identity."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trkfastsim import tables as T
from mu2efastsim.tables import write

spec = importlib.util.spec_from_file_location("tv3_provenance", Path(__file__).parents[1] / "scripts/secondary_tv3_provenance.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def gen():
    return pd.DataFrame({"run": [1, 1, 1, 2], "subrun": 3, "event": [1, 1, 2, 1],
                         "idx": [0, 1, 0, 0], "pdg": [13, 11, -13, 13], "genid": 2,
                         **{c: np.array([1., 2., 3., 4.]) for c in ("x", "y", "z", "px", "py", "pz", "e", "t")}})


def test_exact_subset_is_keyed_and_includes_nonmuons():
    f = gen()
    r = f.iloc[[3, 1, 0]]
    got = A.audit_group(f, f.iloc[::-1], f.sample(frac=1, random_state=4), r)
    assert got["all_event_count"] == 3 and got["generator_particles"] == 4
    assert got["g4_kept_events"] == 2 and got["g4_kept_generator_particles"] == 3
    assert got["fast_merged_original_exact"] and got["g4_retained_all_particles_exact"]
    assert got["all_event_payload_sha256"] == A.payload_hash(A.canonical(f.iloc[::-1]))


@pytest.mark.parametrize("column", ["t", "x", "px", "e"])
def test_one_ulp_payload_change_rejected(column):
    f = gen()
    r = f.iloc[:2].copy()
    r.loc[1, column] = np.nextafter(r.loc[1, column], np.inf)
    with pytest.raises(ValueError, match="payload differs"):
        A.audit_group(f, f, f, r)


def test_missing_nonmuon_and_aliased_event_are_rejected():
    f = gen()
    with pytest.raises(ValueError, match="row count mismatch"):
        A.audit_group(f, f, f, f.iloc[[0]])
    r = f.iloc[[3]].copy()
    r["run"] = 1
    with pytest.raises(ValueError, match="row count mismatch"):
        A.audit_group(f, f, f, r)


@pytest.mark.parametrize("mode", ["duplicate", "gap", "nonfinite", "fractional", "schema"])
def test_malformed_generator_refused(mode):
    f = gen()
    if mode == "duplicate":
        f = pd.concat([f, f.iloc[:1]])
    elif mode == "gap":
        f.loc[1, "idx"] = 2
    elif mode == "nonfinite":
        f.loc[0, "x"] = np.nan
    elif mode == "fractional":
        f["event"] = f.event.astype(float) + .5
    else:
        f = f.drop(columns="t")
    with pytest.raises(ValueError):
        A.canonical(f)


def test_whole_input_and_expected_event_count_checked():
    f = gen()
    other = f.copy()
    other.loc[2, "e"] += 1
    with pytest.raises(ValueError, match="fast vs original"):
        A.audit_group(f, f, other, f.iloc[:2])
    with pytest.raises(ValueError, match="events per subrun"):
        A.audit_group(f, f, f, f.iloc[:2], expected_events=2)


def test_manifest_roundtrip_and_exclusive_output(tmp_path):
    f = gen().iloc[:3]
    sub = pd.DataFrame({"run": [1], "subrun": [3], "livetime": [1.], "primaries": [100]})
    fast = tmp_path / "fasttrkCL.grp.000001_00000003.root"
    merged = tmp_path / "merged" / "gen.grp.000001_00000003.root"
    original = tmp_path / "gen" / "gen.000001_00000003.root"
    ref = tmp_path / "g4.root"
    for path, rows in ((fast, f), (merged, f.iloc[::-1]), (original, f), (ref, f.iloc[:2])):
        write(str(path), {"gen": rows, "subruns": sub}, as_ttree=True)
    comparison = tmp_path / "compare.json"
    comparison.write_text(json.dumps({"paired": True, "input_files": {"steps_fast": [str(fast)], "steps_g4": [str(ref)]}}))
    out = tmp_path / "result"
    result = A.run(comparison, original.parent, merged.parent, out, expected_events=2)
    assert result["status"] == "passed" and result["full_manifest"]
    assert result["totals"]["all_event_count"] == 2
    with pytest.raises(FileExistsError):
        A.run(comparison, original.parent, merged.parent, out)
    changed = f.copy()
    changed.loc[0, "x"] += 1
    write(str(fast), {"gen": changed, "subruns": sub}, as_ttree=True)
    with pytest.raises(ValueError, match="fast vs merged"):
        A.run(comparison, original.parent, merged.parent, tmp_path / "failed", expected_events=2)
    fail = json.loads((tmp_path / "failed" / "provenance.json").read_text())
    assert fail["status"] == "failed" and not fail["groups"]
