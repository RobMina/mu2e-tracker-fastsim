"""Synthetic paired CE core/tail and missing-event tests."""
import numpy as np
import pandas as pd

from trkfastsim.validation import cereco


def test_paired_core_identical_arm_has_zero_difference_and_uncertainty():
    a = np.linspace(-0.5, 0.5, 101)
    result = cereco.paired_core(a, a)
    np.testing.assert_allclose(result["difference"], [0, 0])
    np.testing.assert_allclose(result["paired_error"], [0, 0])
    np.testing.assert_allclose(result["reference"], cereco.core(a))


def test_paired_core_translation_retains_width_and_known_median_difference():
    a = np.linspace(-0.5, 0.5, 101)
    result = cereco.paired_core(a + 0.2, a)
    np.testing.assert_allclose(result["difference"], [0.2, 0], atol=1e-14)
    np.testing.assert_allclose(result["paired_error"], [0, 0], atol=1e-14)


def test_compare_keeps_missing_reco_events_in_yield_denominator():
    n = 10
    d = pd.DataFrame({"fast_track": [True] * 8 + [False, False], "g4_track": [True] * 7 + [False] * 3,
                      "fast_surface": [True] * 8 + [False, False], "g4_surface": [True] * 7 + [False] * 3,
                      "fast_p_reco": np.full(n, 104.1), "g4_p_reco": np.full(n, 104.0),
                      "fast_p_truth": np.full(n, 104.0), "g4_p_truth": np.full(n, 104.0),
                      "fast_nactive": np.full(n, 35), "g4_nactive": np.full(n, 34)})
    r = cereco.compare(d, "fast", "g4")
    assert r["yield"]["gained"] == 1 and r["yield"]["lost"] == 0
    assert r["yield"]["paired_difference"][2] == n
    assert r["at_front"]["paired"] == 7
    np.testing.assert_allclose(r["active_hits_both_tracks"]["paired_difference"][:2], [1, 0])


def test_compare_reports_unique_active_straws_separately_from_hit_count():
    n = 8
    d = pd.DataFrame({f"{arm}_{key}": np.full(n, value) for arm in ("fast", "g4")
                      for key, value in (("track", True), ("surface", True), ("p_reco", 104.0),
                                         ("p_truth", 104.0), ("nactive", 35), ("nstraw", 30))})
    d["fast_nstraw"] = 31
    r = cereco.compare(d, "fast", "g4")
    np.testing.assert_allclose(r["unique_active_straws_both_tracks"]["paired_difference"][:2], [1, 0])
    np.testing.assert_allclose(r["active_hits_both_tracks"]["paired_difference"][:2], [0, 0])


def test_reader_counts_highest_cycle_rows_missing_tracks_and_unique_straws(monkeypatch):
    import awkward as ak
    from types import SimpleNamespace

    front = {"sid": 0, "inbounds": True, "time": 10.0,
             "mom": {"fCoordinates": {"fX": 100.0, "fY": 0.0, "fZ": 30.0}}}
    side = {**front, "sid": 4}
    hit = {"plane": 0, "panel": 1, "straw": 2, "state": 1}
    arrays = {"evtinfo/run": ak.Array([1430, 1430, 1430]), "evtinfo/subrun": ak.Array([0, 0, 0]),
              "evtinfo/event": ak.Array([2, 3, 6]), "trk/trk.status": ak.Array([[], [1], [1]]),
              "trk/trk.nactive": ak.Array([[], [3], [1]])}
    branches = {"trksegs": ak.Array([[], [[front]], [[side]]]),
                "trkhits": ak.Array([[], [[hit, hit, {**hit, "straw": 3}]], [[hit]]])}

    class Tree:
        num_entries = 3

        def arrays(self, *args, **kwargs):
            return arrays

        def __getitem__(self, key):
            return SimpleNamespace(array=lambda: branches[key])

    class File:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, key):
            assert key == cereco.TREE
            return Tree()

    monkeypatch.setattr(cereco.uproot, "open", lambda p: File())
    d, entries = cereco.read("synthetic.root")
    assert entries == 3
    assert d.track.tolist() == [False, True, True]
    assert d.surface.tolist() == [False, True, False]
    assert np.isnan(d.p_reco.iloc[0]) and np.isnan(d.p_reco.iloc[2])
    assert d.p_reco.iloc[1] == np.hypot(100.0, 30.0)
    assert np.isnan(d.nstraw.iloc[0]) and d.nstraw.iloc[1] == 2 and d.nstraw.iloc[2] == 1


def test_assemble_retains_all_births_and_exact_large_ids_and_rejects_foreign_rows():
    import pytest

    den = pd.DataFrame({"run": np.full(3, 1430, dtype=np.uint64), "subrun": np.zeros(3, dtype=np.uint64),
                        "event": np.array([2**54 + 1, 2**54 + 3, 2**54 + 5], dtype=np.uint64)})
    entries = den.iloc[:2].astype(np.int64).assign(px=100.0, py=0.0, pz=30.0)
    f = den.iloc[:1].astype(np.int64).assign(track=True, surface=True, p_reco=104.0, nactive=35)
    frames = {"g4": f, "fast": f.iloc[:0]}
    d = cereco.assemble(den, entries, entries, frames, "g4")
    np.testing.assert_array_equal(d.event, den.event)
    assert len(d) == 3 and d.g4_track.fillna(False).sum() == 1
    r = cereco.compare(d, "fast", "g4")
    assert r["yield"]["lost"] == 1 and r["yield"]["paired_difference"][2] == 3
    with pytest.raises(ValueError, match="outside birth"):
        cereco.assemble(den, entries, entries, {"g4": f.assign(event=1)}, "g4")
    with pytest.raises(ValueError, match="outside birth"):
        cereco.assemble(den, entries.iloc[:1].assign(event=1), entries, frames, "g4")
    with pytest.raises(ValueError, match="duplicate"):
        cereco.assemble(pd.concat([den, den.iloc[:1]]), entries, entries, frames, "g4")


def test_cli_checks_scope_before_reading_ntuple_and_preserves_selected_diagnostic(tmp_path, monkeypatch):
    import json
    import pytest
    from mu2efastsim import tables

    den = pd.DataFrame({"run": [1, 1], "subrun": [0, 0], "event": [1, 2]})
    entries = den.iloc[:1].assign(px=100.0, py=0.0, pz=30.0)
    tables.write(str(tmp_path / "states.root"), {"gen": den, "entries": entries, "g4entries": entries}, as_ttree=True)
    frames = entries[cereco.KEY].assign(track=True, surface=True, p_reco=104.0, nactive=35)
    calls = []

    def read(path):
        calls.append(path)
        return frames.copy(), len(frames)

    monkeypatch.setattr(cereco, "read", read)
    args = ["g4=g4.root", "fast=fast.root", "--states", str(tmp_path / "states.root"),
            "--outdir", str(tmp_path / "out")]
    with pytest.raises(ValueError, match="complete-birth efficiency"):
        cereco.main(args + ["--claim", "complete-birth"])
    assert not calls and not (tmp_path / "out").exists()
    cereco.main(args)
    r = json.loads((tmp_path / "out/cereco.json").read_text())
    assert r["events"] == 2 and r["conditional_on_dts_selection"]
    assert r["compare"]["fast"]["yield"]["paired_difference"][2] == 2
    receipt = dict(mode="chain", generated_starts=2, reference_selection="unfiltered", expected_starts=2,
                   conditional_on_dts_selection=False)
    (tmp_path / "handoff.json").write_text(json.dumps(receipt))
    cereco.main(args + ["--claim", "complete-birth"])
    r = json.loads((tmp_path / "out/cereco.json").read_text())
    assert r["efficiency_scope"]["complete_birth_eligible"] and not r["conditional_on_dts_selection"]


def test_saved_physics_comparison_allows_scope_metadata_not_numerical_changes():
    import copy
    import pytest

    r = dict(events=2, ntuple_entries={"g4": 1, "fast": 1}, truth="own entry",
             compare={"fast": {"metric": [0.0, float("nan"), 2]}})
    assert cereco.unchanged_physics({**r, "efficiency_scope": {}}, copy.deepcopy(r))["unchanged"]
    changed = copy.deepcopy(r)
    changed["compare"]["fast"]["metric"][0] = 1e-15
    with pytest.raises(ValueError, match="physics changed: compare"):
        cereco.unchanged_physics(changed, r)
