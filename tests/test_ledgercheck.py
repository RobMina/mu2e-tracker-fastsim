"""Small synthetic tests of file-local writer diagnostic counts and identity pairing."""
import json

import numpy as np
import pandas as pd
import pytest
import uproot

from trkfastsim.analyses import ledgercheck as lc


def particles():
    return pd.DataFrame({"run": [1, 1, 1], "subrun": [2, 2, 2], "event": [10, 10, 30], "key": [1, 2, 1]})


def steps(n1=12, n2=2):
    sp = np.repeat([1, 2], [n1, n2])
    count = len(sp)
    frame = pd.DataFrame({"run": np.ones(count, dtype=int), "subrun": np.full(count, 2),
                          "event": np.full(count, 10), "sp": sp, "sid": np.arange(count),
                          "px": np.full(count, 100.), "py": np.zeros(count), "pz": np.zeros(count),
                          "eion": np.full(count, .001)})
    frame[lc.GEOMETRY] = np.zeros((count, len(lc.GEOMETRY)))
    return frame


def ledger(frame, emitted=None):
    result = frame[lc.IDENTITY + ["sid"]].copy()
    result["path_mm"] = np.arange(len(frame), dtype=float)
    result["support_loss_MeV"] = 0.
    result["p_start_MeV_c"] = 100.
    result["energy_defect_MeV"] = np.arange(len(frame), dtype=float)
    result["emitted"] = True if emitted is None else emitted
    return result


def test_counts_and_pairing_do_not_conflate_muons():
    before = lc.muon_counts(steps(), particles())
    after = lc.muon_counts(steps(2, 12), particles())
    assert before.loc[(1, 2, 10, 1), "selected"]
    assert not before.loc[(1, 2, 10, 2), "selected"]
    assert before.loc[(1, 2, 30, 1), "crossings"] == 0
    summary = lc.summarize_counts(before)
    assert summary == {"muons": 3, "crossings": 14, "good_steps": 14, "crossing_muons": 2,
                       "selected_muons": 1, "crossing_events": 1, "selected_events": 1}
    paired = lc.pair_counts(before, after)
    result = lc.summarize_pair(paired)
    assert result["selected_muons_lost"] == result["selected_muons_gained"] == 1
    assert result["selected_events_lost"] == result["selected_events_gained"] == 0
    assert result["selected_events_both"] == 1
    assert paired.loc[(1, 2, 10, 1), "crossings_delta"] == -10


def test_selection_cannot_pool_steps_across_muons():
    counts = lc.muon_counts(steps(6, 6), particles())
    assert lc.summarize_counts(counts)["selected_events"] == 0


def test_empty_summaries_and_quantiles():
    empty = steps(0, 0)
    counts = lc.muon_counts(empty, particles())
    assert lc.summarize_counts(counts)["muons"] == 3
    assert lc.summarize_counts(counts)["crossing_events"] == 0
    no_muons = lc.muon_counts(empty, particles().iloc[:0])
    assert all(value == 0 for value in lc.summarize_counts(no_muons).values())
    assert all(value == 0 for value in lc.summarize_pair(lc.pair_counts(no_muons, no_muons)).values())
    defects = lc.defect_quantiles(ledger(empty))
    assert defects["all_entries"]["quantiles_MeV"]["0.5"] is None
    assert lc.compare_crossings(empty, empty, ledger(empty), ledger(empty))["consistent_with_writer_only"]
    json.dumps(defects, allow_nan=False)


def test_pairing_outer_join():
    before = lc.muon_counts(steps(12, 0), particles().iloc[[0]])
    after = lc.muon_counts(steps(0, 12), particles().iloc[[1]])
    result = lc.summarize_pair(lc.pair_counts(before, after))
    assert result["crossed_muons_lost"] == result["crossed_muons_gained"] == 1
    assert result["selected_events_both"] == 1


def test_defect_quantiles_are_cumulative_entries_not_sums():
    entries = ledger(steps(3, 0), [True, True, False])
    entries["energy_defect_MeV"] = [1., 3., 100.]
    result = lc.defect_quantiles(entries)
    assert result["all_entries"]["quantiles_MeV"]["0.5"] == 3.
    assert result["emitted_entries"]["quantiles_MeV"]["0.5"] == 2.
    assert result["removed_entries"] == 1


def test_frozen_geometry_and_stopped_row_removal():
    before = steps(3, 0)
    before["sid"] = 1
    entries = ledger(before)
    stopped = entries.copy()
    stopped.loc[2, "emitted"] = False
    result = lc.compare_crossings(before, before.iloc[:2], entries, stopped)
    assert result["consistent_with_writer_only"]
    assert not result["expect_all_emitted_geometry_equal"]
    assert result["emitted_geometry"]["lost"] == 1
    changed = before.copy()
    changed.loc[0, "x0"] = .1
    assert not lc.compare_crossings(before, changed, entries, entries)["consistent_with_writer_only"]
    assert not lc.compare_crossings(before, before.iloc[:2], entries, entries)["consistent_with_writer_only"]
    changed_entries = entries.copy()
    changed_entries.loc[0, "path_mm"] += .1
    assert not lc.compare_crossings(before, before, entries, changed_entries)["consistent_with_writer_only"]


def write_gen(path):
    with uproot.recreate(path) as root:
        root.mktree("dump/gen", {"run": "int64", "subrun": "int64", "event": "int64"})
        root["dump/gen"].extend({"run": np.array([1]), "subrun": np.array([2]), "event": np.array([999])})
        root.mktree("dump/gen", {"run": "int64", "subrun": "int64", "event": "int64"})
        root["dump/gen"].extend({"run": np.array([1, 1, 1]), "subrun": np.array([2, 2, 2]),
                                 "event": np.array([10, 10, 30])})


def test_read_gen_uses_highest_cycle_and_actual_event_keys(tmp_path):
    path = tmp_path / "gen.root"
    write_gen(path)
    frame, info = lc.read_gen(path)
    assert info == {"object": "dump/gen;2", "cycle": 2, "rows": 3, "events": 2}
    assert frame.event.tolist() == [10, 10, 30]


def test_runner_equal_seeds_settings_manifest_and_refuse_overwrite(tmp_path, monkeypatch):
    path = tmp_path / "gen.root"
    write_gen(path)
    gdml = tmp_path / "geometry.gdml"
    gdml.write_text("synthetic fixture")
    monkeypatch.setattr(lc.geometry, "from_gdml", lambda path: object())
    monkeypatch.setattr(lc.st, "load_world", lambda path, geom: object())
    monkeypatch.setattr(lc.st, "TrackerInterior", lambda path, geom: object())
    draws = []

    def make_steps(gen, geom, rng, **kwargs):
        draws.append(rng.random())
        assert kwargs["stage"] == "C" and kwargs["sliced_msc"]
        assert kwargs["capture_entry_ledger"] and kwargs["all_muons"]
        assert kwargs["exit_scatter"] is False
        assert kwargs["sampler"] is kwargs["offaxis"] is None
        frame = steps()
        make_steps.entry_ledger = ledger(frame)
        return frame, particles()

    monkeypatch.setattr(lc.st, "make_steps", make_steps)
    out = tmp_path / "output"
    kwargs = dict(inputs=[str(path)], gdml=str(gdml), outdir=str(out), seed=17,
                  tracker_ref="pinned-tracker", core_ref="pinned-core")
    result = lc.run(**kwargs)
    assert draws[0] == draws[1] == draws[2]
    assert result["complete"] and result["file_local_gen_events"] == 2
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["tracker_ref"] == "pinned-tracker"
    assert manifest["sources_sha256"]["tracker"]["analyses/ledgercheck.py"] == lc.sha256(lc.__file__)
    report = json.loads((out / "files.jsonl").read_text())
    assert report["gen"]["cycle"] == 2
    assert report["arms"]["energy"]["crossing_comparison"]["consistent_with_writer_only"]
    paired = pd.read_csv(out / "muons.csv")
    assert len(paired) == 9
    assert set(paired["mode"]) == set(lc.entryledger.MODES)
    assert not list(out.glob("*.root"))
    with pytest.raises(FileExistsError):
        lc.run(**kwargs)
