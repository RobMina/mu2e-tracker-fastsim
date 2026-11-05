"""Saved step-record deficits are separated exactly into distinct straws and repeat excess."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import fieldmultiplicity as fm


def fixture():
    p = pd.DataFrame({"run": np.full(3, 1430, dtype=np.uint64), "subrun": np.zeros(3, dtype=np.uint64),
                      "event": np.array([2**54 + 1, 2**54 + 3, 2**54 + 5], dtype=np.uint64), "sp": [1, 1, 1]})
    s = p.iloc[[0, 0, 0, 1]].copy().assign(plane=0, panel=0, straw=[1, 1, 2, 3])
    return p, dict(g4_all=s, g4_clock=s.iloc[:3], fast=s.iloc[[0, 2, 3]])


def test_record_partition_distinct_straws_repeats_zero_particles_and_large_ids():
    p, arms = fixture()
    d = fm.assemble(p, arms)
    np.testing.assert_array_equal(d.event, p.event)
    assert d.g4_clock_records.tolist() == [3, 0, 0]
    assert d.g4_clock_straws.tolist() == [2, 0, 0]
    r = fm.metrics(d)
    assert r["tracks"] == 3 and r["clock_removed_g4_records"] == 1
    delta = r["fast_minus_g4_clock"]
    assert delta["records"]["total_difference"] == 0
    assert delta["straws"]["total_difference"] == 1
    assert delta["repeat_excess"]["total_difference"] == -1
    assert delta["records"]["paired_mean_error_n"][2] == 3
    assert r["arms"]["fast"]["tracks_with_records"] == 2


def test_reject_foreign_particles_duplicate_denominator_and_bad_straw_keys():
    p, arms = fixture()
    with pytest.raises(ValueError, match="outside entry"):
        fm.assemble(p, {**arms, "fast": arms["fast"].assign(sp=2)})
    with pytest.raises(ValueError, match="duplicate entry"):
        fm.assemble(pd.concat([p, p.iloc[:1]]), arms)
    with pytest.raises(ValueError, match="straw identity"):
        fm.assemble(p, {**arms, "fast": arms["fast"].assign(straw=-1)})
    with pytest.raises(ValueError, match="all three"):
        fm.assemble(p, {"fast": arms["fast"]})
    with pytest.raises(ValueError, match="particle identities"):
        fm.assemble(p.assign(sp=1.0), arms)


def test_summary_keeps_pilot_separate_and_rejects_overlap():
    p, arms = fixture()
    d = fm.assemble(p, arms)
    q = d.assign(subrun=np.uint64(100))
    r = fm.summarize({"pilot": d, "confirm": q}, "pilot")
    assert r["confirmation_only"]["tracks"] == 3 and r["including_pilot"]["tracks"] == 6
    assert r["per_parent"]["pilot"]["reference_clock_step_particles"]["tracks"] == 1
    with pytest.raises(ValueError, match="overlapping"):
        fm.summarize({"pilot": d, "confirm": d}, "pilot")
    assert fm.summarize({"pilot": d}, "pilot")["confirmation_only"] is None


def test_empty_steps_have_zero_counts_and_loader_checks_all_entry_tables(tmp_path):
    from mu2efastsim import tables

    p, arms = fixture()
    arms["fast"] = arms["fast"].iloc[:0]
    d = fm.assemble(p, arms)
    assert d.fast_records.sum() == 0
    for label, name in (("g4_all", "g4prim_field.root"), ("g4_clock", "g4fp_field.root"), ("fast", "fasttrk_field.root")):
        tables.write(str(tmp_path / name), {"particles": p.rename(columns={"sp": "key"}),
                                           "sgsteps": fixture()[1][label]}, as_ttree=True)
    pd.testing.assert_frame_equal(fm.load(tmp_path), fm.assemble(*fixture()))
    tables.write(str(tmp_path / "fasttrk_field.root"), {"particles": p.iloc[:1].rename(columns={"sp": "key"}),
                                                       "sgsteps": fixture()[1]["fast"]}, as_ttree=True)
    with pytest.raises(AssertionError):
        fm.load(tmp_path)


def test_count_partition_is_exhaustive_including_gains_and_losses():
    d = fm.assemble(*fixture())
    r = fm.metrics(d)
    assert r["track_presence"] == dict(both=1, g4_only=0, fast_only=1, neither=1)
    for metric in ("records", "straws", "repeat_excess"):
        assert sum(v[metric] for v in r["count_partition"].values()) == r["fast_minus_g4_clock"][metric]["total_difference"]
