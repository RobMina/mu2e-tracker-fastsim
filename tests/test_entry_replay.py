"""Typed entry replay, empty events and source particle identity."""
import numpy as np
import pandas as pd
import pytest
import uproot

from trkfastsim.validation import entry_replay as er


@pytest.fixture
def sample(tmp_path):
    er.smoke(tmp_path)
    return er.read(tmp_path / "entries.root")


def test_empty_and_multiple_events_and_wide_signed_keys(sample, tmp_path):
    ev, pt = sample
    np.testing.assert_array_equal(ev.n_particles, [0, 2, 1])
    np.testing.assert_array_equal(pt.source_sp, [-2147483647, 2**53 + 7, 9])
    er.write(tmp_path / "readback.root", ev, pt, "entryReplay")
    assert er.check(tmp_path / "entries.root", tmp_path / "readback.root") == dict(events=3, particles=3, exact=True)


def test_empty_particle_tree_written(sample, tmp_path):
    ev, pt = sample
    er.write(tmp_path / "empty.root", ev.assign(n_particles=np.uint32(0)), pt.iloc[:0])
    with uproot.open(tmp_path / "empty.root") as root:
        assert root["entry/particles"].classname == "TTree"
        assert root["entry/particles"].num_entries == 0
    out, empty = er.read(tmp_path / "empty.root")
    assert len(out) == 3 and len(empty) == 0


@pytest.mark.parametrize("mutation,match", [
    ("duplicate_event", "duplicate event"), ("duplicate_particle", "duplicate particle"),
    ("count", "counts differ"), ("foreign", "missing from event census"),
    ("index", "contiguous"), ("pdg", "muons only"),
    ("nan", "nonfinite"), ("momentum", "momentum"),
    ("float_id", "must contain integers"), ("negative_event", "outside"),
    ("reserved_event", "outside"), ("zero_event", "positive"),
])
def test_bad_tables_rejected(sample, mutation, match):
    ev, pt = (f.copy() for f in sample)
    if mutation == "duplicate_event":
        ev = pd.concat([ev, ev.iloc[:1]])
    elif mutation == "duplicate_particle":
        pt = pd.concat([pt, pt.iloc[:1]])
    elif mutation == "count":
        ev.loc[0, "n_particles"] = 1
    elif mutation == "foreign":
        pt.loc[2, "event"] = 7
    elif mutation == "index":
        pt.loc[2, "index"] = 1
    elif mutation == "pdg":
        pt.loc[0, "pdg"] = 11
    elif mutation == "nan":
        pt.loc[0, "x"] = np.nan
    elif mutation == "momentum":
        pt.loc[0, ["px", "py", "pz"]] = 0.
    elif mutation == "float_id":
        ev["event"] = ev.event.astype(float)
    elif mutation == "negative_event":
        ev["event"] = ev.event.astype("int64")
        ev.loc[0, "event"] = -1
    elif mutation == "reserved_event":
        ev.loc[0, "event"] = np.iinfo(np.uint32).max
    else:
        ev.loc[0, "event"] = 0
    with pytest.raises(ValueError, match=match):
        er.canonical(ev, pt)


def test_mutated_readback_rejected(sample, tmp_path):
    ev, pt = sample
    pt.loc[0, "t"] = np.nextafter(pt.loc[0, "t"], np.inf)
    er.write(tmp_path / "readback.root", ev, pt, "entryReplay")
    with pytest.raises(AssertionError):
        er.check(tmp_path / "entries.root", tmp_path / "readback.root")


def test_source_art_count_uses_highest_cycle(sample, tmp_path):
    ev, pt = sample
    er.write(tmp_path / "readback.root", ev, pt, "entryReplay")
    source = tmp_path / "source.root"
    with uproot.recreate(source) as root:
        root.mktree("Events", {"dummy": "int32"}).extend({"dummy": np.arange(1, dtype=np.int32)})
        root.mktree("Events", {"dummy": "int32"}).extend({"dummy": np.arange(3, dtype=np.int32)})
    result = er.check(tmp_path / "entries.root", tmp_path / "readback.root", source_art=source)
    assert result["source_art_entries"] == 3
    assert result["empty_events"] == 1
    assert result["multi_particle_events"] == 1
    with uproot.update(source) as root:
        root.mktree("Events", {"dummy": "int32"}).extend({"dummy": np.arange(4, dtype=np.int32)})
    with pytest.raises(ValueError, match="differs from replayed"):
        er.check(tmp_path / "entries.root", tmp_path / "readback.root", source_art=source)


def test_art_invalid_fixture_is_actually_invalid(tmp_path):
    er.smoke(tmp_path)
    with pytest.raises(ValueError, match="counts differ"):
        er.read(tmp_path / "bad-count.root")


def test_select_first_surface_or_vd_keeps_zero_response_particles():
    key = dict(run=[1, 1], subrun=[0, 0], event=[1, 1], sp=[-7, 8], pdg=[13, -13])
    vd = pd.DataFrame(dict(**key, vdid=[13, 13], x=[1., 2.], y=[0., 0.], z=[5., 5.],
                           px=[0., 0.], py=[0., 0.], pz=[1000., 2000.], t=[2., 3.]))
    surf = pd.DataFrame(dict(run=[1], subrun=[0], event=[1], sp=[-7], pdg=[13], surf=[4],
                             x0=[0.], y0=[0.], z0=[0.], x1=[1.], y1=[0.], z1=[0.],
                             px=[0.], py=[0.], pz=[1000.], t=[1.]))
    out = er.select_entries(vd, surf, [10., 20., 30.])
    assert len(out) == 2
    np.testing.assert_array_equal(out["t"], [1., 3.])
    np.testing.assert_array_equal(out["x"], [10., 2.])


def test_wrong_dtype_readback_rejected(sample, tmp_path):
    ev, pt = sample
    with uproot.recreate(tmp_path / "wrong.root") as root:
        root.mktree("entry/events", {**er.EVENT_TYPES, "event": "float64"}).extend(
            {key: ev[key].to_numpy() for key in er.EVENT_TYPES})
        root.mktree("entry/particles", er.PARTICLE_TYPES).extend(
            {key: pt[key].to_numpy() for key in er.PARTICLE_TYPES})
    with pytest.raises(ValueError, match="branch dtype"):
        er.read(tmp_path / "wrong.root")
