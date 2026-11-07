"""Synthetic adapter tests only: no GDML, real field or physics transport is run."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import uproot

from trkfastsim.validation import entry_fast as ef
from trkfastsim.validation import entry_replay as er


@pytest.fixture
def sample(tmp_path):
    er.smoke(tmp_path)
    ev, pt = er.read(tmp_path / "entries.root")
    final = pt[ef.KEY + er.STATE].copy()
    final["sp"] = pt["index"].to_numpy(dtype=np.int64) + 1
    final = final.assign(path=[10., 20., 30000.], chords=[1, 2, 1500], status=[0, 1, 2])
    final["z"] += 10.
    steps = pd.DataFrame({key: np.zeros(2, dtype=dtype) for key, dtype in ef.STEP_TYPES.items()})
    steps[ef.KEY] = np.array([[1, 0, 2], [1, 0, 2]])
    steps["sp"] = [2, 2]
    steps["len"] = [1., 2.]
    return ev, pt, steps, final


def test_zero_hit_states_and_exact_identity(sample, tmp_path):
    ev, pt, steps, final = sample
    frames = ef.injection_tables(ev, pt, steps, final.iloc[::-1])
    parts, mapping = frames["particles"], frames["entry_map"]
    assert len(parts) == 3
    np.testing.assert_array_equal(parts["key"], [1, 2, 1])
    np.testing.assert_array_equal(parts.genidx, [0, 1, 0])
    np.testing.assert_array_equal(mapping.source_sp, [-2147483647, 2**53 + 7, 9])
    np.testing.assert_array_equal(parts.sx, pt.x)
    np.testing.assert_array_equal(parts.ez, final.z)
    np.testing.assert_array_equal(parts.st, pt.t)
    np.testing.assert_array_equal(frames["events"].nsgs, [0, 2, 0])
    np.testing.assert_array_equal(frames["final"].status, [0, 1, 2])
    ef.write_tables(tmp_path / "fast.root", frames)
    with uproot.open(tmp_path / "fast.root") as root:
        assert root["dump/sgsteps"].classname == "TTree"
        assert root["dump/particles"].num_entries == 3
        np.testing.assert_array_equal(root["dump/entry_map/source_sp"].array(library="np"), pt.source_sp)
        assert set(["sx", "sy", "sz", "spx", "spy", "spz", "st", "ex", "ey", "ez", "epx", "epy", "epz", "et"]).issubset(root["dump/particles"].keys())


@pytest.mark.parametrize("all_empty", [False, True])
def test_empty_trees_have_full_injector_schema(sample, tmp_path, all_empty):
    ev, pt, steps, final = sample
    if all_empty:
        ev, pt, final = ev.assign(n_particles=np.uint32(0)), pt.iloc[:0], final.iloc[:0]
    frames = ef.injection_tables(ev, pt, steps.iloc[:0], final)
    ef.write_tables(tmp_path / "empty.root", frames)
    with uproot.open(tmp_path / "empty.root") as root:
        assert root["dump/events"].num_entries == 3
        for name in ("sgsteps", "particles", "entry_map", "final"):
            tree = root[f"dump/{name}"]
            assert tree.classname == "TTree"
            assert set(tree.keys()) == set(frames[name].columns)
        assert root["dump/sgsteps"].num_entries == 0
        assert root["dump/particles"].num_entries == (0 if all_empty else 3)


@pytest.mark.parametrize("bad", ["missing", "duplicate", "foreign", "status", "nan", "step"])
def test_bad_transport_output_rejected(sample, bad):
    ev, pt, steps, final = sample
    if bad == "missing":
        final = final.iloc[:2]
    elif bad == "duplicate":
        final = pd.concat([final.iloc[:2], final.iloc[:1]])
    elif bad == "foreign":
        final.loc[0, "sp"] = 99
    elif bad == "status":
        final.loc[0, "status"] = 3
    elif bad == "nan":
        final.loc[0, "x"] = np.nan
    else:
        steps.loc[0, "sp"] = 99
    with pytest.raises(ValueError):
        ef.injection_tables(ev, pt, steps, final)


def test_map_uses_existing_loader_and_rejects_outside(monkeypatch, tmp_path):
    header = tmp_path / "DSMap.header"
    header.write_text("synthetic")
    header.with_suffix(".bin").write_bytes(b"synthetic")
    lo, spacing, shape = np.zeros(3), np.ones(3), np.array([2, 2, 2])
    grid = np.ones((2, 2, 2, 3))
    monkeypatch.setattr(ef.ehelix, "read_g4bl", lambda path: (lo, spacing, shape, grid))
    field, info = ef.load_field(header)
    np.testing.assert_allclose(field(np.array([[.2, .3, .4]])), [[1., 1., 1.]])
    assert info["configuration"] == "bfgeom_no_tsu_ps_v01.txt"
    for point in ([1., .5, .5], [-.1, .5, .5], [np.nan, 0., 0.]):
        with pytest.raises(ValueError, match="outside"):
            field(np.array([point]))
    with pytest.raises(ValueError, match="stock"):
        ef.load_field(tmp_path / "patched.header")


def mock_run(monkeypatch, tmp_path, sample):
    ev, pt, steps, final = sample
    for name in ef.THREAD_VARS:
        monkeypatch.setenv(name, "1")
    gdml = tmp_path / "saved.gdml"
    gdml.write_text("synthetic: never parsed")
    envelope = SimpleNamespace(axis=np.array(ef.ORIGIN[:2]), r=1000., z0=8000., z1=12000.)
    monkeypatch.setattr(ef.geometry, "from_gdml", lambda path: "mock geometry")
    monkeypatch.setattr(ef.fs, "Envelope", lambda geom, axis_xy: envelope)
    monkeypatch.setattr(ef.fs, "interior", lambda *args, **kwargs: "mock interior")
    monkeypatch.setattr(ef, "load_field", lambda path: (np.array([0., 0., 1.]), {"mock": True}))

    def transport(rng, start, geom, field, support, **kwargs):
        np.testing.assert_array_equal(start.sp, [1, 2, 1])
        np.testing.assert_array_equal(start[er.STATE], pt[er.STATE])
        assert support == "mock interior"
        assert kwargs["complete_outer_wall"] is False
        assert set(kwargs) == {"envelope", "origin", "bnom", "complete_outer_wall"}
        return steps, final

    monkeypatch.setattr(ef.fs, "transport", transport)
    return dict(entries=tmp_path / "entries.root", gdml=gdml, field_map="unused",
                origin=ef.ORIGIN, outdir=tmp_path / "out", expected_events=3, expected_particles=3)


def test_run_keeps_full_census_and_records_payload_only(sample, monkeypatch, tmp_path):
    args = mock_run(monkeypatch, tmp_path, sample)
    clock = iter([100.0, 102.75])
    monkeypatch.setattr(ef.time, "perf_counter", lambda: next(clock))
    result = ef.run(**args)
    assert result["transport_s"] == 2.75
    assert result["source_events"] == result["entry_particles"] == 3
    assert result["empty_entry_events"] == result["multi_entry_events"] == 1
    assert result["zero_hit_primaries"] == 2
    assert result["final_status"] == dict(exited=1, stopped=1, capped=1)
    assert result["transport_s"] >= 0
    assert result["complete_outer_wall"] is False
    assert (tmp_path / "out/entry-fast.json").exists()


@pytest.mark.parametrize("bad", ["origin", "events", "particles", "threads", "metadata"])
def test_run_fail_closed(sample, monkeypatch, tmp_path, bad):
    args = mock_run(monkeypatch, tmp_path, sample)
    if bad == "origin":
        args["origin"] = [0., 0., 0.]
    elif bad in ("events", "particles"):
        args["expected_" + bad] = 9
    elif bad == "threads":
        monkeypatch.delenv("OMP_NUM_THREADS")
    else:
        metadata = tmp_path / "entries.json"
        metadata.write_text('{"origin_mm": [0, 0, 0], "envelope": {}}')
        args["metadata"] = metadata
    with pytest.raises(ValueError):
        ef.run(**args)


def test_cli_uses_map_and_full_census_defaults(monkeypatch, capsys):
    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return {"synthetic": True}

    monkeypatch.setattr(ef, "run", run)
    ef.main(["entries.root", "--gdml", "saved.gdml", "--map", "DSMap.header",
             "--origin", "-3904", "0", "10170.999666398546", "--metadata", "entries.json",
             "--outdir", "output"])
    assert captured["field_map"] == "DSMap.header"
    assert captured["expected_events"] == 52037
    assert captured["expected_particles"] == 43156
    assert captured["origin"] == list(ef.ORIGIN)
    assert "synthetic" in capsys.readouterr().out


def test_multi_primary_step_count_is_per_event(sample):
    ev, pt, steps, final = sample
    steps.loc[0, "sp"] = 1
    frames = ef.injection_tables(ev, pt, steps, final)
    np.testing.assert_array_equal(frames["sgsteps"].nsgs, [2, 2])
    np.testing.assert_array_equal(frames["sgsteps"].sp, [1, 2])
