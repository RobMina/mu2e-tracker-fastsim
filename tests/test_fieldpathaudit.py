"""Actual-chord observation is inert; saved replay attribution requires exact payloads."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import tables
from trkfastsim import geometry
from trkfastsim.sim import fieldsteps as fs
from trkfastsim.validation import fieldcrossing as fc, fieldpathaudit as audit


def geom():
    n = 8
    return geometry.StrawGeometry(
        plane=np.zeros(n, dtype=int), panel=np.repeat([0, 1], 4), straw=np.tile(np.arange(4), 2),
        mid=np.c_[np.zeros(n), np.tile([-6., -2., 2., 6.], 2), np.repeat([0., 8.], 4)],
        dir=np.tile([1., 0., 0.], (n, 1)), halflen=np.full(n, 30.), radius=2.5,
        panel_R=np.tile(np.eye(3), (2, 1, 1)), panel_t=np.zeros((2, 3)),
        plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)), wall_rmax=2.515,
        metal_t=0.0001, wire_r=0.0125, wire_plate_t=0.0002)


def rays(n=3):
    x = np.tile([0., 1., -4.], (n, 1))
    c = np.tile([0., 0., 1.], (n, 1))
    return x, c, np.full(n, 5.)


def test_observer_returns_identical_raw_object_unclipped_and_mutates_nothing(monkeypatch):
    original = fs.ChordIndex
    x, c, length = rays()
    geometry_ = geom()
    raw = original(geometry_).crossings(x, c, length, geometry_)
    assert (raw[:, 3] > length[raw[:, 0].astype(int)]).any()
    monkeypatch.setattr(original, "crossings", lambda self, *args: raw)
    copies = [v.copy() for v in (x, c, length, raw, geometry_.mid, geometry_.dir)]
    obs = audit.CrossingObserver()
    with audit.observing_crossings(obs):
        index = fs.ChordIndex(geometry_)
        assert isinstance(index, original) and isinstance(index, fs.ChordIndex)
        assert index.crossings(x, c, length, geometry_) is raw
    assert fs.ChordIndex is original
    for before, after in zip(copies, (x, c, length, raw, geometry_.mid, geometry_.dir)):
        np.testing.assert_array_equal(before, after)
    assert obs.report()["total_mismatches"] == 0
    assert obs.stats["lanes"] == 3


def test_context_restores_class_on_transport_and_observer_exceptions():
    original = fs.ChordIndex
    with pytest.raises(RuntimeError, match="transport"):
        with audit.observing_crossings(audit.CrossingObserver()):
            raise RuntimeError("transport")
    assert fs.ChordIndex is original

    class Broken:
        def observe(self, *args):
            raise RuntimeError("observer")

    with pytest.raises(RuntimeError, match="observer"):
        with audit.observing_crossings(Broken()):
            fs.ChordIndex(geom()).crossings(*rays(), geom())
    assert fs.ChordIndex is original


def test_full_sets_detect_omission_not_just_hit_flags_and_examples_bounded(monkeypatch):
    original = fs.ChordIndex
    real = original.crossings

    def omit(self, x, c, length, geometry_):
        rows = real(self, x, c, length, geometry_)
        return rows[rows[:, 1] < 4]

    monkeypatch.setattr(original, "crossings", omit)
    x, c, length = rays(30)
    length[:] = 20.
    obs = audit.CrossingObserver(oracle_chords=31)
    with audit.observing_crossings(obs):
        index = fs.ChordIndex(geom())
        for _ in range(2):
            rows = index.crossings(x, c, length, geom())
            assert np.unique(rows[:, 0]).size == 30
    result = obs.report()
    assert result["comparisons"]["A_B"]["kinds"] == {"missing_A": 60}
    assert result["comparisons"]["A_C"]["kinds"] == {"missing_A": 31}
    assert result["comparisons"]["B_C"]["mismatches"] == 0
    assert result["oracle_chords"] == 31 and result["oracle_unchecked_chords"] == 29
    assert len(result["examples"]) == 20
    example = result["examples"][0]
    assert example["chord_call"] == 0 and example["lane"] == 0
    assert "unavailable" in example["identity_scope"] and "event" not in example


def test_oracle_prefix_covers_zero_hit_lanes_and_maximum(monkeypatch):
    calls = []
    real = fc.oracle_crossings

    def record(x, c, length, geometry_):
        calls.append(x.copy())
        return real(x, c, length, geometry_)

    monkeypatch.setattr(fc, "oracle_crossings", record)
    obs = audit.CrossingObserver()
    x, c, length = rays(200)
    x[:, 0] = np.arange(200) + 100.
    with audit.observing_crossings(obs):
        index = fs.ChordIndex(geom())
        for _ in range(3):
            index.crossings(x, c, length, geom())
    assert [len(v) for v in calls] == [200, 56]
    np.testing.assert_array_equal(calls[1], x[:56])
    assert obs.stats["lanes"] == obs.stats["zero_hit_lanes"] == 600
    assert obs.stats["oracle_chords"] == 256
    for value in (-1, 257, True):
        with pytest.raises(ValueError):
            audit.CrossingObserver(value)


def test_seeded_stochastic_transport_and_rng_state_exactly_unchanged():
    geometry_ = geom()
    start = pd.DataFrame(dict(run=1, subrun=0, event=np.arange(8, dtype=np.uint64) + np.uint64(2**54),
                              sp=1, pdg=13, x=np.linspace(-5., 5., 8), y=np.linspace(-5., 5., 8), z=-8.,
                              px=1., py=2., pz=150., t=0.))
    saved = start.copy(deep=True)
    opts = dict(chord_mm=3., envelope=fs.Envelope(geometry_, pad_r=2., pad_z=2., axis_xy=(0., 0.)),
                origin=np.zeros(3), gas=fs.STRAW_GAS_G4STATE, radiative=True)
    rng0, rng1 = np.random.default_rng(491), np.random.default_rng(491)
    baseline = fs.transport(rng0, start, geometry_, np.array([0., 0., 1.]), **opts)
    observer = audit.CrossingObserver(13)
    with audit.observing_crossings(observer):
        observed = fs.transport(rng1, start, geometry_, np.array([0., 0., 1.]), **opts)
    for a, b in zip(baseline, observed):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(start, saved, check_exact=True)
    assert rng0.bit_generator.state == rng1.bit_generator.state
    assert len(baseline[0]) > 0 and observer.report()["total_mismatches"] == 0


def payload():
    ids = np.array([2**54 + 1, 2**54 + 3], dtype=np.uint64)
    return dict(particles=pd.DataFrame(dict(run=1, subrun=0, event=ids, key=ids, pdg=13)),
                sgsteps=pd.DataFrame(dict(run=1, subrun=0, event=ids, sp=ids, len=[1., 2.])))


def write_arms(directory):
    directory.mkdir()
    for filename in fc.FILES.values():
        tables.write(str(directory / filename), payload(), as_ttree=True)


@pytest.mark.parametrize("tree,column,value", [("particles", "key", 2**54 + 2),
                                                ("sgsteps", "len", np.nextafter(1., 2.)),
                                                ("sgsteps", "event", 2**54 + 2)])
def test_exact_saved_comparison_detects_numeric_change_including_large_keys(tmp_path, tree, column, value):
    old, new = tmp_path / "old", tmp_path / "new"
    write_arms(old)
    write_arms(new)
    assert all(c["equal"] for c in audit.compare_saved_tables(old, new).values())
    changed = payload()
    changed[tree].loc[0, column] = value
    tables.write(str(new / fc.FILES["fast"]), changed, as_ttree=True)
    checks = audit.compare_saved_tables(old, new)
    assert not checks[f"fast/{tree}"]["equal"] and sum(c["equal"] for c in checks.values()) == 5


def test_missing_table_or_different_row_order_cannot_reproduce(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    write_arms(old)
    write_arms(new)
    changed = payload()
    changed["sgsteps"] = changed["sgsteps"].iloc[::-1].reset_index(drop=True)
    tables.write(str(new / fc.FILES["fast"]), changed, as_ttree=True)
    assert not audit.compare_saved_tables(old, new)["fast/sgsteps"]["equal"]
    tables.write(str(new / fc.FILES["fast"]), {"particles": payload()["particles"]}, as_ttree=True)
    checks = audit.compare_saved_tables(old, new)
    assert checks["fast/particles"]["equal"] and not checks["fast/sgsteps"]["equal"]


def settings():
    return dict(species=13, inputs=1, seed=19, chord_mm=20., sagitta_mm=.02,
                gas_state="gas", mu_radiative=True, origin_mm=[-4., 2., 17.],
                entry_vds=[11, 12, 13], field="map.header")


@pytest.mark.parametrize("change", [{"species": 11}, {"inputs": 2}, {"seed": None},
                                     {"gas_state": "unknown"}, {"origin_mm": [1, 2]}, {"entry": "vd"}])
def test_frozen_settings_fail_closed(change):
    with pytest.raises((ValueError, TypeError)):
        audit.frozen_settings(settings() | change, "first")
    missing = settings()
    del missing["mu_radiative"]
    with pytest.raises(ValueError, match="missing"):
        audit.frozen_settings(missing, "first")


def replay_fixture(tmp_path, monkeypatch, mode="equal"):
    old = tmp_path / "original"
    write_arms(old)
    (old / "fieldcompare.json").write_text(json.dumps(settings()))
    gdml, map_, input_ = [tmp_path / name for name in ("world.gdml", "map.header", "input.root")]
    for path in (gdml, map_, input_, map_.with_suffix(".bin")):
        path.write_bytes(b"fixture")
    received = []

    def replay(inputs, output, **kw):
        received.append((inputs, kw))
        assert fs.ChordIndex.__name__ == "ObservedChordIndex"
        fs.ChordIndex(geom()).crossings(*rays(), geom())
        write_arms(Path(output))
        if mode == "mismatch":
            changed = payload()
            changed["sgsteps"].loc[0, "len"] = 9.
            tables.write(str(Path(output) / fc.FILES["fast"]), changed, as_ttree=True)
        elif mode == "mutate":
            input_.write_bytes(b"changed")
        elif mode == "error":
            raise RuntimeError("replay failure")

    monkeypatch.setattr(audit.fieldcompare, "run", replay)
    return old, input_, gdml, map_, received


@pytest.mark.parametrize("mode", ["equal", "mismatch", "mutate", "error"])
def test_replay_hashes_frozen_arguments_exact_tables_and_failure_retention(tmp_path, monkeypatch, mode):
    old, input_, gdml, map_, received = replay_fixture(tmp_path, monkeypatch, mode)
    original = fs.ChordIndex
    out = tmp_path / "audit"
    result = audit.run(old, input_, gdml, map_, out, entry="first", oracle_chords=2)
    assert fs.ChordIndex is original
    assert result["passed"] == (mode == "equal")
    assert (out / "FAILED").exists() == (mode != "equal")
    assert (out / "replay" / fc.FILES["fast"]).is_file()
    assert json.loads((out / "fieldpathaudit.json").read_text())["passed"] == result["passed"]
    assert result["inputs_unchanged"] == (mode != "mutate")
    assert "map_adjacent_binary" in result["hashes_before"]
    assert result["explicit_original_entry_cli"] == "first"
    assert "not performed" in result["final_state_comparison"]
    assert received == [([str(input_)], dict(gdml=str(gdml), field_map=str(map_),
                                            **audit.frozen_settings(settings(), "first")))]
    if mode == "mismatch":
        assert not result["reproduction"]["reproduced"] and result["comparisons_diagnostic_only"]
    with pytest.raises(FileExistsError):
        audit.run(old, input_, gdml, map_, out, entry="first")


def test_cli_requires_explicit_original_entry(tmp_path):
    with pytest.raises(SystemExit) as error:
        audit.main(["--field-dir", "old", "--input-table", "in", "--gdml", "g", "--map", "m", "--outdir", "out"])
    assert error.value.code == 2
