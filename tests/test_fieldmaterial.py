"""Report-only material geometry, inert observation, and exact replay attribution gates."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import tables
from trkfastsim import geometry
from trkfastsim.sim import fieldsteps as fs
from trkfastsim.validation import fieldcrossing as fc, fieldmaterial as fm


def geom():
    return geometry.StrawGeometry(
        plane=np.array([0]), panel=np.array([0]), straw=np.array([0]),
        mid=np.zeros((1, 3)), dir=np.array([[0., 0., 1.]]), halflen=np.array([2.]), radius=1.,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)), plane_R=np.eye(3)[None],
        plane_t=np.zeros((1, 3)), wall_rmax=1.2)


def rays(start=(-2., 0., 0.), end=(2., 0., 0.)):
    return fc.endpoint_secants(np.array([start]), np.array([end]), np.zeros(3))[:3]


def observe(x, c, length, geometry_=None, observer=None):
    geometry_ = geom() if geometry_ is None else geometry_
    observer = fm.WallObserver() if observer is None else observer
    sp = fs._straw_pieces(x, c, length, geometry_, 1.2, 0., 0., fs.ChordIndex(geometry_))
    observer.observe(x, c, length, geometry_, 1.2, sp)
    return observer.report(), sp


def test_analytic_central_crossing_and_actual_consumption():
    g = geom()
    x, c, length = rays()
    wall = fm.geometric_wall(x, c, length, g, 1.2)
    np.testing.assert_allclose(wall[["outer_in", "outer_out", "gas_in", "gas_out", "wall_mm"]],
                               [[.8, 3.2, 1., 3., .4]], atol=1e-14, rtol=0)
    report, sp = observe(x, c, length, g)
    assert report["consistent"] and report["missing_rows"] == 0
    assert report["consumed_wall_mm"] == float(sp[4].sum())
    assert report["consumed_wall_mm"] == pytest.approx(.4)
    assert report["geometric_gas_mm"] == report["consumed_gas_mm"] == 2.
    assert report["oracle_chords"] == report["oracle_positive_lanes"] == report["oracle_rows"] == 1
    assert report["oracle_mismatches"] == 0


def test_radial_wall_only_graze_and_expanded_panel_bounds():
    x, c, length = rays((-2., 1.1, 0.), (2., 1.1, 0.))
    g = geom()
    assert fs.ChordIndex(g).crossings(x, c, length, g).shape == (0, 4)
    report, sp = observe(x, c, length, g)
    expected = 2 * np.sqrt(1.2 ** 2 - 1.1 ** 2)
    assert sp is None and report["consistent"]
    assert report["geometric_wall_mm"] == pytest.approx(expected)
    assert report["missing_wall_mm"] == pytest.approx(expected)
    assert report["gas_free_line_graze"] == dict(rows=1, lanes=1, wall_mm=pytest.approx(expected))
    assert report["finite_chord_boundary"]["rows"] == 0
    assert report["oracle_positive_lanes"] == 1
    assert report["missing_wall_is_diagnostic_not_failure"]


def test_segmentation_before_and_after_gas_preserves_geometric_wall_not_consumed():
    breaks = np.array([-2., -1.1, -.9, .9, 1.1, 2.])
    start, end = np.zeros((5, 3)), np.zeros((5, 3))
    start[:, 0], end[:, 0] = breaks[:-1], breaks[1:]
    x, c, length, _ = fc.endpoint_secants(start, end, np.zeros(3))
    segmented, _ = observe(x, c, length)
    whole, _ = observe(*rays())
    assert segmented["consistent"]
    assert segmented["geometric_wall_mm"] == pytest.approx(whole["geometric_wall_mm"])
    assert segmented["geometric_gas_mm"] == pytest.approx(whole["geometric_gas_mm"])
    assert segmented["consumed_wall_mm"] == pytest.approx(.2)
    assert segmented["missing_wall_mm"] == pytest.approx(.2)
    assert segmented["finite_chord_boundary"]["rows"] == 2
    assert segmented["gas_free_line_graze"]["rows"] == 0
    assert segmented["matched_rows"] == 3 and segmented["missing_rows"] == 2
    assert all(abs(v) < 1e-12 for v in segmented["partition_residuals_mm"].values())
    reverse, _ = observe(*fc.endpoint_secants(end, start, np.zeros(3))[:3])
    for key in ("geometric_wall_mm", "consumed_wall_mm", "missing_wall_mm", "consumed_gas_mm"):
        assert reverse[key] == pytest.approx(segmented[key])


def test_finite_cap_clips_outer_and_inner_not_infinite_cylinder():
    x, c, length = rays((-1.5, 0., .5), (.5, 0., 2.5))
    report, _ = observe(x, c, length)
    assert report["consistent"]
    assert report["geometric_wall_mm"] == pytest.approx(.2 * np.sqrt(2))
    assert report["geometric_gas_mm"] == pytest.approx(np.sqrt(2))
    assert report["geometric_outer_mm"] == pytest.approx(1.2 * np.sqrt(2))
    x, c, length = rays((-2., 0., 2.1), (2., 0., 2.1))
    assert fm.geometric_wall(x, c, length, geom(), 1.2).empty
    x, c, length = rays((1.1, 0., 1.8), (0., 0., 4.))
    report, _ = observe(x, c, length)
    assert report["gas_free_line_graze"]["rows"] == 1
    assert report["finite_chord_boundary"]["rows"] == 0


@pytest.mark.parametrize("radius,expected_outer,expected_gas", [(0., 4., 4.), (.5, 4., 4.), (1.1, 4., 0.), (1.3, 0., 0.)])
def test_parallel_inside_gas_wall_and_outside(radius, expected_outer, expected_gas):
    report, _ = observe(*rays((radius, 0., -3.), (radius, 0., 3.)))
    assert report["consistent"]
    assert report["geometric_outer_mm"] == expected_outer
    assert report["geometric_gas_mm"] == expected_gas
    assert report["geometric_wall_mm"] == expected_outer - expected_gas
    report, _ = observe(*rays((radius, 0., 3.), (radius, 0., 4.)))
    assert report["outer_rows"] == 0


def test_nohit_and_empty_batch_and_invalid_inputs():
    for args in (rays((10., 0., 0.), (11., 0., 0.)), (np.empty((0, 3)), np.empty((0, 3)), np.empty(0))):
        report, sp = observe(*args)
        assert sp is None and report["consistent"]
        assert report["outer_rows"] == report["oracle_chords"] == 0
    x, c, length = rays()
    with pytest.raises(ValueError, match="radius"):
        fm.geometric_wall(x, c, length, geom(), .5)
    with pytest.raises(ValueError, match="nonunit"):
        fm.geometric_wall(x, c * 2, length, geom(), 1.2)
    for n in (-1, 257, True):
        with pytest.raises(ValueError):
            fm.WallObserver(n)


def test_batched_analytic_intervals_and_cache_are_consistent():
    g = geom()
    x, c, length = rays()
    x, c, length = np.repeat(x, 17, axis=0), np.repeat(c, 17, axis=0), np.repeat(length, 17)
    a = fm.geometric_wall(x, c, length, g, 1.2, batch=1)
    b = fm.geometric_wall(x, c, length, g, 1.2, batch=100)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    observer = fm.WallObserver(0)
    observe(x, c, length, g, observer)
    observe(x, c, length, g, observer)
    assert len(observer._outer_cache) == 1
    assert observer.report()["consistent"]


def test_wrapper_exact_return_identity_original_once_no_mutation_or_subclass(monkeypatch):
    g = geom()
    x, c, length = rays()
    original, index_class = fs._straw_pieces, fs.ChordIndex
    index = index_class(g)
    sp = original(x, c, length, g, 1.2, 0., 0., index)
    calls = []

    def once(*args):
        calls.append(args)
        return sp

    monkeypatch.setattr(fs, "_straw_pieces", once)
    values = (x, c, length, g.mid, g.dir, g.halflen, *sp)
    before = [v.copy() for v in values]
    observer = fm.WallObserver()
    with fm.observing_material(observer):
        assert fs.ChordIndex is index_class
        assert fs._straw_pieces(x, c, length, g, 1.2, 0., 0., index) is sp
    assert fs._straw_pieces is once and len(calls) == 1
    for left, right in zip(before, values):
        np.testing.assert_array_equal(left, right)
    assert observer.report()["consistent"]


def test_wrapper_restoration_on_observer_original_and_body_exceptions(monkeypatch):
    original = fs._straw_pieces

    class Broken:
        def observe(self, *args):
            raise RuntimeError("observer")

    with pytest.raises(RuntimeError, match="observer"):
        with fm.observing_material(Broken()):
            fs._straw_pieces(*rays(), geom(), 1.2, 0., 0.)
    assert fs._straw_pieces is original
    with pytest.raises(RuntimeError, match="body"):
        with fm.observing_material(fm.WallObserver()):
            raise RuntimeError("body")
    assert fs._straw_pieces is original

    def broken(*args):
        raise RuntimeError("original")

    monkeypatch.setattr(fs, "_straw_pieces", broken)
    with pytest.raises(RuntimeError, match="original"):
        with fm.observing_material(fm.WallObserver()):
            fs._straw_pieces(*rays(), geom(), 1.2, 0., 0.)
    assert fs._straw_pieces is broken


def test_deliberate_wrong_consumed_wall_and_absolute_tolerance():
    g = geom()
    x, c, length = rays()
    sp = fs._straw_pieces(x, c, length, g, 1.2, 0., 0., fs.ChordIndex(g))
    changed = list(sp)
    changed[4] = sp[4] + 2e-7
    observer = fm.WallObserver()
    observer.observe(x, c, length, g, 1.2, changed)
    report = observer.report()
    assert not report["consistent"] and report["matched_wall_mismatches"] == 1
    assert report["matched_wall_abs_discrepancy_mm"] == pytest.approx(2e-7)
    assert report["examples"][0]["kind"] == "matched_wall_discrepancy"
    changed[4] = sp[4] + .5e-7
    observer = fm.WallObserver()
    observer.observe(x, c, length, g, 1.2, changed)
    assert observer.report()["consistent"]
    changed[3] = sp[3] - 2e-7
    observer = fm.WallObserver()
    observer.observe(x, c, length, g, 1.2, changed)
    assert observer.report()["gas_mismatches"] == 1


def test_missing_gas_row_is_failure_but_missing_wall_only_is_not():
    observer = fm.WallObserver()
    observer.observe(*rays(), geom(), 1.2, None)
    report = observer.report()
    assert not report["consistent"]
    assert report["unexpected_missing_gas"]["rows"] == report["gas_mismatches"] == 1


def test_oracle_first_256_outer_hits_after_vacuum_prefix_and_bounded_examples(monkeypatch):
    g = geom()
    observer = fm.WallObserver()
    real = fc.oracle_crossings
    calls = []

    def record(x, c, length, outer):
        calls.append(x.copy())
        assert outer.radius == 1.2
        return real(x, c, length, outer)

    monkeypatch.setattr(fc, "oracle_crossings", record)
    x, c, length = rays((-2., 5., 0.), (2., 5., 0.))
    observer.observe(np.repeat(x, 300, axis=0), np.repeat(c, 300, axis=0), np.repeat(length, 300), g, 1.2, None)
    assert not calls
    x, c, length = rays((-2., 1.1, 0.), (2., 1.1, 0.))
    for _ in range(3):
        observer.observe(np.repeat(x, 200, axis=0), np.repeat(c, 200, axis=0), np.repeat(length, 200), g, 1.2, None)
    report = observer.report()
    assert [len(v) for v in calls] == [200, 56]
    assert report["oracle_chords"] == report["oracle_positive_lanes"] == report["oracle_rows"] == 256
    assert report["oracle_unchecked_outer_hit_lanes"] == 344
    assert report["consistent"] and report["zero_outer_hit_lanes"] == 300
    assert len(report["examples"]) == 20
    example = report["examples"][0]
    assert example["call"] == 1 and example["lane"] == 0 and example["straw_index"] == 0
    assert "unavailable" in example["identity_scope"]
    assert example["x_x"] == -2. and example["end_x"] == 2. and example["L_mm"] == 4.


def test_oracle_compares_complete_outer_sets_and_endpoints(monkeypatch):
    g = geom()
    g = replace(g, plane=np.array([0, 0]), panel=np.array([0, 0]), straw=np.array([0, 1]),
                mid=np.array([[0., 0., 0.], [.2, 0., 0.]]), dir=np.tile(g.dir, (2, 1)), halflen=np.full(2, 2.))
    real = fm._CHORD_INDEX.crossings

    def omit(self, x, c, length, outer):
        rows = real(self, x, c, length, outer)
        return rows[rows[:, 1] == 0]

    monkeypatch.setattr(fm._CHORD_INDEX, "crossings", omit)
    observer = fm.WallObserver()
    observer.observe(*rays((-2., 1.1, 0.), (2., 1.1, 0.)), g, 1.2, None)
    report = observer.report()
    assert not report["consistent"] and report["oracle_mismatches"] == 1
    assert any(e["kind"] == "outer_oracle_discrepancy" for e in report["examples"])


def test_saved_secants_are_batched_original_unwrapped_and_explicitly_nonphysical(monkeypatch):
    frame = pd.DataFrame(dict(x0=[-2., -2., 0.], y0=[0., 1.1, 0.], z0=0.,
                              x1=[2., 2., 0.], y1=[0., 1.1, 0.], z1=0.))
    before = frame.copy(deep=True)
    original = fs._straw_pieces

    def broken(*args):
        raise RuntimeError("wrapped routine must not be used offline")

    monkeypatch.setattr(fs, "_straw_pieces", broken)
    for batch in (1, 10):
        report = fm.audit_saved_secants([frame], geom(), np.zeros(3), batch=batch)
        assert report["consistent"] and report["saved_rows"] == 3 and report["zero_length_rows"] == 1
        assert "nonphysical" in report["coverage"] and "not actual" in report["coverage"]
        assert report["matched_rows"] == report["missing_rows"] == 1
    pd.testing.assert_frame_equal(frame, before, check_exact=True)
    monkeypatch.setattr(fs, "_straw_pieces", original)


def test_seeded_transport_payload_and_rng_exactly_unchanged():
    g = geom()
    start = pd.DataFrame(dict(run=1, subrun=0, event=np.arange(4, dtype=np.uint64) + np.uint64(2**54),
                              sp=1, pdg=13, x=-3., y=np.linspace(-.5, .5, 4), z=0., px=150., py=1., pz=1., t=0.))
    before = start.copy(deep=True)
    opts = dict(chord_mm=.5, envelope=fs.Envelope(g, pad_r=2., pad_z=2., axis_xy=(0., 0.)), origin=np.zeros(3))
    rng0, rng1 = np.random.default_rng(19), np.random.default_rng(19)
    base = fs.transport(rng0, start, g, np.array([0., 0., 1.]), **opts)
    observer = fm.WallObserver(16)
    with fm.observing_material(observer):
        observed = fs.transport(rng1, start, g, np.array([0., 0., 1.]), **opts)
    for left, right in zip(base, observed):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    pd.testing.assert_frame_equal(start, before, check_exact=True)
    assert rng0.bit_generator.state == rng1.bit_generator.state
    assert observer.report()["consistent"] and len(base[0]) > 0


def payload():
    ids = np.array([2**54 + 1, 2**54 + 3], dtype=np.uint64)
    return dict(particles=pd.DataFrame(dict(run=1, subrun=0, event=ids, key=ids, pdg=13)),
                sgsteps=pd.DataFrame(dict(run=1, subrun=0, event=ids, sp=ids, len=[1., 2.])))


def write_arms(directory):
    directory.mkdir()
    for filename in fc.FILES.values():
        tables.write(str(directory / filename), payload(), as_ttree=True)


def settings():
    return dict(species=13, inputs=1, seed=19, chord_mm=20., sagitta_mm=.02,
                gas_state="gas", mu_radiative=True, origin_mm=[0., 0., 0.],
                entry_vds=[11, 12, 13], field="map.header")


@pytest.mark.parametrize("mode", ["equal", "mismatch", "integer", "order", "mutate", "error", "wrong_wall"])
def test_cli_exact_replay_gate_receipts_and_failed_retention(tmp_path, monkeypatch, mode):
    old = tmp_path / "original"
    write_arms(old)
    (old / "fieldcompare.json").write_text(json.dumps(settings()))
    gdml, map_, input_ = [tmp_path / name for name in ("world.gdml", "map.header", "input.root")]
    for path in (gdml, map_, input_, map_.with_suffix(".bin")):
        path.write_bytes(b"fixture")
    original, index_class = fs._straw_pieces, fs.ChordIndex
    received = []

    def replay(inputs, output, **kw):
        received.append((inputs, kw))
        assert fs.ChordIndex is index_class and fs._straw_pieces is not original
        fs._straw_pieces(*rays((-2., 1.1, 0.), (2., 1.1, 0.)), geom(), 1.2, 0., 0.)
        if mode == "wrong_wall":
            sp = original(*rays(), geom(), 1.2, 0., 0.)
            changed = list(sp)
            changed[4] = sp[4] + .01
            observer.observe(*rays(), geom(), 1.2, changed)
        write_arms(Path(output))
        if mode in ("mismatch", "integer", "order"):
            changed = payload()
            if mode == "mismatch":
                changed["sgsteps"].loc[0, "len"] = np.nextafter(1., 2.)
            elif mode == "integer":
                changed["particles"].loc[0, "key"] = 2**54 + 2
            else:
                changed["sgsteps"] = changed["sgsteps"].iloc[::-1].reset_index(drop=True)
            tables.write(str(Path(output) / fc.FILES["fast"]), changed, as_ttree=True)
        elif mode == "mutate":
            input_.write_bytes(b"changed")
        elif mode == "error":
            raise RuntimeError("replay failure")

    observer = fm.WallObserver(2)
    monkeypatch.setattr(fm, "WallObserver", lambda n: observer)
    monkeypatch.setattr(fm.fieldcompare, "run", replay)
    out = tmp_path / "audit"
    result = fm.run(old, input_, gdml, map_, out, entry="first", oracle_chords=2)
    assert fs._straw_pieces is original and fs.ChordIndex is index_class
    assert result["passed"] == (mode == "equal")
    assert (out / "FAILED").exists() == (mode != "equal")
    assert (out / "replay" / fc.FILES["fast"]).is_file()
    assert json.loads((out / "fieldmaterial.json").read_text())["passed"] == result["passed"]
    assert result["inputs_unchanged"] == (mode != "mutate")
    assert "map_adjacent_binary" in result["hashes_before"]
    assert "code/core_tables" in result["hashes_before"]
    assert "original/fasttrk_field.root" in result["hashes_before"]
    assert received == [([str(input_)], dict(gdml=str(gdml), field_map=str(map_),
                                            **fm.pa.frozen_settings(settings(), "first")))]
    if mode in ("mismatch", "integer", "order"):
        assert not result["reproduction"]["reproduced"] and result["comparisons_diagnostic_only"]
    with pytest.raises(FileExistsError):
        fm.run(old, input_, gdml, map_, out, entry="first")


def test_cli_requires_explicit_original_first_entry():
    with pytest.raises(SystemExit) as exc:
        fm.main(["--field-dir", "old", "--input-table", "in", "--gdml", "g", "--map", "m", "--outdir", "out"])
    assert exc.value.code == 2


def test_outer_oracle_endpoint_tolerance_and_no_relative_tolerance(monkeypatch):
    original = fc.oracle_crossings

    def shifted(x, c, length, outer):
        rows = original(x, c, length, outer)
        rows[:, 3] -= 2e-7
        return rows

    monkeypatch.setattr(fc, "oracle_crossings", shifted)
    report, _ = observe(*rays())
    assert report["oracle_mismatches"] == 1 and not report["consistent"]
    example = report["examples"][0]
    assert example["kind"] == "outer_oracle_discrepancy"
    assert example["s_out_left"] - example["s_out_right"] == pytest.approx(2e-7)


def test_consumed_invalid_identity_duplicate_and_outside_chord_fail_closed():
    args = rays()
    sp = fs._straw_pieces(*args, geom(), 1.2, 0., 0.)
    for column, value in ((0, .5), (0, -1), (1, 1), (3, 5.)):
        changed = [a.copy().astype(float) for a in sp]
        changed[column][:] = value
        with pytest.raises(ValueError):
            fm.WallObserver().observe(*args, geom(), 1.2, changed)
    with pytest.raises(ValueError, match="duplicate"):
        fm.WallObserver().observe(*args, geom(), 1.2, [np.repeat(a, 2) for a in sp])


def test_saved_tables_reads_all_three_endpoint_arms(tmp_path):
    frame = pd.DataFrame(dict(x0=[-2.], y0=[1.1], z0=[0.], x1=[2.], y1=[1.1], z1=[0.]))
    for filename in fc.FILES.values():
        tables.write(str(tmp_path / filename), {"sgsteps": frame}, as_ttree=True)
    reports = fm.audit_saved_tables(tmp_path, geom(), np.zeros(3), batch=1)
    assert set(reports) == {"fast", "g4_all", "g4_clock"}
    for report in reports.values():
        assert report["consistent"] and report["saved_rows"] == 1
        assert report["missing_rows"] == 1 and report["oracle_positive_lanes"] == 1
        assert "not actual" in report["coverage"]


def test_cli_requires_both_map_header_and_binary_before_replay(tmp_path):
    old = tmp_path / "original"
    write_arms(old)
    (old / "fieldcompare.json").write_text(json.dumps(settings()))
    header = tmp_path / "map.header"
    header.write_text("fixture")
    with pytest.raises(ValueError, match="adjacent .bin"):
        fm.run(old, tmp_path / "input.root", tmp_path / "world.gdml", header, tmp_path / "audit", entry="first")
    assert not (tmp_path / "audit").exists()
