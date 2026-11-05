"""Saved endpoint search agreement is interval-complete, bounded and independent of transport."""
import json
import numpy as np
import pandas as pd
import pytest

from mu2efastsim import tables
from trkfastsim import crossings, geometry
from trkfastsim.sim import fieldsteps as fs
from trkfastsim.validation import fieldcrossing as fc


def tilted_geometry():
    axis = np.array([[1., 0., 0.], [1., 0., 0.], [0.8, 0., 0.6], [0.8, 0., 0.6],
                     [0., 1., 0.], [0., 1., 0.]])
    return geometry.StrawGeometry(
        plane=np.zeros(6, dtype=np.int64), panel=np.repeat(np.arange(3), 2), straw=np.tile([0, 1], 3),
        mid=np.array([[0., 0., 0.], [0., 3., 0.], [0., 0., 5.], [0., 3., 5.],
                      [0., 0., 12.], [3., 0., 12.]]), dir=axis, halflen=np.array([4., 3., 7., 5., 2., 4.]),
        radius=1., panel_R=np.tile(np.eye(3), (3, 1, 1)), panel_t=np.zeros((3, 3)),
        plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)))


def denominator():
    return pd.DataFrame(dict(run=np.array([1430, 1430], dtype=np.uint64), subrun=np.array([0, 0], dtype=np.uint64),
                             event=np.array([2**54 + 1, 2**54 + 3], dtype=np.uint64), sp=np.array([1, 2], dtype=np.uint64)))


def endpoints():
    start = np.array([[0., 0., -5.], [0., 0., 18.], [-8., -0.2, 0.], [-8., -0.2, 0.],
                      [0., 0., 0.], [0., 0., -2.], [0., 0., -2.], [0., 0., 5.],
                      [0., 1., -2.], [-8., 2., 0.], [20., 20., 20.]])
    end = np.array([[0., 0., 18.], [0., 0., -5.], [8., -0.2, 0.], [-2., -0.2, 0.],
                    [0., 0., 0.25], [0., 0., -1.], [0., 0., -0.5], [5.6, 0., 9.2],
                    [0., 1., 2.], [8., 2., 0.], [21., 20., 20.]])
    return start, end


def step_frame(start=None, end=None, origin=None):
    if start is None:
        start, end = endpoints()
    origin = np.zeros(3) if origin is None else origin
    d = pd.concat([denominator().iloc[:1]] * len(start), ignore_index=True).assign(plane=0, panel=0, straw=0)
    d[["x0", "y0", "z0"]] = start - origin
    d[["x1", "y1", "z1"]] = end - origin
    return d


def chunks(d, size):
    return (d.iloc[i:i + size] for i in range(0, len(d), size))


def three_searches(start, end, geom):
    x, c, length, keep = fc.endpoint_secants(start, end, np.zeros(3))
    assert keep.all()
    a = fc.canonical(fs.ChordIndex(geom).crossings(x, c, length, geom), length, geom.nstraws, "A")
    b = fc.canonical(crossings.straw_crossings(x, c, geom, index=crossings.panel_index(geom)), length, geom.nstraws, "B")
    oracle = fc.oracle_crossings(x, c, length, geom, batch=2, straw_block=1)
    assert fc.compare_rows(a, b, ["A", "B"]).empty
    assert fc.compare_rows(a, oracle, ["A", "C"]).empty
    return a, length


def test_tilted_unequal_z_panels_all_directions_caps_inside_clipping_tangent():
    geom = tilted_geometry()
    lo, hi, _ = crossings.panel_index(geom)
    assert np.unique(hi[:, 2] - lo[:, 2]).size > 1
    rows, length = three_searches(*endpoints(), geom)
    lane = rows[:, 0].astype(int)
    assert np.all(rows[:, 3] <= length[lane])
    np.testing.assert_allclose(rows[(lane == 2) & (rows[:, 1] == 0), 2:], [[4., 12.]], atol=1e-12)
    np.testing.assert_allclose(rows[(lane == 3) & (rows[:, 1] == 0), 2:], [[4., 6.]], atol=1e-12)
    np.testing.assert_allclose(rows[(lane == 4) & (rows[:, 1] == 0), 2:], [[0., 0.25]], atol=1e-12)
    assert not (lane == 5).any()
    np.testing.assert_allclose(rows[(lane == 6) & (rows[:, 1] == 0), 2:], [[1., 1.5]], atol=1e-12)
    np.testing.assert_allclose(rows[(lane == 7) & (rows[:, 1] == 2), 2:], [[0., 7.]], atol=1e-12)
    assert not (lane == 8).any()
    np.testing.assert_allclose(rows[(lane == 9) & (rows[:, 1] == 1), 2:], [[5., 11.]], atol=1e-12)
    assert not (lane == 10).any()


def test_reversal_preserves_complete_straw_sets_and_reflects_intervals():
    start, end = endpoints()
    forward, length = three_searches(start, end, tilted_geometry())
    backward, _ = three_searches(end, start, tilted_geometry())
    np.testing.assert_array_equal(forward[:, :2], backward[:, :2])
    lane = forward[:, 0].astype(int)
    np.testing.assert_allclose(backward[:, 2], length[lane] - forward[:, 3], atol=1e-10, rtol=0)
    np.testing.assert_allclose(backward[:, 3], length[lane] - forward[:, 2], atol=1e-10, rtol=0)


def test_chunk_invariance_zero_rows_complete_denominator_and_fixed_oracle_prefix():
    steps = step_frame()
    steps = pd.concat([steps.iloc[:2], steps.iloc[:1].assign(x1=0., y1=0., z1=-5.), steps.iloc[2:]], ignore_index=True)
    reports = [fc.audit_arm(chunks(steps, size), denominator(), tilted_geometry(), np.zeros(3), oracle_segments=5)
               for size in (1, 3, 512)]
    assert reports[0] == reports[1] == reports[2]
    report = reports[0]
    assert report["rows"] == 12 and report["zero_length"] == 1 and report["secants"] == 11
    assert report["oracle_secants"] == 5
    assert report["comparisons"]["A_B"]["segments"] == 11
    assert report["comparisons"]["A_C"]["segments"] == 5
    assert report["agreement"] and not report["examples"]
    disabled = fc.audit_arm([steps], denominator(), tilted_geometry(), np.zeros(3), oracle_segments=0)
    assert disabled["oracle_secants"] == disabled["crossings"]["C"] == 0


def test_full_straw_set_detects_A_omission_even_when_every_lane_still_has_a_hit():
    geom = tilted_geometry()
    real = fs.ChordIndex(geom)

    class Omit:
        def crossings(self, x, c, length, geom):
            rows = real.crossings(x, c, length, geom)
            return rows[rows[:, 1] != 2]

    steps = step_frame().iloc[:2]
    x, c, length, _ = fc.endpoint_secants(steps[["x0", "y0", "z0"]], steps[["x1", "y1", "z1"]], np.zeros(3))
    np.testing.assert_array_equal(np.unique(real.crossings(x, c, length, geom)[:, 0]),
                                  np.unique(Omit().crossings(x, c, length, geom)[:, 0]))
    report = fc.audit_arm(chunks(steps, 1), denominator(), geom, np.zeros(3), chord_index=Omit())
    assert report["comparisons"]["A_B"]["kinds"] == {"missing_A": 2}
    assert report["comparisons"]["A_C"]["kinds"] == {"missing_A": 2}
    assert report["comparisons"]["B_C"]["mismatches"] == 0
    assert report["total_mismatches"] == 4 and not report["agreement"]
    example = report["examples"][0]
    assert example["event"] == 2**54 + 1 and example["sp"] == 1
    assert example["segment_row"] == 0 and example["current_straw"] == 0 and example["crossed_straw"] == 0
    assert example["crossed_panel"] == 1 and example["kind"] == "missing_A"


def test_explicit_endpoint_tolerance_canonical_sort_duplicates_and_clip_both_searches():
    a = np.array([[0., 1., 0., 2.], [0., 0., 0.5, 5.]])
    clipped = fc.canonical(a, np.array([1.]), 2, "A")
    np.testing.assert_array_equal(clipped, [[0., 0., 0.5, 1.], [0., 1., 0., 1.]])
    assert fc.compare_rows(clipped, clipped + [0, 0, 0.5e-7, 0], ["A", "B"]).empty
    changed = clipped.copy()
    changed[0, 3] += 2e-7
    assert fc.compare_rows(clipped, changed, ["A", "B"]).kind.tolist() == ["interval"]
    with pytest.raises(ValueError, match="duplicate"):
        fc.canonical(np.repeat(a[:1], 2, axis=0), np.array([1.]), 2, "A")
    with pytest.raises(ValueError, match="foreign"):
        fc.canonical(np.array([[1., 0., 0., 1.]]), np.array([1.]), 2, "A")
    with pytest.raises(ValueError, match="invalid"):
        fc.canonical(np.array([[0., 0.5, 0., 1.]]), np.array([1.]), 2, "A")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_endpoints_rejected_not_dropped(bad):
    steps = step_frame()
    steps.loc[3, "x1"] = bad
    with pytest.raises(ValueError, match="nonfinite"):
        fc.audit_arm(chunks(steps, 3), denominator(), tilted_geometry(), np.zeros(3))


@pytest.mark.parametrize("direction,length", [([0., 0., 0.], 1.), ([0., 0., 2.], 1.),
                                             ([0., 0., 1.], 0.), ([np.nan, 0., 1.], 1.)])
def test_invalid_oracle_rays_rejected(direction, length):
    with pytest.raises(ValueError):
        fc.oracle_crossings(np.zeros((1, 3)), np.array([direction]), np.array([length]), tilted_geometry())


def test_exact_integer_unique_identical_particles_foreign_steps_and_straws():
    p = denominator()
    arms = {name: p.copy() for name in fc.FILES}
    pd.testing.assert_frame_equal(fc.verify_particles(arms), p)
    with pytest.raises(ValueError, match="different entry"):
        fc.verify_particles({**arms, "g4_all": p.assign(event=p.event + np.uint64(1))})
    with pytest.raises(ValueError, match="duplicate"):
        fc.verify_particles({**arms, "fast": pd.concat([p, p.iloc[:1]])})
    with pytest.raises(ValueError, match="identities"):
        fc.verify_particles({**arms, "fast": p.assign(event=p.event.astype(float))})
    with pytest.raises(ValueError, match="all three"):
        fc.verify_particles({"fast": p})
    for steps, message in ((step_frame().assign(sp=8), "outside entry"),
                           (step_frame().assign(straw=10), "foreign current straw"),
                           (step_frame().assign(panel=0.5), "invalid current straw")):
        with pytest.raises(ValueError, match=message):
            fc.audit_arm([steps], p, tilted_geometry(), np.zeros(3))


def saved_inputs(tmp_path, monkeypatch):
    parent = tmp_path / "saved"
    parent.mkdir()
    origin = np.array([-4., 2., 17.])
    (parent / "fieldcompare.json").write_text(json.dumps(dict(origin_mm=origin.tolist(), entries=2)))
    for label, filename in fc.FILES.items():
        tables.write(str(parent / filename), {"particles": denominator().rename(columns={"sp": "key"}),
                                              "sgsteps": step_frame(origin=origin)}, as_ttree=True)
    gdml = tmp_path / "test.gdml"
    gdml.write_text("synthetic geometry bound by test fixture")
    monkeypatch.setattr(fc.geometry, "from_gdml", lambda path: tilted_geometry())
    return parent, gdml, origin


def test_saved_root_cli_origin_hashes_and_exclusive_output(tmp_path, monkeypatch, capsys):
    parent, gdml, origin = saved_inputs(tmp_path, monkeypatch)
    out = tmp_path / "audit"
    fc.main(["--field-dir", str(parent), "--gdml", str(gdml), "--outdir", str(out), "--chunk", "2",
             "--oracle-segments", "3"])
    result = json.loads((out / "fieldcrossing.json").read_text())
    assert result["coverage"] == fc.COVERAGE and result["agreement"]
    assert result["origin_mm"] == origin.tolist() and result["denominator_particles"] == 2
    assert result["interval_atol_mm"] == 1e-7 and result["interval_rtol"] == 0
    assert set(result["inputs"]) == {"fieldcompare", "gdml", *fc.FILES}
    for receipt in result["inputs"].values():
        assert receipt["sha256"] == fc._sha256(receipt["path"])
    assert all(arm["oracle_secants"] == 3 for arm in result["arms"].values())
    with pytest.raises(FileExistsError):
        fc.run(parent, gdml, out)
    capsys.readouterr()


def test_empty_saved_step_arm_keeps_particles_and_invalid_cli_limits(tmp_path, monkeypatch):
    parent, gdml, _ = saved_inputs(tmp_path, monkeypatch)
    tables.write(str(parent / fc.FILES["fast"]), {"particles": denominator().rename(columns={"sp": "key"})}, as_ttree=True)
    result = fc.run(parent, gdml, tmp_path / "empty", chunk=1)
    assert result["denominator_particles"] == 2
    assert result["arms"]["fast"]["rows"] == result["arms"]["fast"]["oracle_secants"] == 0
    for opts in (dict(chunk=0), dict(oracle_segments=-1)):
        with pytest.raises(ValueError):
            fc.run(parent, gdml, tmp_path / "bad", **opts)


def test_oracle_random_rays_batch_and_straw_block_invariance():
    rng = np.random.default_rng(712)
    start = rng.uniform([-12., -6., -7.], [12., 6., 19.], (123, 3))
    end = rng.uniform([-12., -6., -7.], [12., 6., 19.], (123, 3))
    a, _ = three_searches(start, end, tilted_geometry())
    x, c, length, _ = fc.endpoint_secants(start, end, np.zeros(3))
    oracle = fc.oracle_crossings(x, c, length, tilted_geometry(), batch=13, straw_block=4)
    assert fc.compare_rows(a, oracle, ["A", "C"]).empty


def test_bounded_first_mismatch_examples_invariant_under_chunking_and_duplicate_A_rejected():
    geom = tilted_geometry()
    real = fs.ChordIndex(geom)

    class Omit:
        def crossings(self, x, c, length, geom):
            rows = real.crossings(x, c, length, geom)
            return rows[rows[:, 1] != 2]

    steps = pd.concat([step_frame().iloc[:2]] * 30, ignore_index=True)
    small = fc.audit_arm(chunks(steps, 1), denominator(), geom, np.zeros(3), chord_index=Omit())
    large = fc.audit_arm(chunks(steps, 512), denominator(), geom, np.zeros(3), chord_index=Omit())
    assert small == large
    assert len(small["examples"]) == fc.EXAMPLES and small["total_mismatches"] == 120

    class Duplicate:
        def crossings(self, x, c, length, geom):
            rows = real.crossings(x, c, length, geom)
            return np.concatenate([rows, rows[:1]])

    with pytest.raises(ValueError, match="duplicate lane/straw"):
        fc.audit_arm([steps], denominator(), geom, np.zeros(3), chord_index=Duplicate())


def test_saved_input_foreign_particles_bad_origin_and_nonfinite_never_write_success(tmp_path, monkeypatch):
    parent, gdml, _ = saved_inputs(tmp_path, monkeypatch)
    p = denominator().assign(event=denominator().event + np.uint64(1))
    tables.write(str(parent / fc.FILES["fast"]), {"particles": p.rename(columns={"sp": "key"}),
                                              "sgsteps": step_frame()}, as_ttree=True)
    with pytest.raises(ValueError, match="different entry"):
        fc.run(parent, gdml, tmp_path / "foreign")
    tables.write(str(parent / fc.FILES["fast"]), {"particles": denominator().rename(columns={"sp": "key"}),
                                              "sgsteps": step_frame().assign(x1=np.nan)}, as_ttree=True)
    out = tmp_path / "nonfinite"
    with pytest.raises(ValueError, match="nonfinite"):
        fc.run(parent, gdml, out)
    assert not (out / "fieldcrossing.json").exists()
    (parent / "fieldcompare.json").write_text(json.dumps(dict(origin_mm=[0, 1], entries=2)))
    with pytest.raises(ValueError, match="origin"):
        fc.run(parent, gdml, tmp_path / "origin")
