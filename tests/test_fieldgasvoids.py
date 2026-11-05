"""Geometry-only finite own-wire proxy: never infer an actual trajectory cause."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import tables
from trkfastsim import geometry
from trkfastsim.validation import fieldgasvoids as fv


def geom():
    return geometry.StrawGeometry(
        plane=np.array([0, 0]), panel=np.array([0, 0]), straw=np.array([0, 1]),
        mid=np.array([[0., 0., 0.], [4., 0., 0.]]),
        dir=np.array([[0., 0., 1.], [0., 0., 1.]]),
        halflen=np.array([1000., 2.]), radius=1., wire_r=.1,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)),
        plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)))


def particles():
    return pd.DataFrame({"run": np.array([1, 1, 1], dtype=np.uint64),
                         "subrun": np.array([2, 2, 2], dtype=np.uint64),
                         "event": np.array([2**54 + 1, 2**54 + 3, 2**63 + 7], dtype=np.uint64),
                         "sp": np.array([1, 1, 2], dtype=np.uint64)})


def steps(origin=None):
    origin = np.zeros(3) if origin is None else np.asarray(origin)
    # A long, nearly axial central wire gap bracketed by gas endpoints; two gas
    # pieces would sum to chord-wire. Then transverse, off-wire and finite caps.
    start = np.array([[-.5, 0., -900.], [-1., 0., 0.], [.5, 0., -2.],
                      [4., 0., -4.], [0., 0., 0.], [0., 0., 0.],
                      [-3., 0., 0.], [4.5, 0., -4.], [8., 0., -2.]])
    end = np.array([[.5, 0., 900.], [1., 0., 0.], [.5, 0., 2.],
                    [4., 0., 4.], [0., 0., 0.], [0., 0., 0.],
                    [3., 0., 0.], [4.5, 0., 4.], [8., 0., 2.]])
    chord = np.linalg.norm(end - start, axis=1)
    s = particles().iloc[[0, 0, 1, 1, 1, 1, 0, 1, 1]].reset_index(drop=True)
    s = s.assign(sid=np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.uint64),
                 plane=0, panel=0, straw=[0, 0, 0, 1, 0, 0, 0, 1, 0],
                 len=[chord[0] * .8, 1.8, 4., 0., 0., 2., 1.8, 4., 0.],
                 px=1., py=2., pz=3.)
    s[["x0", "y0", "z0"]] = start - origin
    s[["x1", "y1", "z1"]] = end - origin
    return s


def audit(s=None, g=None, origin=None):
    return fv.audit_arm(steps() if s is None else s, particles(), geom() if g is None else g,
                        np.zeros(3) if origin is None else origin)


def test_long_axial_segmented_gas_and_transverse_wire_gap_offwire_zero():
    s = steps()
    chord, inner, wire, gas = fv.secant_lengths(s, geom(), np.zeros(3), batch=2)
    assert wire[0] == pytest.approx(chord[0] * .2)
    assert gas[0] == pytest.approx(s.len[0])
    assert wire[1] == pytest.approx(.2)
    assert gas[1] == pytest.approx(1.8)
    assert wire[2] == 0 and gas[2] == 4
    np.testing.assert_array_equal(chord[4:6], [0., 0.])
    np.testing.assert_array_equal(wire[4:6], [0., 0.])
    assert wire[8] == inner[8] == gas[8] == 0  # own SID, not nearby geometry
    per, out, report = audit()
    assert report["saved_rows"] == len(s)
    assert report["zero_secants"] == 2 and report["zero_secants_positive_len"] == 1
    assert report["zero_step_particles"] == 1
    assert per.rowcount.tolist() == [3, 6, 0]
    assert per.event.dtype == np.uint64
    assert per.event.tolist() == particles().event.tolist()
    assert report["raw_short_by1_rows"] == 5
    assert report["raw_short_by1_with_residual_by1_rows"] == 0
    assert report["raw_short_by1_without_residual_by1_rows"] == 5
    assert out.originalrowindex.tolist() == [0, 3, 6, 7, 8]
    assert set(["px", "py", "pz", "x0", "z1"] + fv.TRACK).issubset(out.columns)
    assert report["positive_shortfalls_mm"]["after_inner_and_wire"]["max"] < 1e-9
    assert report["ratio_below"] == {str(t): int((s.len.to_numpy()[chord > 0] / chord[chord > 0] < t).sum())
                                      for t in (.99, .9, .5, .1)}
    for kind, values in (("raw", chord - s.len), ("residual", gas - s.len)):
        assert report["shortfall_beyond_mm"][kind] == {str(t): int((values > t).sum()) for t in fv.THRESHOLDS}
    json.dumps(report, allow_nan=False)


def test_finite_caps_and_outside_inner_no_double_subtraction():
    chord, inner, wire, gas = fv.secant_lengths(steps(), geom(), np.zeros(3))
    assert chord[3] == 8 and inner[3] == 4 and wire[3] == 4 and gas[3] == 0
    assert chord[6] == 6 and inner[6] == 2
    assert wire[6] == pytest.approx(.2) and gas[6] == pytest.approx(1.8)
    assert chord[7] == 8 and inner[7] == 4 and wire[7] == 0 and gas[7] == 4
    assert gas[6] != pytest.approx(chord[6] - wire[6])


def test_residual_remaining_and_complete_unbounded_outliers():
    s = pd.concat([steps().iloc[[2]].assign(len=0.)] * 45, ignore_index=True)
    per, out, report = audit(s)
    assert len(out) == 45 and out.originalrowindex.tolist() == list(range(45))
    assert report["raw_short_by1_with_residual_by1_rows"] == 45
    assert report["positive_shortfalls_mm"]["before_full_chord"]["total"] == 180
    assert report["positive_shortfalls_mm"]["after_inner_and_wire"]["total"] == 180
    assert per.residual_short_by1_rows.tolist() == [0, 45, 0]


def test_allzero_43156_denominator_and_empty_all_arms_uint64():
    n = 43156
    p = pd.DataFrame({"run": np.ones(n, dtype=np.uint64), "subrun": np.zeros(n, dtype=np.uint64),
                      "event": np.arange(n, dtype=np.uint64) + np.uint64(2**63 + 3),
                      "sp": np.ones(n, dtype=np.uint64)})
    ledger, out, report = fv.assemble({a: p for a in fv.FILES},
                                     {a: steps().iloc[:0] for a in fv.FILES}, geom(), np.zeros(3))
    pd.testing.assert_frame_equal(ledger[fv.TRACK], p)
    for a in fv.FILES:
        assert ledger[f"{a}_rowcount"].sum() == 0 and out[a].empty
        assert report["arms"][a]["zero_step_particles"] == n
        assert report["arms"][a]["len_over_chord"]["max"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("change,message", [({"sid": 999}, "foreign saved SID"),
    ({"sid": 1}, "labels disagree"), ({"plane": 1}, "labels disagree"),
    ({"straw": .5}, "integer identities"), ({"event": 9}, "outside entry"),
    ({"event": float(2**54)}, "identities"), ({"len": -1.}, "negative saved"),
    ({"x0": np.inf}, "nonfinite"), ({"px": np.nan}, "nonfinite")])
def test_invalid_mapping_keys_lengths_and_any_nonfinite(change, message):
    with pytest.raises(ValueError, match=message):
        audit(steps().assign(**change))


@pytest.mark.parametrize("wire_r", [0., -1., np.nan, np.inf, 1., 2.])
def test_invalid_wire_geometry(wire_r):
    with pytest.raises(ValueError, match="wire outer radius"):
        audit(g=replace(geom(), wire_r=wire_r))


def test_duplicate_mapping_and_foreign_particle_denominator():
    with pytest.raises(ValueError, match="duplicate geometry"):
        audit(g=replace(geom(), straw=np.array([0, 0])))
    p = {a: particles() for a in fv.FILES}
    p["g4_all"] = p["g4_all"].assign(event=p["g4_all"].event + np.uint64(1))
    with pytest.raises(ValueError, match="different entry"):
        fv.assemble(p, {a: steps() for a in fv.FILES}, geom(), np.zeros(3))


def test_native_origin_translation_and_unchanged_objects():
    origin = np.array([-4., 3., 17.])
    s, p, g = steps(origin), particles(), geom()
    old_s, old_p, old_mid = s.copy(deep=True), p.copy(deep=True), g.mid.copy()
    per, out, report = fv.audit_arm(s, p, g, origin)
    expected, expected_out, expected_report = audit()
    pd.testing.assert_frame_equal(per, expected)
    assert report == expected_report
    np.testing.assert_allclose(out[["x0", "y0", "z0"]] + origin,
                               expected_out[["x0", "y0", "z0"]])
    pd.testing.assert_frame_equal(s, old_s)
    pd.testing.assert_frame_equal(p, old_p)
    np.testing.assert_array_equal(g.mid, old_mid)


def saved(tmp_path, monkeypatch):
    parent = tmp_path / "saved"
    parent.mkdir()
    origin = [-4., 3., 17.]
    (parent / "fieldcompare.json").write_text(json.dumps({"origin_mm": origin, "entries": 3}))
    for a, filename in fv.FILES.items():
        payload = {"particles": particles().rename(columns={"sp": "key"})}
        if a != "g4_clock":
            payload["sgsteps"] = steps(origin)
        tables.write(str(parent / filename), payload, as_ttree=True)
    gdml = tmp_path / "geom.gdml"
    gdml.write_text("synthetic geometry fixture")
    monkeypatch.setattr(fv.geometry, "from_gdml", lambda _: geom())
    return parent, gdml


def test_cli_strict_json_exact_csv_exclusive_output_and_hashes(tmp_path, monkeypatch, capsys):
    parent, gdml = saved(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in [*parent.iterdir(), gdml]}
    out = tmp_path / "audit"
    fv.main(["--field-dir", str(parent), "--gdml", str(gdml), "--outdir", str(out)])
    report = json.loads((out / "fieldgasvoids.json").read_text(),
                        parse_constant=lambda x: pytest.fail(f"non-strict JSON: {x}"))
    assert report["denominator_particles"] == 3
    assert report["arms"]["g4_clock"]["saved_step_tree_present"] is False
    assert report["limitations"] == list(fv.LIMITATIONS)
    for receipt in report["inputs"].values():
        assert receipt["sha256_before"] == receipt["sha256_after"] == fv.fg._sha256(receipt["path"])
    assert {"code/fieldgasledger", "code/fieldmaterial", "code/fieldcrossing", "code/cecontract"}.issubset(report["inputs"])
    for receipt in report["outputs"].values():
        assert receipt["sha256"] == fv.fg._sha256(receipt["path"])
    ledger = pd.read_csv(out / "fieldgasvoids_particles.csv.gz", dtype={k: "uint64" for k in fv.TRACK})
    pd.testing.assert_frame_equal(ledger[fv.TRACK], particles())
    rows = pd.read_csv(out / "fieldgasvoids_fast_outliers.csv.gz", dtype={k: "uint64" for k in fv.TRACK + ["sid"]})
    assert rows.originalrowindex.tolist() == [0, 3, 6, 7, 8]
    assert rows.event.tolist() == steps().iloc[[0, 3, 6, 7, 8]].event.tolist()
    assert all(p.read_bytes() == b for p, b in before.items())
    with pytest.raises(FileExistsError):
        fv.run(parent, gdml, out)
    capsys.readouterr()


@pytest.mark.parametrize("when", ["assemble", "output"])
def test_input_mutation_failclosed_before_and_after_outputs(tmp_path, monkeypatch, when):
    parent, gdml = saved(tmp_path, monkeypatch)
    def mutate():
        with (parent / fv.FILES["fast"]).open("ab") as stream:
            stream.write(b"mutation during audit")
    if when == "assemble":
        original = fv.assemble
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            mutate()
            return result
        monkeypatch.setattr(fv, "assemble", wrapped)
    else:
        original = pd.DataFrame.to_csv
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            mutate()
            return result
        monkeypatch.setattr(pd.DataFrame, "to_csv", wrapped)
    out = tmp_path / "mutated"
    with pytest.raises(ValueError, match="changed during"):
        fv.run(parent, gdml, out)
    assert not (out / "fieldgasvoids.json").exists()


def test_exact_axis_wire_offwire_tangent_and_rotated_coordinates():
    s = steps().iloc[[2, 2, 2]].copy().reset_index(drop=True)
    s[["x0", "y0", "z0"]] = [[0., 0., -3.], [.5, 0., -3.], [-1., .1, 0.]]
    s[["x1", "y1", "z1"]] = [[0., 0., 3.], [.5, 0., 3.], [1., .1, 0.]]
    s["len"] = [0., 6., 2. * np.sqrt(.99)]
    chord, inner, wire, gas = fv.secant_lengths(s, geom(), np.zeros(3))
    np.testing.assert_allclose(wire, [6., 0., 0.], atol=1e-12)
    np.testing.assert_allclose(gas, s.len, atol=1e-12)
    # A rigid permutation tests pairing with the actual assigned axis and midpoint.
    rotation = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    shift = np.array([10., -20., 30.])
    rotated = s.copy()
    for end in (0, 1):
        cols = [f"{a}{end}" for a in "xyz"]
        rotated[cols] = s[cols].to_numpy() @ rotation.T + shift
    g = replace(geom(), mid=geom().mid @ rotation.T + shift, dir=geom().dir @ rotation.T)
    for actual, expected in zip(fv.secant_lengths(rotated, g, np.zeros(3)), (chord, inner, wire, gas)):
        np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("origin", [[1, 2], [1, 2, float("nan")]])
def test_bad_origin_failclosed(tmp_path, monkeypatch, origin):
    parent, gdml = saved(tmp_path, monkeypatch)
    (parent / "fieldcompare.json").write_text(json.dumps({"origin_mm": origin, "entries": 3}))
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="origin"):
        fv.run(parent, gdml, out)
    assert not (out / "fieldgasvoids.json").exists()
