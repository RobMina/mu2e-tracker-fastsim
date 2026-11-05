"""Saved SGS length/geometry grouping is descriptive, exact-keyed and read-only."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import tables
from trkfastsim import geometry
from trkfastsim.validation import fieldgasledger as fg


def geom():
    return geometry.StrawGeometry(
        plane=np.array([0, 0]), panel=np.array([0, 0]), straw=np.array([0, 1]),
        mid=np.array([[0., 0., 0.], [4., 0., 0.]]), dir=np.array([[0., 0., 1.], [0., 0., 1.]]),
        halflen=np.array([3., 2.]), radius=1., panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)),
        plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)))


def particles():
    return pd.DataFrame({"run": np.array([1, 1, 1], dtype=np.uint64),
                         "subrun": np.array([2, 2, 2], dtype=np.uint64),
                         "event": np.array([2**54 + 1, 2**54 + 3, 2**63 + 7], dtype=np.uint64),
                         "sp": np.array([1, 1, 2], dtype=np.uint64)})


def steps(origin=None):
    origin = np.zeros(3) if origin is None else np.asarray(origin)
    p = particles().iloc[[0, 0, 0, 1, 1, 1]].reset_index(drop=True)
    p = p.assign(sid=np.array([0, 0, 1, 0, 0, 0], dtype=np.uint64), plane=0, panel=0,
                 straw=[0, 0, 1, 0, 0, 0], len=[2., np.pi, 4., 0., .5, 3.])
    p[["x0", "y0", "z0"]] = np.array([[-1., 0., 0.], [-1., 0., 0.], [4., 0., -2.],
                                           [0., 0., 0.], [0., 0., 0.], [1.02, 0., 3.02]]) - origin
    p[["x1", "y1", "z1"]] = np.array([[1., 0., 0.], [1., 0., 0.], [4., 0., 2.],
                                           [0., 0., 0.], [1., 0., 0.], [1.02, 0., 3.02]]) - origin
    return p


def assemble(arms=None, origin=None):
    return fg.assemble({a: particles() for a in fg.FILES},
                       {a: steps() for a in fg.FILES} if arms is None else arms,
                       geom(), np.zeros(3) if origin is None else origin)


def test_vectorized_straight_curved_grouped_zero_and_shorter_rows():
    ledger, grouped, report = assemble()
    r = report["arms"]["fast"]
    assert r["saved_rows"] == 6 and r["distinct_particle_sid_pairs"] == 3
    assert r["repeat_rows"] == 3 and r["repeated_particle_sid_pairs"] == 2
    assert r["zero_step_particles"] == 1
    m = r["saved_row_lengths"]
    assert m["zero_secants"] == 2 and m["nonzero_secants"] == 4
    assert m["zero_secants_positive_len"] == 1
    assert m["ratio_above"] == {"1.01": 1, "1.1": 1, "2": 0}
    assert m["len_shorter_than_chord_beyond_0.001mm"] == 1
    np.testing.assert_allclose(list(m["len_over_chord"]["quantiles"].values()),
                               np.quantile([1., np.pi / 2, 1., .5], fg.QUANTILES))
    assert r["endpoint_residuals"]["entry"]["radial_mm"]["outside_beyond_0.01mm"] == 1
    assert r["endpoint_residuals"]["entry"]["axial_mm"]["outside_beyond_0.01mm"] == 1
    assert r["endpoint_residuals"]["exit"]["outside_finite_cylinder_beyond_0.01mm"] == 1
    np.testing.assert_allclose(ledger.fast_gaslength_mm, [6 + np.pi, 3.5, 0])
    np.testing.assert_allclose(ledger.fast_chordlength_mm, [8, 1, 0])
    assert ledger.fast_distinct_straws.tolist() == [2, 1, 0]
    assert ledger.fast_rows.tolist() == [3, 3, 0]
    assert ledger.event.tolist() == particles().event.tolist()
    assert grouped["fast"].rows.tolist() == [2, 1, 3]
    assert all(v["per_particle"]["difference_left_minus_right"]["rows"]["total"] == 0
               for v in report["paired"].values())
    json.dumps(report, allow_nan=False)


def test_symmetric_pairs_and_empty_arm_keep_complete_denominator():
    s = steps()
    ledger, groups, report = assemble({"fast": s, "g4_all": s.iloc[:1], "g4_clock": s.iloc[:0]})
    assert len(ledger) == 3 and len(groups["g4_clock"]) == 0
    r = report["arms"]["g4_clock"]
    assert r["zero_step_particles"] == 3 and r["per_particle"]["rows"]["n"] == 3
    assert r["saved_row_lengths"]["len_over_chord"]["quantiles"]["0"] is None
    assert ledger.g4_clock_rows.tolist() == [0, 0, 0]
    assert report["paired"]["fast_minus_g4_clock"]["per_particle_sid"]["left_only"] == 3
    assert report["paired"]["fast_minus_g4_all"]["per_particle"]["difference_left_minus_right"]["rows"]["total"] == 5
    assert len(report["paired"]) == 3


def test_all43156_zero_step_denominator_and_uint64_identity_safety():
    n = 43156
    p = pd.DataFrame({"run": np.ones(n, dtype=np.uint64), "subrun": np.zeros(n, dtype=np.uint64),
                      "event": np.arange(n, dtype=np.uint64) + np.uint64(2**63 + 3),
                      "sp": np.ones(n, dtype=np.uint64)})
    ledger, _, report = fg.assemble({a: p for a in fg.FILES}, {a: steps().iloc[:0] for a in fg.FILES},
                                    geom(), np.zeros(3))
    pd.testing.assert_frame_equal(ledger[fg.TRACK], p)
    assert report["denominator_particles"] == n
    assert report["arms"]["fast"]["zero_step_particles"] == n


def test_foreign_same_count_particles_and_exact_high_keys_rejected():
    p = particles()
    with pytest.raises(ValueError, match="different entry"):
        fg.assemble({"fast": p, "g4_all": p.assign(event=p.event + np.uint64(1)), "g4_clock": p},
                    {a: steps() for a in fg.FILES}, geom(), np.zeros(3))
    for s in (steps().assign(event=steps().event + np.uint64(1)), steps().assign(sp=5)):
        with pytest.raises(ValueError, match="outside entry"):
            fg.audit_arm(s, p, geom(), np.zeros(3))
    with pytest.raises(ValueError, match="identities"):
        fg.audit_arm(steps().assign(event=steps().event.astype(float)), p, geom(), np.zeros(3))


@pytest.mark.parametrize("change,message", [({"sid": 999}, "foreign saved SID"),
                                           ({"sid": 1}, "labels disagree"),
                                           ({"panel": 1}, "labels disagree"),
                                           ({"straw": 0.5}, "integer identities")])
def test_never_proximity_foreign_or_inconsistent_own_sid(change, message):
    with pytest.raises(ValueError, match=message):
        fg.audit_arm(steps().assign(**change), particles(), geom(), np.zeros(3))


def test_own_sid_is_used_even_when_endpoints_lie_in_other_straw():
    s = steps().iloc[[2]].assign(sid=0, straw=0)
    per, grouped, report = fg.audit_arm(s, particles(), geom(), np.zeros(3))
    assert grouped.sid.tolist() == [0]
    assert per.distinct_straws.tolist() == [1, 0, 0]
    residual = report["endpoint_residuals"]
    assert residual["entry"]["radial_mm"]["quantiles"]["0.5"] == 3
    assert residual["exit"]["radial_mm"]["outside_beyond_0.01mm"] == 1


def test_exact_group_union_distinguishes_high_keys_present_only_in_one_arm():
    s = steps()
    other = particles().iloc[[2]].assign(sid=0, plane=0, panel=0, straw=0, len=1.,
                                        x0=0., y0=0., z0=0., x1=1., y1=0., z1=0.)
    ledger, _, report = assemble({"fast": s, "g4_all": other, "g4_clock": s.iloc[:0]})
    paired = report["paired"]["fast_minus_g4_all"]["per_particle_sid"]
    assert paired["paired_identities"] == 4
    assert paired["left_only"] == 3 and paired["right_only"] == 1
    assert ledger.fast_minus_g4_all_rows.tolist() == [3, 3, -1]
    assert ledger.event.dtype == np.uint64


def test_duplicate_geom_mapping_and_missing_payload_rejected():
    g = replace(geom(), straw=np.array([0, 0]))
    with pytest.raises(ValueError, match="duplicate geometry"):
        fg.audit_arm(steps(), particles(), g, np.zeros(3))
    with pytest.raises(ValueError, match="missing required"):
        fg.audit_arm(steps().drop(columns="sid"), particles(), geom(), np.zeros(3))
    with pytest.raises(ValueError, match="negative saved"):
        fg.audit_arm(steps().assign(len=-1.), particles(), geom(), np.zeros(3))


@pytest.mark.parametrize("column", ["x0", "z1", "len", "px"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_any_payload_rejected_no_silent_drops(column, bad):
    s = steps()
    s[column] = bad
    with pytest.raises(ValueError, match="nonfinite"):
        fg.audit_arm(s, particles(), geom(), np.zeros(3))


def test_native_origin_translation_and_no_inmemory_mutations():
    origin = np.array([-4., 3., 17.])
    s, p, g = steps(origin), particles(), geom()
    saved_s, saved_p, saved_mid = s.copy(deep=True), p.copy(deep=True), g.mid.copy()
    _, _, r = fg.audit_arm(s, p, g, origin)
    _, _, expected = fg.audit_arm(steps(), p, g, np.zeros(3))
    actual_residuals, expected_residuals = r.pop("endpoint_residuals"), expected.pop("endpoint_residuals")
    assert r == expected
    for end in ("entry", "exit"):
        assert actual_residuals[end]["outside_finite_cylinder_beyond_0.01mm"] == 1
        for axis in ("radial_mm", "axial_mm"):
            actual, reference = actual_residuals[end][axis], expected_residuals[end][axis]
            assert actual["outside_beyond_0.01mm"] == reference["outside_beyond_0.01mm"]
            assert actual["quantiles"] == pytest.approx(reference["quantiles"], abs=1e-12)
            assert actual["total"] == pytest.approx(reference["total"], abs=1e-12)
    pd.testing.assert_frame_equal(s, saved_s)
    pd.testing.assert_frame_equal(p, saved_p)
    np.testing.assert_array_equal(g.mid, saved_mid)


def saved(tmp_path, monkeypatch, empty=False):
    parent = tmp_path / "saved"
    parent.mkdir()
    origin = np.array([-4., 3., 17.])
    (parent / "fieldcompare.json").write_text(json.dumps({"origin_mm": origin.tolist(), "entries": 3}))
    for label, filename in fg.FILES.items():
        payload = {"particles": particles().rename(columns={"sp": "key"}).assign(pdg=13, parent=-1, isprim=1, ismcprim=1)}
        if not (empty and label == "g4_clock"):
            payload["sgsteps"] = steps(origin)
        tables.write(str(parent / filename), payload, as_ttree=True)
    gdml = tmp_path / "geom.gdml"
    gdml.write_text("synthetic geometry fixture")
    monkeypatch.setattr(fg.geometry, "from_gdml", lambda _: geom())
    return parent, gdml, origin


def test_cli_strict_json_compressed_ledgers_hashes_unchanged_exclusive(tmp_path, monkeypatch, capsys):
    parent, gdml, origin = saved(tmp_path, monkeypatch, empty=True)
    before = {p: p.read_bytes() for p in [*parent.iterdir(), gdml]}
    out = tmp_path / "ledger"
    fg.main(["--field-dir", str(parent), "--gdml", str(gdml), "--outdir", str(out)])
    report = json.loads((out / "fieldgasledger.json").read_text(),
                        parse_constant=lambda x: pytest.fail(f"non-strict JSON: {x}"))
    assert report["origin_mm"] == origin.tolist() and report["denominator_particles"] == 3
    assert report["arms"]["g4_clock"]["saved_step_tree_present"] is False
    assert report["limitations"] == list(fg.LIMITATIONS)
    assert "not entry" in report["momentum_bins"]
    for receipt in report["inputs"].values():
        assert receipt["sha256_before"] == receipt["sha256_after"] == fg._sha256(receipt["path"])
    for receipt in report["outputs"].values():
        assert receipt["sha256"] == fg._sha256(receipt["path"])
    ledger = pd.read_csv(out / "fieldgasledger_particles.csv.gz", dtype={k: "uint64" for k in fg.TRACK})
    pd.testing.assert_frame_equal(ledger[fg.TRACK], particles())
    assert ledger.g4_clock_rows.sum() == 0
    assert all(p.read_bytes() == b for p, b in before.items())
    with pytest.raises(FileExistsError):
        fg.run(parent, gdml, out)
    capsys.readouterr()


def test_cli_payload_hash_mutation_failclosed(tmp_path, monkeypatch):
    parent, gdml, _ = saved(tmp_path, monkeypatch)
    original = fg.assemble

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with (parent / fg.FILES["fast"]).open("ab") as stream:
            stream.write(b"mutation during audit")
        return result

    monkeypatch.setattr(fg, "assemble", mutate)
    out = tmp_path / "mutated"
    with pytest.raises(ValueError, match="changed during"):
        fg.run(parent, gdml, out)
    assert not (out / "fieldgasledger.json").exists()


@pytest.mark.parametrize("origin", [[1, 2], [1, 2, float("nan")]])
def test_cli_invalid_old_report_origin_never_writes_success(tmp_path, monkeypatch, origin):
    parent, gdml, _ = saved(tmp_path, monkeypatch)
    (parent / "fieldcompare.json").write_text(json.dumps({"origin_mm": origin, "entries": 3}))
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="origin"):
        fg.run(parent, gdml, out)
    assert not (out / "fieldgasledger.json").exists()
