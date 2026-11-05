"""Saved sampled-polylines only: exact genealogy, geometric runs and fail-closed receipts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import uproot

from mu2efastsim import tables
from trkfastsim import geometry
from trkfastsim.validation import fieldgaptrajectory as audit


KEY_VALUES = [2**54 + 1, 2**53 + 3, 2**63 + 7]
SP = 2**54 + 17


def keys(n, sp=SP):
    return pd.DataFrame({**{k: np.full(n, v, dtype="uint64") for k, v in zip(audit.KEY, KEY_VALUES)},
                         "sp": np.full(n, sp, dtype="uint64")})


def geom():
    return geometry.StrawGeometry(
        plane=np.array([0, 0]), panel=np.array([0, 0]), straw=np.array([0, 1]),
        mid=np.array([[0., 0., 0.], [0., 4., 0.]]), dir=np.array([[0., 0., 1.], [0., 0., 1.]]),
        halflen=np.array([10., 10.]), radius=1., wire_r=.1,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)),
        plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)))


def points(x=(-2., 2.), y=0., ipt=None, sp=SP):
    n = len(x)
    return keys(n, sp).assign(ipt=np.arange(n, dtype="uint64") if ipt is None else np.array(ipt, dtype="uint64"),
                              x=np.array(x, dtype=float), y=float(y), z=0., t=np.arange(n, dtype=float), ke=10.)


def raw_sgs():
    row = keys(1).assign(sid=np.uint64(0), t=np.float32(.25), len=np.float32(.12345679))
    for end in (0, 1):
        for axis in "xyz":
            row[f"{axis}{end}"] = np.float32(0.)
    row["x0"], row["x1"] = np.float32(-2.1234567), np.float32(2.1234567)
    return row


def saved_outliers():
    raw = raw_sgs()
    saved = audit.source.csv_representation(raw)
    saved["savedlen_mm"] = raw["len"].to_numpy(float)
    saved["originalsp"] = saved.sp
    saved["sp"] = np.uint64(1)
    saved["sourceoriginalrowindex"] = np.int64(0)
    both = pd.concat([saved.assign(arm=arm) for arm in audit.ARMS], ignore_index=True)
    return both


def targets():
    out = audit.targets_from_source(saved_outliers(), {"outlier_rows": dict(g4_all=1, g4_clock=1)})
    return audit.bind_sgs(out, raw_sgs(), np.zeros(3))


def summarize(p, target=None, **kwargs):
    audit.validate_points(p)
    result, count = audit.summarize(p, targets() if target is None else target, geom(), "full", **kwargs)
    return result.iloc[0], count


def test_both_arms_deduplicated_exact_large_uint64_identity():
    target = targets()
    assert len(target) == 1
    assert target.g4_all.tolist() == [True] and target.g4_clock.tolist() == [True]
    assert target.originalsp.iloc[0] == SP
    assert target.event.iloc[0] == KEY_VALUES[-1]
    assert all(target[k].dtype == np.uint64 for k in audit.TRACK)
    assert target.raw_sgs_t.iloc[0] == .25
    assert target.raw_sgs_x0_mu2e_mm.iloc[0] == float(raw_sgs().x0.iloc[0])
    with pytest.raises(ValueError, match="duplicate"):
        audit.targets_from_source(pd.concat([saved_outliers(), saved_outliers().iloc[[0]]]),
                                  {"outlier_rows": dict(g4_all=2, g4_clock=1)})
    bad = saved_outliers()
    bad.loc[1, "x0"] += 1e-8
    with pytest.raises(ValueError, match="payload"):
        audit.targets_from_source(bad, {"outlier_rows": dict(g4_all=1, g4_clock=1)})


def test_sgs_raw_float32_serialization_and_exact_binding():
    target = targets()
    assert target.x0.iloc[0] != float(raw_sgs().x0.iloc[0])
    for name, value in (("originalsp", SP + 1), ("sid", 1), ("event", 2), ("sourceoriginalrowindex", 1)):
        bad = target.copy()
        bad[name] = np.array([value], dtype="uint64")
        with pytest.raises(ValueError):
            audit.bind_sgs(bad, raw_sgs(), np.zeros(3))
    bad = target.copy()
    bad["x0"] += 1e-9
    with pytest.raises(ValueError, match="serialized endpoint"):
        audit.bind_sgs(bad, raw_sgs(), np.zeros(3))
    shifted = audit.bind_sgs(target, raw_sgs(), np.array([10., 20., 30.]))
    assert shifted.raw_sgs_x0_mu2e_mm.iloc[0] == float(raw_sgs().x0.iloc[0]) + 10


@pytest.mark.parametrize("p", [points(()), points((0.,))])
def test_zero_one_points_are_unavailable_not_failure(p):
    row, count = summarize(p)
    assert row.full_points == len(p)
    assert row.full_segments == count == 0
    assert row.full_polyline_gas_mm == 0
    assert row.full_unresolved
    assert "no exact" in row.full_availability or "one exact" in row.full_availability


def test_wire_split_and_broken_loop_geometric_runs_only():
    row, count = summarize(points((-2., 2., -2.)))
    assert row.full_segments == count == 2
    assert row.full_polyline_gas_mm == pytest.approx(3.6)
    assert row.full_positive_intervals == 4
    assert row.full_disconnected_runs == 4
    assert row.full_gas_segments == 2
    assert row.full_unresolved


def test_adjacent_segments_connected_and_sparse_indices_split_runs():
    p = points((-2., 0., 2.), y=.5)
    row, _ = summarize(p)
    assert row.full_polyline_gas_mm == pytest.approx(2 * np.sqrt(.75))
    assert row.full_positive_intervals == 2
    assert row.full_disconnected_runs == 1
    sparse = p.assign(ipt=np.array([0, 2, 3], dtype="uint64"))
    row, _ = summarize(sparse)
    assert row.full_disconnected_runs == 2
    assert row.full_polyline_gas_mm == pytest.approx(2 * np.sqrt(.75))
    huge = p.assign(ipt=np.array([2**63 + 1, 2**63 + 2, 2**63 + 3], dtype="uint64"))
    row, _ = summarize(huge)
    assert row.full_disconnected_runs == 1
    assert int(row.full_min_ipt) == 2**63 + 1
    assert int(row.full_max_ipt) == 2**63 + 3


def test_outside_gap_and_batch_boundary_do_not_merge_runs():
    row, _ = summarize(points((-2., 2., -2.), y=.5), batch=1)
    assert row.full_positive_intervals == 2
    assert row.full_disconnected_runs == 2
    row, _ = summarize(points((-2., 0., 2.), y=.5), batch=1)
    assert row.full_disconnected_runs == 1


def test_exact_particle_separation_and_own_target_many_to_many():
    # If grouped shift accidentally bridges particles, this creates a false hit.
    p = pd.concat([points((-2.,), sp=SP), points((2.,), sp=SP + 1)], ignore_index=True)
    alltargets = targets()
    other = alltargets.copy().assign(originalsp=np.uint64(SP + 1), target=1)
    alltargets = pd.concat([alltargets, other], ignore_index=True)
    rows, count = audit.summarize(p, alltargets, geom(), "full")
    assert count == 0 and rows.full_segments.tolist() == [0, 0]
    assert rows.full_points.tolist() == [1, 1]
    # Every targeted straw, but only for the exact same particle, gets each segment.
    other = targets().assign(sid=np.uint64(1), target=1)
    alltargets = pd.concat([targets(), other], ignore_index=True)
    rows, count = audit.summarize(points(), alltargets, geom(), "full")
    assert count == 2
    assert rows.full_polyline_gas_mm.tolist() == pytest.approx([1.8, 0.])
    assert rows.full_gas_segments.tolist() == [1, 0]
    # No proximity fallback to a nearby but different original particle key.
    row, count = summarize(points(sp=SP + 1))
    assert row.full_points == count == 0


def test_fixed_time_view_does_not_interpolate_or_invent_exit_clock():
    p = points((-2., 0., 2.), y=.5)
    row, count = summarize(p, after_entry=True)
    assert row.full_points == 2 and row.full_segments == count == 1
    assert row.full_min_t == 1 and row.full_max_t == 2
    assert row.full_polyline_gas_mm == pytest.approx(np.sqrt(.75))


@pytest.mark.parametrize("kind", ["floatkey", "negativeipt", "duplicate", "nan", "negativeke", "zerosp"])
def test_native_point_validation_fail_closed(kind):
    p = points()
    if kind == "floatkey":
        p["event"] = p.event.astype(float)
    elif kind == "negativeipt":
        p["ipt"] = np.array([-1, 0], dtype="int64")
    elif kind == "duplicate":
        p["ipt"] = np.uint64(0)
    elif kind == "nan":
        p.loc[0, "x"] = np.nan
    elif kind == "negativeke":
        p.loc[0, "ke"] = -1.
    else:
        p["sp"] = np.uint64(0)
    with pytest.raises(ValueError):
        audit.validate_points(p)


def make_source(tmp_path, p=None, empty_tree=False):
    table = tmp_path / "original.root"
    tables.write(str(table), {"sgsteps": raw_sgs(), **({"trajpts": p} if p is not None and len(p) else {})}, as_ttree=True)
    if empty_tree:
        with uproot.update(table) as root:
            root.mktree(f"{tables.DIR}/trajpts", {c: ("uint64" if c in audit.KEY + ["sp", "ipt"] else "float64")
                                               for c in audit.POINT_COLUMNS})
    art, gdml, old, mapping = [tmp_path / name for name in ("source.art", "geometry.gdml", "fieldcompare.json", "map.csv")]
    art.write_bytes(b"hash-only native ART fixture; no raw payload decoded")
    gdml.write_text("geometry fixture")
    old.write_text(json.dumps({"origin_mm": [0., 0., 0.]}))
    mapping.write_text("frozen genealogy receipt")
    inventory = dict(path=str(art), branches=[dict(name="MCTrajectories", typename="mu2e::MCTrajectories")],
                     raw_payload_read=False, raw_decoded=False, native_art_exact_link_certified=False,
                     presence_scope="branch presence only; neither nonempty nor decoded nor linked")
    invpath = tmp_path / "inventory.json"
    invpath.write_text(json.dumps(inventory))
    outliers = tmp_path / "outliers.csv.gz"
    saved_outliers().to_csv(outliers, index=False, compression="gzip")
    inputs = {"input_table": table, "source_art": art, "voids/inputs/gdml": gdml,
              "voids/inputs/fieldcompare": old}
    outputs = {"all_outliers": outliers, "renumber_map": mapping, "art_inventory": invpath}
    report = dict(passed=True, inputs_unchanged=True, origin_mm=[0., 0., 0.], art_inventory=inventory,
                  outlier_rows=dict(g4_all=1, g4_clock=1),
                  inputs={k: dict(path=str(v), sha256_before=audit.fg._sha256(v), sha256_after=audit.fg._sha256(v))
                          for k, v in inputs.items()},
                  outputs={k: dict(path=str(v), sha256=audit.fg._sha256(v)) for k, v in outputs.items()})
    path = tmp_path / "fieldsourceaudit.json"
    path.write_text(json.dumps(report))
    return path, report


@pytest.mark.parametrize("mode", ["missing", "empty", "one", "loop"])
def test_run_saved_only_availability_exclusive_outputs_and_hashes(tmp_path, monkeypatch, mode):
    p = points((0.,)) if mode == "one" else points((-2., 2., -2.)) if mode == "loop" else None
    path, source_report = make_source(tmp_path, p, empty_tree=mode == "empty")
    monkeypatch.setattr(audit.geometry, "from_gdml", lambda _: geom())
    out = tmp_path / "output"
    result = audit.run(path, out)
    assert result["passed"] and result["inputs_unchanged"] and result["all_outliers_unresolved"]
    assert result["unique_targets"] == 1 and result["source_outlier_rows"] == 2
    assert result["arm_memberships"] == dict(g4_all=1, g4_clock=1)
    assert not result["physical_passages_certified"] and not result["raw_art_decoded"]
    if mode == "missing":
        assert result["trajectory_availability"] == "original dump/trajpts tree absent"
    if mode == "empty":
        assert result["trajectory_availability"] == "original dump/trajpts tree present but zero points"
    if mode in ("missing", "empty"):
        assert result["targets_with_zero_points"] == 1
    if mode == "one":
        assert result["targets_with_one_point"] == 1
    if mode == "loop":
        assert result["targets_with_sampled_polyline"] == 1
    json.dumps(result, allow_nan=False)
    assert json.loads((out / "fieldgaptrajectory.json").read_text()) == result
    receipt = result["outputs"]["targets"]
    assert audit.fg._sha256(receipt["path"]) == receipt["sha256"]
    csv = pd.read_csv(receipt["path"], dtype={k: "uint64" for k in audit.TRACK})
    assert csv.originalsp.iloc[0] == SP and csv.event.iloc[0] == KEY_VALUES[-1]
    assert len(csv) == 1
    assert set(p.name for p in out.iterdir()) == {"fieldgaptrajectory.json", "fieldgaptrajectory_targets.csv.gz"}
    with pytest.raises(FileExistsError):
        audit.run(path, out)


@pytest.mark.parametrize("kind", ["failed", "changed", "missinginputs", "missingoutputs", "outputmutation", "inputmutation", "inventoryscope"])
def test_source_mandatory_receipts_and_inventory_scope(tmp_path, kind):
    path, report = make_source(tmp_path)
    if kind == "failed":
        report["passed"] = False
    elif kind == "changed":
        report["inputs_unchanged"] = False
    elif kind == "missinginputs":
        del report["inputs"]
    elif kind == "missingoutputs":
        del report["outputs"]
    elif kind == "outputmutation":
        Path(report["outputs"]["renumber_map"]["path"]).write_text("mutation")
    elif kind == "inputmutation":
        Path(report["inputs"]["source_art"]["path"]).write_bytes(b"mutation")
    else:
        report["art_inventory"]["raw_decoded"] = True
    with pytest.raises(ValueError):
        audit.verify_source(report)


def test_mutation_during_read_prevents_publication(tmp_path, monkeypatch):
    path, report = make_source(tmp_path, points())
    monkeypatch.setattr(audit.geometry, "from_gdml", lambda _: geom())
    original = audit.read_points

    def mutating_read(table):
        p, status = original(table)
        Path(report["inputs"]["voids/inputs/gdml"]["path"]).write_text("changed while reading")
        return p, status

    monkeypatch.setattr(audit, "read_points", mutating_read)
    with pytest.raises(ValueError, match="changed"):
        audit.run(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_cli_only_saved_source_report_and_exclusive_outdir(monkeypatch, capsys):
    monkeypatch.setattr(audit, "run", lambda **kwargs: dict(passed=True, **kwargs))
    audit.main(["--source-report", "source.json", "--outdir", "new-output"])
    result = json.loads(capsys.readouterr().out)
    assert result == dict(passed=True, source_report="source.json", outdir="new-output")


def test_bad_wire_radius_fails_even_without_trajectory_segments(tmp_path, monkeypatch):
    from dataclasses import replace
    path, _ = make_source(tmp_path)
    monkeypatch.setattr(audit.geometry, "from_gdml", lambda _: replace(geom(), wire_r=-.1))
    with pytest.raises(ValueError, match="wire outer radius"):
        audit.run(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_native_trajectory_coordinates_are_not_translated_by_sgs_origin():
    target = audit.bind_sgs(targets(), raw_sgs(), np.array([100., 200., 300.]))
    row, _ = summarize(points(), target)
    assert row.full_polyline_gas_mm == pytest.approx(1.8)


def test_no_outlier_targets_is_a_valid_empty_inventory():
    empty = audit.targets_from_source(saved_outliers().iloc[:0], {"outlier_rows": dict(g4_all=0, g4_clock=0)})
    empty = audit.bind_sgs(empty, raw_sgs(), np.zeros(3))
    result, count = audit.summarize(points(), empty, geom(), "full")
    assert result.empty and count == 0
