"""Exact genealogy and metadata-only native ART inventory; no transport or raw streamer reads."""
import json

import numpy as np
import pandas as pd
import pytest
import uproot

from mu2efastsim import tables
from trkfastsim import geometry
from trkfastsim.validation import fieldsourceaudit as audit


def fixture_frames():
    keys = pd.DataFrame({"run": np.array([2**54 + 1] * 3, dtype="uint64"),
                         "subrun": np.array([2**53 + 3] * 3, dtype="uint64"),
                         "event": np.array([2**63 + 7] * 3, dtype="uint64"),
                         "sp": np.array([2**32 + 11, 2**54 + 13, 2**54 + 17], dtype="uint64")})
    part = keys.assign(pdg=np.int32(13))
    vd = part.iloc[[1, 0, 2]].copy().reset_index(drop=True)
    vd = vd.assign(x=1., y=2., z=3., px=np.float32(1.234567), py=np.float32(2.), pz=np.float32(3.),
                   t=[20., 10., 8.], vdid=np.int32(13))
    surf = part.iloc[[0, 1]].copy().reset_index(drop=True)
    surf = surf.assign(surf=np.int32(0), t=[5., 30.], px=np.float32(7.), py=np.float32(8.), pz=np.float32(9.))
    for end in (0, 1):
        for axis in "xyz":
            surf[f"{axis}{end}"] = np.float32(.125 * (end + 1))
    steps = part.iloc[[1, 0, 1, 0, 2]].copy().reset_index(drop=True)
    steps = steps.assign(nsgs=np.int32(5), sid=np.array([1, 0, 0, 1, 0], dtype="uint64"),
                         plane=np.int32(0), panel=np.int32(0), straw=np.array([1, 0, 0, 1, 0], dtype="int32"),
                         upanel=np.int32(0), shape=np.int32(1), ion=np.int32(1), eion=np.float32(.01),
                         len=np.float32(.12345679), width=np.float32(.03), t=[22., 6., 19., 4., 10.],
                         px=np.float32(1.234567), py=np.float32(2.), pz=np.float32(3.))
    for axis in "xyz":
        steps[f"{axis}0"] = np.float32(0.)
        steps[f"{axis}1"] = np.float32(0.)
    steps["x0"] = np.float32(.5)
    steps["x1"] = np.float32(.5)
    steps["z0"] = np.float32(-2.)
    steps["z1"] = np.float32(2.)
    source = steps.assign(sourceoriginalrowindex=np.array([8, 10, 11, 20, 25], dtype="int64"))
    return part, steps, vd, surf, source


def rebuild():
    p, s, v, u, source = fixture_frames()
    return audit.reconstruct(p, s, v, u, np.zeros(3), [13], source)


def geom():
    return geometry.StrawGeometry(
        plane=np.array([0, 0]), panel=np.array([0, 0]), straw=np.array([0, 1]),
        mid=np.zeros((2, 3)), dir=np.array([[0., 0., 1.], [0., 0., 1.]]),
        halflen=np.array([10., 10.]), radius=1., wire_r=.1,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)),
        plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)))


def test_firstentry_frozen_concat_and_renumber_source_uint64():
    r = rebuild()
    assert r["entry"].t.tolist() == [5., 8., 20.]
    assert r["entry"].vdid.tolist() == [-1, 13, 13]
    assert r["entry"].x0.notna().tolist() == [True, False, True]
    assert r["genealogy"].originalsp.tolist() == [2**32 + 11, 2**54 + 13, 2**54 + 17]
    assert r["genealogy"].injectorkey.tolist() == [1, 2, 3]
    assert all(r["genealogy"][k].dtype == np.uint64 for k in audit.KEY + ["originalsp"])
    assert r["genealogy"].entryvd.tolist() == [-1, 13, 13]
    assert r["genealogy"].entrypx.tolist() == [7., float(np.float32(1.234567)), float(np.float32(1.234567))]
    assert r["source"].sourceoriginalrowindex.tolist() == [8, 10, 25]
    assert r["source"].originalsp.tolist() == [2**54 + 13, 2**32 + 11, 2**54 + 17]
    assert r["sgsteps"].sp.tolist() == [2, 1, 3]
    assert r["sgsteps"].nsgs.tolist() == [3, 3, 3]
    assert list(r["sgsteps"].columns) == audit.EXPORT
    assert "sourceoriginalrowindex" not in r["sgsteps"]
    assert r["sgsteps"].px.dtype == np.float32


def test_exact_ordered_frozen_export_roundtrip_no_dtype_relaxation(tmp_path):
    r = rebuild()
    path = tmp_path / "frozen.root"
    tables.write(str(path), {k: r[k] for k in ("particles", "sgsteps")}, as_ttree=True)
    for tree in ("particles", "sgsteps"):
        audit.exact(tables.read(path, tree), r[tree])
    corrupt = r["sgsteps"].copy()
    corrupt.loc[0, "px"] = np.nextafter(corrupt.px.iloc[0], np.float32(np.inf))
    with pytest.raises(AssertionError):
        audit.exact(corrupt, r["sgsteps"])
    with pytest.raises(AssertionError):
        audit.exact(r["sgsteps"].astype({"px": "float64"}), r["sgsteps"])
    with pytest.raises(AssertionError):
        audit.exact(r["sgsteps"].iloc[::-1].reset_index(drop=True), r["sgsteps"])
    with pytest.raises(AssertionError):
        audit.exact(r["sgsteps"].reindex(columns=sorted(audit.EXPORT)), r["sgsteps"])


def clock():
    all_steps = rebuild()["sgsteps"]
    cut = all_steps.iloc[[0, 2]].reset_index(drop=True).assign(nsgs=np.int64(2))
    return all_steps, cut


def test_clock_exact_unique_index_and_per_event_nsgs():
    all_steps, cut = clock()
    assert audit.clock_index(all_steps, cut).tolist() == [0, 2]
    with pytest.raises(ValueError, match="duplicate"):
        audit.clock_index(pd.concat([all_steps, all_steps.iloc[[0]]]), cut)
    with pytest.raises(ValueError, match="duplicate"):
        audit.clock_index(all_steps, pd.concat([cut, cut.iloc[[0]]]))
    with pytest.raises(ValueError, match="foreign"):
        audit.clock_index(all_steps, cut.assign(event=cut.event + np.uint64(1)))
    assert audit.clock_index(all_steps, cut.iloc[::-1].reset_index(drop=True)).tolist() == [2, 0]
    with pytest.raises(ValueError, match="column/order"):
        audit.clock_index(all_steps, cut.reindex(columns=sorted(cut.columns)))
    with pytest.raises(AssertionError):
        audit.clock_index(all_steps, cut.assign(nsgs=np.int64(3)))
    with pytest.raises(AssertionError):
        audit.clock_index(all_steps, cut.assign(t=cut.t + 1e-12))


def test_complete_outlier_csv_roundtrip_float32_and_large_keys(tmp_path):
    r = rebuild()
    den = r["particles"].rename(columns={"key": "sp"})
    _, expected, _ = audit.voids.audit_arm(r["sgsteps"], den, geom(), np.zeros(3))
    path = tmp_path / "outliers.csv.gz"
    expected.to_csv(path, index=False)
    got, report = audit.validate_outliers(path, r["sgsteps"], den, geom(), np.zeros(3), "test")
    assert report["all_outlier_rows"] == 3
    assert got.event.tolist() == [2**63 + 7] * 3
    assert got.originalrowindex.tolist() == [0, 1, 2]
    assert got.px.iloc[0] != float(r["sgsteps"].px.iloc[0])
    audit.exact(got, audit.csv_representation(expected))
    for column, value in (("x0", .6), ("savedlen_mm", .2), ("residual_mm", 0.), ("sp", 999), ("originalrowindex", 1)):
        bad = got.copy()
        bad.loc[0, column] = value
        bad.to_csv(path, index=False)
        with pytest.raises(AssertionError):
            audit.validate_outliers(path, r["sgsteps"], den, geom(), np.zeros(3), "test")
    expected.iloc[::-1].to_csv(path, index=False)
    with pytest.raises(AssertionError):
        audit.validate_outliers(path, r["sgsteps"], den, geom(), np.zeros(3), "test")


def test_native_inventory_highest_events_branches_absences_never_arrays(tmp_path, monkeypatch):
    path = tmp_path / "source.art"
    with uproot.recreate(path) as root:
        root.mktree("Events", {"old_StepPointMC": "int32"})
        root.mktree("MetaData", {"productRegistry": "int32"})
    with uproot.update(path) as root:
        root.mktree("Events", {"mu2e_StrawGasSteps_g4": "int32", "mu2e_SimParticles_g4": "int32",
                               "mu2e_Assns_g4": "int32", "mu2e_StatusG4_g4": "int32"})
    def no_arrays(*args, **kwargs):
        pytest.fail("native ART payload must not be decoded")
    monkeypatch.setattr(uproot.behaviors.TTree.TTree, "arrays", no_arrays)
    result = audit.art_inventory(path)
    assert result["events_cycle"] == "Events;2" and result["num_entries"] == 0
    assert result["candidate_branches"]["StrawGasStep"]
    assert result["candidate_branches"]["Assns"]
    assert "StepPointMC" in result["absent_candidates"]
    assert "SurfaceStep" in result["absent_candidates"]
    assert result["productRegistry_metadata_present"]
    assert all(b["typename"] == "int32_t" for b in result["branches"])
    assert result["raw_decoded"] is False and result["native_art_exact_link_certified"] is False
    assert "neither nonempty" in result["presence_scope"]


def saved(tmp_path, monkeypatch):
    p, s, v, u, _ = fixture_frames()
    original = tmp_path / "original.root"
    parts = p.rename(columns={"sp": "key"}).assign(parent=np.int64(-1), isprim=np.int32(1), ismcprim=np.int32(1))
    tables.write(str(original), {"particles": parts, "sgsteps": s.drop(columns="pdg"),
                                 "vdsteps": v.drop(columns="pdg"), "surfsteps": u.drop(columns="pdg")}, as_ttree=True)
    s = s[[c for c in s.columns if c != "pdg"] + ["pdg"]]
    source = s.assign(sourceoriginalrowindex=np.arange(len(s), dtype="int64"))
    r = audit.reconstruct(p, s, v, u, np.zeros(3), [13], source)
    field = tmp_path / "frozen"
    field.mkdir()
    settings = dict(species=13, inputs=1, seed=42, chord_mm=20., sagitta_mm=.02,
                    gas_state="gas", mu_radiative=False, origin_mm=[0., 0., 0.], entry_vds=[13],
                    field="DS.map", entries=3)
    (field / "fieldcompare.json").write_text(json.dumps(settings))
    for label, filename in audit.voids.FILES.items():
        frame = r["sgsteps"] if label != "g4_clock" else r["sgsteps"].iloc[[0, 2]].reset_index(drop=True).assign(nsgs=np.int64(2))
        tables.write(str(field / filename), {"particles": r["particles"], "sgsteps": frame}, as_ttree=True)
    gdml = tmp_path / "geometry.gdml"
    gdml.write_text("synthetic")
    monkeypatch.setattr(audit.geometry, "from_gdml", lambda _: geom())
    voiddir = tmp_path / "voids"
    audit.voids.run(field, gdml, voiddir)
    art = tmp_path / "original.art"
    with uproot.recreate(art) as root:
        root.mktree("Events", {"mu2e_StrawGasSteps_g4": "int32"})
    return original, field, voiddir, art


def test_cli_exact_source_map_and_receipts_no_overwrite(tmp_path, monkeypatch, capsys):
    original, field, voiddir, art = saved(tmp_path, monkeypatch)
    out = tmp_path / "audit"
    audit.main(["--input-table", str(original), "--field-dir", str(field), "--outlier-dir", str(voiddir),
                "--source-art", str(art), "--outdir", str(out)])
    report = json.loads((out / "fieldsourceaudit.json").read_text())
    assert report["passed"] and report["inputs_unchanged"]
    assert report["exact_g4all_rows"] == 3 and report["exact_clock_subset_rows"] == 2
    rows = pd.read_csv(out / "fieldsourceaudit_outliers.csv.gz", dtype={k: "uint64" for k in audit.KEY + ["originalsp"]})
    assert rows.sourceoriginalrowindex.tolist() == [0, 1, 4, 0, 4]
    assert rows.originalsp.tolist() == [2**54 + 13, 2**32 + 11, 2**54 + 17, 2**54 + 13, 2**54 + 17]
    assert not rows.native_art_exact_link_certified.any()
    assert rows.originalparent.tolist() == [-1] * 5
    for receipt in report["inputs"].values():
        assert receipt["sha256_before"] == receipt["sha256_after"] == audit.voids.fg._sha256(receipt["path"])
    with pytest.raises(FileExistsError):
        audit.run(original, field, voiddir, art, out)
    capsys.readouterr()


@pytest.mark.parametrize("change", ["outlier", "saved", "prior_input_hash", "foreign_path"])
def test_failclosed_prior_receipts(tmp_path, monkeypatch, change):
    original, field, voiddir, art = saved(tmp_path, monkeypatch)
    prior_path = voiddir / "fieldgasvoids.json"
    prior = json.loads(prior_path.read_text())
    if change == "outlier":
        with (voiddir / "fieldgasvoids_g4_all_outliers.csv.gz").open("ab") as stream:
            stream.write(b"corrupt")
    elif change == "saved":
        with (field / audit.voids.FILES["g4_all"]).open("ab") as stream:
            stream.write(b"corrupt")
    elif change == "prior_input_hash":
        prior["inputs"]["g4_all"]["sha256_after"] = "wrong"
        prior_path.write_text(json.dumps(prior))
    else:
        prior["outputs"]["g4_all_outliers"] = prior["outputs"]["g4_clock_outliers"]
        prior_path.write_text(json.dumps(prior))
    out = tmp_path / "audit"
    with pytest.raises(ValueError, match="prior void-audit"):
        audit.run(original, field, voiddir, art, out)
    assert not (out / "fieldsourceaudit.json").exists()


@pytest.mark.parametrize("when", ["reconstruct", "output"])
def test_hash_beforeafter_failclosed(tmp_path, monkeypatch, when):
    original, field, voiddir, art = saved(tmp_path, monkeypatch)
    def mutate():
        with art.open("ab") as stream:
            stream.write(b"changed")
    if when == "reconstruct":
        old = audit.reconstruct
        def wrapped(*args, **kwargs):
            result = old(*args, **kwargs)
            mutate()
            return result
        monkeypatch.setattr(audit, "reconstruct", wrapped)
    else:
        old = pd.DataFrame.to_csv
        def wrapped(*args, **kwargs):
            result = old(*args, **kwargs)
            if "compression" in kwargs:
                mutate()
            return result
        monkeypatch.setattr(pd.DataFrame, "to_csv", wrapped)
    out = tmp_path / "audit"
    with pytest.raises(ValueError, match="changed during"):
        audit.run(original, field, voiddir, art, out)
    assert not (out / "fieldsourceaudit.json").exists()


def test_source_row_position_precedes_primary_selection(tmp_path, monkeypatch):
    original, field, voiddir, art = saved(tmp_path, monkeypatch)
    raw = tables.read(original, "sgsteps")
    foreign = raw.iloc[[0]].copy().assign(sp=np.uint64(999))
    trees = {tree: tables.read(original, tree) for tree in ("particles", "vdsteps", "surfsteps")}
    trees["sgsteps"] = pd.concat([foreign, raw.iloc[:2], foreign, raw.iloc[2:]], ignore_index=True)
    tables.write(str(original), trees, as_ttree=True)
    out = tmp_path / "audit"
    with pytest.warns(UserWarning, match="merging on int and float"):
        audit.run(original, field, voiddir, art, out)
    rows = pd.read_csv(out / "fieldsourceaudit_outliers.csv.gz")
    assert rows.sourceoriginalrowindex.tolist() == [1, 2, 6, 1, 6]


@pytest.mark.parametrize("change", ["value", "columns", "rows", "same_count_foreign", "particle_value"])
def test_payload_failclosed_even_with_updated_prior_receipts(tmp_path, monkeypatch, change):
    original, field, voiddir, art = saved(tmp_path, monkeypatch)
    path = field / audit.voids.FILES["g4_all"]
    steps = tables.read(path, "sgsteps")
    parts = tables.read(path, "particles")
    if change == "value":
        steps.loc[0, "px"] = np.nextafter(steps.px.iloc[0], np.float32(np.inf))
    elif change == "columns":
        steps = steps.reindex(columns=sorted(steps.columns))
    elif change == "rows":
        steps = steps.iloc[::-1].reset_index(drop=True)
    elif change == "same_count_foreign":
        steps.loc[0, "sp"] = 999
    else:
        parts.loc[0, "pdg"] = -13
    tables.write(str(path), {"particles": parts, "sgsteps": steps}, as_ttree=True)
    prior_path = voiddir / "fieldgasvoids.json"
    prior = json.loads(prior_path.read_text())
    receipt = prior["inputs"]["g4_all"]
    receipt["sha256_before"] = receipt["sha256_after"] = audit.voids.fg._sha256(path)
    prior_path.write_text(json.dumps(prior))
    out = tmp_path / "audit"
    with pytest.raises(AssertionError):
        audit.run(original, field, voiddir, art, out)
    assert not (out / "fieldsourceaudit.json").exists()


@pytest.mark.parametrize("selection", [dict(species=11), dict(entry="surface"), dict(census={"self_pass": {}}),
                                       dict(census={"unbiased": {}}), dict(max_events=1)])
def test_unsupported_frozen_selection(tmp_path, monkeypatch, selection):
    original, field, voiddir, art = saved(tmp_path, monkeypatch)
    path = field / "fieldcompare.json"
    settings = json.loads(path.read_text())
    settings.update(selection)
    path.write_text(json.dumps(settings))
    prior_path = voiddir / "fieldgasvoids.json"
    prior = json.loads(prior_path.read_text())
    receipt = prior["inputs"]["fieldcompare"]
    receipt["sha256_before"] = receipt["sha256_after"] = audit.voids.fg._sha256(path)
    prior_path.write_text(json.dumps(prior))
    out = tmp_path / "audit"
    with pytest.raises(ValueError):
        audit.run(original, field, voiddir, art, out)
    assert not (out / "fieldsourceaudit.json").exists()


def test_inventory_absent_registry_and_products_and_missing_events(tmp_path):
    art = tmp_path / "empty.art"
    with uproot.recreate(art) as root:
        root.mktree("Events", {"unrelated": "float32"})
    report = audit.art_inventory(art)
    assert report["absent_candidates"] == list(audit.PRODUCTS)
    assert not report["productRegistry_metadata_present"]
    assert not report["raw_payload_read"]
    with uproot.recreate(art) as root:
        root.mktree("Other", {"unrelated": "float32"})
    with pytest.raises(ValueError, match="no Events"):
        audit.art_inventory(art)


def test_empty_outlier_csv_complete_schema(tmp_path):
    r = rebuild()
    den = r["particles"].rename(columns={"key": "sp"})
    steps = r["sgsteps"].assign(len=np.float32(4.))
    _, expected, _ = audit.voids.audit_arm(steps, den, geom(), np.zeros(3))
    assert expected.empty
    path = tmp_path / "empty.csv.gz"
    expected.to_csv(path, index=False)
    actual, report = audit.validate_outliers(path, steps, den, geom(), np.zeros(3), "empty")
    assert actual.empty and report["all_outlier_rows"] == 0
