"""Saved-only exposure summaries: exact keys, fixed masks, closed receipts, unchanged inputs."""
import json

import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import fieldmaterialsummary as summary


def frames():
    n = 9
    p = pd.DataFrame({"run": np.full(n, 2**54 + 1, dtype="uint64"),
                      "subrun": np.full(n, 2**53 + 3, dtype="uint64"),
                      "event": np.array([2**63 + 7] * 2 + list(range(21, 28)), dtype="uint64"),
                      "sp": np.arange(2**54 + 11, 2**54 + 11 + n, dtype="uint64"),
                      "pentry_MeV_c": [0., 299.999, 300., 999.999, 1000., 1999.999, 2000., 4999.999, 5000.]})
    for c in summary.mp.COUNTS:
        p[c] = np.zeros(n, dtype="int64")
    for c in summary.mp.LENGTHS + summary.mp.MODELS:
        p[c] = np.zeros(n)
    p["missing_wall_mm"] = [0., 2., 0., 1., 0., 3., 0., 4., 5.]
    p["consumed_wall_mm"] = np.arange(n, dtype=float)
    p["geometric_wall_mm"] = p.missing_wall_mm + p.consumed_wall_mm
    p["boundary_wall_mm"] = p.missing_wall_mm
    for stem in ("missing", "consumed", "geometric"):
        p[stem + "_wall_x0"] = p[stem + "_wall_mm"] * .01
    p["mean_missing_collision_loss_MeV"] = p.missing_wall_mm * .2
    g = p[summary.TRACK].rename(columns={"sp": "originalsp"}).copy()
    g["injectorkey"] = np.array([1, 2] + [1] * (n - 2), dtype="uint64")
    g["entryp"] = p.pentry_MeV_c + .00001
    g["entrypx"] = np.float32(1.234567)
    g["originalparent"] = -1
    # Reverse baseline order deliberately and retain an event without entry particles.
    b = p[summary.KEY].drop_duplicates().iloc[::-1].reset_index(drop=True)
    b = pd.concat([b.iloc[:2], b.iloc[[0]].assign(event=np.uint64(999)), b.iloc[2:]], ignore_index=True)
    b["p_first"] = [2000., 4999.999, 2500., 5000., 1999.999, 300., 3000., 4500., 2000.]
    b["fast_ntrk"] = [1, 1, 1, 1, 0, 1, 0, 1, 1]
    b["g4_ntrk"] = [1, 1, 1, 0, 1, 1, 0, 1, 1]
    for arm in ("fast", "g4"):
        b[arm + "_sid"] = 1
        b[arm + "_nactive"] = 20
        b[arm + "_p_reco"] = 2500.
        b[arm + "_p_first_own"] = 2490.
    b.loc[5, "fast_sid"] = 2
    b["retained_note"] = np.arange(len(b))
    return p, g, b


def test_exact_uint64_one_to_one_multiplicity_and_order():
    p, g, b = frames()
    originals = [x.copy(deep=True) for x in (p, g, b)]
    joined = summary.join_particles(p, g.iloc[::-1], len(p))
    assert joined.originalsp.tolist() == p.sp.tolist()
    assert joined.source_minus_material_entryp_MeV_c.abs().max() > 0
    d, outside = summary.join_events(joined, b)
    pd.testing.assert_frame_equal(d[list(b.columns)], b)
    assert outside == 0
    multi = d[d.event == np.uint64(2**63 + 7)].iloc[0]
    assert multi.material_entry_particles == 2
    assert multi.material_missing_wall_mm == 2.
    zero = d[d.event == 999].iloc[0]
    assert zero.material_entry_particles == 0 and zero.material_missing_wall_mm == 0
    assert zero.material_zero_missing_wall and zero.material_no_entry_particles
    report = summary.event_summary(d)
    assert report["all_baseline"]["multi_entry"] == 1
    assert report["all_baseline"]["no_entry"] == 1
    assert sum(v["events"] for v in report["legacy_firstgood_joint"].values()) == len(b)
    assert "not primary-matched" in report["label"]
    for original, actual in zip(originals, (p, g, b)):
        pd.testing.assert_frame_equal(original, actual)


def test_fixed_material_bins_exact_edges_and_event_proxy_mask():
    p, g, b = frames()
    joined = summary.join_particles(p, g, len(p))
    report = summary.particle_summary(joined)
    assert report["edges"] == [0., 300., 1000., 2000., 5000., 1e9]
    assert [r["rows"] for r in report["bins"]] == [2, 2, 2, 2, 1]
    assert report["population"]["zero_missing"] == 4
    assert report["population"]["nonzero_missing"] == 5
    assert report["population"]["distributions"]["missing_wall_mm"]["total"] == 15.
    assert report["population"]["missing_fraction_of_total_geometric_wall"] == 15 / 51
    assert sum(r["fraction_population_missing_wall"] for r in report["bins"]) == pytest.approx(1.)
    assert sum(r["fraction_population_geometric_wall"] for r in report["bins"]) == pytest.approx(1.)
    d, _ = summary.join_events(joined, b)
    assert d.legacy_paired_2to5GeV_ownfirststraw_proxy.tolist() == [True, True, True, False, False, False, False, True, True]
    paired = summary.event_summary(d)["paired_2to5GeV_ownfirststraw_proxy"]
    assert paired["all"]["events"] == 5
    assert paired["all"]["events"] == paired["no_omitted"]["events"] + paired["has_omitted"]["events"]
    assert paired["all"]["multi_entry"] == 1
    assert paired["all"]["no_entry"] == 1
    b.loc[0, "fast_p_first_own"] = 0.
    b.loc[1, "g4_p_reco"] = np.nan
    d, _ = summary.join_events(joined, b)
    assert not d.legacy_paired_2to5GeV_ownfirststraw_proxy.iloc[:2].any()
    p.loc[0, "pentry_MeV_c"] = 1e9
    report = summary.particle_summary(summary.join_particles(p, g, len(p)))
    assert report["outside_fixed_bins"] == 1
    assert report["bins"][0]["rows"] == 1


def test_empty_zero_exposure_nulls_and_outside_baseline():
    p, g, b = frames()
    for c in summary.mp.LENGTHS + summary.mp.MODELS:
        p[c] = 0.
    joined = summary.join_particles(p, g, len(p))
    allzero = summary.particle_summary(joined)
    assert allzero["population"]["missing_fraction_of_total_geometric_wall"] is None
    assert all(r["fraction_population_missing_wall"] is None for r in allzero["bins"])
    empty = summary.exposure_summary(joined.iloc[:0])
    assert empty["distributions"]["missing_wall_mm"]["mean"] is None
    assert empty["distributions"]["missing_wall_mm"]["max"] is None
    assert set(empty["distributions"]["missing_wall_mm"]["quantiles"].values()) == {None}
    d, outside = summary.join_events(joined, b[b.event == 999])
    assert outside == 8 and len(d) == 1 and d.material_no_entry_particles.all()
    r = summary.event_summary(d)
    assert r["legacy_firstgood_joint"]["lost"]["events"] == 0
    json.dumps(r, allow_nan=False)
    d, outside = summary.join_events(joined, b.iloc[:0])
    assert d.empty and outside == 8
    json.dumps(summary.event_summary(d), allow_nan=False)


@pytest.mark.parametrize("change", ["foreign", "duplicate", "injectorzero", "injectorduplicate", "floatkey", "negative", "nonfinite"])
def test_reject_equal_count_foreign_and_invalid_identity(change):
    p, g, _ = frames()
    if change == "foreign":
        g.loc[0, "originalsp"] = np.uint64(77)
    elif change == "duplicate":
        g.loc[1, "originalsp"] = g.originalsp.iloc[0]
    elif change == "injectorzero":
        g.loc[0, "injectorkey"] = 0
    elif change == "injectorduplicate":
        g.loc[1, "injectorkey"] = 1
    elif change == "floatkey":
        p["event"] = p.event.astype(float)
    elif change == "negative":
        p.loc[0, "missing_wall_mm"] = -1.
    else:
        g.loc[0, "entryp"] = np.nan
    with pytest.raises(ValueError):
        summary.join_particles(p, g, len(p))


def fixture_evidence(tmp_path):
    p, g, b = frames()
    field, material, source = [tmp_path / n for n in ("field", "material", "source")]
    for directory in (field, material, source):
        directory.mkdir()
    (field / "fieldcompare.json").write_text(json.dumps(dict(entries=len(p))))
    for filename in summary.fc.FILES.values():
        (field / filename).write_text("retained payload receipt")
    original = tmp_path / "original.root"
    original.write_bytes(b"original native dumper")
    other = tmp_path / "other-receipt"
    other.write_text("verify even receipts not used directly")
    particlepaths = {"input_table": original, "other": other,
                     "original/fieldcompare.json": field / "fieldcompare.json"}
    particlepaths.update({f"original/{f}": field / f for f in summary.fc.FILES.values()})
    rec = summary.pa.receipts(particlepaths)
    particle = dict(passed=True, interpretation_allowed=True, inputs_unchanged=True,
                    original_field_dir=str(field), hashes_before=rec, hashes_after=rec,
                    reproduction=dict(reproduced=True, tables={f"{a}/{t}": dict(equal=True)
                                      for a in summary.fc.FILES for t in ("particles", "sgsteps")}),
                    global_material_comparison=dict(equal=True),
                    particle_material_stats=dict(consistent=True, entries=len(p),
                        totals={c: int(p[c].sum()) if c in summary.mp.COUNTS else float(p[c].sum())
                                for c in summary.mp.COUNTS + summary.mp.LENGTHS + summary.mp.MODELS}))
    p.to_csv(material / "particles.csv", index=False)
    g.to_csv(source / "fieldsourceaudit_renumber.csv.gz", index=False)
    inventory = source / "art.json"
    inventory.write_text("{}")
    inpaths = {"input_table": original, "unrelated": other,
               "voids/inputs/fieldcompare": field / "fieldcompare.json"}
    inpaths.update({"voids/inputs/" + a: field / f for a, f in summary.fc.FILES.items()})
    ins = {k: dict(path=str(v), sha256_before=summary.pa.sha256(v), sha256_after=summary.pa.sha256(v))
           for k, v in inpaths.items()}
    outliers = source / "outliers.csv.gz"
    outliers.write_text("unused output still verified")
    outs = summary.pa.receipts(dict(renumber_map=source / "fieldsourceaudit_renumber.csv.gz",
                                   art_inventory=inventory, all_outliers=outliers))
    src = dict(passed=True, inputs_unchanged=True, entry_particles=len(p), g4all_ordered_payload_exact=True,
               clock_payload_exact_except_event_nsgs=True,
               all_outlier_csv_payloads_and_recomputed_selection_exact=True, inputs=ins, outputs=outs)
    particlepath, sourcepath = material / "fieldmaterialparticles.json", source / "fieldsourceaudit.json"
    particlepath.write_text(json.dumps(particle))
    sourcepath.write_text(json.dumps(src))
    manifest = material.parent / "closed.sha256"
    manifest.write_text("".join(f"{summary.pa.sha256(path)}  {path.name}\n"
                                for path in (particlepath, material / "particles.csv")))
    baseline = tmp_path / "oldevents.csv.gz"
    b.to_csv(baseline, index=False)
    return particlepath, sourcepath, baseline


def test_cli_receipts_no_input_mutation_and_exclusive_output(tmp_path, capsys):
    args = fixture_evidence(tmp_path)
    snapshots = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    out = tmp_path / "summary"
    summary.main(["--particle-report", str(args[0]), "--source-report", str(args[1]),
                  "--baseline-events", str(args[2]), "--outdir", str(out)])
    report = json.loads((out / "fieldmaterialsummary.json").read_text())
    assert report["passed"] and report["inputs_unchanged"]
    assert report["hashes_before"] == report["hashes_after"]
    for receipt in report["hashes_before"].values():
        assert summary.pa.sha256(receipt["path"]) == receipt["sha256"]
    assert report["hashes_before"]["particles_csv"]["path"] == str(args[0].parent / "particles.csv")
    for path, content in snapshots.items():
        assert path.read_bytes() == content
    saved = summary.read_csv(out / "events.csv.gz", summary.KEY, legacy=True)
    original = summary.read_csv(args[2], summary.KEY, legacy=True)
    pd.testing.assert_frame_equal(saved[list(original.columns)], original)
    joined = summary.read_csv(out / "particles.csv.gz", summary.ORIGINAL + ["injectorkey"])
    assert joined.originalsp.tolist() == frames()[0].sp.tolist()
    for receipt in report["outputs"].values():
        assert summary.pa.sha256(receipt["path"]) == receipt["sha256"]
    with pytest.raises(FileExistsError):
        summary.run(*args, out)
    with pytest.raises(ValueError, match="exclusive"):
        summary.run(*args, args[0].parent / "nested")
    capsys.readouterr()


@pytest.mark.parametrize("change", ["particlepassed", "interpretation", "sourcepassed", "sixpayload", "beforeafter",
                                    "unrelatedreceipt", "sourceoutput", "fielddir", "inputpath", "foreignmap"])
def test_failclosed_corrupt_evidence(tmp_path, change):
    args = fixture_evidence(tmp_path)
    p, s = [json.loads(path.read_text()) for path in args[:2]]
    if change == "particlepassed":
        p["passed"] = False
    elif change == "interpretation":
        p["interpretation_allowed"] = False
    elif change == "sourcepassed":
        s["passed"] = False
    elif change == "sixpayload":
        p["reproduction"]["tables"]["fast/particles"]["equal"] = False
    elif change == "beforeafter":
        p["hashes_after"]["other"]["sha256"] = "bad"
    elif change == "unrelatedreceipt":
        (tmp_path / "other-receipt").write_text("corrupt")
    elif change == "sourceoutput":
        (args[1].parent / "art.json").write_text("corrupt")
    elif change == "fielddir":
        s["inputs"]["voids/inputs/fast"] = s["inputs"]["voids/inputs/g4_all"]
    elif change == "inputpath":
        s["inputs"]["input_table"] = s["inputs"]["unrelated"]
    else:
        path = args[1].parent / "fieldsourceaudit_renumber.csv.gz"
        g = summary.read_csv(path, summary.ORIGINAL + ["injectorkey"])
        g.loc[0, "originalsp"] = np.uint64(77)
        g.to_csv(path, index=False)
        s["outputs"]["renumber_map"]["sha256"] = summary.pa.sha256(path)
    args[0].write_text(json.dumps(p))
    args[1].write_text(json.dumps(s))
    out = tmp_path / "summary"
    with pytest.raises(ValueError):
        summary.run(*args, out)
    assert not (out / "fieldmaterialsummary.json").exists()


@pytest.mark.parametrize("when", ["join", "output"])
def test_input_mutation_during_processing_fails_closed(tmp_path, monkeypatch, when):
    args = fixture_evidence(tmp_path)
    target = tmp_path / "other-receipt"
    if when == "join":
        old = summary.join_particles
        def mutate(*a, **kw):
            result = old(*a, **kw)
            target.write_text("changed during processing")
            return result
        monkeypatch.setattr(summary, "join_particles", mutate)
    else:
        old = pd.DataFrame.to_csv
        def mutate(*a, **kw):
            result = old(*a, **kw)
            target.write_text("changed during publication")
            return result
        monkeypatch.setattr(pd.DataFrame, "to_csv", mutate)
    out = tmp_path / "summary"
    with pytest.raises(ValueError, match="changed during"):
        summary.run(*args, out)
    assert not (out / "fieldmaterialsummary.json").exists()


def test_decimal_uint64_max_roundtrip_no_float_normalization(tmp_path):
    path = tmp_path / "native.csv"
    path.write_text("run,subrun,event,sp\n18446744073709551615,9007199254740993,9223372036854775815,18446744073709551614\n")
    got = summary.read_csv(path, summary.TRACK)
    assert got.run.iloc[0] == np.uint64(2**64 - 1)
    assert got.sp.iloc[0] == np.uint64(2**64 - 2)
    assert all(got[c].dtype == np.uint64 for c in summary.TRACK)
    path.write_text("run,subrun,event,sp\n1,2,9007199254740993.0,4\n")
    with pytest.raises(ValueError, match="invalid identity"):
        summary.read_csv(path, summary.TRACK)
    path.write_text("run,subrun,event,sp\n1,2,18446744073709551616,4\n")
    with pytest.raises(ValueError, match="out-of-range"):
        summary.read_csv(path, summary.TRACK)


@pytest.mark.parametrize("change", ["missing_source_output", "total", "count", "csv"])
def test_full_saved_population_and_unreceipted_ledger_binding(tmp_path, change):
    args = fixture_evidence(tmp_path)
    particle, source = [json.loads(path.read_text()) for path in args[:2]]
    if change == "missing_source_output":
        del source["outputs"]["all_outliers"]
    elif change == "total":
        particle["particle_material_stats"]["totals"]["missing_wall_mm"] += 1
    elif change == "count":
        source["entry_particles"] -= 1
    else:
        path = args[0].parent / "particles.csv"
        ledger = summary.read_csv(path, summary.TRACK)
        ledger.loc[0, "mean_missing_collision_loss_MeV"] += 1
        ledger.to_csv(path, index=False)
    args[0].write_text(json.dumps(particle))
    args[1].write_text(json.dumps(source))
    out = tmp_path / "summary"
    with pytest.raises(ValueError):
        summary.run(*args, out)
    assert not (out / "fieldmaterialsummary.json").exists()


def test_closed_output_receipt_rejects_redistributed_particle_exposure(tmp_path):
    args = fixture_evidence(tmp_path)
    path = args[0].parent / "particles.csv"
    p = summary.read_csv(path, summary.TRACK)
    cols = list(summary.mp.LENGTHS + summary.mp.MODELS)
    before = p[cols].sum()
    p.loc[[0, 1], cols] = p.loc[[1, 0], cols].to_numpy()
    pd.testing.assert_series_equal(p[cols].sum(), before)
    p.to_csv(path, index=False)
    with pytest.raises(ValueError, match="closed-manifest hash mismatch"):
        summary.run(*args, tmp_path / "rejected")


def test_closed_output_manifest_required(tmp_path):
    args = fixture_evidence(tmp_path)
    (args[0].parent.parent / "closed.sha256").unlink()
    with pytest.raises(FileNotFoundError):
        summary.run(*args, tmp_path / "rejected")
