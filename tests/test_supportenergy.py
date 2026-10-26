"""Hash-pinned energy runner consumes complete paired paths without regeneration or silent filtering."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trkfastsim import pairedenergy as energy, transportledger
from trkfastsim.analyses import supportenergy as runner


@pytest.fixture
def saved(tmp_path):
    directory = tmp_path / "envelope"
    directory.mkdir()
    gdml = tmp_path / "geometry.gdml"
    gdml.write_text("immutable geometry identity; not loaded by energy")
    identity = dict(run=[1, 1], subrun=[2, 2], event=[9007199254740993]*2, sp=[1, 2])
    seeds = pd.DataFrame(identity | dict(x=[0., 0.], y=[0., 0.], z=[0., 0.], px=[200., 400.], py=[0., 0.], pz=[0., 0.],
                                      length_mm=[5., 5.], reference_surface_row=[13, 14], surf=[0, 2], t=[2., 3.]))
    seeds.to_csv(directory / "seeds.csv", index=False)
    prefix = "envelope_1e-10mm"
    intervals = pd.DataFrame(dict(lane=[0, 0, 1, 1], s0_mm=[0., 2., 0., 2.], s1_mm=[2., 5., 2., 5.], material=["G4_Al"]*4))
    intervals = intervals.merge(seeds[energy.KEY].assign(lane=[0, 1]), on="lane", validate="many_to_one")
    for arm in ("inner", "outer"):
        intervals.to_csv(directory / f"{prefix}_{arm}_intervals.csv", index=False)
    decisions = seeds[energy.KEY].assign(lane=[0, 1], requested_length_mm=5., bounded_pair_admitted=True, admitted_for_transport=False)
    decisions.to_csv(directory / f"{prefix}_decisions.csv", index=False)
    pd.DataFrame(columns=["lane", "reason"] + energy.KEY).to_csv(directory / f"{prefix}_refusals.csv", index=False)
    summary = dict(schema="trkfastsim.support-envelope-replay.v1", complete=True, mode="support-navigation-torus-envelope-v1",
        admitted_for_transport=False, candidates=[dict(endpoint_budget_mm=1e-10, all_pairs_admitted=True, refused_rays=0,
            bounded_pair_rays=2, inner_interval_rows=4, outer_interval_rows=4)],
        gdml=dict(path=str(gdml), sha256=energy.sha256(gdml)), seed_rows=2, seed_domain=dict(selected_muons=2),
        inputs={"seeds.csv": dict(path=str(directory / "seeds.csv"), sha256=energy.sha256(directory / "seeds.csv"))},
        control_inputs={}, settings=dict(test=True), sources_sha256={})
    (directory / "summary.json").write_text(json.dumps(summary))
    pin = tmp_path / "pin.json"
    info = energy.pin_inputs(directory, pin, endpoint_budget_mm=1e-10)
    return dict(pin_path=pin, pin_sha256=info["sha256"], outdir=tmp_path / "energy", tracker_ref="1"*40, core_ref="2"*40)


def test_complete_paired_run_mean_no_sampling_full_identities_and_no_overwrite(saved, monkeypatch):
    from mu2efastsim import eloss
    monkeypatch.setattr(eloss, "sample_loss", lambda *a: pytest.fail("mean mode must not sample"))
    result = runner.run(**saved)
    assert result["seed_rows"] == 2 and result["complete"]
    assert result["comparison"]["max_absolute_final_ke_difference_MeV"] == 0.
    assert result["arms"]["inner"]["canonical_interval_rows"] == 2
    summary = json.loads((saved["outdir"] / "summary.json").read_text())
    assert not summary["production_admitted"] and not summary["geometry_admitted_for_transport"]
    assert summary["settings"]["loss_model"] == "mean" and summary["settings"]["timing"] == "unknown"
    for name, digest in summary["outputs_sha256"].items():
        assert energy.sha256(saved["outdir"] / name) == digest
    for arm in ("inner", "outer"):
        table = pd.read_csv(saved["outdir"] / f"{arm}_final.csv", dtype={"event": "int64"})
        assert table.event.tolist() == [9007199254740993]*2 and table.sp.tolist() == [1, 2]
        assert table.reference_surface_row.tolist() == [13, 14]
        history = transportledger.SegmentHistory.load(saved["outdir"] / f"{arm}_history.npz")
        assert history.metadata["settings"]["geometry_envelope"] == arm
        assert np.isnan(history["deposited_MeV"]).all()
        assert np.isnan(history["dt_ns"]).all()
        assert history.accounting()["primary_residual_MeV"].max() == 0.
    with pytest.raises(FileExistsError):
        runner.run(**saved)


def test_pin_cannot_be_overwritten(saved):
    pin = json.loads(saved["pin_path"].read_text())
    with pytest.raises(FileExistsError):
        energy.pin_inputs(pin["directory"], saved["pin_path"], endpoint_budget_mm=1e-10)


@pytest.mark.parametrize("file", ["seeds.csv", "summary.json", "envelope_1e-10mm_inner_intervals.csv", "envelope_1e-10mm_outer_intervals.csv",
                                    "envelope_1e-10mm_decisions.csv", "envelope_1e-10mm_refusals.csv"])
def test_mutated_pinned_inputs_abort_before_output(saved, file):
    pin = json.loads(saved["pin_path"].read_text())
    with (Path(pin["directory"]) / file).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.run(**saved)
    assert not saved["outdir"].exists()


def repin_mutation(saved, filename, change):
    from pathlib import Path
    pin = json.loads(saved["pin_path"].read_text())
    directory = Path(pin["directory"])
    path = directory / filename
    change(path)
    pin["files"][filename] = energy.sha256(path)
    if filename == "seeds.csv":
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["inputs"]["seeds.csv"]["sha256"] = energy.sha256(path)
        summary_path.write_text(json.dumps(summary))
        pin["files"]["summary.json"] = energy.sha256(summary_path)
    saved["pin_path"].write_text(json.dumps(pin))
    saved["pin_sha256"] = energy.sha256(saved["pin_path"])


@pytest.mark.parametrize("column,value", [("bounded_pair_admitted", False), ("admitted_for_transport", True),
    ("requested_length_mm", 6.), ("lane", 9), ("sp", 9)])
def test_forged_but_repinned_decision_schema_still_refused(saved, column, value):
    def change(path):
        table = pd.read_csv(path)
        table.loc[0, column] = value
        table.to_csv(path, index=False)
    repin_mutation(saved, "envelope_1e-10mm_decisions.csv", change)
    with pytest.raises(ValueError):
        runner.run(**saved)
    assert not saved["outdir"].exists()


def test_wrong_pin_and_upstream_gdml_refused(saved):
    with pytest.raises(ValueError, match="pin SHA256"):
        runner.run(**(saved | dict(pin_sha256="0"*64)))
    (saved["pin_path"].parent / "geometry.gdml").write_text("changed geometry")
    with pytest.raises(ValueError, match="GDML hash"):
        runner.run(**saved)
    assert not saved["outdir"].exists()


def test_model_cap_failure_writes_no_completion(saved):
    with pytest.raises(RuntimeError, match="max_steps"):
        runner.run(**saved, max_steps=1, max_step_mm=.01)
    assert not saved["outdir"].exists()


def test_second_arm_failure_cannot_leave_success_product(saved, monkeypatch):
    actual = energy.orderedtransport.transport_intervals
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("second arm failed")
        return actual(*args, **kwargs)
    monkeypatch.setattr(energy.orderedtransport, "transport_intervals", fail_second)
    with pytest.raises(RuntimeError, match="second arm"):
        runner.run(**saved)
    assert not saved["outdir"].exists()


def test_original_seed_provenance_cannot_be_relabelled(saved):
    def change(path):
        table = pd.read_csv(path)
        table.loc[0, "sp"] = 7
        table.to_csv(path, index=False)
    repin_mutation(saved, "seeds.csv", change)
    with pytest.raises(ValueError, match="identity"):
        runner.run(**saved)
    assert not saved["outdir"].exists()


@pytest.mark.parametrize("name", ["event", "lane", "material", "s0_mm", "s1_mm"])
def test_duplicate_raw_csv_headers_are_rejected_before_pandas_mangles_them(name):
    with pytest.raises(ValueError, match="raw CSV headers"):
        energy._read_csv(f"{name},{name}\n1,2\n".encode(), 10, [])


def test_extreme_seed_momentum_cannot_mislabel_restricted_cut_as_unrestricted(saved):
    def change(path):
        table = pd.read_csv(path)
        table.loc[0, "px"] = 1e12
        table.to_csv(path, index=False)
    repin_mutation(saved, "seeds.csv", change)
    with pytest.raises(ValueError, match="effective unrestricted"):
        runner.run(**saved)
    assert not saved["outdir"].exists()


def test_complete_runner_empty_population(tmp_path):
    from test_pairedenergy import saved as fixture_factory
    maker = fixture_factory.__wrapped__(tmp_path)
    _, pin = maker([], lengths=(), kinetic=())
    result = runner.run(pin["pin"], pin_sha256=pin["sha256"], outdir=tmp_path / "empty-energy",
                        tracker_ref="1"*40, core_ref="2"*40)
    assert result["seed_rows"] == 0 and result["complete"]
    assert result["arms"]["inner"]["final_rows"] == result["arms"]["outer"]["history_rows"] == 0
