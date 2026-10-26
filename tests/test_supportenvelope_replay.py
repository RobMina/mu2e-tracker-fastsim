"""Quantified envelope runner provenance, controls and no-overwrite tests."""
import json

import numpy as np
import pandas as pd
import pytest

from test_supportnavigation_replay import saved as audit_fixture
from trkfastsim.analyses import supportenvelope as runner, supportnavigation as refusal


@pytest.fixture
def saved(audit_fixture):
    refusal.run(**audit_fixture)
    return dict(audit_dir=audit_fixture["audit_dir"], gdml=audit_fixture["gdml"],
        candidate_dir=audit_fixture["outdir"], outdir=audit_fixture["outdir"].parent / "envelopes",
        max_component_mm=1e-6, max_ray_unknown_mm=1e-5)


def test_envelope_replay_preserves_full_identities_and_no_transport(saved):
    result = runner.run(**saved)
    assert result["complete"] and result["seed_rows"] == 2
    manifest = json.loads((saved["outdir"] / "summary.json").read_text())
    assert manifest["seed_events"] == 1 and not manifest["admitted_for_transport"]
    assert manifest["settings"]["max_component_mm"] == 1e-6
    assert len(manifest["sources_sha256"]["tracker"]["supportenvelope.py"]) == 64
    for item in manifest["candidates"]:
        assert item["bounded_pair_rays"] == 2
        assert item["saved_refusal_control_identical"] and item["zero_unknown_budget_control_identical"]
        assert item["invariance"]["reversed_evidence_identical"]
        assert item["resolution_comparison"]["all_bounds_overlap"]
        assert all(p["same_lane_decisions"] for p in item["stored_endpoint_perturbations"])
    for arm in ("inner", "outer"):
        table = pd.read_csv(saved["outdir"] / f"envelope_0mm_{arm}_intervals.csv")
        assert table.sp.tolist() == [4, 5] and table.event.tolist() == [3, 3]
        assert table.material.tolist() == ["vacuum", "vacuum"]
    with pytest.raises(FileExistsError):
        runner.run(**saved)


@pytest.mark.parametrize("change,match", [
    ("schema", "completed refusal-first"), ("mode", "recognized"), ("transport", "recognized"),
    ("input", "different saved audit"), ("gdml", "GDML differs"), ("budgets", "saved refusal control"),
    ("settings", "primary probe settings"), ("table", "different"), ("reports", "reports differ")])
def test_control_corruption_or_mismatch_refuses_completion(saved, change, match):
    path = saved["candidate_dir"] / "summary.json"
    summary = json.loads(path.read_text())
    if change == "schema":
        summary["schema"] = "wrong"
    elif change == "mode":
        summary["mode"] = "not-refusal"
    elif change == "transport":
        summary["admitted_for_transport"] = True
    elif change == "input":
        summary["inputs"]["seeds.csv"]["sha256"] = "0" * 64
    elif change == "gdml":
        summary["gdml"]["sha256"] = "0" * 64
    elif change == "budgets":
        saved["budgets"] = [1e-8]
    elif change == "settings":
        saved["min_cell_mm"] = 2e-7
    elif change == "table":
        tablepath = saved["candidate_dir"] / "candidate_0mm_intervals.csv"
        table = pd.read_csv(tablepath)
        table.loc[0, "material"] = "forged"
        table.to_csv(tablepath, index=False)
    else:
        (saved["candidate_dir"] / "omitted_placements.json").write_text('[{"source":"forged"}]')
    path.write_text(json.dumps(summary))
    with pytest.raises((ValueError, AssertionError), match=match):
        runner.run(**saved)
    assert not saved["outdir"].exists()


@pytest.mark.parametrize("key,value", [("max_component_mm", -1.), ("max_component_mm", np.nan),
    ("max_component_mm", True), ("max_ray_unknown_mm", np.inf), ("max_ray_unknown_mm", -1.),
    ("compare_cell_mm", 1e-8), ("max_pairs", 0), ("budgets", [])])
def test_settings_refused_before_output(saved, key, value):
    saved[key] = value
    with pytest.raises(ValueError):
        runner.run(**saved)
    assert not saved["outdir"].exists()


def test_refusal_first_sources_not_modified_by_pair_runner(saved):
    from trkfastsim import supportnavigation, omittedpath
    before = {module.__file__: runner.sha256(module.__file__) for module in (supportnavigation, omittedpath)}
    runner.run(**saved)
    assert before == {path: runner.sha256(path) for path in before}


def test_zero_uncertainty_budget_is_explicit(saved):
    saved["max_component_mm"] = saved["max_ray_unknown_mm"] = 0.
    result = runner.run(**saved)
    assert all(row["zero_unknown_budget_control_identical"] for row in result["candidates"])
    assert all(row["max_component_mm"] == 0. and row["max_ray_unknown_mm"] == 0. for row in result["candidates"])
