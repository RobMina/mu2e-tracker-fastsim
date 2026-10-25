"""Saved-arm matching and actual highest-cycle slab event census."""
from copy import deepcopy

import numpy as np
import pytest
import uproot

from trkfastsim.analyses.slabcompare import audit_references, compare


def _row():
    return {"material": "G4_Al", "g_cm2": 10., "p_MeV": 300, "n_g4": 2,
            "surv_g4": 1., "g4": [1., 2., 3.], "p_exit_frac": .8,
            "fast_over_g4": [1., 1., 1.], "fast_cal_over_g4": [.9, .8, .7]}


def test_compare_requires_same_reference_and_reports_domain():
    before = {"rows": [_row()]}
    after = deepcopy(before)
    after["rows"][0]["fast_over_g4"] = [1.02, 1.03, 1.04]
    result = compare(before, after)
    assert result["n_matched_points"] == 1
    assert result["baseline_settings"] is None
    assert result["summary"][0]["arms"]["after"]["fast"]["n_width_within_3pct"] == 1
    after["rows"][0]["g4"][0] = 2.
    with pytest.raises(ValueError, match="different G4"):
        compare(before, after)


def test_compare_rejects_different_or_duplicate_points():
    with pytest.raises(ValueError, match="coverage"):
        compare({"rows": [_row()]}, {"rows": []})
    with pytest.raises(ValueError, match="duplicate"):
        compare({"rows": [_row(), _row()]}, {"rows": [_row()]})


def test_reference_audit_uses_highest_cycle(tmp_path):
    directory = tmp_path / "G4_Al"
    directory.mkdir()
    path = directory / "slab.t10_p300.root"
    columns = {"run": np.ones(2, dtype="i4"), "subrun": np.ones(2, dtype="i4"),
               "event": np.arange(2, dtype="i4"), "isprim": np.ones(2, dtype="i4"),
               "pdg": np.full(2, 13, dtype="i4"), "epz": np.full(2, 280.)}
    with uproot.recreate(path) as root:
        root.mktree("dump/particles", {key: value.dtype for key, value in columns.items()})
        root["dump/particles"].extend({key: value[:1] for key, value in columns.items()})
    with uproot.update(path) as root:
        root.mktree("dump/particles", {key: value.dtype for key, value in columns.items()})
        root["dump/particles"].extend(columns)
    result = audit_references([str(path)], [_row()])
    assert result["primary_muon_events"] == 2
    assert result["files"][0]["particle_tree_cycle"] == 2
    wrong = _row()
    wrong["n_g4"] = 1
    with pytest.raises(ValueError, match="entries disagree"):
        audit_references([str(path)], [wrong])
