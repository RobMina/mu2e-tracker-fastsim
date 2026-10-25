import importlib.util
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("rejection_trace_check", Path(__file__).parents[1] / "scripts/rejection_trace_check.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def tables(rejected=False):
    row = {key: 0 for key in module.KEYS + module.SAME}
    row.update(full_schedule=True, stage=-1, maxit=-1, capture_enabled=False, rejected=0)
    candidate = dict(row, capture_enabled=True)
    if rejected:
        row["status"] = candidate["status"] = 4
        row["goodseed"] = candidate["goodseed"] = 1
        candidate["rejected"] = 1
        p = np.zeros((4, 6))
        p[[0, 2], 0] = 2.0
        for target in (row, candidate):
            target.update(pf=np.zeros(6), pb=np.zeros(6), cf=np.eye(6).ravel(), cb=np.eye(6).ravel())
        candidate.update(rejected_parameters=p.ravel(), rejected_covariances=np.tile(np.eye(6), (4, 1, 1)).ravel(),
                         rejected_steps=(p[[0, 2]] - p[[1, 3]]).ravel(), rejected_metrics=[4.0, 4.0, 1.0, 0.0, 0.0],
                         rejected_ranges=np.tile([0.0, 1.0], 4), rejected_reference_indices=[0, 0])
    return ak.Array([row]), ak.Array([candidate])


def test_identical_capture_and_projection():
    a, b = tables(True)
    result = module.compare(a, b)
    assert result["n_rejected"] == 1
    assert result["recorded_capture_equivalent"]
    np.testing.assert_allclose(result["reference_min_eigenvalue_q10_50_90"], 1.0)


def test_no_rejections_still_checks_history():
    a, b = tables()
    assert module.compare(a, b)["n_rejected"] == 0


@pytest.mark.parametrize("field,value", [("chi2", 1.0), ("stage", 3), ("capture_enabled", False), ("event", 2)])
def test_changed_history_or_contract_rejected(field, value):
    a, b = tables()
    b = ak.with_field(b, ak.Array([value]), field)
    with pytest.raises(ValueError):
        module.compare(a, b)


def test_duplicate_identity_rejected():
    a, b = tables()
    with pytest.raises(ValueError, match="duplicate"):
        module.compare(ak.concatenate([a, a]), ak.concatenate([b, b]))


def test_empty_trace_cannot_validate():
    a, b = tables()
    with pytest.raises(ValueError, match="empty"):
        module.compare(a[:0], b[:0])


def test_missing_extension_rejection_snapshot_rejected():
    a, b = tables(True)
    b = ak.with_field(b, ak.Array([0]), "rejected")
    with pytest.raises(ValueError, match="coverage"):
        module.compare(a, b)


def test_momentum_not_weak_geometry_dominates():
    a, b = tables(True)
    p = np.zeros((4, 6))
    p[[0, 2], 5] = 0.2
    cov = np.tile(np.eye(6), (4, 1, 1))
    cov[:, 0, 1] = cov[:, 1, 0] = 1 - 1e-6
    cov[:, 5, 5] = 1e-8
    b = ak.with_field(b, ak.Array([p.ravel()]), "rejected_parameters")
    b = ak.with_field(b, ak.Array([cov.ravel()]), "rejected_covariances")
    b = ak.with_field(b, ak.Array([(p[[0, 2]] - p[[1, 3]]).ravel()]), "rejected_steps")
    b = ak.with_field(b, ak.Array([[4e6, 4e6, 1e6, 0, 0]]), "rejected_metrics")
    result = module.compare(a, b)["mode_diagnosis"]
    assert result["n_weakest_mode_dominates"] == 0
    assert result["n_momentum_diagonal_above_threshold"] == 2
    np.testing.assert_allclose(result["momentum_diagonal_over_full_q10_50_90"], 1.0)


def test_indefinite_positive_diagonal_reference_rejected():
    a, b = tables(True)
    cov = np.tile(np.eye(6), (4, 1, 1))
    cov[:, 1, 2] = cov[:, 2, 1] = 2.0
    b = ak.with_field(b, ak.Array([cov.ravel()]), "rejected_covariances")
    with pytest.raises(ValueError, match="nonpositive reference eigenvalue"):
        module.compare(a, b)


def test_wrong_recorded_step_rejected():
    a, b = tables(True)
    b = ak.with_field(b, ak.Array([[0.0] * 12]), "rejected_steps")
    with pytest.raises(AssertionError):
        module.compare(a, b)
