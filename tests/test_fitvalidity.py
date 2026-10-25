import json

import numpy as np
import pytest

from trkfastsim.reco.fitvalidity import covariance_checks, fit_checks


def test_covariance_adversarial_cases():
    cov = np.tile(np.eye(6), (7, 1, 1))
    cov[1, 0, 1] = cov[1, 1, 0] = 2.0
    cov[2, 0, 1] = np.nan
    cov[3, 2, 2] = np.inf
    cov[4, 0, 1] = 0.1
    cov[5, 0, 0] = -1.0
    cov[6] = 0.0
    out = covariance_checks(cov)
    np.testing.assert_array_equal(out["valid"], [True, False, False, False, False, False, True])
    assert not out["psd"][1]
    assert not out["symmetric"][4]
    assert np.isnan(out["min_scaled_eigenvalue"][[2, 3]]).all()


def test_weak_positive_direction_is_not_a_condition_number_cut():
    cov = np.eye(6)
    cov[0, 1] = cov[1, 0] = 1.0 - 1e-12
    scale = np.array([1e4, 1e-4, 1, 1e-2, 1e2, 1e-3])
    out = covariance_checks(cov * scale[:, None] * scale[None, :])
    assert out["valid"]
    assert 0 < out["min_scaled_eigenvalue"] < 2e-12


def test_validity_counts_do_not_change_acceptance_and_ignore_padding():
    x = np.zeros((4, 2, 6))
    cov = np.tile(np.eye(6), (4, 2, 1, 1))
    mask = np.tile([True, False], (4, 1))
    x[:, 1] = np.nan
    cov[:, 1] = np.nan
    cov[1, 0, 0, 1] = cov[1, 0, 1, 0] = 2.0
    tx, tc = np.zeros((4, 6)), np.tile(np.eye(6), (4, 1, 1))
    tc[2, 0, 1] = np.nan
    good, stockok = np.array([True, True, False, False]), np.array([True, False, True, False])
    before = good.copy()
    checks, summary = fit_checks(x, cov, mask, tx, tc, [1, 1, 1, np.nan], [1, 1, 1, 1], good, stockok)
    np.testing.assert_array_equal(checks["valid"], [True, False, False, False])
    np.testing.assert_array_equal(good, before)
    pop = summary["populations"]
    assert pop["f1_good"]["n_tracks"] == 2
    assert pop["f1_only"]["failures"]["covariance_psd"] == 1
    assert pop["stock_only"]["failures"]["covariance_finite"] == 1
    assert pop["neither"]["failures"]["finite_statistics"] == 1
    json.dumps(summary, allow_nan=False)


def test_empty_batch_is_typed_and_has_zero_denominators():
    checks, summary = fit_checks(np.empty((0, 1, 6)), np.empty((0, 1, 6, 6)), np.empty((0, 1), bool),
                                np.empty((0, 6)), np.empty((0, 6, 6)), [], [], [], [])
    assert checks["valid"].shape == (0,)
    assert checks["valid"].dtype == np.bool_
    assert all(p["n_tracks"] == p["n_valid"] == 0 for p in summary["populations"].values())
    json.dumps(summary, allow_nan=False)


def test_no_effects_is_invalid_even_with_finite_t0():
    checks, _ = fit_checks(np.empty((1, 0, 6)), np.empty((1, 0, 6, 6)), np.empty((1, 0), bool),
                          np.zeros((1, 6)), np.eye(6)[None], [1], [1], [True], [False])
    assert not checks["valid"][0]
    assert not checks["has_effect"][0]


@pytest.mark.parametrize("tolerance", [-1, np.nan, np.inf])
def test_bad_tolerance_rejected(tolerance):
    with pytest.raises(ValueError):
        covariance_checks(np.eye(6), tolerance)


def test_bad_shapes_rejected():
    with pytest.raises(ValueError):
        covariance_checks(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        fit_checks(np.zeros((1, 2, 6)), np.zeros((1, 2, 6, 6)), np.ones((1, 3), bool),
                   np.zeros((1, 6)), np.eye(6)[None], [1], [1], [True], [True])
