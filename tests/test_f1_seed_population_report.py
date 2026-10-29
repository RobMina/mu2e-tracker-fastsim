"""Paired predicate report tests on analytic full-key arrays, not native fits."""
from copy import deepcopy

import numpy as np
import pytest

import f1_seed_population_report as R


def sample():
    return dict(keys=np.array([[1, 0, 2, 0], [1, 1, 2, 0], [2, 0, 2, 0]]),
                good=np.ones(3, bool), stockok=np.ones(3, bool), kl=np.zeros((3, 6)),
                stock_params=np.zeros((3, 6)), stock_errors=np.ones((3, 6)))


def test_full_key_shuffle_and_nonfinite_predicates():
    left = sample()
    right = deepcopy(left)
    left["kl"][0, 5] = .2
    right["kl"][1, 5] = np.nan
    right = {k: v[::-1] for k, v in right.items()}
    result = R.compare(left, right)
    assert result["acceptance_identical"]
    assert result["common_both_good"] == 3
    momentum = result["parameters"]["mom"]
    assert momentum["n_changed"] == 2 and momentum["gains"] == momentum["losses"] == 1
    assert momentum["gain_keys"] == [[1, 1, 2, 0]]
    assert momentum["loss_keys"] == [[1, 0, 2, 0]]
    assert not momentum["numeric_exact"]


def test_rejected_and_invalid_error_not_silently_counted():
    left, right = sample(), sample()
    left["good"][0] = False
    left["stock_errors"][1, 5] = right["stock_errors"][1, 5] = np.inf
    left["kl"][:, 5] = np.nan
    result = R.compare(left, right)
    assert not result["acceptance_identical"] and result["common_both_good"] == 2
    assert result["parameters"]["mom"]["loss_keys"] == [[2, 0, 2, 0]]


@pytest.mark.parametrize("field", ["keys", "stockok", "stock_params", "stock_errors"])
def test_mismatch_rejected(field):
    left, right = sample(), sample()
    right[field].flat[0] += 1 if right[field].dtype.kind != "b" else False
    if field == "stockok":
        right[field][0] = False
    with pytest.raises(ValueError):
        R.compare(left, right)


def test_phi_wrapping_and_strict_cut():
    left, right = sample(), sample()
    left["kl"][:, 1] = 2*np.pi
    result = R.compare(left, right)
    assert result["parameters"]["phi0"]["n_changed"] == 0
    left["kl"][0, 1] = .1+1e-10
    result = R.compare(left, right)
    assert result["parameters"]["phi0"]["losses"] == 1
