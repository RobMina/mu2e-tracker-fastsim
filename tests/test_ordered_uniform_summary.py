"""Weak convergence reports refuse same-seed or mismatched ensemble comparisons."""
import copy

import numpy as np
import pytest

from scripts.ordered_uniform_summary import compare


def test_independent_mean_error_and_settings_gate():
    moments = dict(mean=2., mean_se=.5)
    first = dict(mode="ordered-stored-sampled-v1", material="Al6061", stop_ke_MeV=.5,
        samples_per_energy=2, scatter=False, seed=1,
        energies=[dict(initial_ke_MeV=5., stopped=2, range_mm=moments, time_ns=moments)])
    second = copy.deepcopy(first)
    second["seed"] = 2
    second["energies"][0]["range_mm"] = dict(mean=3., mean_se=.5)
    values = dict(initial_ke_MeV=np.array([5., 5.]), path=np.array([1., 3.]), dt=np.array([1., 3.]))
    result = compare(first, second, values, values)[0]["observables"]["range_mm"]
    np.testing.assert_allclose(result["independent_mean_error"], np.sqrt(.5))
    assert result["delta"] == 1.
    with pytest.raises(ValueError, match="distinct seeds"):
        compare(first, first, values, values)
    second["stop_ke_MeV"] = .25
    with pytest.raises(ValueError, match="stop_ke_MeV"):
        compare(first, second, values, values)
