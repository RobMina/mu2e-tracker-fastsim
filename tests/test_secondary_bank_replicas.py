"""Nested donor uncertainty retains noisy negative estimates instead of claiming reconstructed errors."""
import numpy as np
import pytest

from scripts.secondary_bank_replicas import variance_components


def test_nested_variance_subtracts_draw_noise_and_retains_negative_raw():
    result = variance_components([[1., 3.], [2., 4.], [3., 5.]])
    assert result["within_bank_draw_variance"] == 2.
    assert result["variance_of_bank_draw_means"] == 1.
    assert result["donor_variance_raw"] == result["donor_variance_nonnegative"] == 0.
    result = variance_components([[0., 2.], [0., 2.]])
    assert result["donor_variance_raw"] == -1.
    assert result["donor_sigma_nonnegative"] == 0.
    with pytest.raises(ValueError, match="at least two"):
        variance_components([[1., 2.]])
    with pytest.raises(ValueError, match="finite"):
        variance_components([[1., 2.], [3., np.nan]])


def test_bank_jackknife_agrees_with_direct_delete_one_variance():
    values = np.array([[1., 2., 4.], [2., 3., 6.], [4., 5., 7.], [1., 5., 9.]])
    replicas = []
    for omit in range(len(values)):
        v = np.delete(values, omit, axis=0)
        replicas.append(np.var(v.mean(axis=1), ddof=1) - np.var(v, axis=1, ddof=1).mean() / v.shape[1])
    result = variance_components(values)
    expected = np.sqrt(3 / 4 * np.square(replicas - np.mean(replicas)).sum())
    np.testing.assert_allclose(result["donor_variance_jackknife_se"], expected)


def test_current_reference_never_permits_primary_or_energy_change():
    import pandas as pd
    from scripts.secondary_bank_replicas import coordinate_only_reference
    from scripts.secondary_replay import ReplayIdentityError
    baseline = dict(sgsteps=pd.DataFrame(dict(run=1, subrun=1, event=1, sp=[1, 1000], sid=[7, 7],
        nsgs=2, x0=[0., 0.], eion=[.01, .001])),
        particles=pd.DataFrame(dict(run=1, subrun=1, event=1, key=[1, 1000], pdg=[13, 11])))
    current = {k: v.copy() for k, v in baseline.items()}
    current["sgsteps"].loc[1, "x0"] = 1e-14
    result = coordinate_only_reference(baseline, current)
    assert result["sgsteps"]["x0"]["count"] == 1
    current["sgsteps"].loc[0, "x0"] = 1e-14
    with pytest.raises(ReplayIdentityError, match="primary changed"):
        coordinate_only_reference(baseline, current)
    current["sgsteps"].loc[0, "x0"] = 0.
    current["sgsteps"].loc[1, "eion"] = .002
    with pytest.raises(ReplayIdentityError, match="coordinate recomputation only"):
        coordinate_only_reference(baseline, current)
