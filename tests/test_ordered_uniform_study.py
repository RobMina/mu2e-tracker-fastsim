"""Actual uniform-medium transport checks mean convergence against independent quadrature."""
import numpy as np
import pytest

from mu2efastsim import materials
from scripts.ordered_uniform_study import reference, run
from trkfastsim import orderedinterior


def test_mean_range_and_clock_refine_to_continuous_existing_model(tmp_path):
    coarse = run("Al6061", tmp_path / "coarse", mode=orderedinterior.MODE, max_step_mm=10.,
                 fractional_loss=.05, samples=2)
    fine = run("Al6061", tmp_path / "fine", mode=orderedinterior.MODE, max_step_mm=1.,
               fractional_loss=.01, samples=2)
    for first, second in zip(coarse["energies"], fine["energies"]):
        assert second["stopped"] == 2 and second["not_stopped"] == 0
        assert abs(second["range_mean_minus_continuous_mm"]) < abs(first["range_mean_minus_continuous_mm"])
        assert abs(second["time_mean_minus_continuous_ns"]) < abs(first["time_mean_minus_continuous_ns"])
        np.testing.assert_allclose(second["final_ke_MeV"]["mean"], .5, atol=1e-12)
    assert fine["primary_balance_max_abs_MeV"] < 1e-10


def test_sampled_stops_absorb_and_no_output_overwrite(tmp_path):
    result = run("Al6061", tmp_path / "sampled", mode=orderedinterior.SAMPLED_MODE,
                 max_step_mm=1., samples=2)
    assert all(row["not_stopped"] == 0 for row in result["energies"])
    assert all(row["range_mm"]["std"] > 0 for row in result["energies"])
    with pytest.raises(FileExistsError):
        run("Al6061", tmp_path / "sampled", mode=orderedinterior.SAMPLED_MODE, max_step_mm=1., samples=2)


def test_reference_cutoff_reduces_range_and_clock():
    low = reference(materials.ALUMINIUM_6061, 20., .25)
    high = reference(materials.ALUMINIUM_6061, 20., .5)
    assert low["range_mm"] > high["range_mm"] > 0
    assert low["time_ns"] > high["time_ns"] > 0


def test_opt_in_midpoint_removes_leading_mean_range_bias(tmp_path):
    entry = run("Al6061", tmp_path / "entry", mode=orderedinterior.MODE, max_step_mm=10., samples=2)
    midpoint = run("Al6061", tmp_path / "midpoint", mode=orderedinterior.MODE, max_step_mm=10.,
                   samples=2, stopping_evaluation="midpoint")
    for first, second in zip(entry["energies"], midpoint["energies"]):
        assert abs(second["range_mean_minus_continuous_mm"]) < .1 * abs(first["range_mean_minus_continuous_mm"])
        assert abs(second["time_mean_minus_continuous_ns"]) < .1 * abs(first["time_mean_minus_continuous_ns"])
        assert second["stopped"] == 2
