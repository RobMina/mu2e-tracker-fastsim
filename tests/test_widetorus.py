"""Wider spatial arithmetic preserves binary64 parameter cells/caps and analytic occupancy."""
import numpy as np
import pytest

from trkfastsim import widetorus as P


def probe(origin, direction, length, **kwargs):
    return P.torus_intervals(np.asarray(origin, np.longdouble).reshape(-1, 3),
        np.asarray(direction, np.longdouble).reshape(-1, 3), length, rtor=10., rmin=0., rmax=2., **kwargs)


def test_capability_is_genuinely_wider():
    P.require_wider()
    assert np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant


def test_analytic_chord_hole_and_inner_start():
    result = probe([[10., 0., -5.], [0., 0., -5.], [10., 0., 0.]], [[0., 0., 1.]]*3, [10., 10., 1.])
    assert result.lower_mm[0] <= 4. <= result.upper_mm[0]
    assert result.upper_mm[1] == 0.
    assert result.lower_mm[2] <= 1. <= result.upper_mm[2]
    cells = result.intervals
    assert cells.s0_mm.dtype == cells.s1_mm.dtype == np.dtype(float)
    assert (cells.s1_mm > cells.s0_mm).all()
    rows = cells.sort_values(["lane", "s0_mm"])
    previous = rows.groupby("lane").s1_mm.shift()
    assert not (rows.s0_mm < previous).any()
    assert not result.counts["cap_reached"]


def test_actual_cap_and_stall_evidence_never_declared_outside():
    result = probe([[10., 0., -5.]], [[0., 0., 1.]], [10.], max_nodes=0)
    assert result.counts["cap_reached"] and (result.intervals.status == "unresolved").all()
    assert result.upper_mm[0] >= 4.
    assert set(result.intervals.reason) == {"max_nodes"}


def test_far_origin_guard_and_finer_cells_keep_analytic_chord():
    for width in (1e-7, 1e-9):
        result = probe([[10., 0., -1000000.]], [[0., 0., 1.]], [1000010.], min_width_mm=width)
        assert result.lower_mm[0] <= 4. <= result.upper_mm[0]
        assert not result.counts["cap_reached"]


def test_tangent_is_not_assumed_finite_chord():
    result = probe([[12., 0., -5.]], [[0., 0., 1.]], [10.])
    assert result.lower_mm[0] == 0.
    assert (result.intervals.status == "unresolved").all()


def test_sector_opposite_quadrants_not_full_circle():
    result = probe([[10., 1., -5.], [-10., -1., -5.]], [[0., 0., 1.]]*2, [10., 10.], deltaphi=np.pi/2)
    assert result.lower_mm[0] > 3. and result.upper_mm[1] == 0.


def test_empty_zero_and_nonunit_refusal():
    result = probe([], [], [])
    assert result.intervals.empty and not result.counts["cap_reached"]
    result = probe([[10., 0., 0.]], [[0., 0., 1.]], [0.])
    assert result.intervals.empty and result.upper_mm[0] == 0.
    with pytest.raises(ValueError, match="unit"):
        probe([[10., 0., 0.]], [[0., 0., 2.]], [1.])


def test_initial_active_cap_retains_and_counts_original_frontier():
    result = probe([[10., 0., -5.]]*2, [[0., 0., 1.]]*2, [10., 10.], max_active=1)
    assert result.counts["cap_reached"] and result.counts["peak_active_cells"] == 2
    assert result.unresolved_count.tolist() == [1, 1]
