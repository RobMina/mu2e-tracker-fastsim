"""Exact initial plane contact differs from upward, upstream or positive-path contact."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.guardednavigation import replacement_domain


@pytest.mark.parametrize("start,y,dy,expected", [
    (0., 10., -1., True), (0., np.nextafter(10., np.inf), -1., False),
    (0., 10., 1., False), (0., 10., 0., False), (1., 11., -1., False),
    (0., 9., -1., True), (0., 9., 1., False)])
def test_exact_closed_initial_contact_only(start, y, dy, expected):
    cells = pd.DataFrame(dict(lane=[0], s0_mm=[start], s1_mm=[start+2.]))
    o, d = np.array([[0., y, 0.]]), np.array([[0., dy, 0.]])
    allowed, strict, evidence = replacement_domain(cells, o, d, 10., allow_start_contact=True)
    assert allowed == expected
    default, _, _ = replacement_domain(cells, o, d, 10.)
    assert default == strict
    if allowed and not strict:
        assert evidence.exact_start_contact.all() and (evidence.s0_mm == 0).all()
