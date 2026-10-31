"""Bounded query audit rejects shortened near-plane spans and inconsistent crossing evidence."""
import pandas as pd
import pytest

from scripts.ordered_lookahead_audit import audit_queries


def fixture():
    rows = pd.DataFrame(dict(lane=[0], iteration=[4], full_requested_mm=[2000.], query_length_mm=[100.],
        crossing_mm=[1000.], x=[0.], y=[0.], z=[0.], ux=[0.], uy=[1.], uz=[0.]))
    receipt = dict(effective_settings=dict(max_step_mm=10.), handoff=dict(plane_y_mm=1000.))
    return rows, receipt


def test_saved_bounded_query_is_not_physical_terminal():
    result = audit_queries(*fixture())
    assert result["queries"] == 1 and result["maximum_query_mm"] == 100.


@pytest.mark.parametrize("field,value", [("crossing_mm", 5.), ("query_length_mm", 1000.), ("y", 1.), ("uy", -1.)])
def test_inconsistent_horizon_refuses(field, value):
    rows, receipt = fixture()
    rows[field] = value
    with pytest.raises(ValueError):
        audit_queries(rows, receipt)
