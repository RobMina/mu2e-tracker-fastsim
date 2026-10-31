"""Response-draw summaries preserve negative shifts and never call them event errors."""
import numpy as np
import pandas as pd
import pytest

from scripts.ordered_response_summary import summarize


def fixture():
    response = dict(generated_events=1000, rows=[dict(arm=arm, draw=draw, metrics=dict(n=value + draw))
        for arm, value in (("merged", 3.), ("actual_slices", 2.), ("straightened_slices", 1.)) for draw in range(2)])
    lanes = pd.DataFrame(dict(interior_recorded=[True, True, False], gas_entries=[1, 0, 0]))
    for name in ("loader_aabb", "vd_cylinder"):
        lanes[f"{name}_inside_mm"] = [2., 0., 0.]
        lanes[f"{name}_outside_mm"] = [1., 3., 0.]
    return response, lanes


def test_independent_draw_errors_and_empty_lanes_retained():
    response, lanes = fixture()
    result = summarize(response, lanes)
    contrast = result["independent_draw_contrasts"]["actual_slices_minus_merged"]["n"]
    assert contrast["difference"] == -1.
    assert contrast["independent_response_draw_mean_se"] == pytest.approx(np.sqrt(.5))
    assert result["domain_summary"]["loader_aabb"]["active_with_any_inside"] == 1
    assert result["domain_summary"]["loader_aabb"]["active_lanes"] == 2
    assert not result["admitted_for_production"]


def test_incomplete_or_duplicate_trials_refuse():
    response, lanes = fixture()
    response["rows"].pop()
    with pytest.raises(ValueError, match="rectangle"):
        summarize(response, lanes)
    response, lanes = fixture()
    response["rows"][-1] = response["rows"][0]
    with pytest.raises(ValueError, match="unique"):
        summarize(response, lanes)
