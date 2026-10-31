"""Explicit xtru plane-offset certificate admission, unchanged strict controls and budgets."""
from copy import deepcopy
import os

import numpy as np
import pandas as pd
import pytest

from scripts.ordered_torus_pair_probe import load_pair_sources
from trkfastsim.mixedenvelope import MixedEnvelope


@pytest.fixture(scope="module")
def modules():
    source = os.environ.get("TRKFS_REVIEWED_GEOMETRY_SOURCE")
    if source is None:
        pytest.skip("explicit reviewed geometry source required")
    return load_pair_sources(source)


def fixture(modules):
    source = "world[0]/block[0]"
    cells = pd.DataFrame(dict(lane=[0, 0, 0], s0_mm=[2., 2.+1e-8, 4.], s1_mm=[2.+1e-8, 4., 4.+1e-8],
        material=["concrete"]*3, state=["unresolved", "inside", "unresolved"], ancestors_verified=[True]*3,
        leaf_verified=[True]*3, replaces_material=["vacuum"]*3, source=[source]*3, scope=["external"]*3,
        reason=["plane_offset", "certificate", "plane_offset"]))
    record = dict(placement_path=source, classification="unsupported", material="concrete", solid_tag="xtru", scope="external", is_leaf=True)
    report = dict(source=source, material="concrete", solid_tag="xtru", scope="external", leaf=True, resolved=True,
        parent_material="Vacuum", solid_lower_mm=modules["omittedpath"].length_bound(cells[cells.state == "inside"], upper=False),
        solid_upper_mm=modules["omittedpath"].length_bound(cells, upper=True),
        hierarchy_verified_lower_mm=modules["omittedpath"].length_bound(cells[cells.state == "inside"], upper=False),
        hit_lanes=1, possible_lanes=1, method=dict(method="core prism triangulation with explicit inner/outer plane-offset bands",
            certificate_scheme="xtru-plane-offset-v1", guard_mm=1e-10, capped=False, lanes=1, positive_length_lanes=1,
            inside_cells=1, unresolved_cells=2, unresolved_lanes=1, unresolved_reasons={"plane_offset": 2}))
    raw = pd.DataFrame({k: pd.Series(dtype=t) for k, t in dict(lane="int64", s0_mm="float64", s1_mm="float64", material="object", sign="int64").items()})
    return raw, np.array([10.]), cells, [report], [record]


def compose(modules, args, **kw):
    return MixedEnvelope(modules["supportenvelope"], modules["supportnavigation"]).compose(*args,
        endpoint_budget_mm=0., min_cell_mm=1e-9, max_component_mm=kw.get("component", 1e-6), max_ray_unknown_mm=1e-5)


def test_explicit_xtru_admits_pair_not_strict_or_original_torus_policy(modules):
    args = fixture(modules)
    result = compose(modules, args)
    assert result["summary"]["all_pairs_admitted"] and not result["summary"]["admitted_for_transport"]
    assert result["summary"]["authenticated_xtru_placements"] == 1
    assert not result["control"]["decisions"].candidate_resolved.item()
    original = modules["supportenvelope"].compose(*args, endpoint_budget_mm=0., min_cell_mm=1e-9,
                                               max_component_mm=1e-6, max_ray_unknown_mm=1e-5)
    assert not original["summary"]["all_pairs_admitted"]
    assert "non-torus-unresolved-boundary" in original["refusals"].reason.to_numpy()


@pytest.mark.parametrize("fault", ["scheme", "method", "cap", "alias", "count", "boolean_count", "reason"])
def test_forged_or_capped_certificate_refuses(modules, fault):
    args = fixture(modules)
    method = args[3][0]["method"]
    if fault == "scheme":
        method["certificate_scheme"] = "torus-min-width"
    elif fault == "method":
        method["method"] = "approximate bbox"
    elif fault == "cap":
        method["capped"] = True
    elif fault == "alias":
        method["cap_reached"] = True
    elif fault == "count":
        method["unresolved_cells"] = 1
    elif fault == "boolean_count":
        method["lanes"] = True
    else:
        args[2].loc[0, "reason"] = "min_width"
    with pytest.raises(ValueError):
        compose(modules, args)


def test_same_unknown_budget_refuses_even_authenticated_xtru(modules):
    result = compose(modules, fixture(modules), component=1e-9)
    assert not result["summary"]["all_pairs_admitted"]
    assert "unknown-component-budget-exceeded" in result["refusals"].reason.to_numpy()
