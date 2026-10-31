"""Inclusive certificates never forge leaves or discard a possible nonleaf placement."""
from copy import deepcopy
import os

import numpy as np
import pandas as pd
import pytest

from scripts.ordered_torus_pair_probe import load_pair_sources
from trkfastsim import inclusiveenvelope as P, wideomission


@pytest.fixture(scope="module")
def modules():
    source = os.environ.get("TRKFS_REVIEWED_GEOMETRY_SOURCE")
    if source is None:
        pytest.skip("explicit reviewed geometry source required")
    return load_pair_sources(source)


def fixture(tmp_path, modules, *, y=0., nonleaf_x=100.):
    gdml = tmp_path / "geometry.gdml"
    gdml.write_text(f'''<gdml><solids>
      <box name="worldbox" x="1000" y="1000" z="1000"/>
      <box name="a" x="10" y="10" z="10"/><box name="b" x="2" y="2" z="20"/>
      <box name="cable" x="10" y="10" z="10"/><box name="child" x="2" y="2" z="2"/>
      <subtraction name="hole"><first ref="a"/><second ref="b"/></subtraction>
      </solids><structure>
      <volume name="World"><materialref ref="Vacuum"/><solidref ref="worldbox"/>
      <physvol name="block"><volumeref ref="Block"/></physvol>
      <physvol name="cable"><volumeref ref="Cable"/><position x="{nonleaf_x}"/></physvol></volume>
      <volume name="Block"><materialref ref="CONCRETE_MARS"/><solidref ref="hole"/></volume>
      <volume name="Cable"><materialref ref="Copper"/><solidref ref="cable"/>
      <physvol name="child"><volumeref ref="Child"/></physvol></volume>
      <volume name="Child"><materialref ref="Fiber"/><solidref ref="child"/></volume>
      </structure><setup><world ref="World"/></setup></gdml>''')
    census = modules["navigationcensus"].census(gdml, np.array([[-200.]*3, [200.]*3]))
    probe = P.CertifiedOmission(wideomission.WideOmission(modules["omittedpath"], authenticate_xtru=True), modules["omittedpath"])
    origin, direction, lengths = np.array([[-10., y, 0.]]), np.array([[1., 0., 0.]]), np.array([20.])
    cells, reports = probe.probe(gdml, census["records"], origin, direction, lengths)
    raw = pd.DataFrame({k: pd.Series(dtype=t) for k, t in dict(lane="int64", s0_mm="float64", s1_mm="float64", material="object", sign="int64").items()})
    return probe, [raw, lengths, cells, reports, census["records"]]


def compose(modules, probe, args, *, budget=1e-6):
    return P.InclusiveEnvelope(modules["supportenvelope"], modules["supportnavigation"], probe).compose(*args,
        endpoint_budget_mm=0., min_cell_mm=1e-9, max_component_mm=budget, max_ray_unknown_mm=1e-5)


def test_full_census_misses_distinct_from_leaf_csg_and_original_controls(tmp_path, modules):
    probe, args = fixture(tmp_path, modules)
    result = compose(modules, probe, args)
    assert result["summary"]["all_pairs_admitted"]
    assert result["summary"]["evaluated_box_misses"] == 2
    assert result["summary"]["authenticated_csg_placements"] == 1
    assert not result["original_strict_control"]["summary"]["all_rays_resolved"]
    assert not result["original_probe_strict_control"]["summary"]["all_rays_resolved"]
    nonleaf = [r for r in args[3] if not r["leaf"]]
    assert len(nonleaf) == 1 and nonleaf[0]["resolved"] is False
    assert nonleaf[0]["evaluation_status"] == "evaluated-miss"
    assert not args[2].source.isin([nonleaf[0]["source"]]).any()
    assert not result["summary"]["admitted_for_transport"]


def test_possible_nonleaf_keeps_global_refusal(tmp_path, modules):
    probe, args = fixture(tmp_path, modules, nonleaf_x=0.)
    result = compose(modules, probe, args)
    assert not result["summary"]["all_pairs_admitted"]
    assert result["control"]["summary"]["global_refusals"]


@pytest.mark.parametrize("target", ["reports", "cells", "records", "lengths", "missing"])
def test_unbound_mutated_certificate_evidence_refuses(tmp_path, modules, target):
    probe, args = fixture(tmp_path, modules)
    if target == "reports":
        args[3][0]["method"]["lanes"] = True
    elif target == "cells":
        args[2].loc[args[2].index[0], "ancestors_verified"] = False
    elif target == "records":
        args[4] = deepcopy(args[4][:-1])
    elif target == "lengths":
        args[1] = args[1] + 1
    else:
        probe.latest = None
    with pytest.raises(ValueError, match="unbound"):
        compose(modules, probe, args)


def test_coplanar_csg_unknown_budget_not_waived(tmp_path, modules):
    probe, args = fixture(tmp_path, modules, y=5.)
    result = compose(modules, probe, args)
    assert not result["summary"]["all_pairs_admitted"]
    assert "unknown-component-budget-exceeded" in result["refusals"].reason.to_numpy()


def test_a_new_query_invalidates_previous_ray_certificates(tmp_path, modules):
    probe, args = fixture(tmp_path, modules)
    probe.probe(tmp_path / "geometry.gdml", args[4], np.array([[-10., 6., 0.]]),
                np.array([[1., 0., 0.]]), args[1])
    with pytest.raises(ValueError, match="stale"):
        compose(modules, probe, args)


def test_wrong_census_leaf_flag_never_relabels_physical_nonleaf(tmp_path, modules):
    probe, args = fixture(tmp_path, modules)
    records = deepcopy(args[4])
    nonleaf = next(r for r in records if not r["is_leaf"])
    nonleaf["is_leaf"] = True
    with pytest.raises(ValueError, match="census"):
        probe.probe(tmp_path / "geometry.gdml", records, np.array([[-10., 0., 0.]]),
                    np.array([[1., 0., 0.]]), args[1])


def test_exclusion_overflow_refuses_instead_of_miss():
    with pytest.raises(ValueError, match="overflow"):
        P.bounds_exposure(np.array([[-1e308, 0., 0.]]), np.array([[1., 0., 0.]]), np.array([1.]),
                          np.array([[1e308, -1., -1.], [1.1e308, 1., 1.]]))


def test_box_bounds_do_not_turn_tangency_into_miss():
    possible, guard = P.bounds_exposure(np.array([[-10., 5., 0.]]), np.array([[1., 0., 0.]]), np.array([20.]),
                                         np.array([[-5.]*3, [5.]*3]))
    assert possible.item() and guard > 0
