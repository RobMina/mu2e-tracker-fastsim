"""Actual GDML torus/parent plus analytic straw evolution with explicit reviewed sources."""
import os

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import eloss, materials
from scripts.ordered_torus_pair_probe import load_pair_sources
from trkfastsim import geometry, guardednavigation as G, ledger, orderedinterior as O, steps


@pytest.fixture(scope="module")
def reviewed_modules():
    source = os.environ.get("TRKFS_REVIEWED_GEOMETRY_SOURCE")
    if source is None:
        pytest.skip("explicit reviewed geometry source required for external-source integration")
    return load_pair_sources(source)


@pytest.mark.parametrize("mode", O.MODES)
@pytest.mark.parametrize("arm", ["inner", "outer"])
@pytest.mark.parametrize("scatter", [False, True])
def test_actual_torus_parent_straw_evolution(tmp_path, monkeypatch, reviewed_modules, mode, arm, scatter):
    material = materials.ALUMINIUM_6061.name
    gdml = tmp_path / "torus.gdml"
    gdml.write_text(f'''<gdml><solids><box name="world" x="100" y="100" z="100"/>
    <box name="parent" x="60" y="60" z="60"/>
    <torus name="ring" rtor="10" rmin="0" rmax="2" startphi="0" deltaphi="360" aunit="deg" lunit="mm"/>
    </solids><structure><volume name="World"><materialref ref="Vacuum"/><solidref ref="world"/>
    <physvol name="p"><volumeref ref="Parent"/></physvol></volume>
    <volume name="Parent"><materialref ref="Vacuum"/><solidref ref="parent"/>
    <physvol name="ring"><volumeref ref="Leaf"/></physvol></volume>
    <volume name="Leaf"><materialref ref="{material}"/><solidref ref="ring"/></volume>
    </structure><setup><world ref="World"/></setup></gdml>''')
    records = [dict(placement_path="p[0]/ring[0]", classification="unsupported", material=material,
                    solid_tag="torus", scope="support", is_leaf=True, placement_assumptions=[])]
    census = dict(schema="tracker-navigation-census-v1", records=records, placement_warnings=[])
    geom = geometry.StrawGeometry(plane=np.array([0]), panel=np.array([0]), straw=np.array([0]),
        mid=np.array([[10., 0., 4.5]]), dir=np.array([[1., 0., 0.]]), halflen=np.array([20.]), radius=.2,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)), plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)),
        wall_rmax=.215, wire_r=0.)
    raw = pd.DataFrame({k: pd.Series(dtype=t) for k, t in dict(lane="int64", s0_mm="float64", s1_mm="float64",
        material="object", source="object", sign="int64").items()})
    provider = G.GuardedNavigator(arm=arm, gdml=gdml, census=census, modules=reviewed_modules,
        raw_hits=lambda *a: raw.copy(), world_plane_y_mm=40., endpoint_budget_mm=1e-10,
        min_cell_mm=1e-7, max_component_mm=1e-6, max_ray_unknown_mm=1e-5)
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.full_like(p, .01))
    monkeypatch.setattr(eloss, "sample_loss", lambda rng, mat, p, length, cut: length * .01)
    monkeypatch.setattr(steps, "_kink_core", lambda rng, v, a, b:
        (np.tile([.001 if scatter else 0., 0.], (len(v), 1)), np.zeros((len(v), 2)), np.zeros(len(v))))
    obj = type("Interior", (), {"shells": None})()
    rows = O.transport(obj, np.random.default_rng(19), np.array([[10., 0., -5.]]), np.array([[0., 0., 1.]]),
        np.array([80.]), geom, tail=False, tail_k=1., mode=mode, max_path_mm=10., max_step_mm=1.,
        signed_endpoint_budget_mm=1e-10, navigation_provider=provider)
    sl = obj.last_slices
    assert len(rows["it"]) == 1
    assert obj.last_final["lost_support"][0] > .039
    assert obj.last_final["lost_support"][0] < .041
    gas = sl["material"] == steps.STRAW_GAS.name
    assert gas.any() and sl["ke0"][gas][0] < ledger.kinetic(80.) - .039
    np.testing.assert_allclose(sl["pos1"][:-1], sl["pos0"][1:], atol=1e-12)
    np.testing.assert_array_equal(sl["end"][:-1], sl["start"][1:])
    np.testing.assert_allclose(sl["ke1"][:-1], sl["ke0"][1:], atol=1e-12)
    np.testing.assert_allclose(sl["t1"][:-1], sl["t0"][1:], atol=1e-12)
    np.testing.assert_allclose(obj.last_final["ke"] + obj.last_final["lost_support"] + obj.last_final["lost_thin"], ledger.kinetic(80.))
    assert len(provider.calls) > 1 if scatter else len(provider.calls) == 1
    assert all(c["status"] == "complete" and c["replacements_strictly_below_plane"] for c in provider.calls)
    if scatter:
        assert any(not np.array_equal(c["direction"], provider.calls[0]["direction"]) for c in provider.calls[1:])


def test_actual_capped_probe_refuses_with_original_evidence(tmp_path, reviewed_modules):
    gdml = tmp_path / "cap.gdml"
    material = materials.ALUMINIUM_6061.name
    gdml.write_text(f'''<gdml><solids><box name="w" x="100" y="100" z="100"/>
    <torus name="ring" rtor="10" rmin="0" rmax="2" deltaphi="360" aunit="deg"/></solids>
    <structure><volume name="World"><materialref ref="Vacuum"/><solidref ref="w"/>
    <physvol name="r"><volumeref ref="Leaf"/></physvol></volume>
    <volume name="Leaf"><materialref ref="{material}"/><solidref ref="ring"/></volume></structure>
    <setup><world ref="World"/></setup></gdml>''')
    records = [dict(placement_path="r[0]", classification="unsupported", material=material,
                    solid_tag="torus", scope="support", is_leaf=True, placement_assumptions=[])]
    raw = pd.DataFrame({k: pd.Series(dtype=t) for k, t in dict(lane="int64", s0_mm="float64", s1_mm="float64",
        material="object", source="object", sign="int64").items()})
    provider = G.GuardedNavigator(arm="inner", gdml=gdml,
        census=dict(schema="tracker-navigation-census-v1", records=records), modules=reviewed_modules,
        raw_hits=lambda *a: raw.copy(), world_plane_y_mm=40., endpoint_budget_mm=1e-10,
        min_cell_mm=1e-7, max_component_mm=1e-6, max_ray_unknown_mm=1e-5, max_cells=1)
    with pytest.raises(ValueError, match="pair refused"):
        provider(None, None, np.array([[10., 0., -5.]]), np.array([[0., 0., 1.]]), np.array([10.]), endpoint_budget_mm=1e-10)
    call = provider.calls[-1]
    assert call["status"] == "refused"
    assert call["reports"][0]["method"]["cap_reached"]
    assert "capped-placement-probe" in call["result"]["refusals"].reason.to_numpy()
    assert not call["result"]["summary"]["all_pairs_admitted"]
