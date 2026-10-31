"""Actual ordered evolution coupled to complete native planar pair and fresh-ray certificates."""
import os

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import eloss, materials
from scripts.ordered_torus_pair_probe import load_pair_sources
from trkfastsim import geometry, guardednavigation as G, inclusiveenvelope, nativeenvelope, orderedinterior as O, wideomission


@pytest.fixture(scope="module")
def modules():
    source = os.environ.get("TRKFS_REVIEWED_GEOMETRY_SOURCE")
    if source is None:
        pytest.skip("explicit reviewed geometry source required")
    return load_pair_sources(source)


@pytest.mark.parametrize("arm", ["inner", "outer"])
def test_native_scatter_requeries_both_projections(tmp_path, monkeypatch, modules, arm):
    mat = materials.ALUMINIUM_6061.name
    gdml = tmp_path / "native.gdml"
    gdml.write_text(f'''<gdml><solids><box name="world" x="100" y="100" z="100"/>
      <xtru name="slab"><twoDimVertex x="-2" y="-2"/><twoDimVertex x="2" y="-2"/>
      <twoDimVertex x="2" y="2"/><twoDimVertex x="-2" y="2"/><section zPosition="-1"/><section zPosition="1"/></xtru>
      </solids><structure><volume name="World"><materialref ref="Vacuum"/><solidref ref="world"/>
      <physvol name="slab"><volumeref ref="Slab"/></physvol></volume>
      <volume name="Slab"><materialref ref="{mat}"/><solidref ref="slab"/></volume>
      </structure><setup><world ref="World"/></setup></gdml>''')
    census = modules["navigationcensus"].census(gdml, np.array([[-20.]*3, [20.]*3]))
    probe = nativeenvelope.NativeOmission(inclusiveenvelope.CertifiedOmission(
        wideomission.WideOmission(modules["omittedpath"], authenticate_xtru=True), modules["omittedpath"]), modules["omittedpath"])
    combined = dict(modules, omittedpath=probe, supportenvelope=nativeenvelope.NativeEnvelope(modules["supportenvelope"], modules["supportnavigation"], probe))
    raw = pd.DataFrame({k: pd.Series(dtype=t) for k, t in dict(lane="int64", s0_mm="float64", s1_mm="float64", material="object", source="object", sign="int64").items()})
    provider = G.GuardedNavigator(arm=arm, gdml=gdml, census=census, modules=combined, raw_hits=lambda *a: raw.copy(),
        world_plane_y_mm=40., endpoint_budget_mm=1e-10, min_cell_mm=1e-9, max_component_mm=1e-6, max_ray_unknown_mm=1e-5,
        forward_only=True, allow_start_contact=True)
    geom = geometry.StrawGeometry(plane=np.array([0]), panel=np.array([0]), straw=np.array([0]),
        mid=np.array([[0., 0., 4.5]]), dir=np.array([[1., 0., 0.]]), halflen=np.array([20.]), radius=.2,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)), plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)), wall_rmax=.215, wire_r=0.)
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.full_like(p, .01))
    monkeypatch.setattr(eloss, "sample_loss", lambda rng, mat, p, length, cut: length * .01)
    from trkfastsim import steps
    monkeypatch.setattr(steps, "_kink_core", lambda rng, v, a, b:
        (np.tile([.0001, 0.], (len(v), 1)), np.zeros((len(v), 2)), np.zeros(len(v))))
    obj = type("Interior", (), {"shells": None})()
    rows = O.transport(obj, np.random.default_rng(19), np.array([[0., 0., -5.]]), np.array([[0., 0., 1.]]),
        np.array([80.]), geom, tail=False, tail_k=1., mode=O.MODE, max_path_mm=10., max_step_mm=1.,
        signed_endpoint_budget_mm=1e-10, navigation_provider=provider)
    assert len(rows["it"]) == 1 and obj.last_final["lost_support"][0] > .019
    assert len(provider.calls) > 1
    assert all(c["status"] == "complete" and c["result"]["summary"]["all_pairs_admitted"] for c in provider.calls)
