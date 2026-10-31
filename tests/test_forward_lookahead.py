"""A query horizon is not a terminal path or permission to cross the forward split."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import eloss, materials
from trkfastsim import geometry, orderedinterior as O, steps


class Uniform:
    policy = "fixture-uniform-medium"
    def __init__(self, name):
        self.name, self.calls = name, []
    def __call__(self, shells, geom, o, d, lengths, **kw):
        self.calls.append((o.copy(), d.copy(), lengths.copy()))
        return pd.DataFrame(dict(lane=np.arange(len(o)), s0_mm=np.zeros(len(o)), s1_mm=lengths,
            material=self.name, kind="vacuum" if self.name == "vacuum" else "support"))


def run(monkeypatch, *, y=0., plane=1000., length=2000., material="vacuum", loss=.01, momentum=80., bounded=True):
    geom = geometry.StrawGeometry(plane=np.array([0]), panel=np.array([0]), straw=np.array([0]),
        mid=np.array([[100., 100., 100.]]), dir=np.array([[1., 0., 0.]]), halflen=np.array([1.]), radius=.2,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)), plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)), wall_rmax=.215, wire_r=0.)
    provider = Uniform(material)
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.full_like(p, loss))
    monkeypatch.setattr(eloss, "sample_loss", lambda rng, mat, p, l, cut: np.zeros_like(p))
    monkeypatch.setattr(steps, "_kink_core", lambda rng, v, a, b:
        (np.zeros((len(v), 2)), np.zeros((len(v), 2)), np.zeros(len(v))))
    obj = SimpleNamespace(shells=None)
    O.transport(obj, np.random.default_rng(19), np.array([[0., y, 0.]]), np.array([[0., 1., 0.]]), np.array([momentum]),
        geom, tail=False, tail_k=1., mode=O.MODE, max_path_mm=length, navigation_provider=provider,
        forward_plane_y_mm=plane, bounded_forward_lookahead=bounded)
    return obj, provider


def test_short_query_reaches_true_requested_arc_not_early_terminal(monkeypatch):
    obj, provider = run(monkeypatch, length=900., plane=1000.)
    assert obj.last_final["path"][0] == 900. and obj.last_stats["bounded_lookahead_queries"] == 0


def test_material_stops_before_distant_lookahead_crossing(monkeypatch):
    obj, provider = run(monkeypatch, material=materials.ALUMINIUM_6061.name, loss=.1, momentum=12.)
    assert obj.last_final["stopped"][0] and obj.last_final["path"][0] < 10.
    assert provider.calls[0][2][0] == 100.
    assert obj.last_stats["bounded_lookahead_queries"] == 1
    assert obj.last_final["o"][0, 1] < 1000.


def test_query_exhaustion_requeries_and_still_refuses_actual_crossing(monkeypatch):
    with pytest.raises(ValueError, match="crosses world handoff"):
        run(monkeypatch)


def test_default_full_remaining_leg_still_refuses(monkeypatch):
    with pytest.raises(ValueError, match="crosses world handoff"):
        run(monkeypatch, bounded=False)


def test_near_crossing_within_one_max_step_never_shortened(monkeypatch):
    with pytest.raises(ValueError, match="crosses world handoff"):
        run(monkeypatch, y=995.)
