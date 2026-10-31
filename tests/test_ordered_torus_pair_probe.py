"""Torus-pair probes retain raw signed source ownership; no material or transport promotion."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import ordered_torus_pair_probe as P


def test_raw_sources_keep_labels_and_negative_parent_terms(monkeypatch):
    monkeypatch.setattr(P.world, "load_tubes", lambda *a, **k: [("steel", 1, 2)])
    monkeypatch.setattr(P.world, "CoaxialShells", lambda *a: object())
    monkeypatch.setattr(P.materialpath, "tube_hits", lambda *a: pd.DataFrame(dict(lane=[0], s0_mm=[1.], s1_mm=[2.], material=["tube:0|steel"])))
    monkeypatch.setattr(P.world, "signed_subtree", lambda *a, **k: ([(1, "mother|steel")], {}))
    packed = SimpleNamespace(mats=["calo:mother|steel"], mat=np.array([0]))
    monkeypatch.setattr(P.world, "SignedPieceSet", lambda *a, **k: object())
    monkeypatch.setattr(P.convexpath, "_sets", lambda *a: [(1, packed), (-1, packed)])
    monkeypatch.setattr(P.convexpath, "_hits", lambda *a: iter([(np.array([0]), np.array([0]), np.array([3.]), np.array([4.]))]))
    result = P.stored_hits("fixture", None, np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([5.]))
    assert result.source.tolist() == ["tube:0", "calo:mother", "calo:mother"]
    assert result.material.tolist() == ["steel"] * 3
    assert result.sign.tolist() == [1, 1, -1]
    assert len(result) == 3


def test_geometry_source_loader_refuses_existing_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "load_geometry_sources", lambda *a: {})
    monkeypatch.setitem(P.sys.modules, "trkfastsim.supportnavigation", object())
    with pytest.raises(ValueError, match="refusing imported"):
        P.load_pair_sources(tmp_path)


def test_cached_labelled_query_matches_existing_reader_and_loads_once(monkeypatch):
    calls = []
    monkeypatch.setattr(P.world, "load_tubes", lambda *a, **k: calls.append("load") or [("steel", 1, 2)])
    monkeypatch.setattr(P.world, "CoaxialShells", lambda *a: object())
    monkeypatch.setattr(P.materialpath, "tube_hits", lambda *a: pd.DataFrame(dict(lane=[0], s0_mm=[1.], s1_mm=[2.], material=["tube:0|steel"])))
    monkeypatch.setattr(P.world, "signed_subtree", lambda *a, **k: ([], {}))
    query = P.StoredHitQuery("fixture", None)
    args = (np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([5.]))
    first = query(*args)
    pd.testing.assert_frame_equal(first, query(*args), check_exact=True)
    assert calls == ["load"]
    pd.testing.assert_frame_equal(first, P.stored_hits("fixture", None, *args), check_exact=True)
