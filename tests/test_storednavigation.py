"""Same-medium union is an explicit set policy, never a tolerance."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim import materialpath, storednavigation as N


def hits(rows):
    return pd.DataFrame(rows, columns=["lane", "s0_mm", "s1_mm", "material"])


def test_union_roundoff_and_genuine_same_material_overlap_once():
    h = hits([(0, 1., 2., "a"), (0, np.nextafter(2., 0.), 3., "a"),
              (0, 2.5, 4., "a"), (1, 1., 2., "b")])
    r = N.union_support(h, [5., 5.])
    assert r[r.material == "a"][["s0_mm", "s1_mm"]].to_numpy().tolist() == [[1., 4.]]
    pd.testing.assert_frame_equal(r, N.union_support(h.iloc[::-1], [5., 5.]))


def test_gap_is_never_coalesced_even_one_ulp():
    end = np.nextafter(2., 3.)
    r = N.union_support(hits([(0, 1., 2., "a"), (0, end, 3., "a")]), [4.])
    assert ((r.material == "vacuum") & (r.s0_mm == 2.) & (r.s1_mm == end)).any()


def test_cross_material_overlap_refused_even_one_ulp():
    with pytest.raises(materialpath.IntervalOverlap):
        N.union_support(hits([(0, 1., 2., "a"), (0, np.nextafter(2., 0.), 3., "b")]), [4.])


def test_explicit_endpoint_arm_applies_before_tube_material_sweep(monkeypatch):
    from mu2efastsim.world import CoaxialShells
    h = hits([(0, 0., 1., "Al"), (0, np.nextafter(1., 0.), 2., "Cu")])
    monkeypatch.setattr(materialpath, "tube_hits", lambda *a: h.copy())
    empty = CoaxialShells([], [0., 0., 0.])
    with pytest.raises(materialpath.IntervalOverlap):
        N.support_intervals(empty, [[0., 0., 0.]], [[1., 0., 0.]], [2.])
    evidence = []
    result = N.support_intervals(empty, [[0., 0., 0.]], [[1., 0., 0.]], [2.],
                                 endpoint_budget_mm=1e-10, endpoint_evidence=evidence.append)
    assert result.material.tolist() == ["Al", "Cu"]
    assert evidence[0]["summary"]["changed_endpoints"] > 0
