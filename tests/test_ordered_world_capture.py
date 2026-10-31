"""World capture addressing retains full keys and reproduces ordered height-band query selection."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts import ordered_world_capture as P


def test_expected_query_full_identity_and_empty_bands(monkeypatch):
    gen = pd.DataFrame(dict(run=1, subrun=2, event=[3, 4], pdg=13,
        x=0., y=[10., 10.], z=0., px=0., py=-100., pz=0.))
    monkeypatch.setattr(P.steps, "plane_y", lambda geom: 0.)
    monkeypatch.setattr(P.steps, "near_tracker", lambda *a, **kw: np.array([True, False]))
    world = SimpleNamespace(edges=np.array([10., 5., 0.]), band=lambda k: [object()] if k == 0 else [])
    mu, queries = P.expected_queries(gen, None, world)
    assert len(mu) == 2 and len(queries) == 1
    assert queries[0]["labels"][P.KEY].to_numpy().tolist() == [[1, 2, 3, 1]]
    np.testing.assert_array_equal(queries[0]["origin"], [[0., 10., 0.]])
    np.testing.assert_array_equal(queries[0]["direction"], [[0., -1., 0.]])
    np.testing.assert_array_equal(queries[0]["length"], [5.])
