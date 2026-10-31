"""Tube provenance retains loader indices even when straw gas is excluded."""
import json

import numpy as np
import pandas as pd

from scripts import ordered_tube_witness as W


def test_skipped_gas_does_not_renumber_sources(tmp_path, monkeypatch):
    query = tmp_path / "query"
    query.mkdir()
    np.savez(query / "rays.npz", source_selection_bbox_mm=np.array([[-10.]*3, [10.]*3]))
    pd.DataFrame(dict(source=["tube:1"], material=["steel"])).to_csv(query / "adjusted_hits.csv", index=False)
    gdml = tmp_path / "fixture.gdml"
    gdml.write_text('<gdml><solids><tube name="t" rmax="2" z="4"/></solids><structure>'
        '<volume name="World"><materialref ref="Vacuum"/><solidref ref="t"/></volume>'
        '<volume name="Shell"><materialref ref="steel"/><solidref ref="t"/></volume>'
        '</structure></gdml>')
    def load(*args, names, **kwargs):
        names.extend(["World/Gas", "World/Shell"])
        return [(m, np.eye(3), np.zeros(3), 0., 2., 2., 0., 2*np.pi) for m in ("StrawGas", "steel")]
    monkeypatch.setattr(W.world, "load_tubes", load)
    out = tmp_path / "receipt.json"
    result = W.run(query, gdml, out)
    assert result["sources"][0]["logical_path"] == "World/Shell"
    assert result["sources"][0]["source"] == "tube:1"
    assert not result["admitted_for_physics"]
    assert json.loads(out.read_text())["sources"][0]["logical_matches"][0]["volume"] == "Shell"
