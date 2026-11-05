"""Synthetic event-key and identity checks for the CE tracker hand-off."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import tables
from trkfastsim.validation import fieldce


def test_result_entries_retains_integer_event_identity_and_plane_selection():
    g = pd.DataFrame({"run": [1430, 1430, 1430], "subrun": [0, 100, 200],
                      "event": [2**54 + 1, 2**54 + 3, 2**54 + 5]})
    r = SimpleNamespace(fate=np.array([1, 2, 1]), T=np.array([104.0, 0.0, 100.0]),
                        x=np.arange(9).reshape(3, 3), u=np.tile([0.0, 0.0, 1.0], (3, 1)),
                        t=np.array([100.0, 200.0, 300.0]))
    ent = fieldce.result_entries(g, r, 1)
    np.testing.assert_array_equal(ent.event, g.event.iloc[[0, 2]])
    np.testing.assert_array_equal(ent.subrun, [0, 200])
    assert (ent.pdg == 11).all() and (ent.sp == 1).all()
    np.testing.assert_allclose(ent.pz, np.sqrt(r.T[[0, 2]] * (r.T[[0, 2]] + 2 * fieldce.fs.eloss.ME)))
    with pytest.raises(ValueError, match="lengths differ"):
        fieldce.result_entries(g.iloc[:2], r, 1)


def test_entries_select_primary_identity_before_first_crossing(tmp_path):
    path = tmp_path / "reference.root"
    base = dict(run=1430, subrun=0, event=2)
    parts = pd.DataFrame([{**base, "key": 7, "pdg": 11, "isprim": 1, "ismcprim": 1},
                          {**base, "key": 8, "pdg": 11, "isprim": 0, "ismcprim": 0}])
    v = pd.DataFrame([{**base, "sp": sp, "vdid": vd, "x": 0.0, "y": 0.0, "z": 8539.9,
                       "px": 90.0, "py": 0.0, "pz": 40.0, "t": t}
                      for sp, vd, t in [(8, 13, 1.0), (7, 13, 5.0), (7, 13, 9.0), (7, 11, 2.0)]])
    tables.write(str(path), {"particles": parts, "vdsteps": v}, as_ttree=True)
    ent = fieldce.primary_entries([str(path)])
    assert len(ent) == 1 and ent.sp.iloc[0] == 7 and ent.t.iloc[0] == 5.0


def test_starts_refuses_multiple_electrons_per_event(tmp_path):
    path = tmp_path / "bad.root"
    g = pd.DataFrame({"run": [1, 1], "subrun": [0, 0], "event": [2, 2], "pdg": [11, 11]})
    tables.write(str(path), {"gen": g}, as_ttree=True)
    with pytest.raises(ValueError, match="exactly one"):
        fieldce.starts([str(path)])


def test_starts_falls_back_to_primary_simparticle_when_gen_holds_parent_stop(tmp_path):
    path = tmp_path / "resampled.root"
    base = {"run": 1430, "subrun": 0, "event": 2, "pdg": 11, "isprim": 1, "ismcprim": 1}
    p = pd.DataFrame([{**base, "sx": -3904.0, "sy": 0.0, "sz": 6000.0,
                       "spx": 100.0, "spy": 0.0, "spz": 30.0, "st": 500.0}])
    g = pd.DataFrame([{"run": 1430, "subrun": 0, "event": 2, "pdg": 13}])
    tables.write(str(path), {"gen": g, "particles": p}, as_ttree=True)
    s = fieldce.starts([str(path)])
    assert s.event.iloc[0] == 2 and s.x.iloc[0] == -3904.0 and s.t.iloc[0] == 500.0


def test_unfiltered_reference_requires_external_event_count_and_preserves_default_label():
    g = pd.DataFrame({"event": [1, 2]})
    assert fieldce.reference_contract(g, "selected-dts")["conditional_on_dts_selection"]
    with pytest.raises(ValueError, match="externally audited"):
        fieldce.reference_contract(g, "unfiltered")
    with pytest.raises(ValueError, match="externally audited"):
        fieldce.reference_contract(g, "unfiltered", 3)
    assert not fieldce.reference_contract(g, "unfiltered", 2)["conditional_on_dts_selection"]


def test_unfiltered_truncation_is_rejected_before_transport(tmp_path, monkeypatch):
    g = pd.DataFrame({"run": [1, 1], "subrun": [0, 0], "event": [1, 2]})
    monkeypatch.setattr(fieldce, "starts", lambda paths: g)
    monkeypatch.setattr(fieldce.tables, "expand", lambda paths: paths)
    args = ["unused.root", "--outdir", str(tmp_path), "--gdml", "unused.gdml", "--maps", "unused.map",
            "--mode", "chain", "--reference-selection", "unfiltered", "--expected-starts", "1", "--max-events", "1"]
    with pytest.raises(SystemExit):
        fieldce.main(args)
    assert not (tmp_path / "handoff.json").exists()
