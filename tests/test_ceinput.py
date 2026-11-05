"""Unfiltered CE input census keeps no-step and no-entry events explicit."""
import numpy as np
import pandas as pd
import pytest

from mu2efastsim import tables
from trkfastsim.validation import ceinput


def test_census_primary_identity_zero_steps_and_side_entry_warning(tmp_path):
    base = pd.DataFrame({"run": np.full(3, 1430, dtype=np.uint32), "subrun": np.zeros(3, dtype=np.uint32),
                         "event": np.arange(1, 4, dtype=np.uint32)})
    g = base.assign(pdg=11, x=0.0, y=0.0, z=6000.0, px=100.0, py=0.0, pz=30.0, t=1.0)
    p = base.assign(key=7, pdg=11, isprim=1, ismcprim=1)
    v = g.iloc[:1].drop(columns="pdg").assign(sp=7, vdid=13, t=10.0)
    s = pd.concat([base.iloc[:2].assign(sp=7, sid=1, t=11.0), base.iloc[:1].assign(sp=8, sid=2, t=11.0)])
    path = str(tmp_path / "table.root")
    tables.write(path, {"gen": g, "particles": p, "vdsteps": v, "sgsteps": s}, as_ttree=True)
    report, d = ceinput.census([path], 3)
    assert report["generated_starts"] == 3 and report["primary_vd13"] == 1
    assert report["primary_step_events"] == 2 and report["primary_step_events_without_vd13"] == 1
    assert report["other_particle_steps"] == 1 and (d.primary_steps == 0).sum() == 1
    with pytest.raises(ValueError, match="externally audited"):
        ceinput.census([path], 4)
    bound, _ = ceinput.census([path], 3, base.iloc[::-1])
    assert bound["closed_census_keys_exact"]
    assert bound["birth_keys_sha256"] == report["birth_keys_sha256"]
    foreign = base.copy()
    foreign.loc[2, "event"] = 99
    with pytest.raises(AssertionError):
        ceinput.census([path], 3, foreign)
    with pytest.raises(AssertionError):
        ceinput.census([path], 3, base.iloc[:2])
    with pytest.raises(ValueError, match="duplicate"):
        ceinput.census([path], 3, pd.concat([base.iloc[:2], base.iloc[:1]]))
    with pytest.raises(ValueError, match="identities"):
        ceinput.census([path], 3, base.astype(float))
