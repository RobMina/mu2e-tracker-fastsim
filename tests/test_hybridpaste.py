"""hybridpaste: Geant4 primary-muon steps kept, other steps removed, muon keys renumbered 1..n per event."""
import numpy as np
import pandas as pd

from trkfastsim.validation import hybridpaste as HP


def test_primary_muons_keeps_primary_muon_steps_only(tmp_path):
    n = 6
    key = {"run": np.full(n, 1400), "subrun": np.full(n, 21), "event": np.array([1, 1, 1, 1, 2, 2])}
    sg = {**key, "nsgs": np.full(n, 3), "sid": np.arange(n), "plane": np.arange(n), "panel": np.zeros(n, int),
          "straw": np.arange(n), "upanel": np.arange(n) * 6, "shape": np.zeros(n, int), "ion": np.zeros(n, int),
          "sp": np.array([40, 7, 40, 52, 3, 9]), "eion": np.full(n, 1e-3), "len": np.full(n, 5.0), "width": np.zeros(n),
          "t": np.array([2.0, 1.0, 1.0, 3.0, 0.0, 1.0]), **{c: np.zeros(n) for c in
                                                             ("x0", "y0", "z0", "x1", "y1", "z1", "px", "pz")},
          "py": np.full(n, -1e3)}
    pa = {"run": np.full(5, 1400), "subrun": np.full(5, 21), "event": np.array([1, 1, 1, 2, 2]),
          "key": np.array([7, 40, 52, 3, 9]), "pdg": np.array([11, 13, -13, 13, 11]), "isprim": np.array([0, 1, 1, 1, 0])}
    from mu2efastsim import tables as core_tables
    f = tmp_path / "t.root"
    core_tables.write(str(f), {"sgsteps": pd.DataFrame(sg), "particles": pd.DataFrame(pa)}, as_ttree=True)
    steps, parts = HP.primary_muons(str(f))
    assert len(steps) == 4                                         # the e- steps are gone
    assert steps[steps.event == 1].sp.tolist() == [1, 1, 2]        # keys 40, 52 -> 1, 2, time-ordered per muon
    assert steps[steps.event == 1].t.tolist() == [1.0, 2.0, 3.0]
    assert steps.nsgs.tolist() == [3, 3, 3, 1]
    assert parts.key.tolist() == [1, 2, 1] and parts.pdg.tolist() == [13, -13, 13]
    assert (parts.parent == -1).all() and (parts.isprim == 1).all()
