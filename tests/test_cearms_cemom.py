"""validation.cearms (Geant4 CE reference arm) and validation.cemom (paired CE comparison): synthetic tables, no data files."""
import numpy as np
import pandas as pd
import pytest

from mu2efastsim import tables
from trkfastsim.validation import cearms, cemom


def _tables(path):
    k = dict(run=[1430] * 4, subrun=[0] * 4)
    part = pd.DataFrame({**k, "event": [5, 5, 5, 6], "key": [1, 7, 9, 7], "pdg": [2212, 11, 11, 11],
                         "isprim": [1, 0, 0, 0], "ismcprim": [0, 1, 0, 1]})
    vd = pd.DataFrame({"run": 1430, "subrun": 0, "event": [5, 5, 5, 6], "vdid": [13, 13, 13, 12], "sp": [7, 7, 9, 7],
                       "x": 0.0, "y": 0.0, "z": 8539.9, "px": 0.0, "py": 60.0, "pz": [80.0, 70.0, 10.0, 80.0],
                       "t": [1000.0, 1100.0, 1001.0, 900.0]})
    n = 5
    sg = pd.DataFrame({c: np.zeros(n) for c in cearms.STEP_COLS})
    sg["run"], sg["subrun"], sg["event"] = 1430, 0, [5, 5, 5, 5, 6]
    sg["sp"] = [7, 7, 9, 7, 7]
    sg["t"] = [999.0, 1003.0, 1004.0, 1105.0, 950.0]
    tables.write(str(path), {"particles": part, "vdsteps": vd, "sgsteps": sg}, as_ttree=True)


def test_reference_arm_takes_the_primary_electron_from_its_first_vd13_crossing(tmp_path):
    f = tmp_path / "t.root"
    _tables(f)
    ent = cearms.primary_entries([str(f)])
    assert list(ent.event) == [5]                       # event 6 has no VD13 crossing of its primary
    assert ent.t.iloc[0] == 1000.0 and ent.pz.iloc[0] == 80.0
    st = cearms.primary_steps([str(f)], ent)
    assert sorted(st.t) == [1003.0, 1105.0]             # the step before VD13 and the other particle's step are left out
    assert (st.sp == 1).all() and list(st.columns) == cearms.STEP_COLS


def test_gauss_core_recovers_a_gaussian_and_pulls_are_zero_for_identical_arms():
    rng = np.random.default_rng(1)
    x = rng.normal(-0.05, 0.2, 20000)
    m, s = cemom.gauss_core(x)
    assert m == pytest.approx(-0.05, abs=0.01) and s == pytest.approx(0.2, rel=0.03)
    p = cemom.binned_pulls(x, x.copy(), cemom.RES_EDGES)
    assert p["chi2_paired"] == 0.0 and not np.any(p["diff"])


def test_paired_pulls_detect_a_shift():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.2, 20000)
    p = cemom.binned_pulls(x + 0.1, x, cemom.RES_EDGES)
    assert p["chi2_paired"] / p["ndf"] > 5


def test_ce_selection_needs_every_cut():
    base = dict(fit=True, pdg=11.0, pz=60.0, fitcon=0.5, tandip=0.8, d0=20.0, rmax=500.0)
    rows = [base, {**base, "fit": False}, {**base, "pdg": -11.0}, {**base, "pz": -60.0}, {**base, "fitcon": 1e-5},
            {**base, "tandip": 1.2}, {**base, "d0": 150.0}, {**base, "rmax": 700.0}, {**base, "pz": np.nan}]
    assert list(cemom.ce_selected(pd.DataFrame(rows))) == [True] + [False] * 8


def test_assemble_keeps_events_without_a_track_and_rejects_foreign_events():
    den = pd.DataFrame({"run": np.uint64(1), "subrun": np.uint64(0), "event": np.arange(4, dtype=np.uint64)})
    arm = den.iloc[[0, 2]].assign(fit=True, ce=[True, False], p_reco=[104.0, 103.0])
    tr = den.assign(p_true=104.1)
    d = cemom.assemble(den, {"g4": arm}, {"g4": tr})
    assert len(d) == 4 and list(d.g4_ce) == [True, False, False, False]
    assert d.g4_res.iloc[0] == pytest.approx(-0.1)
    foreign = arm.assign(event=np.uint64(9))
    with pytest.raises(ValueError):
        cemom.assemble(den, {"g4": foreign}, {"g4": tr})
