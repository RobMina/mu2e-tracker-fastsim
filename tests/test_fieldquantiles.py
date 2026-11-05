"""Cosmic paired quantiles keep own-truth and common-surface contracts."""
import numpy as np
import pandas as pd

from trkfastsim.validation.fieldquantiles import report


def test_relative_quantiles_identical_and_surface_exclusion():
    n = 20
    d = pd.DataFrame({"run": 1430, "subrun": 0, "event": np.arange(n), "p_first": 1500.0,
                      "fast_ntrk": 1, "g4_ntrk": 1, "fast_sid": 0, "g4_sid": 0,
                      "fast_p_first_own": 1500.0, "g4_p_first_own": 1500.0,
                      "fast_p_reco": np.linspace(1490, 1510, n), "g4_p_reco": np.linspace(1490, 1510, n)})
    d.loc[0, "fast_sid"] = 4
    d.loc[1, "fast_ntrk"] = 0
    r = report(d)
    assert r["common_reco_surface_events"] == n - 2
    b = r["bins"][1]
    assert b["events"] == n - 2
    np.testing.assert_array_equal(b["difference"], [0, 0])
    np.testing.assert_array_equal(b["paired_error"], [0, 0])
