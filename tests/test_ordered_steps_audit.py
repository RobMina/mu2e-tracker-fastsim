"""Physical state audit rejects extension, clock mismatch and artificial escape."""
import numpy as np
import pandas as pd
import pytest

from scripts.ordered_steps_audit import audit
from trkfastsim.flighttime import linear_kinetic_dt_ns
from trkfastsim import ledger


def frames():
    mu = pd.DataFrame(dict(run=[1], subrun=[1], event=[1], sp=[1], path_interior_mm=[10.],
        p_plane_MeV_c=[ledger.momentum(20.)], ke_final_MeV=[19.], lost_support_MeV=[0.],
        lost_thin_MeV=[1.], stopped_interior=[0]))
    dt = linear_kinetic_dt_ns(10., 20., 19.)
    v = pd.DataFrame(dict(run=[1, 1], subrun=[1, 1], event=[1, 1], sp=[1, 1], path=[0., 10.],
                         kinetic_MeV=[20., 19.], time_ns=[7., 7. + dt]))
    seg = pd.DataFrame(dict(run=[1], subrun=[1], event=[1], sp=[1], sid=[0], path_mm=[0.],
        deposited_gas_MeV=[.5], unassigned_gas_MeV=[.5], gas_primary_debit_MeV=[1.],
        escaping_gas_MeV=[np.nan], t_integrated_ns=[7.]))
    return mu, seg, v


def test_audit_pass_and_failures():
    mu, seg, v = frames()
    assert audit(mu, seg, v)["entries"] == 1
    for column, value, match in [("path", 11., "integral|extension"), ("time_ns", 8., "integral")]:
        bad = v.copy()
        bad.loc[1, column] = value
        with pytest.raises(ValueError, match=match):
            audit(mu, seg, bad)
    bad = seg.assign(escaping_gas_MeV=0.)
    with pytest.raises(ValueError, match="escaping"):
        audit(mu, bad, v)
