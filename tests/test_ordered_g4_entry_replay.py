"""G4 entrance replay keeps secondary exclusion and full primary/straw identity."""
import numpy as np
import pandas as pd

from scripts.ordered_g4_entry_replay import entrances, observables


def test_primary_initial_state_and_nontrivial_shared_interval():
    reference = pd.DataFrame(dict(run=1, subrun=1, event=[1, 1, 1], sp=[2, 2, 9], sid=[7, 8, 7],
        t=[10., 11., 9.], px=0., py=100., pz=0., x0=0., y0=0., z0=0.))
    particles = pd.DataFrame(dict(run=1, subrun=1, event=1, key=[2, 9], pdg=[13, 13], ismcprim=[1, 0]))
    first, selected = entrances(reference, particles, 8)
    assert first.sp.tolist() == [2] and selected.sid.tolist() == [7, 8]
    entries = dict(it=np.array([0, 0]), dt=np.array([0., 1.1]), path=np.array([0., 300.]), lost=np.array([0., .1]))
    result, paired = observables(entries, pd.DataFrame(dict(sid=[7, 8])), first, selected)
    assert result["shared_straws"] == 2 and result["later_shared_straws"] == 1
    np.testing.assert_allclose(result["later_time_ns"]["mean"], .1)
    assert paired.sp.tolist() == [2, 2]
