"""Unit tests: the PrimaryFilter tracker term on synthetic StrawGasStep rows."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trkfastsim import tables as T                                      # noqa: E402


def _steps(n, sp, eion=1e-3, pz=3000.0, event=1):
    return pd.DataFrame({"run": 1400, "subrun": 1, "event": event, "sp": sp, "eion": np.full(n, eion),
                         "px": 0.0, "py": 0.0, "pz": pz})


def test_twelve_good_steps_from_one_particle_pass():
    f = T.trk_filter(_steps(12, 1))
    assert f.pass_trk.all() and (f.nmax == 12).all()


def test_steps_split_between_particles_fail():
    f = T.trk_filter(pd.concat([_steps(6, 1), _steps(6, 2)]))
    assert not f.pass_trk.any()


def test_cuts_on_eion_and_momentum():
    s = pd.concat([_steps(11, 1), _steps(1, 1, eion=1e-7), _steps(1, 1, pz=40.0)])
    assert not T.trk_filter(s).pass_trk.any()
