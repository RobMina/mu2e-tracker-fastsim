"""Tracker reference tables: readers for the StrawGasStepTableDumper trees (art/TrkFastSim) and the fast sim.

Trees in ``dump/`` (columns in art/README.md): ``sgsteps`` (StrawGasStep, Mu2e frame), ``surfsteps`` (SurfaceStep,
detector frame), ``vdsteps``, ``trajpts`` (MCTrajectory points), ``particles`` (SimParticle + ismcprim), ``gen``,
``subruns``. Events are keyed by (run, subrun, event); ``sp`` is the SimParticle key (``particles.key``).
"""
import numpy as np

from mu2efastsim.tables import KEY, expand, livetime, read, read_many  # noqa: F401

# production PrimaryFilter, tracker term (Production JobConfig/primary/prolog.fcl, DetectorStepFilter)
FILTER_MIN_TRK_STEPS = 12
FILTER_MIN_STEP_EION = 1.0e-6     # MeV
FILTER_MIN_MOM = 50.0             # MeV/c
FILTER_MAX_MOM = 1.0e6            # MeV/c
# production PrimaryFilter, CRV term (ExtractedCRY.fcl adds CrvSteps)
FILTER_MIN_CRV_STEPS = 1
FILTER_MIN_CRV_EDEP = 0.1         # MeV, visible


def good_steps(sgs):
    """Mask of StrawGasSteps that count for the PrimaryFilter tracker term (no time cut, any PDG)."""
    p = np.sqrt(sgs.px ** 2 + sgs.py ** 2 + sgs.pz ** 2)
    return (sgs.eion > FILTER_MIN_STEP_EION) & (p > FILTER_MIN_MOM) & (p < FILTER_MAX_MOM)


def trk_filter(sgs):
    """Per event: max good-step count of any one SimParticle and the tracker-term decision (DataFrame indexed by KEY)."""
    n = sgs[good_steps(sgs)].groupby(KEY + ["sp"]).size().groupby(KEY).max().rename("nmax")
    out = n.to_frame()
    out["pass_trk"] = out.nmax >= FILTER_MIN_TRK_STEPS
    return out
