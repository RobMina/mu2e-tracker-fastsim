"""Fast StrawGasSteps for primary muons (tier 1a): straight lines through the straw gas, Geant4-matched ionisation.

Per primary muon, from its generated vertex (no world material and no multiple scattering yet, as CRV stage A):
  1. straw gas crossings (crossings.straw_crossings), ordered along the track;
  2. |p| at each straw = |p0| minus the unrestricted air loss to that point (the tracker is in air; the gas and walls
     remove < 1 MeV over the whole stack);
  3. eIon = mu2efastsim.eloss.sample_loss in the straw gas with the tracker region cut, path = chord length,
     time = vertex time + path to entry / (beta c);
  4. step type as MakeStrawGasSteps::setStepType with zero field: line (charged), minion if beta*gamma > 0.5 and
     KE > 20 MeV, else highion; width 0.
Delta rays, other secondaries and steps of non-primary particles are not produced (decision T2: G4 library, later).
Positions are written in the detector frame, as Geant4's StrawSD stores them (geometry.to_detector).
Output rows follow the sgsteps tree of StrawGasStepTableDumper, so the reference readers and comparisons apply.
"""
import numpy as np
import pandas as pd

from mu2efastsim import eloss, materials
from mu2efastsim.tables import KEY, primary_muons

from . import crossings, geometry

C_MM_PER_NS = 299.792458
MINION_BG, MINION_KE = 0.5, 20.0          # MakeStrawGasSteps minionBetaGamma, minionKineticEnergy defaults
TRACKER_CUT_MM = 0.001                    # ExtractedCRY minRangeRegionCuts TrackerMother

# StrawGas from the v04 GDML export (Ar/CO2 80/20 by volume: mass fractions Ar 0.784, O 0.157, C 0.0589)
STRAW_GAS = materials.Material("StrawGas", 0.001696, 158.62097145526,
                               {"Ar": 0.784057035890315, "O": 0.15700963928136, "C": 0.0589333248283256},
                               cuts_e_MeV={"tracker": 990e-6})


def delta_cut_mev(mat=STRAW_GAS):
    """Delta-electron production threshold [MeV] in the tracker region: the 0.001 mm range cut is below Geant4's
    990 eV lower edge, so the edge applies (assumption, to be checked against the reference delta spectrum)."""
    return mat.cuts_e_MeV["tracker"]


def make_steps(gen, geom, rng, air_loss=True):
    """Return (sgsteps, particles) DataFrames for the primary muons in `gen` (one CRY muon per event)."""
    mu = primary_muons(gen)
    p0 = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy()
    o = mu[["x", "y", "z"]].to_numpy()
    d = mu[["px", "py", "pz"]].to_numpy() / p0[:, None]
    xs = crossings.straw_crossings(o, d, geom)
    it = xs[:, 0].astype(np.int64)
    js = xs[:, 1].astype(np.int64)
    s_in, s_out = xs[:, 2], xs[:, 3]
    path = s_out - s_in
    p = p0[it]
    if air_loss:
        p = np.maximum(p - eloss.dedx_restricted(materials.AIR, p0[it], 1e9) * s_in, 1.0)
    eion = eloss.sample_loss(rng, STRAW_GAS, p, path, delta_cut_mev())
    e = np.sqrt(p ** 2 + eloss.MMU ** 2)
    bg = p / eloss.MMU
    ke = e - eloss.MMU
    t = mu.t.to_numpy()[it] + s_in / ((p0[it] / np.sqrt(p0[it] ** 2 + eloss.MMU ** 2)) * C_MM_PER_NS)
    pos0 = geometry.to_detector(o[it] + d[it] * s_in[:, None])
    pos1 = geometry.to_detector(o[it] + d[it] * s_out[:, None])
    steps = pd.DataFrame({k: mu[k].to_numpy()[it] for k in KEY})
    steps["nsgs"] = np.bincount(it, minlength=len(mu))[it]
    steps["sid"] = geom.sid[js]
    steps["plane"] = geom.plane[js]
    steps["panel"] = geom.panel[js]
    steps["straw"] = geom.straw[js]
    steps["upanel"] = geom.upanel[js]
    steps["shape"] = 0
    steps["ion"] = np.where((bg > MINION_BG) & (ke > MINION_KE), 0, 1)
    steps["sp"] = 1
    steps["eion"] = eion
    steps["len"] = path
    steps["width"] = 0.0
    steps["t"] = t
    for j, c in enumerate("xyz"):
        steps[f"{c}0"] = pos0[:, j]
        steps[f"{c}1"] = pos1[:, j]
        steps[f"p{c}"] = d[it, j] * p
    parts = pd.DataFrame({k: mu[k].to_numpy() for k in KEY})
    parts["key"] = 1
    parts["parent"] = -1
    parts["pdg"] = mu.pdg.to_numpy()
    parts["isprim"] = 1
    parts["ismcprim"] = 1
    return steps, parts
