"""Fast StrawGasSteps for primary muons (tier 1a): straight lines through the straw gas, Geant4-matched ionisation.

Stages (as the CRV transport): A = straight line from the generated vertex, air mean loss, no MSC; C = the GDML world
material above the tracker (building, overburden, CRV modules, electronics) in height bands down to a plane just above
the straws (mu2efastsim.transport.world_loss), energy loss per material and correlated Highland scattering to that
plane (msc_to_plane), then straight inside the tracker volume.

Per primary muon:
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
from mu2efastsim import transport as tr
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


Y_PLANE_MARGIN = 20.0        # mm above the highest straw box: the MSC plane
FOOTPRINT_MARGIN = 1500.0    # mm around the straw boxes' x-z footprint at the plane (as the CRV's)
# height bands for the world column above the tracker: CRV taggers and modules (4.2-4.7 m), hatch concrete
# (3.2-5.5 m), overburden above, tracker-area electronics (0.6-0.9 m) just above the straws
BAND_EDGES_Y = np.array([15400.0, 12000.0, 10000.0, 8000.0, 6000.0, 5000.0, 4500.0, 4000.0, 3000.0, 2000.0, 1000.0])


def plane_y(geom):
    return float(crossings.panel_boxes(geom)[1][:, 1].max()) + Y_PLANE_MARGIN


def near_tracker(o, d, geom, y_plane, margin=FOOTPRINT_MARGIN):
    """Downward tracks whose straight line crosses y = y_plane within `margin` of the straws' x-z footprint."""
    lo, hi = crossings.panel_boxes(geom)
    lo, hi = lo.min(0), hi.max(0)
    down = d[:, 1] < 0
    L = np.where(down, (o[:, 1] - y_plane) / -np.where(down, d[:, 1], -1.0), 0.0)
    x = o[:, 0] + d[:, 0] * L
    z = o[:, 2] + d[:, 2] * L
    return down & (x > lo[0] - margin) & (x < hi[0] + margin) & (z > lo[2] - margin) & (z < hi[2] + margin)


def load_world(gdml_path, geom):
    """GDML world pieces above the MSC plane, without the straw gas (the tracker itself is handled by crossings)."""
    from mu2efastsim import world as world_mod
    return [q for q in world_mod.load(gdml_path, y_min=plane_y(geom)) if q[0] != "StrawGas"]


def to_plane(gen, geom, rng, world):
    """Stage C transport of the primary muons to the plane above the tracker.

    Returns (mu, origin [N,3], direction [N,3], |p| at the origin [N], path already travelled [N], alive [N]); tracks
    not near the tracker keep their generated state (they cannot reach a straw).
    """
    mu = primary_muons(gen)
    p0 = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy()
    o = mu[["x", "y", "z"]].to_numpy()
    d = mu[["px", "py", "pz"]].to_numpy() / p0[:, None]
    y_pl = plane_y(geom)
    edges = np.r_[BAND_EDGES_Y[BAND_EDGES_Y > y_pl], y_pl]
    near = near_tracker(o, d, geom, y_pl)
    p = p0.copy()
    L = np.zeros(len(mu))
    alive = np.ones(len(mu), bool)
    if near.any():
        pw, xx0, ok, bxx0, bh = tr.world_loss(rng, o[near], d[near], p0[near], world, edges)
        p_msc = np.maximum(np.sqrt(p0[near] * np.maximum(pw, 1.0)), 1.0)   # geometric mean over the column
        on, dn, Ln = tr.msc_to_plane(rng, o[near], d[near], p_msc, y_pl, eloss.radiation_length_mm(materials.AIR),
                                     xx0, bxx0, bh)
        o, d = o.copy(), d.copy()
        o[near], d[near], L[near], p[near] = on, dn, Ln, pw
        alive[np.flatnonzero(near)[~ok]] = False
    return mu, o, d, p, L, alive


def make_steps(gen, geom, rng, air_loss=True, stage="A", world=None):
    """Return (sgsteps, particles) DataFrames for the primary muons in `gen` (one CRY muon per event)."""
    if stage == "C":
        mu, o, d, p_start, t_off, alive = to_plane(gen, geom, rng, world)
        p0 = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy()
    else:
        mu = primary_muons(gen)
        p0 = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy()
        o = mu[["x", "y", "z"]].to_numpy()
        d = mu[["px", "py", "pz"]].to_numpy() / p0[:, None]
        p_start, t_off, alive = p0.copy(), np.zeros(len(mu)), np.ones(len(mu), bool)
    xs = crossings.straw_crossings(o, d, geom)
    xs = xs[alive[xs[:, 0].astype(np.int64)]] if len(xs) else xs
    it = xs[:, 0].astype(np.int64)
    js = xs[:, 1].astype(np.int64)
    s_in, s_out = xs[:, 2], xs[:, 3]
    path = s_out - s_in
    p = p_start[it]
    if air_loss:
        p = np.maximum(p - eloss.dedx_restricted(materials.AIR, p_start[it], 1e9) * s_in, 1.0)
    eion = eloss.sample_loss(rng, STRAW_GAS, p, path, delta_cut_mev())
    e = np.sqrt(p ** 2 + eloss.MMU ** 2)
    bg = p / eloss.MMU
    ke = e - eloss.MMU
    t = mu.t.to_numpy()[it] + (s_in + t_off[it]) / ((p0[it] / np.sqrt(p0[it] ** 2 + eloss.MMU ** 2)) * C_MM_PER_NS)
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
