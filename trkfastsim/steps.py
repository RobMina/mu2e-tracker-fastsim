"""Fast StrawGasSteps for primary muons (tier 1a): straight lines through the straw gas, Geant4-matched ionisation.

Stages (as the CRV transport): A = straight line from the generated vertex, air mean loss, no MSC; C = the GDML world
material above the tracker (building, overburden, CRV modules, electronics) in height bands down to a plane just above
the straws (mu2efastsim.transport.world_loss), energy loss per material and correlated Highland scattering to that
plane (msc_to_plane), then straight inside the tracker volume. sliced_msc: each height band scatters with its own
|p| (msc_to_plane band_p) instead of one geometric-mean |p| for the column.

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
    """GDML world pieces above the MSC plane, without the straw gas (the tracker itself is handled by crossings),
    pre-split into the stage C height bands and packed for vectorised columns (transport.BandedWorld)."""
    from mu2efastsim import world as world_mod
    pieces = [q for q in world_mod.load(gdml_path, y_min=plane_y(geom)) if q[0] != "StrawGas"]
    y_pl = plane_y(geom)
    return tr.BandedWorld(pieces, np.r_[BAND_EDGES_Y[BAND_EDGES_Y > y_pl], y_pl])


INTERIOR_SKIP = ("TrackerStraw", "TrackerWire")   # the straws themselves are handled by crossings/eIon


class TrackerInterior:
    """The tracker's own support structure (rings, ribs, beams, manifolds, electronics) for the fast transport.

    Built once from the GDML (tube and polycone solids inside the tracker region, without the straws) as coaxial
    shells about the tracker axis (mu2efastsim.world.CoaxialShells). For each muon crossing it supplies the material
    column between the plane above the tracker and each straw entry, and the energy lost in it (sliced, sampled with
    the Urban model per material), so |p| drops along the track as in G4. Scattering in this material is not applied
    (the muon stays straight inside the tracker volume). Off: pass interior=None to make_steps (CLI --no-interior).
    """

    def __init__(self, gdml_path, geom, nsub=8, pad=300.0):
        from mu2efastsim import world as world_mod
        lo, hi = crossings.panel_boxes(geom)
        bbox = (lo.min(0) - pad, hi.max(0) + pad)
        tubes = world_mod.load_tubes(gdml_path, bbox=bbox, skip_names=INTERIOR_SKIP)
        tubes = [t for t in tubes if t[0] != "StrawGas"]           # gas manifolds: negligible, not a loss material
        self.shells = world_mod.CoaxialShells(tubes, axis_point=geometry.DETECTOR_ORIGIN_EXTRACTED)
        self.nsub = nsub
        self.unknown = sorted({t[0] for t in tubes} - set(materials.BY_NAME))

    def loss_to_straws(self, rng, o, d, p, it, s_in):
        """Energy lost [MeV] in the interior between each muon's origin o (on the plane) and each straw entry.

        o, d, p: per muon [N]; it, s_in: per crossing (muon index, path length to the straw entry), sorted by muon then
        s_in. The column to the farthest straw of each muon is integrated once in up to `nseg` segments ending at the
        crossings; the loss is sampled segment by segment and accumulated."""
        if not len(it):
            return np.zeros(0), {}
        out = np.zeros(len(it))
        first = np.r_[True, it[1:] != it[:-1]]
        prev_s = np.where(first, 0.0, np.r_[0.0, s_in[:-1]])
        seg_len = s_in - prev_s
        seg_o = o[it] + d[it] * prev_s[:, None]
        col = self.shells.column(seg_o, d[it], seg_len, nsub=self.nsub)
        pk = p[it].copy()
        lost_seg = np.zeros(len(it))
        stats = {}
        for name, c in col.items():
            if not c.any():
                continue
            mat = materials.BY_NAME.get(name, materials.ALUMINIUM)
            hit = np.flatnonzero(c > 0)
            lost_seg[hit] += tr.slab_loss(rng, mat, np.maximum(pk[hit], 1.0), c[hit])
            stats[name] = float(c.sum())
        # cumulative loss along each muon
        grp = np.cumsum(first) - 1
        cum = np.cumsum(lost_seg)
        base = np.r_[0.0, cum][np.flatnonzero(first)][grp]
        out = cum - base
        return out, stats


def to_plane(gen, geom, rng, world, sliced_msc=False, near_margin=FOOTPRINT_MARGIN):
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
    near = near_tracker(o, d, geom, y_pl, margin=near_margin)
    p = p0.copy()
    L = np.zeros(len(mu))
    alive = np.ones(len(mu), bool)
    if near.any():
        pw, xx0, ok, bxx0, bh, bp = tr.world_loss(rng, o[near], d[near], p0[near], world, edges, return_band_p=True)
        p_msc = np.maximum(np.sqrt(p0[near] * np.maximum(pw, 1.0)), 1.0)   # geometric mean over the column
        on, dn, Ln = tr.msc_to_plane(rng, o[near], d[near], p_msc, y_pl, eloss.radiation_length_mm(materials.AIR),
                                     xx0, bxx0, bh, band_p=bp if sliced_msc else None)
        o, d = o.copy(), d.copy()
        o[near], d[near], L[near], p[near] = on, dn, Ln, pw
        alive[np.flatnonzero(near)[~ok]] = False
    return mu, o, d, p, L, alive


SECONDARY_KEY0 = 1000   # SimParticle keys of library secondaries: 1000 + n within the event (the muon is key 1)


def add_instraw_secondaries(steps, parts, sampler):
    """Append library secondary steps sharing the fast muon's straws (decision T2, in-straw component).

    Each muon crossing draws a G4 crossing of the same (|p|, path) bin and copies its secondary steps: same straw and
    time as the muon step, start/end on the muon chord (the G4 secondaries' own geometry inside the straw is not
    kept; the digitiser uses eIon, length and type), eion/len/ion/shape from the library. Each copied step becomes a
    SimParticle (pdg from the library, parent = the muon).
    """
    steps = steps.reset_index(drop=True)
    mu = steps[steps.sp == 1].reset_index(drop=True)          # muon crossings only, never pasted secondaries
    p = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy()
    _, st = sampler.draw(p, mu.len.to_numpy())
    if not len(st):
        return steps, parts
    src = mu.iloc[st.i.to_numpy()].reset_index(drop=True)
    new = src.copy()
    for c in ("eion", "len", "ion", "shape"):
        new[c] = st[c].to_numpy().astype(src[c].dtype)
    new["px"] = new["py"] = new["pz"] = 0.0      # secondary momentum is not in the library (digi uses step type)
    rank = new.groupby(KEY).cumcount().to_numpy()
    new["sp"] = SECONDARY_KEY0 + rank
    out = pd.concat([steps, new], ignore_index=True)
    out["nsgs"] = out.groupby(KEY)["sid"].transform("size")
    sp = new[KEY + ["sp"]].rename(columns={"sp": "key"})
    sp["parent"] = 1
    sp["pdg"] = st.pdg.to_numpy()
    sp["isprim"] = 0
    sp["ismcprim"] = 0
    return out, pd.concat([parts, sp], ignore_index=True)


NPLANES_, NPANELS_, NSTRAWS_ = 36, 6, 96


def _mirror(x, n):
    """Reflect indices once at the lattice boundaries: -1 -> 1, n -> n - 2 (still outside afterwards: unchanged)."""
    x = np.asarray(x)
    y = np.where(x < 0, -x, np.where(x >= n, 2 * (n - 1) - x, x))
    return np.where((y >= 0) & (y < n), y, x)
OFFAXIS_KEY0 = 100000   # SimParticle keys of pasted off-axis secondaries: 100000 + (G4 particle rank within the set)


def add_offaxis_secondaries(steps, parts, sampler, geom):
    """Append library off-axis secondary steps (decision T2, off-axis component) around each fast muon.

    Per event with muon steps, a G4 event matched in |p|, |cos theta_y| and the muon's plane extent is drawn
    (OffAxisSampler). Each of its off-axis steps is placed by its lattice offsets from an anchor muon step: anchor =
    the fast muon's straw at the same relative position along the muon (k / straws crossed in the library event,
    times the fast muon's count), then plane + dpl, panel (+ dpa) mod 6,
    straw + dst. A target plane outside 0..35 or straw outside 0..95 is mirrored back once at the boundary (-1 -> 1,
    36 -> 34; counted as reflected); still outside after the mirror, or in a straw the fast muon crosses, the step
    is dropped (the in-straw component covers the muon's straws). The step is laid along the target straw axis with the library length, centred on the straw
    midpoint; time = anchor time + library delay. Steps of one G4 secondary stay one SimParticle (parent = the muon).
    Returns (steps, parts, stats).
    """
    mu = steps[steps.sp == 1].sort_values(KEY + ["t"]).reset_index(drop=True)
    if not len(mu):
        return steps, parts, {"drawn": 0, "placed": 0}
    mu["k"] = mu.groupby(KEY).cumcount()
    nmu = mu.groupby(KEY, sort=False).size()
    f = mu.groupby(KEY, sort=False).first().reset_index()
    nk = nmu.reindex(pd.MultiIndex.from_frame(f[KEY])).to_numpy()
    p = np.sqrt(f.px ** 2 + f.py ** 2 + f.pz ** 2).to_numpy()
    pl = mu.groupby(KEY, sort=False).plane.agg(["min", "max"]).reindex(pd.MultiIndex.from_frame(f[KEY]))
    _, st = sampler.draw(p, (f.py / p).to_numpy(), pl["min"].to_numpy(), pl["max"].to_numpy())
    stats = {"drawn": int(len(st))}
    if not len(st):
        stats["placed"] = 0
        return steps, parts, stats
    i = st.i.to_numpy()
    # anchor by relative position along the muon (k / straws crossed in the library event), so a set recorded on a
    # long muon spreads over the fast muon's straws instead of piling onto its last one when k is clipped
    if "nstraw_mu" in st:
        k = np.floor(st.k.to_numpy() / np.maximum(st.nstraw_mu.to_numpy(), 1) * nk[i]).astype(np.int64)
    else:
        k = st.k.to_numpy()
    k = np.clip(k, 0, nk[i] - 1)
    anc = pd.DataFrame({**{c: f[c].to_numpy()[i] for c in KEY}, "k": k}).merge(
        mu[KEY + ["k", "plane", "panel", "straw", "t"]], on=KEY + ["k"], how="left")
    plane0 = anc.plane.to_numpy() + st.dpl.to_numpy()
    panel = (anc.panel.to_numpy() + st.dpa.to_numpy()) % NPANELS_
    straw0 = anc.straw.to_numpy() + st.dst.to_numpy()
    plane, straw = _mirror(plane0, NPLANES_), _mirror(straw0, NSTRAWS_)
    reflected = (plane != plane0) | (straw != straw0)
    ok = (plane >= 0) & (plane < NPLANES_) & (straw >= 0) & (straw < NSTRAWS_)
    js = np.where(ok, (plane * NPANELS_ + panel) * NSTRAWS_ + straw, 0).astype(np.int64)
    ev = anc[KEY]
    mu_st = mu[KEY + ["sid"]].drop_duplicates().assign(_m=1)
    in_mu = ev.assign(sid=geom.sid[js]).merge(mu_st, on=KEY + ["sid"], how="left")._m.notna().to_numpy() & ok
    keep = ok & ~in_mu
    stats.update({"reflected": int((reflected & ok).sum()), "off_lattice": int((~ok).sum()),
                  "in_muon_straw": int(in_mu.sum()), "placed": int(keep.sum()),
                  "draw_level_counts": np.bincount(np.maximum(sampler.last_level, 0), minlength=4).tolist()
                  if getattr(sampler, "last_level", None) is not None else [],
                  **getattr(sampler, "last_concentration", {})})
    if not keep.any():
        return steps, parts, stats
    st, i, js, t_anc = st[keep].reset_index(drop=True), i[keep], js[keep], anc.t.to_numpy()[keep]
    half = 0.5 * np.minimum(st.len.to_numpy(), 2 * geom.halflen[js])
    new = pd.DataFrame({c: f[c].to_numpy()[i] for c in KEY})
    new["sid"] = geom.sid[js]
    new["plane"], new["panel"], new["straw"], new["upanel"] = geom.plane[js], geom.panel[js], geom.straw[js], geom.upanel[js]
    new["shape"] = st["shape"].to_numpy()
    new["ion"] = st["ion"].to_numpy()
    new["eion"] = st.eion.to_numpy()
    new["len"] = st.len.to_numpy()
    new["width"] = 0.0
    new["t"] = t_anc + st.dt.to_numpy()
    p0 = geometry.to_detector(geom.mid[js] - geom.dir[js] * half[:, None])
    p1 = geometry.to_detector(geom.mid[js] + geom.dir[js] * half[:, None])
    for j, c in enumerate("xyz"):
        new[f"{c}0"], new[f"{c}1"] = p0[:, j], p1[:, j]
        new[f"p{c}"] = 0.0
    gkey = pd.DataFrame({"i": i, "gsp": st.sp.to_numpy()})
    new["sp"] = (OFFAXIS_KEY0 + gkey.groupby("i").gsp.rank(method="dense").astype(np.int64).to_numpy() - 1).astype(steps.sp.dtype)
    new["pdg_"] = st.pdg.to_numpy()
    # one step per particle per straw, as MakeStrawGasSteps: a mirrored offset can put a second step of the same
    # G4 particle into a straw it already has; merge them (eIon and length summed, earliest time and geometry kept)
    n0 = len(new)
    new = new.sort_values("t", kind="stable")
    g = new.groupby(KEY + ["sid", "sp"], sort=False)
    merged = g.first()
    merged["eion"] = g.eion.sum()
    merged["len"] = g.len.sum()
    new = merged.reset_index()
    stats["merged_same_particle"] = int(n0 - len(new))
    new["nsgs"] = 0
    out = pd.concat([steps, new[steps.columns]], ignore_index=True)
    out["nsgs"] = out.groupby(KEY)["sid"].transform("size")
    sp = new[KEY + ["sp", "pdg_"]].rename(columns={"pdg_": "pdg"}).drop_duplicates(KEY + ["sp"]).rename(columns={"sp": "key"})
    sp["parent"] = 1
    sp["isprim"] = 0
    sp["ismcprim"] = 0
    return out, pd.concat([parts, sp[parts.columns]], ignore_index=True), stats


def make_steps(gen, geom, rng, air_loss=True, stage="A", world=None, sampler=None, offaxis=None, sliced_msc=False,
               interior=None, near_margin=FOOTPRINT_MARGIN):
    """Return (sgsteps, particles) DataFrames for the primary muons in `gen` (one CRY muon per event).

    sampler: an InStrawSampler (trkfastsim.library) to add the library's in-straw secondaries; None = muons only.
    offaxis: an OffAxisSampler to add the library's off-axis secondaries (needs the muon steps first).
    interior: a TrackerInterior to lose energy in the tracker's support structure between straws (None = off)."""
    if stage == "C":
        mu, o, d, p_start, t_off, alive = to_plane(gen, geom, rng, world, sliced_msc=sliced_msc, near_margin=near_margin)
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
    if interior is not None and len(it):
        lost, make_steps.interior_stats = interior.loss_to_straws(rng, o, d, p_start, it, s_in)
        tk = np.sqrt(p ** 2 + eloss.MMU ** 2) - eloss.MMU - lost
        stopped = tk <= 0.5
        p = np.sqrt(np.clip((np.maximum(tk, 0.5) + eloss.MMU) ** 2 - eloss.MMU ** 2, 1.0, None))
        if stopped.any():   # a muon ranging out in the support structure makes no further straw steps
            keep = ~stopped
            it, js, s_in, s_out, path, p = it[keep], js[keep], s_in[keep], s_out[keep], path[keep], p[keep]
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
    if offaxis is not None:   # before the in-straw pasting, so the frame uses muon steps only
        steps, parts, make_steps.offaxis_stats = add_offaxis_secondaries(steps, parts, offaxis, geom)
    if sampler is not None:
        steps, parts = add_instraw_secondaries(steps, parts, sampler)
    return steps, parts
