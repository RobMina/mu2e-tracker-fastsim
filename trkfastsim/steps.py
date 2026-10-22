"""Fast StrawGasSteps for primary muons (tier 1a): straight lines through the straw gas, Geant4-matched ionisation.

Stages (as the CRV transport): A = straight line from the generated vertex, air mean loss, no MSC; C = the GDML world
material above the tracker (building, overburden, CRV modules, electronics) in height bands down to a plane just above
the straws (mu2efastsim.transport.world_loss), energy loss per material and correlated Highland scattering to that
plane (msc_to_plane), then through the tracker with energy loss and scattering in its support structure segment by
segment between straws (TrackerInterior.transport; straight between kinks). sliced_msc: each height band scatters with
its own |p| (msc_to_plane band_p) instead of one geometric-mean |p| for the column. msc_tail: single-scattering
(Rutherford) tail above the Gaussian core, in the world column and in the interior.

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
    """Downward tracks whose straight line comes within `margin` of the straws' x-z footprint anywhere between the
    plane y = y_plane and the bottom of the straw stack (a shallow muon can pass the plane outside the footprint and
    still cross the stack lower down)."""
    lo, hi = crossings.panel_boxes(geom)
    lo, hi = lo.min(0), hi.max(0)
    down = d[:, 1] < 0
    dy = -np.where(down, d[:, 1], -1.0)
    xs, zs = [], []
    for y in (y_plane, lo[1]):
        L = np.where(down, (o[:, 1] - y) / dy, 0.0)
        xs.append(o[:, 0] + d[:, 0] * L)
        zs.append(o[:, 2] + d[:, 2] * L)
    x0, x1 = np.minimum(*xs), np.maximum(*xs)
    z0, z1 = np.minimum(*zs), np.maximum(*zs)
    return down & (x1 > lo[0] - margin) & (x0 < hi[0] + margin) & (z1 > lo[2] - margin) & (z0 < hi[2] + margin)


def load_world(gdml_path, geom):
    """GDML world pieces above the MSC plane, without the straw gas (the tracker itself is handled by crossings),
    pre-split into the stage C height bands and packed for vectorised columns (transport.BandedWorld)."""
    from mu2efastsim import world as world_mod
    pieces = [q for q in world_mod.load(gdml_path, y_min=plane_y(geom)) if q[0] != "StrawGas"]
    y_pl = plane_y(geom)
    return tr.BandedWorld(pieces, np.r_[BAND_EDGES_Y[BAND_EDGES_Y > y_pl], y_pl])


INTERIOR_SKIP = ("TrackerStraw", "TrackerWire")   # the straws themselves are handled by crossings/eIon
KINK_XX0 = 1e-3      # x/X0 of thin material (straws, gaps) accumulated before the in-tracker transport kinks the track
KINK_BACKOFF_MM = 0.05   # the in-tracker kink is applied this far before the kink straw's gas entry
GAP_MATERIAL = None  # medium between straws and support inside the tracker: the GDML's DSVacuum (None = no material)


class TrackerInterior:
    """The tracker's own support structure (rings, ribs, beams, manifolds, electronics) for the fast transport.

    Built once from the GDML (tube and polycone solids inside the tracker region, without the straws) as coaxial
    shells about the tracker axis (mu2efastsim.world.CoaxialShells). For each muon crossing it supplies the material
    column between the plane above the tracker and each straw entry, and the energy lost in it (sliced, sampled with
    the Urban model per material), so |p| drops along the track as in G4. Off: pass interior=None to make_steps (CLI
    --no-interior).

    transport() also scatters, segment by segment: in this material and in the thin material between and in the
    straws (straw walls, gas, sense wire; optionally a gap medium, none by default), Gaussian core (Highland, log term on the cumulative in-tracker x/X0) plus
    optionally the single-scattering tail; the track kinks after each support segment and every KINK_XX0 of thin
    material, and its straw crossings are recomputed from there. loss_to_straws() is the straight-line (loss only) form.
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

    def transport(self, rng, o, d, p, geom, tail=True, tail_k=None, kink_xx0=KINK_XX0, gap=GAP_MATERIAL, wire=True,
                  max_iter=400, stats=None):
        """Straw crossings of muons scattered and slowed inside the tracker, segment by segment.

        o, d, p: per muon [N] at the plane above the tracker. Each crossing row k of a muon carries the material in front
        of its straw entry: the previous straw of this pass (two wall chords, Mylar + outer metal from the gas chord and
        wall_rmax; its gas chord; the sense wire, W core and Au plating, at the track's distance from the straw axis, if
        `wire`), then the segment from that straw's exit (or the pass origin) to this entry:
        support structure (shells column) and the gap medium `gap` in the rest (default none: the tracker volumes are
        DSVacuum in the GDML). Energy loss is sampled in the support material
        (slab_loss) and applies from this straw on. Scattering: rows are grouped from the pass origin until their summed
        x/X0 reaches kink_xx0 or a row holds support material; the group kinks at the entry of that row's straw with
        the correlated (angle, displacement) Gaussian, Highland with the log term on the cumulative in-tracker x/X0
        (transport.highland_var_increment), each piece at its own lever arm (support and air at the segment midpoint,
        straw material at its chord midpoint), plus the single-scattering tail on the group's nuclear chi_c^2 if
        `tail`: per material on nuclei (each with its form factor) and on atomic electrons (up to ME_OVER_MMU), above
        the angle where the summed single-scattering density crosses the Gaussian. Rows before the kink row are final; crossings are then recomputed from the kinked track (straws already
        recorded for that muon are skipped). The material behind a muon's last straw is not scattered. Returns a dict of per-crossing arrays sorted by muon then path: it, js, pos0, pos1 (Mu2e frame),
        dir, path (to the entry, from the plane), lost (interior loss before the entry, MeV).
        """
        n = len(p)
        o, d = o.copy(), d.copy()
        tk_start = np.sqrt(p ** 2 + eloss.MMU ** 2) - eloss.MMU
        lost = np.zeros(n)
        travelled = np.zeros(n)
        xcum = np.zeros(n)
        restarted = np.zeros(n, bool)                # muon restarted before its kink straw (back-off segment counted)
        act = np.arange(n)
        rows = []
        kinks = []                                   # kink points per muon, for the surface crossings (no rng use)
        done = np.zeros(0, np.int64)                 # muon * nstraw + straw of the recorded crossings
        ns = len(geom.sid)
        r_in = geom.radius
        r_out = geom.wall_rmax if geom.wall_rmax is not None else r_in + 0.015
        f_metal = min(max(geom.metal_t / max(r_out - r_in, 1e-9), 0.0), 1.0)
        gapm = gap if gap is not None else materials.AIR          # placeholder material; zero length when gap is None
        thin = [(gapm, "air"), (materials.MYLAR, "mylar"), (materials.ALUMINIUM_6061, "metal"), (STRAW_GAS, "gas"),
                (materials.TUNGSTEN, "wire"), (materials.GOLD, "plate")]
        rw, rw_in = geom.wire_r if wire else 0.0, max(geom.wire_r - geom.wire_plate_t, 0.0) if wire else 0.0
        x0 = {m.name: eloss.radiation_length_mm(m) for m, _ in thin}
        st = ({"kinks": 0, "tail_scatters": 0, "stopped": 0, "iterations": 0, "capped": 0, "wire_hits": 0}
              if stats is None else stats)
        for _ in range(max_iter):
            if not len(act):
                break
            st["iterations"] += 1
            xs = crossings.straw_crossings(o[act], d[act], geom)
            if not len(xs):
                break
            loc = xs[:, 0].astype(np.int64)
            keep = (xs[:, 3] > 1e-9) & ~np.isin(act[loc] * ns + xs[:, 1].astype(np.int64), done)
            xs, loc = xs[keep], loc[keep]
            if not len(xs):
                break
            it = act[loc]
            js = xs[:, 1].astype(np.int64)
            s_in, s_out = xs[:, 2], xs[:, 3]
            nr = len(it)
            idx = np.arange(nr)
            first = np.r_[True, it[1:] != it[:-1]]
            grp = np.cumsum(first) - 1
            start = np.flatnonzero(first)
            row_end = np.r_[start[1:], nr]
            prev = np.where(first, 0.0, np.r_[0.0, s_out[:-1]])
            # the back-off segment in front of a restarted muon's first straw was counted in the pass before
            seg = np.where(first & restarted[it], 0.0, np.clip(s_in - prev, 0.0, None))
            # support structure in the segment before each straw
            col = self.shells.column(o[it] + d[it] * prev[:, None], d[it], seg, nsub=self.nsub)
            col = {m: c for m, c in col.items() if c.any()}
            solid = np.zeros(nr)
            x_sup = np.zeros(nr)
            for m, c in col.items():
                solid += c
                x_sup += c / eloss.radiation_length_mm(materials.BY_NAME.get(m, materials.ALUMINIUM))
            # thin material: air in the segment, the straw's walls (both sides) and gas
            gas = s_out - s_in
            ua = np.sqrt(np.clip(1 - np.einsum("ij,ij->i", d[it], geom.dir[js]) ** 2, 1e-12, None))
            q = 0.5 * gas * ua
            wall = 2 * (np.sqrt(r_out ** 2 - r_in ** 2 + q ** 2) - q) / ua
            lens = {"air": np.clip(seg - solid, 0, None) * (gap is not None), "mylar": wall * (1 - f_metal),
                    "metal": wall * f_metal, "gas": gas, "wire": np.zeros(nr), "plate": np.zeros(nr)}
            if rw > 0:
                # sense wire: chord at the track's distance b from the straw axis (on the wire's own line)
                nx = np.cross(d[it], geom.dir[js])
                b = np.abs(np.einsum("ij,ij->i", o[it] - geom.mid[js], nx)) / np.maximum(np.linalg.norm(nx, axis=1), 1e-12)
                c_out = 2 * np.sqrt(np.clip(rw ** 2 - b ** 2, 0, None)) / ua
                c_in = 2 * np.sqrt(np.clip(rw_in ** 2 - b ** 2, 0, None)) / ua
                hitw = gas > 0
                lens["wire"] = np.where(hitw, c_in, 0.0)
                lens["plate"] = np.where(hitw, c_out - c_in, 0.0)
                st["wire_hits"] += int((hitw & (c_out > 0)).sum())
            # straw material moves to the next row of the same muon (it lies in front of that row's straw)
            x_straw_own = sum(lens[k] / x0[m.name] for m, k in thin if k != "air")
            x_air = lens["air"] / x0[gapm.name]
            prow = np.where(first, -1, idx - 1)                        # previous row of this muon in this pass
            x_str = np.where(first, 0.0, x_straw_own[np.maximum(prow, 0)])
            h_mid_str = np.where(first, 0.0, 0.5 * (s_in + s_out)[np.maximum(prow, 0)])
            xrow = x_sup + x_air + x_str
            # group from the start of this iteration: kink at the first row reaching kink_xx0 or holding support
            cx = np.cumsum(xrow)
            cx = cx - np.r_[0.0, cx][start][grp]
            trig = (cx >= kink_xx0) | (x_sup > 1e-9)
            fm = np.minimum.reduceat(np.where(trig, idx, nr), start)
            kinked = fm < row_end
            upto = idx <= np.where(kinked, fm, row_end - 1)[grp]     # rows whose support loss is sampled
            fin = idx < np.where(kinked, fm, row_end)[grp]           # final rows: before the kink row
            # energy loss in the support material of rows up to the kink row, at the |p| entering this iteration
            pk = _p_of(tk_start[it] - lost[it])
            lseg = np.zeros(nr)
            for m, c in col.items():
                h = np.flatnonzero((c > 0) & upto)
                if len(h):
                    lseg[h] += tr.slab_loss(rng, materials.BY_NAME.get(m, materials.ALUMINIUM), np.maximum(pk[h], 1.0), c[h])
            cum = np.cumsum(lseg)
            cum = cum - np.r_[0.0, cum][start][grp]
            loss_at = lost[it] + cum
            dead = tk_start[it] - loss_at <= 0.5
            dead_from = np.full(n, nr)
            if dead.any():
                np.minimum.at(dead_from, it[dead], idx[dead])
                st["stopped"] += int(len(np.unique(it[dead])))
            ok = fin & (idx < dead_from[it])
            if ok.any():
                rows.append({"it": it[ok], "js": js[ok], "pos0": o[it[ok]] + d[it[ok]] * s_in[ok, None],
                             "pos1": o[it[ok]] + d[it[ok]] * s_out[ok, None], "dir": d[it[ok]].copy(),
                             "path": travelled[it[ok]] + s_in[ok], "lost": loss_at[ok]})
                done = np.r_[done, it[ok] * ns + js[ok]]
            kr = fm[kinked]
            km = it[kr]
            go = dead_from[km] > kr
            kr, km = kr[go], km[go]
            if not len(km):
                act = np.zeros(0, np.int64)
                break
            # group rows of each kinking muon: its first row of this pass to its kink row (km is sorted, as it)
            member = np.flatnonzero(upto & kinked[grp] & np.isin(it, km))
            gk = np.searchsorted(km, it[member])
            s_k = s_in[kr]
            # lever arms to the kink point: support and air at the segment midpoint, straw material at its chord midpoint
            h_seg = s_k[gk] - (prev[member] + 0.5 * seg[member])
            h_str = s_k[gk] - h_mid_str[member]
            xs_m = x_sup[member] + x_air[member]
            xt_m = x_str[member]
            X = np.bincount(gk, xrow[member], minlength=len(km))
            S1 = np.bincount(gk, xs_m * h_seg + xt_m * h_str, minlength=len(km))
            S2 = np.bincount(gk, xs_m * h_seg ** 2 + xt_m * h_str ** 2, minlength=len(km))
            w1, w2 = S1 / np.maximum(X, 1e-30), S2 / np.maximum(X, 1e-30)
            p_k = _p_of(tk_start[km] - lost[km] - cum[kr])
            V = tr.highland_var_increment(p_k, xcum[km], xcum[km] + X)
            a = np.sqrt(V)
            b = np.sqrt(np.clip(V * w2 - (V * w1) ** 2 / np.maximum(V, 1e-30), 0, None))
            z_t = rng.standard_normal((len(km), 2))
            z_y = rng.standard_normal((len(km), 2))
            dth = z_t * a[:, None]
            dy = z_t * (V * w1 / np.maximum(a, 1e-30))[:, None] + z_y * b[:, None]
            if tail:
                chic2 = np.zeros(len(km))
                thw = np.zeros(len(km))
                pm = p_k[gk]
                pr = np.maximum(prow[member], 0)
                own = ~first[member]
                parts = [(materials.BY_NAME.get(m, materials.ALUMINIUM), c[member]) for m, c in col.items()]
                parts += [(gapm, lens["air"][member])]
                parts += [(m, np.where(own, lens[k][pr], 0.0)) for m, k in thin if k != "air"]
                nuc, che = [], np.zeros(len(km))
                for mat, ln in parts:
                    x2 = np.bincount(gk, tr.chi_c2(mat, ln, pm, electrons=False), minlength=len(km))
                    if x2.any():
                        nuc.append((mat, x2))
                        chic2 += x2
                    che += np.bincount(gk, tr.chi_c2_electron(mat, ln, pm), minlength=len(km))
                th_min = tr.tail_threshold(chic2 + che, a, tail_k)
                tt = np.zeros((len(km), 2))
                for mat, x2 in nuc:                      # each material with its own form factor
                    tt += tr.single_scatter_tail(rng, x2, th_min, tr.theta_nuclear(mat, p_k))
                tt += tr.single_scatter_tail(rng, che, th_min, None, tr.ME_OVER_MMU)
                st["tail_scatters"] += int((tt != 0).any(1).sum())
                dth += tt
                dy += tt * w1[:, None]
            # kink at the end of the segment in front of the kink straw, KINK_BACKOFF_MM before its gas entry (but in
            # the segment), so the displaced restart point stays outside that straw and its crossing is whole
            back = np.minimum(KINK_BACKOFF_MM, 0.5 * seg[kr])
            end = o[km] + d[km] * (s_k - back)[:, None]
            pos, dn = _kink(end, d[km], dth, dy)
            # a displacement can still land the point inside a straw's gas: move it back out along the new direction
            pos = _exit_gas_backwards(pos, dn, geom)
            kinks.append({"it": km.copy(), "end": end.copy(), "pos": pos.copy(), "dir": dn.copy(),
                          "path": travelled[km] + s_k - back, "p": p_k.copy()})
            o[km] = pos
            d[km] = dn
            restarted[km] = True
            travelled[km] += s_k - back
            lost[km] += cum[kr]
            xcum[km] += X
            st["kinks"] += len(km)
            act = km
        else:
            st["capped"] += len(act)
        self.last_stats = st
        self.last_kinks = ({k: np.concatenate([q[k] for q in kinks]) for k in kinks[0]} if kinks else
                           {"it": np.zeros(0, np.int64), "end": np.zeros((0, 3)), "pos": np.zeros((0, 3)),
                            "dir": np.zeros((0, 3)), "path": np.zeros(0), "p": np.zeros(0)})
        if not rows:
            z3 = np.zeros((0, 3))
            return {"it": np.zeros(0, np.int64), "js": np.zeros(0, np.int64), "pos0": z3, "pos1": z3, "dir": z3,
                    "path": np.zeros(0), "lost": np.zeros(0)}
        out = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        order = np.lexsort((out["path"], out["it"]))
        return {k: v[order] for k, v in out.items()}


def _back_to_interior_top(o, d, t_off, alive, interior, geom):
    """Move downward origins on the plane back up along their direction to the top of the interior shells.

    The world column above the plane (world_loss) contains no tubes, and the interior shells (TrackerInterior) start
    below the top of the support rings, so between the plane and that top nothing is double counted. Air over that
    stretch (~130 mm) is negligible. Returns (origin, path offset), the offset reduced by the added stretch so the
    times stay those of the straight path."""
    top = max([g["rmax"] for g in interior.shells.groups] + [0.0]) + interior.shells.c[1]
    y_pl = plane_y(geom)
    if top <= y_pl:
        return o, t_off
    o, t_off = o.copy(), t_off.copy()
    m = alive & (d[:, 1] < 0) & (np.abs(o[:, 1] - y_pl) < 1e-3)
    back = (top - y_pl) / -d[m, 1]
    o[m] = o[m] - d[m] * back[:, None]
    t_off[m] -= back
    return o, t_off


def _p_of(tk):
    """|p| [MeV/c] of a muon with kinetic energy tk [MeV] (floored at 0.5 MeV)."""
    tk = np.maximum(tk, 0.5)
    return np.sqrt((tk + eloss.MMU) ** 2 - eloss.MMU ** 2)


def _exit_gas_backwards(pos, d, geom):
    """Points inside a straw's gas are moved back along -d to just outside it (its entry on the line through pos)."""
    js = crossings.point_straw(pos, geom)
    inside = np.flatnonzero(js >= 0)
    if not len(inside):
        return pos
    j = js[inside]
    a, u = geom.dir[j], d[inside]
    w = pos[inside] - geom.mid[j]
    uu = u - np.einsum("ij,ij->i", u, a)[:, None] * a
    ww = w - np.einsum("ij,ij->i", w, a)[:, None] * a
    A = np.einsum("ij,ij->i", uu, uu)
    B = 2 * np.einsum("ij,ij->i", uu, ww)
    C = np.einsum("ij,ij->i", ww, ww) - geom.radius ** 2      # < 0: inside
    t = (-B - np.sqrt(np.clip(B * B - 4 * A * C, 0, None))) / (2 * np.maximum(A, 1e-15))   # entry (t <= 0)
    out = pos.copy()
    out[inside] = pos[inside] + u * (np.minimum(t, 0.0) - 1e-4)[:, None]
    return out


def _kink(pos, d, dth, dy):
    """Displace pos by dy and turn d by dth in the two transverse planes of d (as transport._apply_msc, no plane)."""
    ref = np.where(np.abs(d[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))
    u = np.cross(d, ref)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(d, u)
    dn = d + u * np.tan(dth[:, [0]]) + v * np.tan(dth[:, [1]])
    dn /= np.linalg.norm(dn, axis=1)[:, None]
    return pos + u * dy[:, [0]] + v * dy[:, [1]], dn


def to_plane(gen, geom, rng, world, sliced_msc=False, near_margin=FOOTPRINT_MARGIN, msc_tail=True, tail_k=None,
             band_spread=False):
    """Stage C transport of the primary muons to the plane above the tracker.

    Returns (mu, origin [N,3], direction [N,3], |p| at the origin [N], path already travelled [N], alive [N]). Tracks
    not near the tracker (straight line outside the footprint margin at the plane, or upward) are not transported and
    are returned with alive = False: their generated state has crossed no material, so a straw they happen to cross
    below the plane would be reached with no energy loss and no scattering. band_spread: spread each height band's
    scattering uniformly along its path (transport.msc_to_plane band_len) instead of at the band centre.
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
    alive = near.copy()
    if near.any():
        res = tr.world_loss(rng, o[near], d[near], p0[near], world, edges, return_band_p=True, return_tail=msc_tail,
                           return_band_len=band_spread)
        pw, xx0, ok, bxx0, bh, bp = res[:6]
        chic2, thn = res[6:8] if msc_tail else (None, None)
        blen = res[-1] if band_spread else None
        p_msc = np.maximum(np.sqrt(p0[near] * np.maximum(pw, 1.0)), 1.0)   # geometric mean over the column
        on, dn, Ln = tr.msc_to_plane(rng, o[near], d[near], p_msc, y_pl, eloss.radiation_length_mm(materials.AIR),
                                     xx0, bxx0, bh, band_p=bp if sliced_msc else None,
                                     band_chic2=chic2, band_thn=thn, tail_k=tail_k, band_len=blen)
        o, d = o.copy(), d.copy()
        o[near], d[near], L[near], p[near] = on, dn, Ln, pw
        alive[np.flatnonzero(near)[~ok]] = False
    return mu, o, d, p, L, alive


SECONDARY_KEY0 = 1000   # SimParticle keys of library secondaries: 1000 + n within the event (the muon is key 1)


def _paste_local(st, m, a, half, radius, q0, q1, pmu, uref_fn):
    """Library v2 step ends (du, r, phi, mub in `st`) rebuilt around a fast muon: straw (m, a, half), muon line through
    q0 -> q1 (Mu2e frame) with momentum pmu; uref_fn(m, a, half, q0, q1, d) gives the along-wire reference. Detector frame."""
    from . import library as L
    d = L.muon_dir(pmu, q0, q1)
    s, e1, e2 = L.straw_frame(a, d)
    flip = np.where(np.sign(L._dot(q0 - m, e2)) * np.sign(st.mub.to_numpy()) < 0, -1.0, 1.0)  # muon on the other side
    uref = uref_fn(m, a, half, q0, q1, d)
    if "cn0" in st:   # chord coordinates (in-straw): e2 mirrored with the side, so h keeps its sense towards/away from the wire
        b = L._dot(q0 - m, e2)
        return [geometry.to_detector(L.from_chord(st[f"du{e}"].to_numpy(), st[f"cn{e}"].to_numpy(), flip * st[f"h{e}"].to_numpy(),
                                                  m, a, s, e1, e2, uref, half, b, radius)) for e in "01"]
    return [geometry.to_detector(L.from_local(st[f"du{e}"].to_numpy(), st[f"r{e}"].to_numpy(), flip * st[f"phi{e}"].to_numpy(),
                                              m, a, s, e1, e2, uref, half, radius)) for e in "01"]


def _xyz_mu2e(df, end):
    return geometry.to_mu2e(df[[f"x{end}", f"y{end}", f"z{end}"]].to_numpy(float))


def add_instraw_secondaries(steps, parts, sampler, geom=None):
    """Append library secondary steps sharing the fast muon's straws (decision T2, in-straw component).

    Each muon crossing draws a G4 crossing of the same (|p|, path) bin and copies its secondary steps: same straw and
    time as the muon step, eion/len/ion/shape from the library. Placement: v1 library (or no geom), start/end on the muon
    chord; v2 library with geom, each end at the stored along-wire offset from the fast chord's point nearest the wire
    and at the stored chord coordinates (cn, h) about the fast chord (library.to_chord; (r, phi) when the wire offsets
    agree), in the fast muon's straw frame (library.straw_frame). Each
    copied step becomes a SimParticle (pdg from the library, parent = the muon).
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
    if geom is not None and "du0" in st:
        from . import library as L
        js = (src.upanel.to_numpy() * NSTRAWS_ + src.straw.to_numpy()).astype(np.int64)
        x0, x1 = _paste_local(st, geom.mid[js], geom.dir[js], geom.halflen[js], geom.radius, _xyz_mu2e(src, "0"),
                              _xyz_mu2e(src, "1"), src[["px", "py", "pz"]].to_numpy(float),
                              lambda m, a, h, q0, q1, d: L.chord_u(m, a, q0, q1))
        for j, c in enumerate("xyz"):
            new[f"{c}0"], new[f"{c}1"] = x0[:, j], x1[:, j]
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
    is dropped (the in-straw component covers the muon's straws). Placement: v1 library, along the target straw axis with
    the library length, centred on the straw midpoint; v2 library, each end at the stored along-wire offset from the point
    where the fast anchor step's line passes closest to the target straw axis (clipped to the half length) and at the
    stored (r, phi) about the wire, phi from the anchor direction's transverse part. Time = anchor time + library delay. Steps of one G4 secondary stay one SimParticle (parent = the muon).
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
    v2 = "du0" in st
    acols = ["x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"] if v2 else []
    anc = pd.DataFrame({**{c: f[c].to_numpy()[i] for c in KEY}, "k": k}).merge(
        mu[KEY + ["k", "plane", "panel", "straw", "t"] + acols], on=KEY + ["k"], how="left")
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
    st, i, js, t_anc, anc = st[keep].reset_index(drop=True), i[keep], js[keep], anc.t.to_numpy()[keep], anc[keep]
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
    if v2:
        from . import library as L
        p0, p1 = _paste_local(st, geom.mid[js], geom.dir[js], geom.halflen[js], geom.radius, _xyz_mu2e(anc, "0"),
                              _xyz_mu2e(anc, "1"), anc[["px", "py", "pz"]].to_numpy(float),
                              lambda m, a, h, q0, q1, d: L.line_u(m, a, h, q0, d))
    else:
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


PATH_EXTEND = 4000.0   # straight extension before the interior start and after the last kink [mm]


def _path_vertices(mu, sel, o, d, p_start, t_off, p0_all, kinks):
    """The transported muon path as ordered vertices (detector frame), for the tracker VD crossings (surfaces.py).

    Per muon: a point PATH_EXTEND before the interior start, the start (top of the support rings), then for each kink the
    segment end and the displaced restart point with the new direction, and a point PATH_EXTEND beyond the last one. Each
    vertex carries the direction of the segment leaving it, |p|, the path length from the interior start, and the time
    offset and speed used for the StrawGasStep times (so crossing times use the same convention)."""
    nk = len(kinks["it"])
    i0 = np.r_[sel, sel, sel[kinks["it"]], sel[kinks["it"]]]
    pos = np.r_[o[sel] - PATH_EXTEND * d[sel], o[sel], kinks["end"], kinks["pos"]]
    dirs = np.r_[d[sel], d[sel], np.zeros((nk, 3)), kinks["dir"]]
    path = np.r_[np.full(len(sel), -PATH_EXTEND), np.zeros(len(sel)), kinks["path"], kinks["path"] + 1e-9]
    p = np.r_[p_start[sel], p_start[sel], kinks["p"], kinks["p"]]
    order = np.lexsort((path, i0))
    i0, pos, dirs, path, p = i0[order], pos[order], dirs[order], path[order], p[order]
    # an end point continues the incoming direction; the last vertex of each muon gets a far point along its direction
    last = np.r_[i0[1:] != i0[:-1], True]
    inc = np.r_[np.zeros((1, 3)), dirs[:-1]]
    dirs = np.where((np.linalg.norm(dirs, axis=1) == 0)[:, None], inc, dirs)
    tail_pos = pos[last] + PATH_EXTEND * dirs[last]
    i0 = np.r_[i0, i0[last]]
    pos = np.r_[pos, tail_pos]
    dirs = np.r_[dirs, dirs[last]]
    path = np.r_[path, path[last] + PATH_EXTEND]
    p = np.r_[p, p[last]]
    order = np.lexsort((path, i0))
    v = pd.DataFrame({k: mu[k].to_numpy()[i0[order]] for k in KEY})
    xyz = geometry.to_detector(pos[order])
    for j, c in enumerate("xyz"):
        v[c] = xyz[:, j]
        v[f"u{c}"] = dirs[order][:, j]
    v["p"] = p[order]
    v["path"] = path[order]
    p0 = p0_all[i0[order]]
    v["t0"] = mu.t.to_numpy()[i0[order]] + t_off[i0[order]] / ((p0 / np.sqrt(p0 ** 2 + eloss.MMU ** 2)) * C_MM_PER_NS)
    v["beta"] = p0 / np.sqrt(p0 ** 2 + eloss.MMU ** 2)
    return v


def make_steps(gen, geom, rng, air_loss=True, stage="A", world=None, sampler=None, offaxis=None, sliced_msc=False,
               interior=None, near_margin=FOOTPRINT_MARGIN, interior_msc=True, msc_tail=True, tail_k=None,
               kink_xx0=KINK_XX0, band_spread=False):
    """Return (sgsteps, particles) DataFrames for the primary muons in `gen` (one CRY muon per event).

    sampler: an InStrawSampler (trkfastsim.library) to add the library's in-straw secondaries; None = muons only.
    offaxis: an OffAxisSampler to add the library's off-axis secondaries (needs the muon steps first).
    interior: a TrackerInterior to lose energy in the tracker's support structure between straws (None = off).
    interior_msc: with interior, also scatter in it segment by segment (TrackerInterior.transport); False = the muon
    stays straight inside the tracker (loss only). msc_tail (default on): the single-scattering tail on the Gaussian core, in
    the world column above the plane (stage C) and in the interior; tail_k: tail threshold in units of theta0 (None =
    where the tail density crosses the Gaussian); kink_xx0: x/X0 of straw material between in-tracker kinks."""
    make_steps.path_vertices = None   # set below only by the interior transport of this call
    if stage == "C":
        mu, o, d, p_start, t_off, alive = to_plane(gen, geom, rng, world, sliced_msc=sliced_msc, near_margin=near_margin,
                                                   msc_tail=msc_tail, tail_k=tail_k, band_spread=band_spread)
        p0 = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy()
    else:
        mu = primary_muons(gen)
        p0 = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy()
        o = mu[["x", "y", "z"]].to_numpy()
        d = mu[["px", "py", "pz"]].to_numpy() / p0[:, None]
        p_start, t_off, alive = p0.copy(), np.zeros(len(mu)), np.ones(len(mu), bool)
    if stage == "C":   # muon state at the plane, for fate diagnostics (analyses/fatecheck.py)
        make_steps.last_plane = {"key": mu[KEY].to_numpy(), "alive": alive.copy(), "p_plane": p_start.copy(),
                                 "o": o.copy(), "d": d.copy()}
    if interior is not None and stage == "C":
        # the interior shells reach above the plane (support rings to r ~ 820 mm about the axis): start the interior
        # transport where they begin, so their top part is crossed (the world column stops at the plane)
        o, t_off = _back_to_interior_top(o, d, t_off, alive, interior, geom)
    if interior is not None and interior_msc:
        sel = np.flatnonzero(alive)
        tr_ = interior.transport(rng, o[sel], d[sel], p_start[sel], geom, tail=msc_tail, tail_k=tail_k, kink_xx0=kink_xx0)
        make_steps.interior_stats = interior.last_stats
        make_steps.path_vertices = _path_vertices(mu, sel, o, d, p_start, t_off, p0_all=np.sqrt(
            mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy(), kinks=interior.last_kinks)
        it = sel[tr_["it"]]
        js = tr_["js"]
        w0, w1, dirs = tr_["pos0"], tr_["pos1"], tr_["dir"]
        s_in = tr_["path"]
        path = np.linalg.norm(w1 - w0, axis=1)
        lost = tr_["lost"]
        p = p_start[it]
        if air_loss:
            p = np.maximum(p - eloss.dedx_restricted(materials.AIR, p_start[it], 1e9) * s_in, 1.0)
        tk = np.sqrt(p ** 2 + eloss.MMU ** 2) - eloss.MMU - lost
        p = np.sqrt(np.clip((np.maximum(tk, 0.5) + eloss.MMU) ** 2 - eloss.MMU ** 2, 1.0, None))
    else:
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
        w0 = o[it] + d[it] * s_in[:, None]
        w1 = o[it] + d[it] * s_out[:, None]
        dirs = d[it]
    eion = eloss.sample_loss(rng, STRAW_GAS, p, path, delta_cut_mev())
    e = np.sqrt(p ** 2 + eloss.MMU ** 2)
    bg = p / eloss.MMU
    ke = e - eloss.MMU
    t = mu.t.to_numpy()[it] + (s_in + t_off[it]) / ((p0[it] / np.sqrt(p0[it] ** 2 + eloss.MMU ** 2)) * C_MM_PER_NS)
    pos0 = geometry.to_detector(w0)
    pos1 = geometry.to_detector(w1)
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
        steps[f"p{c}"] = dirs[:, j] * p
    parts = pd.DataFrame({k: mu[k].to_numpy() for k in KEY})
    parts["key"] = 1
    parts["parent"] = -1
    parts["pdg"] = mu.pdg.to_numpy()
    parts["isprim"] = 1
    parts["ismcprim"] = 1
    if offaxis is not None:   # before the in-straw pasting, so the frame uses muon steps only
        steps, parts, make_steps.offaxis_stats = add_offaxis_secondaries(steps, parts, offaxis, geom)
    if sampler is not None:
        steps, parts = add_instraw_secondaries(steps, parts, sampler, geom)
    return steps, parts
