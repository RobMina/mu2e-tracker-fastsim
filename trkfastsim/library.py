"""Geant4 secondary library for the tracker fast sim (decision T2): non-muon StrawGasSteps of the reference, stored
relative to the primary muon so they can be pasted onto fast muons.

Built from StrawGasStepTableDumper tables of a Geant4 run (g4ref_v1). Only events whose MC primary is a muon with at
least one straw crossing contribute; secondaries are every non-muon StrawGasStep of the event (99.7 % descend from the
muon: delta rays from muIoni, then their eIoni/eBrem/compt/phot progeny, plus showers from the material above).

Two components, written as tables in one ROOT file (directory dump/); provenance in <file>.meta.json:

  ``instraw``  one row per muon straw crossing: muon |p|, path length in the straw, the number of secondary steps in
               that same straw (nsec), and their summed eion; ``instraw_steps`` lists those steps (crossing index,
               eion, len, ion type, pdg). Pasting draws, for each fast muon straw, a G4 crossing from the same
               (|p|, path) bin and copies its secondary steps into the fast straw.
  ``offaxis``  one row per non-muon step NOT in a muon straw, in straw-lattice coordinates relative to an anchor muon
               step (the muon step nearest in plane, then in time): anchor rank k along the muon (0 = first straw),
               plane offset dpl, panel offset dpa (mod 6), straw offset dst, delay after the anchor (dt), and the
               step's eion, len, ion type, shape, pdg and G4 SimParticle key (sp). ``events`` holds per-event keys
               (muon |p|, direction, straws crossed) for drawing whole events, including those with no off-axis steps.
               Lattice coordinates, not a continuous muon frame: only 11 % of the straw annulus is gas (144 layers
               21 mm apart in z), so a continuously transported step lands in a straw by chance (22 % of draws,
               measured); the secondaries' straw pattern relative to the muon's straws is what transfers.

Format v2 (build with a geometry; meta "format": 2): every library step also carries its placement inside ITS straw, in
a frame set by the straw and the relevant muon direction d (straw_frame): u' = s a (a = straw axis, s = sign(d . a)),
e1 = unit projection of d on the straw's transverse plane, e2 = u' x e1. Per end (0 = start, 1 = end): du = along-wire
offset along u' from a reference point, r = distance from the wire, phi = azimuth from e1 towards e2; mub = the muon
line's signed offset from the wire along e2 (pasting flips phi when the fast muon passes on the other side).
In-straw steps also carry chord coordinates per end: cn = e1 component over the muon chord's transverse half length
sqrt(R^2 - mub^2), h = e2 component minus mub (offset from the chord). Pasting uses these (to_chord / from_chord): a
point on the G4 chord lands on the fast chord whatever the fast muon's wire offset (pure r, phi would move delta rays
born on the muon track off it: measured 29 % vs G4 16 % of shared straws with a secondary nearer the wire).
  in-straw  reference = the muon chord's point nearest the wire (chord_u), d = the muon step's direction; plus dt =
            delay of the step after the muon step.
  off-axis  reference = where the anchor muon step's line passes closest to the target straw axis, clipped to the half
            length (line_u), d = the anchor muon step's direction.
steps.add_instraw_secondaries / add_offaxis_secondaries rebuild the ends around the fast muon (from_local). v1
libraries (no du0 column) keep the v1 placement.
"""
import numpy as np
import pandas as pd

from mu2efastsim import tables as core_tables

from . import tables as T
from .geometry import NSTRAWS, to_mu2e

P_BINS = np.array([0.0, 300.0, 600.0, 1000.0, 2000.0, 4000.0, 10000.0, 1e9])        # MeV/c
LEN_BINS = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 8.0, 1e9])             # mm, path in one straw
PLANE_BIN = 6                                                                         # planes per plane-extent bin
FORMAT_VERSION = 2
LOCAL_COLS = ["du0", "r0", "phi0", "du1", "r1", "phi1"]


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1), 1e-300)[:, None]


def muon_dir(p, p0, p1):
    """Unit muon direction from momentum rows p [N,3], falling back to the chord p1 - p0 where |p| = 0."""
    n = np.linalg.norm(p, axis=1)
    return _unit(np.where((n > 0)[:, None], p, p1 - p0))


def straw_frame(a, d):
    """Muon-relative straw frame: (s, e1, e2), s = sign(d . a), e1 = unit transverse part of d, e2 = (s a) x e1."""
    da = _dot(d, a)
    s = np.where(da < 0, -1.0, 1.0)
    t = d - da[:, None] * a
    alt = np.cross(a, np.where(np.abs(a[:, [1]]) < 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])))
    e1 = _unit(np.where((np.linalg.norm(t, axis=1) < 1e-9)[:, None], alt, t))
    return s, e1, np.cross(s[:, None] * a, e1)


def chord_u(m, a, p0, p1):
    """Along-wire coordinate (from the straw midpoint m along a) of the chord p0-p1's point nearest the wire."""
    w0, w1 = p0 - m, p1 - m
    t0 = w0 - _dot(w0, a)[:, None] * a
    dt = (w1 - _dot(w1, a)[:, None] * a) - t0
    t = np.clip(-_dot(t0, dt) / np.maximum(_dot(dt, dt), 1e-12), 0.0, 1.0)
    return _dot(w0 + t[:, None] * (w1 - w0), a)


def line_u(m, a, half, p, d):
    """Along-wire coordinate of the target axis point closest to the line p + t d, clipped to +-half."""
    w = m - p
    b = _dot(a, d)
    den = 1.0 - b * b
    u = np.where(den > 1e-9, (b * _dot(d, w) - _dot(a, w)) / np.maximum(den, 1e-9), -_dot(a, w))
    return np.clip(u, -half, half)


def to_local(x, m, a, s, e1, e2, uref):
    """Points x [N,3] -> (du, r, phi) in the straw frame (straw_frame) about the along-wire reference uref."""
    v = x - m
    ua = _dot(v, a)
    t = v - ua[:, None] * a
    return s * (ua - uref), np.linalg.norm(t, axis=1), np.arctan2(_dot(t, e2), _dot(t, e1))


def from_local(du, r, phi, m, a, s, e1, e2, uref, half, rmax=None):
    """(du, r, phi) -> points [N,3]; beyond a straw end the along-wire coordinate is mirrored back once, then clipped."""
    u = uref + s * np.asarray(du, float)
    u = np.clip(np.where(u > half, 2 * half - u, np.where(u < -half, -2 * half - u, u)), -half, half)
    r = np.asarray(r, float) if rmax is None else np.minimum(np.asarray(r, float), rmax)
    phi = np.asarray(phi, float)
    return m + u[:, None] * a + (r * np.cos(phi))[:, None] * e1 + (r * np.sin(phi))[:, None] * e2


def to_chord(x, m, a, s, e1, e2, uref, b, R):
    """Points x -> (du, cn, h) relative to the muon chord at wire offset b (along e2) in a straw of radius R."""
    v = x - m
    ua = _dot(v, a)
    t = v - ua[:, None] * a
    return s * (ua - uref), _dot(t, e1) / np.sqrt(np.maximum(R * R - b * b, 1e-6)), _dot(t, e2) - b


def from_chord(du, cn, h, m, a, s, e1, e2, uref, half, b, R):
    """(du, cn, h) -> points about a muon chord at wire offset b; transverse radius capped just inside R."""
    c1 = np.asarray(cn, float) * np.sqrt(np.maximum(R * R - b * b, 1e-6))
    c2 = b + np.asarray(h, float)
    rr = np.hypot(c1, c2)
    f = np.where(rr > R, R * (1 - 1e-6) / np.maximum(rr, 1e-300), 1.0)
    return from_local(du, rr * f, np.arctan2(c2, c1), m, a, s, e1, e2, uref, half)


def version(lib):
    """Library format: 2 if the steps carry straw-local placement, else 1."""
    return 2 if "du0" in lib["instraw_steps"] and "du0" in lib["offaxis"] else 1


def _xyz(df, end, prefix=""):
    return df[[f"{prefix}{c}{end}" for c in "xyz"]].to_numpy(float)


def _local_cols(out, x0, x1, m, a, s, e1, e2, uref, q):
    """Write du/r/phi of both step ends and the muon line's offset mub (q = a point of the muon line) into out."""
    out["mub"] = _dot(q - m, e2)
    for end, x in (("0", x0), ("1", x1)):
        du, r, ph = to_local(x, m, a, s, e1, e2, uref)
        out[f"du{end}"], out[f"r{end}"], out[f"phi{end}"] = du, r, ph


def anchor_offsets(off, mu, geom=None):
    """Lattice offsets of steps `off` from their anchor muon step (`mu` sorted by event and time, with rank k); with
    geom also the v2 straw-local placement about the anchor line's closest approach to the step's straw axis."""
    acols = ["x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"] if geom is not None else []
    a = mu[T.KEY + ["k", "plane", "panel", "straw", "t"] + acols].rename(
        columns={"plane": "aplane", "panel": "apanel", "straw": "astraw", "t": "ta", **{c: "a" + c for c in acols}})
    x = off.reset_index().merge(a, on=T.KEY)
    x["_key"] = (x.plane - x.aplane).abs() * 1e6 + (x.t - x.ta).abs()
    x = x.sort_values("_key", kind="stable").groupby("index", sort=True).first().reset_index()
    out = x[T.KEY + ["eion", "len", "ion", "shape", "pdg", "sp", "k"]].copy()
    out["dpl"] = (x.plane - x.aplane).astype(np.int32)
    out["dpa"] = ((x.panel - x.apanel) % 6).astype(np.int32)
    out["dst"] = (x.straw - x.astraw).astype(np.int32)
    out["dt"] = x.t - x.ta
    if geom is not None:
        js = (x.upanel.to_numpy() * NSTRAWS + x.straw.to_numpy()).astype(np.int64)
        m, ax, half = geom.mid[js], geom.dir[js], geom.halflen[js]
        p = to_mu2e(_xyz(x, "0", "a"))
        d = muon_dir(x[["apx", "apy", "apz"]].to_numpy(float), p, to_mu2e(_xyz(x, "1", "a")))
        s, e1, e2 = straw_frame(ax, d)
        _local_cols(out, to_mu2e(_xyz(x, "0")), to_mu2e(_xyz(x, "1")), m, ax, s, e1, e2, line_u(m, ax, half, p, d), p)
    return out


def build(paths, geom=None):
    """Return the library tables from reference tables (StrawGasStep positions in the detector frame); with geom
    (geometry.StrawGeometry, Mu2e frame) format v2, the straw-local placement of every step; without, format v1."""
    s = T.read_many(paths, "sgsteps")
    p = T.read_many(paths, "particles", T.KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    prim = p[(p.ismcprim == 1) & (p.pdg.abs() == 13)][T.KEY + ["sp"]].rename(columns={"sp": "mu"})
    s = s.merge(prim, on=T.KEY, how="inner").merge(p[T.KEY + ["sp", "pdg"]], on=T.KEY + ["sp"], how="left")
    is_mu = (s.sp == s.mu).to_numpy()
    mu, sec = s[is_mu].copy(), s[~is_mu].copy()
    mu["p"] = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2)

    # in-straw component: muon crossings, and the secondary steps sharing their straw
    ins = mu[T.KEY + ["sid", "p", "len", "eion"]].rename(columns={"len": "mu_len", "eion": "mu_eion"}).reset_index(drop=True)
    ins["xing"] = np.arange(len(ins), dtype=np.int64)
    st = sec.merge(ins[T.KEY + ["sid", "xing"]], on=T.KEY + ["sid"], how="inner")
    ins_steps = st[["xing", "eion", "len", "ion", "shape", "pdg"]].reset_index(drop=True)
    if geom is not None:
        mr = mu.reset_index(drop=True).iloc[st.xing.to_numpy()]          # the muon step of each secondary's crossing
        js = (mr.upanel.to_numpy() * NSTRAWS + mr.straw.to_numpy()).astype(np.int64)
        m, ax = geom.mid[js], geom.dir[js]
        q0, q1 = to_mu2e(_xyz(mr, "0")), to_mu2e(_xyz(mr, "1"))
        s_, e1, e2 = straw_frame(ax, muon_dir(mr[["px", "py", "pz"]].to_numpy(float), q0, q1))
        _local_cols(ins_steps, to_mu2e(_xyz(st, "0")), to_mu2e(_xyz(st, "1")), m, ax, s_, e1, e2, chord_u(m, ax, q0, q1), q0)
        ins_steps["dt"] = st.t.to_numpy() - mr.t.to_numpy()
        for end in "01":
            _, cn, h = to_chord(to_mu2e(_xyz(st, end)), m, ax, s_, e1, e2, 0.0, ins_steps.mub.to_numpy(), geom.radius)
            ins_steps[f"cn{end}"], ins_steps[f"h{end}"] = cn, h
    cnt = ins_steps.groupby("xing").agg(nsec=("eion", "size"), eion_sec=("eion", "sum"))
    ins = ins.merge(cnt.reset_index(), on="xing", how="left")
    ins["nsec"] = ins.nsec.fillna(0).astype(np.int32)
    ins["eion_sec"] = ins.eion_sec.fillna(0.0)
    ins["pbin"] = np.digitize(ins.p, P_BINS) - 1
    ins["lbin"] = np.digitize(ins.mu_len, LEN_BINS) - 1

    # off-axis component, in straw-lattice coordinates relative to an anchor muon step
    mu = mu.sort_values(T.KEY + ["t"]).reset_index(drop=True)
    mu["k"] = mu.groupby(T.KEY).cumcount()
    in_mu = sec.merge(mu[T.KEY + ["sid"]].drop_duplicates().assign(_inmu=1), on=T.KEY + ["sid"], how="left")
    off = sec[in_mu._inmu.isna().to_numpy()].reset_index(drop=True)
    off_out = anchor_offsets(off, mu, geom)

    # per-event keys for drawing
    first = mu.groupby(T.KEY, sort=False).first().reset_index()
    ev = first[T.KEY + ["p"]].copy()
    for c in ("x", "y", "z"):
        ev[f"u{c}"] = (first[f"p{c}"] / first.p).to_numpy()
    ev["nstraw_mu"] = mu.groupby(T.KEY).size().reindex(pd.MultiIndex.from_frame(ev[T.KEY])).to_numpy()
    pl = mu.groupby(T.KEY).plane.agg(["min", "max"]).reindex(pd.MultiIndex.from_frame(ev[T.KEY]))
    ev["plane_min"] = pl["min"].to_numpy().astype(np.int32)
    ev["plane_max"] = pl["max"].to_numpy().astype(np.int32)
    noff = off_out.groupby(T.KEY).size()
    ev["noff"] = noff.reindex(pd.MultiIndex.from_frame(ev[T.KEY])).fillna(0).astype(np.int32).to_numpy()
    ev["pbin"] = np.digitize(ev.p, P_BINS) - 1
    return {"instraw": ins, "instraw_steps": ins_steps, "offaxis": off_out.reset_index(drop=True), "events": ev}


def write(path, lib, provenance):
    """Library trees to `path`; provenance (inputs, counts, format) goes beside it as <path>.meta.json (no string trees)."""
    import json
    provenance = {**provenance, "format": version(lib)}
    core_tables.write(path, lib, as_ttree=False)
    with open(path.replace(".root", ".meta.json"), "w") as fh:
        json.dump(provenance, fh, indent=1)


def read(path):
    return {t: core_tables.read(path, t) for t in ("instraw", "instraw_steps", "offaxis", "events")}


def plane_bin(pmin, pmax, width=PLANE_BIN):
    """Coarse bin of a muon's plane extent: (first plane bin, last plane bin) packed into one integer."""
    nb = 36 // width + 1
    return (np.asarray(pmin) // width) * nb + np.asarray(pmax) // width


def draw_concentration(ev_index, count):
    """How concentrated a set of draws is on few library events: (distinct events drawn among those with steps,
    share of drawn steps from the 10 most drawn events)."""
    ev_index = np.asarray(ev_index)
    with_steps = ev_index[count[ev_index] > 0]
    if not len(with_steps):
        return {"distinct_with_steps": 0, "top10_step_share": 0.0}
    u, n = np.unique(with_steps, return_counts=True)
    steps = n * count[u]
    return {"distinct_with_steps": int(len(u)), "top10_step_share": float(np.sort(steps)[::-1][:10].sum() / steps.sum())}


class OffAxisSampler:
    """Draw whole G4 events' off-axis secondary sets for fast muons.

    Key: muon |p| bin, |cos theta_y| bin and the muon's plane extent (first and last plane in bins of
    plane_bin_width planes; 0 = no plane matching), so a lattice offset is applied to a muon crossing the same part of the tracker it was recorded in.
    Every library event of a key is a candidate, including events with no off-axis steps (most), so the occupancy is
    drawn with the right frequency. Empty keys fall back hierarchically: neighbouring plane-extent bins at the same
    (|p|, angle), then any plane extent at the same (|p|, angle), then neighbouring angle bins. The drawn set carries
    lattice offsets (k, dpl, dpa, dst, dt) that steps.add_offaxis_secondaries applies to the fast muon's straws.
    """

    COS_BINS = np.array([0.0, 0.5, 0.7, 0.85, 0.95, 1.01])

    def __init__(self, lib, rng, plane_bin_width=PLANE_BIN):
        ev = lib["events"].reset_index(drop=True)
        self.rng = rng
        self.use_planes = bool(plane_bin_width) and "plane_min" in ev
        self.pbw = int(plane_bin_width) if self.use_planes else 36
        self.nc = len(self.COS_BINS)
        self.npb = (36 // self.pbw + 1) ** 2
        cb = np.clip(np.digitize(np.abs(ev.uy), self.COS_BINS) - 1, 0, self.nc - 2)
        pa = ev.pbin.to_numpy() * self.nc + cb
        pb = plane_bin(ev.plane_min, ev.plane_max, self.pbw) if self.use_planes else np.zeros(len(ev), np.int64)
        order = np.lexsort((pb, pa))
        self.pa_sorted, self.pb_sorted = pa[order], pb[order]
        self.full_sorted = self.pa_sorted * self.npb + self.pb_sorted
        self.ev = ev.iloc[order].reset_index(drop=True)
        off = lib["offaxis"]
        ek = self.ev[T.KEY].copy()
        ek["ei"] = np.arange(len(ek))
        if "nstraw_mu" in self.ev:
            ek["nstraw_mu"] = self.ev.nstraw_mu.to_numpy()
        off = off.merge(ek, on=T.KEY, how="inner").sort_values("ei", kind="stable").reset_index(drop=True)
        self.off = off
        ei = off.ei.to_numpy()
        self.first = np.searchsorted(ei, np.arange(len(ek)), "left")
        self.count = np.searchsorted(ei, np.arange(len(ek)), "right") - self.first
        self.last_level = None

    def _range(self, arr, key):
        return np.searchsorted(arr, key, "left"), np.searchsorted(arr, key, "right")

    def draw(self, p, uy, plane_min=None, plane_max=None):
        """For fast muons (|p| and direction y component at the first straw, plane extent): (library event index per
        muon, step table with column `i` = muon index). `last_level` records the fallback level used per muon
        (0 = exact key, 1 = neighbouring plane-extent bin, 2 = any plane extent, 3 = neighbouring angle bin)."""
        pbn = np.clip(np.digitize(p, P_BINS) - 1, 0, len(P_BINS) - 2)
        cb = np.clip(np.digitize(np.abs(uy), self.COS_BINS) - 1, 0, self.nc - 2)
        pa = pbn * self.nc + cb
        n = len(pa)
        level = np.full(n, -1)
        lo, hi = np.zeros(n, np.int64), np.zeros(n, np.int64)
        if self.use_planes and plane_min is not None:
            nb = 36 // self.pbw + 1
            b0, b1 = np.asarray(plane_min) // self.pbw, np.asarray(plane_max) // self.pbw
            for lev, shifts in ((0, [(0, 0)]), (1, [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)])):
                for d0, d1 in shifts:
                    e = level < 0
                    if not e.any():
                        break
                    pbk = np.clip(b0[e] + d0, 0, nb - 1) * nb + np.clip(b1[e] + d1, 0, nb - 1)
                    l2, h2 = self._range(self.full_sorted, pa[e] * self.npb + pbk)
                    got = h2 > l2
                    idx = np.flatnonzero(e)[got]
                    lo[idx], hi[idx], level[idx] = l2[got], h2[got], lev
        for lev, dcs in ((2, [0]), (3, [1, -1, 2, -2])):
            for dc in dcs:
                e = level < 0
                if not e.any():
                    break
                k2 = pbn[e] * self.nc + np.clip(cb[e] + dc, 0, self.nc - 2)
                l2, h2 = self._range(self.pa_sorted, k2)
                got = h2 > l2
                idx = np.flatnonzero(e)[got]
                lo[idx], hi[idx], level[idx] = l2[got], h2[got], lev
        self.last_level = level
        ev = np.minimum(lo + (self.rng.random(n) * np.maximum(hi - lo, 1)).astype(np.int64), len(self.ev) - 1)
        cnt = self.count[ev]
        i = np.repeat(np.arange(n), cnt)
        src = np.repeat(self.first[ev], cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        st = self.off.iloc[src].reset_index(drop=True)
        st["i"] = i
        self.last_concentration = draw_concentration(ev, self.count)
        return ev, st


class InStrawSampler:
    """Draw secondary in-straw content for fast muon straw crossings from the library, by (|p| bin, path bin)."""

    def __init__(self, lib, rng):
        ins = lib["instraw"]
        self.rng = rng
        key = ins.pbin.to_numpy() * len(LEN_BINS) + ins.lbin.to_numpy()
        order = np.argsort(key, kind="stable")
        self.key_sorted = key[order]
        self.xing_sorted = ins.xing.to_numpy()[order]
        st = lib["instraw_steps"].sort_values("xing", kind="stable").reset_index(drop=True)
        self.steps = st
        self.first = np.searchsorted(st.xing.to_numpy(), np.arange(len(ins)), "left")
        self.count = np.searchsorted(st.xing.to_numpy(), np.arange(len(ins)), "right") - self.first
        filled = np.unique(key)
        self.filled = filled

    def _ranges(self, key):
        lo = np.searchsorted(self.key_sorted, key, "left")
        hi = np.searchsorted(self.key_sorted, key, "right")
        return lo, hi

    def draw(self, p, length):
        """For arrays (|p|, path) of fast muon crossings: (crossing index of each draw, step table of the secondaries
        with a column `i` = index into the input arrays)."""
        pb = np.clip(np.digitize(p, P_BINS) - 1, 0, len(P_BINS) - 2)
        lb = np.clip(np.digitize(length, LEN_BINS) - 1, 0, len(LEN_BINS) - 2)
        key = pb * len(LEN_BINS) + lb
        # empty bins (rare): move to the nearest populated key of the same |p| bin, vectorised over a few shifts
        lo, hi = self._ranges(key)
        for dl in (1, -1, 2, -2, 3, -3, 4, -4):
            e = hi <= lo
            if not e.any():
                break
            k2 = pb[e] * len(LEN_BINS) + np.clip(lb[e] + dl, 0, len(LEN_BINS) - 2)
            l2, h2 = self._ranges(k2)
            lo[e], hi[e] = l2, h2
        idx = lo + (self.rng.random(len(key)) * np.maximum(hi - lo, 1)).astype(np.int64)
        xing = self.xing_sorted[np.minimum(idx, len(self.xing_sorted) - 1)]
        n = self.count[xing]
        i = np.repeat(np.arange(len(key)), n)
        src = np.repeat(self.first[xing], n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        st = self.steps.iloc[src].reset_index(drop=True)
        st["i"] = i
        return xing, st
