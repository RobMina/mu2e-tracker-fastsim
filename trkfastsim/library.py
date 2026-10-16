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
  ``offaxis``  one row per non-muon step NOT in a muon straw, in the muon frame: event id, distance along the muon
               line from its first straw entry (u), signed transverse offsets (v, w) of the step midpoint, the step's
               eion, len, ion type, pdg; plus per-event keys (muon |p|, direction) for drawing whole events.

Frames: the muon line is fitted through its own step midpoints (G4, detector frame); u is along the muon direction,
v is horizontal and perpendicular to it, w completes the right-handed frame.
"""
import numpy as np
import pandas as pd

from mu2efastsim import tables as core_tables

from . import tables as T

P_BINS = np.array([0.0, 300.0, 600.0, 1000.0, 2000.0, 4000.0, 10000.0, 1e9])        # MeV/c
LEN_BINS = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 8.0, 1e9])             # mm, path in one straw


def _muon_frame(mu_steps):
    """Per event: origin (first entry), unit direction u, and v, w (detector frame) from the muon's own steps."""
    m = mu_steps.sort_values("t")
    first = m.groupby(T.KEY, sort=False).first()
    last = m.groupby(T.KEY, sort=False).last()
    o = first[["x0", "y0", "z0"]].to_numpy()
    d = last[["x1", "y1", "z1"]].to_numpy() - o
    n = np.linalg.norm(d, axis=1)
    # single-straw muons: use the step's own chord
    d = np.where(n[:, None] > 1e-6, d, first[["x1", "y1", "z1"]].to_numpy() - o)
    u = d / np.linalg.norm(d, axis=1)[:, None]
    v = np.cross(u, [0.0, 1.0, 0.0])
    vn = np.linalg.norm(v, axis=1)
    v = np.where(vn[:, None] > 1e-6, v / np.maximum(vn, 1e-12)[:, None], [1.0, 0.0, 0.0])
    w = np.cross(u, v)
    f = first.reset_index()[T.KEY]
    return f, o, u, v, w


def build(paths):
    """Return (instraw, offaxis, events) DataFrames from reference tables."""
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
    cnt = ins_steps.groupby("xing").agg(nsec=("eion", "size"), eion_sec=("eion", "sum"))
    ins = ins.merge(cnt.reset_index(), on="xing", how="left")
    ins["nsec"] = ins.nsec.fillna(0).astype(np.int32)
    ins["eion_sec"] = ins.eion_sec.fillna(0.0)
    ins["pbin"] = np.digitize(ins.p, P_BINS) - 1
    ins["lbin"] = np.digitize(ins.mu_len, LEN_BINS) - 1

    # off-axis component, in the muon frame
    keys, o, u, v, w = _muon_frame(mu)
    fr = keys.copy()
    for j, c in enumerate("xyz"):
        fr[f"o{c}"], fr[f"u{c}"], fr[f"v{c}"], fr[f"w{c}"] = o[:, j], u[:, j], v[:, j], w[:, j]
    in_mu = sec.merge(mu[T.KEY + ["sid"]].drop_duplicates().assign(_inmu=1), on=T.KEY + ["sid"], how="left")
    off = sec[in_mu._inmu.isna().to_numpy()].merge(fr, on=T.KEY)
    mid = 0.5 * (off[["x0", "y0", "z0"]].to_numpy() + off[["x1", "y1", "z1"]].to_numpy())
    rel = mid - off[["ox", "oy", "oz"]].to_numpy()
    off_out = off[T.KEY + ["eion", "len", "ion", "shape", "pdg"]].copy()
    for name, ax in (("u", "u"), ("v", "v"), ("w", "w")):
        off_out[name] = np.einsum("ij,ij->i", rel, off[[f"{ax}x", f"{ax}y", f"{ax}z"]].to_numpy())

    # per-event keys for drawing
    first = mu.sort_values("t").groupby(T.KEY, sort=False).first().reset_index()
    ev = first[T.KEY + ["p"]].merge(fr[T.KEY + ["ux", "uy", "uz"]], on=T.KEY)
    ev["nstraw_mu"] = mu.groupby(T.KEY).size().reindex(pd.MultiIndex.from_frame(ev[T.KEY])).to_numpy()
    noff = off_out.groupby(T.KEY).size()
    ev["noff"] = noff.reindex(pd.MultiIndex.from_frame(ev[T.KEY])).fillna(0).astype(np.int32).to_numpy()
    ev["pbin"] = np.digitize(ev.p, P_BINS) - 1
    return {"instraw": ins, "instraw_steps": ins_steps, "offaxis": off_out.reset_index(drop=True), "events": ev}


def write(path, lib, provenance):
    """Library trees to `path`; provenance (inputs, counts) goes beside it as <path>.meta.json (no string trees)."""
    import json
    core_tables.write(path, lib, as_ttree=False)
    with open(path.replace(".root", ".meta.json"), "w") as fh:
        json.dump(provenance, fh, indent=1)


def read(path):
    return {t: core_tables.read(path, t) for t in ("instraw", "instraw_steps", "offaxis", "events")}


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
