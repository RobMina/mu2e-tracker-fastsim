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

"""
import numpy as np
import pandas as pd

from mu2efastsim import tables as core_tables

from . import tables as T

P_BINS = np.array([0.0, 300.0, 600.0, 1000.0, 2000.0, 4000.0, 10000.0, 1e9])        # MeV/c
LEN_BINS = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 8.0, 1e9])             # mm, path in one straw


def anchor_offsets(off, mu):
    """Lattice offsets of steps `off` from their anchor muon step (`mu` sorted by event and time, with rank k)."""
    a = mu[T.KEY + ["k", "plane", "panel", "straw", "t"]].rename(
        columns={"plane": "aplane", "panel": "apanel", "straw": "astraw", "t": "ta"})
    x = off.reset_index().merge(a, on=T.KEY)
    x["_key"] = (x.plane - x.aplane).abs() * 1e6 + (x.t - x.ta).abs()
    x = x.sort_values("_key", kind="stable").groupby("index", sort=True).first().reset_index()
    out = x[T.KEY + ["eion", "len", "ion", "shape", "pdg", "sp", "k"]].copy()
    out["dpl"] = (x.plane - x.aplane).astype(np.int32)
    out["dpa"] = ((x.panel - x.apanel) % 6).astype(np.int32)
    out["dst"] = (x.straw - x.astraw).astype(np.int32)
    out["dt"] = x.t - x.ta
    return out


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

    # off-axis component, in straw-lattice coordinates relative to an anchor muon step
    mu = mu.sort_values(T.KEY + ["t"]).reset_index(drop=True)
    mu["k"] = mu.groupby(T.KEY).cumcount()
    in_mu = sec.merge(mu[T.KEY + ["sid"]].drop_duplicates().assign(_inmu=1), on=T.KEY + ["sid"], how="left")
    off = sec[in_mu._inmu.isna().to_numpy()].reset_index(drop=True)
    off_out = anchor_offsets(off, mu)

    # per-event keys for drawing
    first = mu.groupby(T.KEY, sort=False).first().reset_index()
    ev = first[T.KEY + ["p"]].copy()
    for c in ("x", "y", "z"):
        ev[f"u{c}"] = (first[f"p{c}"] / first.p).to_numpy()
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


class OffAxisSampler:
    """Draw whole G4 events' off-axis secondary sets for fast muons, by muon |p| bin and |cos theta_y| bin.

    Every library event of the bin is a candidate, including events with no off-axis steps (most), so the occupancy
    is drawn with the right frequency. A drawn set carries lattice offsets (k, dpl, dpa, dst, dt) that
    steps.add_offaxis_secondaries applies to the fast muon's own straws.
    """

    COS_BINS = np.array([0.0, 0.5, 0.7, 0.85, 0.95, 1.01])

    def __init__(self, lib, rng):
        ev = lib["events"].reset_index(drop=True)
        self.rng = rng
        cb = np.clip(np.digitize(np.abs(ev.uy), self.COS_BINS) - 1, 0, len(self.COS_BINS) - 2)
        key = ev.pbin.to_numpy() * len(self.COS_BINS) + cb
        order = np.argsort(key, kind="stable")
        self.key_sorted = key[order]
        self.ev = ev.iloc[order].reset_index(drop=True)
        off = lib["offaxis"]
        ek = self.ev[T.KEY].copy()
        ek["ei"] = np.arange(len(ek))
        off = off.merge(ek, on=T.KEY, how="inner").sort_values("ei", kind="stable").reset_index(drop=True)
        self.off = off
        ei = off.ei.to_numpy()
        self.first = np.searchsorted(ei, np.arange(len(ek)), "left")
        self.count = np.searchsorted(ei, np.arange(len(ek)), "right") - self.first

    def draw(self, p, uy):
        """For fast muons (|p| at the first straw, direction y component): (library event index per muon, step table
        with column `i` = muon index)."""
        pb = np.clip(np.digitize(p, P_BINS) - 1, 0, len(P_BINS) - 2)
        cb = np.clip(np.digitize(np.abs(uy), self.COS_BINS) - 1, 0, len(self.COS_BINS) - 2)
        key = pb * len(self.COS_BINS) + cb
        lo = np.searchsorted(self.key_sorted, key, "left")
        hi = np.searchsorted(self.key_sorted, key, "right")
        for dc in (1, -1, 2, -2):
            e = hi <= lo
            if not e.any():
                break
            k2 = pb[e] * len(self.COS_BINS) + np.clip(cb[e] + dc, 0, len(self.COS_BINS) - 2)
            lo[e], hi[e] = np.searchsorted(self.key_sorted, k2, "left"), np.searchsorted(self.key_sorted, k2, "right")
        ev = np.minimum(lo + (self.rng.random(len(key)) * np.maximum(hi - lo, 1)).astype(np.int64), len(self.ev) - 1)
        n = self.count[ev]
        i = np.repeat(np.arange(len(key)), n)
        src = np.repeat(self.first[ev], n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        st = self.off.iloc[src].reset_index(drop=True)
        st["i"] = i
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
