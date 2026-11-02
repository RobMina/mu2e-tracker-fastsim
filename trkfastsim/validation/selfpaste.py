"""selfpaste -- paste each Geant4 event's own off-axis secondaries back onto its own muon (closure of the placement).

The library is built in memory from the input reference tables (library.build, format 2). Each event's MC-primary
muon steps become the recipient, and the sampler returns that same event's off-axis set (no binning, no draw). A
perfect paster reproduces every step's straw. Reported per class of place relative to the muon (same plane, one to
two planes, farther) and per muon plane-extent width: the steps of the reference, the pasted steps, those that land on
their own straw, and the paste's rejections (off the lattice, in a muon straw, merged).
--anchor selects the anchor rule: default (relative crossing rank), edge, or topology (needs buildlib --topology
columns, built here with topology=True).
Output: <outdir>/selfpaste.json.
"""
import json
import os

import numpy as np
import pandas as pd

from trkfastsim import geometry, tables as T
from trkfastsim.sim import library as L, steps as S
from trkfastsim.validation.shardhist import _place

CLASSES = ("same plane", "1-2 planes", ">= 3 planes")


class OwnEventSampler(L.OffAxisSampler):
    """OffAxisSampler whose draw returns, for each recipient muon, the library event of the same (run, subrun, event)."""

    def __init__(self, lib, keys, **kw):
        super().__init__(lib, np.random.default_rng(0), **kw)
        self._keys = keys

    def draw(self, p, uy, plane_min=None, plane_max=None):
        k = self._keys.merge(self.ev[T.KEY].reset_index().rename(columns={"index": "ei"}), on=T.KEY, how="left")
        if k.ei.isna().any():
            raise ValueError("recipient without its own library event")
        ev = k.ei.to_numpy(np.int64)
        self.last_level = np.zeros(len(ev), np.int64)
        cnt = self.count[ev]
        i = np.repeat(np.arange(len(ev)), cnt)
        src = np.repeat(self.first[ev], cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        st = self.off.iloc[src].reset_index(drop=True)
        st["i"] = i
        self.last_concentration = L.draw_concentration(ev, self.count)
        return ev, st


def _recipients(path):
    """MC-primary muon steps (one muon per event, key 1) and the event's other steps, from one reference file."""
    s = T.read(path, "sgsteps")
    p = T.read(path, "particles", T.KEY + ["key", "pdg", "isprim", "ismcprim"]).rename(columns={"key": "sp"})
    s = s.merge(p, on=T.KEY + ["sp"], how="left")
    prim = p[(p.ismcprim == 1) & (p.pdg.abs() == 13)]
    one = prim.groupby(T.KEY).size()
    prim = prim.set_index(T.KEY).loc[one[one == 1].index].reset_index()[T.KEY + ["sp"]].rename(columns={"sp": "mu"})
    s = s.merge(prim, on=T.KEY, how="inner")
    mu = s[s.sp == s.mu].copy()
    mu["sp"] = 1
    other = s[s.sp != s.mu].copy()
    return mu, other


def run(inputs, outdir, gdml=None, anchor="default", **kw):
    from trkfastsim.sim.run import default_gdml
    os.makedirs(outdir, exist_ok=True)
    geom = geometry.from_gdml(gdml or default_gdml())
    paths = T.expand(inputs)
    tot = {}
    for path in paths:
        lib = L.build([path], geom, topology=(anchor == "topology"))
        mu, other = _recipients(path)
        mu = mu.merge(lib["events"][T.KEY], on=T.KEY, how="inner")
        if not len(mu):
            continue
        cols = [c for c in mu.columns if c in ("run", "subrun", "event", "sid", "plane", "panel", "straw", "upanel",
                                                 "shape", "ion", "sp", "eion", "len", "width", "t", "x0", "y0", "z0",
                                                 "x1", "y1", "z1", "px", "py", "pz", "nsgs")]
        steps = mu[cols].reset_index(drop=True)
        parts = pd.DataFrame({**{c: steps.drop_duplicates(T.KEY)[c].to_numpy() for c in T.KEY}})
        parts["key"], parts["parent"], parts["pdg"], parts["isprim"], parts["ismcprim"] = 1, 0, 13, 1, 1
        f = steps.sort_values(T.KEY + ["sp", "t"]).groupby(T.KEY + ["sp"], sort=False).first().reset_index()
        sampler = OwnEventSampler(lib, f[T.KEY], edge_anchor=(anchor == "edge"), topology_anchor=(anchor == "topology"))
        out, _, stats = S.add_offaxis_secondaries(steps, parts, sampler, geom)
        pasted = out[out.sp >= S.OFFAXIS_KEY0]
        # reference: the event's off-axis steps (not in a straw of its muon)
        ms = steps[T.KEY + ["sid"]].drop_duplicates().assign(_m=1)
        ref = other.merge(lib["events"][T.KEY], on=T.KEY).merge(ms, on=T.KEY + ["sid"], how="left")
        ref = ref[ref._m.isna()].drop(columns="_m")
        width = steps.groupby(T.KEY).plane.agg(["min", "max"]).pipe(lambda b: (b["max"] - b["min"]).rename("w")).reset_index()
        for name, d in (("reference", ref), ("pasted", pasted)):
            a = pd.concat([steps[T.KEY + ["sid", "plane"]].assign(mu=True), d[T.KEY + ["sid", "plane"]].assign(mu=False)],
                          ignore_index=True)
            place, _ = _place(a)
            w = a[~a.mu][T.KEY].merge(width, on=T.KEY, how="left").w.to_numpy()
            wb = np.minimum(w, 6).astype(np.int64)
            ok = (place >= 1) & (place <= 3)                   # in a muon straw or without a muon: not off-axis
            c = np.bincount((place[ok] - 1) * 7 + wb[ok], minlength=21)
            tot[f"{name}_not_off_axis"] = tot.get(f"{name}_not_off_axis", 0) + int((~ok).sum())
            tot[name] = tot.get(name, 0) + c
        same = pasted[T.KEY + ["sid"]].drop_duplicates().merge(ref[T.KEY + ["sid"]].drop_duplicates(), on=T.KEY + ["sid"])
        tot["pasted_on_a_reference_straw"] = tot.get("pasted_on_a_reference_straw", 0) + len(same)
        for k in ("drawn", "placed", "in_muon_straw", "off_lattice", "reflected", "merged_same_particle"):
            tot[k] = tot.get(k, 0) + int(stats.get(k, 0))
        tot["events"] = tot.get("events", 0) + int(len(f))
    res = {"anchor": anchor, "inputs": [os.path.basename(p) for p in paths], "width_bins": "0..5, 6 = 6 or more",
           "classes": list(CLASSES)}
    for name in ("reference", "pasted"):
        c = np.asarray(tot.get(name, np.zeros(21))).reshape(3, 7)
        res[name] = {"by_class": dict(zip(CLASSES, c.sum(1).tolist())), "by_class_and_width": c.tolist()}
    r, q = np.asarray(res["reference"]["by_class_and_width"], float), np.asarray(res["pasted"]["by_class_and_width"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        res["pasted_over_reference"] = {"by_class": (q.sum(1) / r.sum(1)).tolist(), "by_class_and_width": (q / r).tolist()}
    res.update({k: v for k, v in tot.items() if k not in ("reference", "pasted")})
    with open(os.path.join(outdir, "selfpaste.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=int)
    print(json.dumps({k: res[k] for k in ("anchor", "pasted_over_reference")}, indent=1))
    return res
