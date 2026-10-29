"""Shared-draw, lattice-only off-axis migration diagnostics (not physics output).

Input is one file's saved baseline steps, with production primary keys <1000,
plus a topology-annotated OffAxisSampler. draw_context consumes its RNG once;
trace/aggregate/classify are pure and never modify inputs or consume randomness.
Keep the returned st to compare all four anchor combinations. draw_row identifies
one sampled occurrence within that file, not a unique donor library row.

Both anchors follow production ordering, including input-order time ties and the
legacy floating-point relative-rank formula. Hybrids change only the lattice:
time uses the default anchor except in full topology mode. No endpoints are
constructed; occupancy must use final (KEY, sid), and location final lattice
coordinates. Invalid rows have sid/upanel=-1 and classify as -1. The event-wide
muon veto includes every primary, not just the draw's parent. No edge-anchor mode.
"""
import numpy as np
import pandas as pd

from .. import steps as S, tables as T
from ..placement import TOPOLOGY, topology_anchor

MK = T.KEY + ["sp"]


def draw_context(primary_steps, sampler):
    """Return (mu, first, st); first row index is the draw's recipient i.

    mu includes per-primary crossing rank k. first includes donor_index, even for
    draws with zero secondary rows. st includes file-local unique draw_row.
    Sampler placement flags do not affect drawing; modes are explicit in trace.
    """
    mu = primary_steps[primary_steps.sp < S.SECONDARY_KEY0].sort_values(MK + ["t"]).reset_index(drop=True).copy()
    mu["k"] = mu.groupby(MK).cumcount()
    first = mu.groupby(MK, sort=False).first().reset_index()
    if len(mu):
        pl = mu.groupby(MK, sort=False).plane.agg(["min", "max"]).reindex(pd.MultiIndex.from_frame(first[MK]))
        p = np.sqrt(first.px ** 2 + first.py ** 2 + first.pz ** 2).to_numpy()
        donor, st = sampler.draw(p, (first.py / p).to_numpy(), pl["min"].to_numpy(), pl["max"].to_numpy())
        first["donor_index"] = donor
    else:
        first["donor_index"] = np.zeros(0, dtype=np.int64)
        st = getattr(sampler, "off", pd.DataFrame(columns=TOPOLOGY +
                     ["sp", "k", "dpl", "dpa", "dst", "dt", "eion", "len"])).iloc[:0].copy()
        st["i"] = np.zeros(0, dtype=np.int64)
    st = st.reset_index(drop=True).copy()
    st["draw_row"] = np.arange(len(st), dtype=np.int64)
    return mu, first, st


def trace(mu, first, st, geom, plane_mode="default", transverse_mode="default"):
    """One output row per raw sampled occurrence, including invalid/vetoed rows.

    anchor_{default,topology}_{k,plane,panel,straw,t} record both choices.
    target_plane/target_straw are unreflected; plane/panel/straw/sid/upanel are
    final lattice addresses. reflected_* means an actual successful one-bounce
    change, as in production; valid is the final lattice bounds check.
    """
    if plane_mode not in ("default", "topology") or transverse_mode not in ("default", "topology"):
        raise ValueError("anchor modes must be default or topology")
    # The existing helper validates annotations and recipient identities, too.
    kt = topology_anchor(mu, MK, first, st)
    i = st.i.to_numpy(dtype=np.int64)
    nk = mu.groupby(MK, sort=False).size().reindex(pd.MultiIndex.from_frame(first[MK])).to_numpy()
    kd = st.k.to_numpy()
    if "nstraw_mu" in st:
        # Do not replace with integer division: it changes floating-point ties.
        kd = np.floor(kd / np.maximum(st.nstraw_mu.to_numpy(), 1) * nk[i]).astype(np.int64)
    kd = np.clip(kd, 0, nk[i] - 1)
    out = st.reset_index(drop=True).rename(columns={**{c: "donor_" + c for c in T.KEY},
                                                    "sp": "gsp", "k": "donor_k"}).copy()
    if "draw_row" not in out:
        out["draw_row"] = np.arange(len(out), dtype=np.int64)
    if out.draw_row.isna().any() or out.draw_row.duplicated().any():
        raise ValueError("draw_row must uniquely identify each sampled occurrence")
    for c in T.KEY:
        out[c] = first[c].to_numpy()[i]
    out["parent"] = first.sp.to_numpy()[i]
    crossings = mu.set_index(MK + ["k"])
    for mode, k in (("default", kd), ("topology", kt)):
        query = pd.DataFrame({**{c: first[c].to_numpy()[i] for c in MK}, "k": k})
        anchor = crossings.reindex(pd.MultiIndex.from_frame(query))
        out[f"anchor_{mode}_k"] = k
        for c in ("plane", "panel", "straw", "t"):
            out[f"anchor_{mode}_{c}"] = anchor[c].to_numpy()
    plane0 = out[f"anchor_{plane_mode}_plane"].to_numpy() + st.dpl.to_numpy()
    straw0 = out[f"anchor_{transverse_mode}_straw"].to_numpy() + st.dst.to_numpy()
    panel = (out[f"anchor_{transverse_mode}_panel"].to_numpy() + st.dpa.to_numpy()) % S.NPANELS_
    plane, straw = S._mirror(plane0, S.NPLANES_), S._mirror(straw0, S.NSTRAWS_)
    valid = (plane >= 0) & (plane < S.NPLANES_) & (straw >= 0) & (straw < S.NSTRAWS_)
    js = np.where(valid, (plane * S.NPANELS_ + panel) * S.NSTRAWS_ + straw, 0).astype(np.int64)
    out["target_plane"], out["target_straw"] = plane0, straw0
    out["reflected_plane"], out["reflected_straw"] = plane != plane0, straw != straw0
    out["plane"] = np.where(valid, geom.plane[js], plane)
    out["panel"] = np.where(valid, geom.panel[js], panel)
    out["straw"] = np.where(valid, geom.straw[js], straw)
    out["sid"] = np.where(valid, geom.sid[js], -1)
    out["upanel"] = np.where(valid, geom.upanel[js], -1)
    out["valid"] = valid
    out["in_muon"] = valid & pd.MultiIndex.from_frame(out[T.KEY + ["sid"]]).isin(
        pd.MultiIndex.from_frame(mu[T.KEY + ["sid"]]))
    out["keep"] = valid & ~out.in_muon.to_numpy()
    time_mode = "topology" if plane_mode == transverse_mode == "topology" else "default"
    out["t"] = out[f"anchor_{time_mode}_t"] + st.dt.to_numpy()
    out["plane_mode"], out["transverse_mode"] = plane_mode, transverse_mode
    return out


def aggregate(traced):
    """Kept rows collapsed by KEY+(i,gsp,sid), exactly production particle keys.

    eion/len sum, earliest t (stable ties) and representative draw_row retained;
    raw_n is the number of raw survivors in each merged step. Particle numbering
    uses surviving (i,gsp) pairs only. Returned order matches production's stable
    time sort then first group occurrence. Sum(raw_n)-len(result) is merge loss,
    not a veto/lattice loss. Deduplicate KEY+sid separately for occupancy.
    """
    out = traced[traced.keep].copy()
    pair = out.groupby(["i", "gsp"], sort=True).ngroup()
    out["sp"] = S.OFFAXIS_KEY0 + pair.groupby([out[c] for c in T.KEY]).rank(method="dense").astype(np.int64) - 1
    out = out.sort_values("t", kind="stable")
    g = out.groupby(T.KEY + ["i", "gsp", "sid"], sort=False)
    result = g.first()
    result["eion"], result["len"], result["raw_n"] = g.eion.sum(), g.len.sum(), g.size()
    return result.reset_index()


def classify(traced, mu):
    """Return positional int loc array from final lattice (raw or aggregated).

    Event-wide priority: same muon straw=0; same panel within three straws=1;
    same plane otherwise=2; nearest muon plane 1-2 away=3; >=3 away=4.
    Invalid=-1. A valid row in an event without muon planes is 4, as seccensus.
    """
    s = traced.reset_index(drop=True)
    loc = np.full(len(s), -1, dtype=np.int64)
    inmu = pd.MultiIndex.from_frame(s[T.KEY + ["sid"]]).isin(pd.MultiIndex.from_frame(mu[T.KEY + ["sid"]]))
    valid = s.valid.to_numpy(dtype=bool)
    loc[valid & inmu] = 0
    rest = s.loc[valid & ~inmu, T.KEY + ["plane", "upanel", "straw"]].copy()
    rest["row"] = rest.index
    if rest.empty:
        return loc
    planes = mu[T.KEY + ["plane"]].drop_duplicates().rename(columns={"plane": "mpl"})
    y = rest.merge(planes, on=T.KEY, how="left")
    d = (y.plane - y.mpl).abs().groupby(y.row).min().reindex(rest.index).to_numpy()
    panels = mu[T.KEY + ["upanel", "straw"]].drop_duplicates().rename(columns={"straw": "mst"})
    z = rest.merge(panels, on=T.KEY + ["upanel"], how="inner")
    ds = (z.straw - z.mst).abs().groupby(z.row).min().reindex(rest.index).to_numpy()
    loc[rest.index] = np.where(d == 0, np.where(ds <= 3, 1, 2), np.where(d <= 2, 3, 4))
    return loc
