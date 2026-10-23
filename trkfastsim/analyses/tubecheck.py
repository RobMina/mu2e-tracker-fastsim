"""tubecheck -- the tube and polycone material the fast world leaves out above the tracker, along the muons G4 delivers.

world.load (the stage C world above the plane) keeps box and xtru solids only; world.load_tubes reads tubes and
polycones, which the fast transport uses inside the tracker (TrackerInterior) but not above the plane. For the events
in worldcheck_events.csv (from `worldcheck`) whose primary muon passes the tracker term in both arms, this takes the
straight generated line from the CRY vertex to the plane above the straws and integrates the tube/polycone column
along it (world.column_tubes on the shells whose bounding box the line's box overlaps, excluding the tracker interior
below the plane), per material, in g/cm2 and as the Bethe mean loss at the muon's generated |p|. It then sets that
against the paired loss difference (G4 minus fast) of the same events: if the omitted tubes carry the deficit, the
difference grows with the tube column.

    tubecheck WORLDCHECK_DIR --gdml mu2e_extracted_v04.gdml --outdir DIR
"""
import os

import numpy as np
import pandas as pd

from mu2efastsim import eloss, materials, plots
from mu2efastsim import world as world_mod
from .. import geometry, steps as st

TUBE_EDGES = np.array([-1.0, 0.01, 1.0, 5.0, 20.0, 1e9])          # g/cm2 of omitted tube material
DENSITY_FALLBACK = 7.9                                            # g/cm3 (steel) for materials without constants


def _density(name, gdml_rho):
    m = materials.BY_NAME.get(name)
    return m.rho if m else gdml_rho.get(name, DENSITY_FALLBACK)


def _gdml_densities(path):
    import re
    import xml.etree.ElementTree as ET
    out = {}
    for m in ET.parse(path).getroot().find("materials").findall("material"):
        d = m.find("D")
        if d is not None:
            out[re.sub(r"0x[0-9a-f]+$", "", m.get("name"))] = float(d.get("value"))
    return out


def _cull(tubes, lo, hi):
    """Shells whose global bounding box (from rmax, half z and the rotation) overlaps the box [lo, hi]."""
    keep = []
    for t in tubes:
        mat, R, tc, rmin, rmax, hz, phi0, dphi = t
        ext = np.abs(R) @ np.array([rmax, rmax, hz])
        if np.all(tc + ext >= lo) and np.all(tc - ext <= hi):
            keep.append(t)
    return keep


def interior_top_y(interior, y_plane):
    """Highest y [mm] of the tracker-interior shells (coaxial about the horizontal tracker axis: axis y + rmax)."""
    top = max([g["rmax"] for g in interior.shells.groups] + [0.0]) + interior.shells.c[1]
    return float(max(top, y_plane))


def run(inputs, outdir, gdml=None, chunk=500, **kw):
    from .steps import default_gdml
    gdml = gdml or default_gdml()
    geom = geometry.from_gdml(gdml)
    y_pl = st.plane_y(geom)
    ev = pd.read_csv(os.path.join(inputs[0], "worldcheck_events.csv"))
    ev = ev[ev.p_first.notna() & ev.p_first_g4.notna()].reset_index(drop=True)
    o = ev[["x", "y", "z"]].to_numpy()
    p = ev.p_gen.to_numpy()
    d = ev[["px", "py", "pz"]].to_numpy() / p[:, None]
    L = (o[:, 1] - y_pl) / -d[:, 1]
    end = o + d * L[:, None]
    lo, hi = np.minimum(o, end).min(0), np.maximum(o, end).max(0)
    tubes = [t for t in world_mod.load_tubes(gdml, bbox=(lo, hi)) if t[0] != "StrawGas"]
    # only material above the plane: shells whose box lies entirely below y_pl cannot be crossed
    tubes = [t for t in tubes if t[2][1] + (np.abs(t[1]) @ np.array([t[4], t[4], t[5]]))[1] > y_pl]
    rho = _gdml_densities(gdml)
    gcm2 = np.zeros(len(ev))
    mev = np.zeros(len(ev))
    per_mat = {}
    for c0 in range(0, len(ev), chunk):
        sl = slice(c0, c0 + chunk)
        clo = np.minimum(o[sl], end[sl]).min(0)
        chi = np.maximum(o[sl], end[sl]).max(0)
        sub = _cull(tubes, clo, chi)
        if not sub:
            continue
        col = world_mod.column_tubes(sub, o[sl], d[sl], L[sl], nsub=8)
        for name, c in col.items():
            r = _density(name, rho)
            gcm2[sl] += c * r / 10.0
            m = materials.BY_NAME.get(name, materials.MILD_STEEL)
            mev[sl] += eloss.dedx_restricted(m, p[sl], 1e9) * c * (r / m.rho)
            per_mat[name] = per_mat.get(name, 0.0) + float((c * r / 10.0).sum())
    ev["tube_gcm2"], ev["tube_bethe_mev"] = gcm2, mev
    # the part of it that belongs to the tracker interior model (TrackerInterior shells), which the fast transport
    # starts only at the plane: its shells reach up to r ~ 820 mm about the axis, ~100 mm above the plane
    interior = st.TrackerInterior(gdml, geom)
    hi_y = interior_top_y(interior, y_pl)
    oy = o + d * ((o[:, 1] - hi_y) / -d[:, 1])[:, None]                    # the line at y = interior top
    seg = (hi_y - y_pl) / -d[:, 1]
    icol = interior.shells.column(oy, d, seg, nsub=interior.nsub)
    ig, imev = np.zeros(len(ev)), np.zeros(len(ev))
    for name, c in icol.items():
        m = materials.get(name)
        ig += c * m.rho / 10.0
        imev += eloss.dedx_restricted(m, p, 1e9) * c
    ev["interior_above_gcm2"], ev["interior_above_mev"] = ig, imev
    ev["dloss"] = (ev.p_gen - ev.p_first_g4) - (ev.p_gen - ev.p_first)       # G4 minus fast
    out = {"events_both": int(len(ev)), "tubes_considered": len(tubes),
           "share_crossing_tubes": float(np.mean(gcm2 > 0.01)),
           "tube_gcm2_q": np.quantile(gcm2, [0.5, 0.9, 0.99]).round(3).tolist(), "tube_gcm2_mean": float(gcm2.mean()),
           "tube_bethe_mev_mean": float(mev.mean()),
           "per_material_gcm2_mean": {k: v / len(ev) for k, v in sorted(per_mat.items(), key=lambda kv: -kv[1])[:12]},
           "interior_above_plane": {"top_y": float(hi_y), "share": float(np.mean(ig > 0.01)),
                                    "gcm2_q": np.quantile(ig, [0.5, 0.9, 0.99]).round(3).tolist(), "gcm2_mean": float(ig.mean()),
                                    "bethe_mev_mean": float(imev.mean())},
           "by_tube_column": []}
    b = np.clip(np.digitize(gcm2, TUBE_EDGES) - 1, 0, len(TUBE_EDGES) - 2)
    for k in range(len(TUBE_EDGES) - 1):
        m = b == k
        if m.sum() < 10:
            continue
        x = ev[m]
        dl = x.dloss.to_numpy()
        out["by_tube_column"].append({"tube_gcm2": [float(TUBE_EDGES[k]), float(TUBE_EDGES[k + 1])], "n": int(m.sum()),
                                      "dloss_median": float(np.median(dl)),
                                      "dloss_median_err": float(1.2533 * dl.std() / np.sqrt(len(dl))),
                                      "tube_bethe_mev_median": float(np.median(x.tube_bethe_mev)),
                                      "world_gcm2_median": float(x.gcm2.median())})
    print(f"events passing in both: {out['events_both']:,}  tube/polycone shells above the plane in the lines' box: "
          f"{out['tubes_considered']:,}")
    print(f"share crossing omitted tubes {out['share_crossing_tubes']:.3f}  tube column q50/q90/q99 {out['tube_gcm2_q']} g/cm2"
          f"  mean {out['tube_gcm2_mean']:.2f} g/cm2, Bethe mean {out['tube_bethe_mev_mean']:.1f} MeV")
    print("  per material (mean g/cm2 over all events):", {k: round(v, 3) for k, v in out["per_material_gcm2_mean"].items()})
    ia = out["interior_above_plane"]
    print(f"of which tracker-interior shells between the plane (y {y_pl:.0f}) and their top (y {ia['top_y']:.0f}): share {ia['share']:.3f}"
          f"  q50/q90/q99 {ia['gcm2_q']} g/cm2  mean {ia['gcm2_mean']:.2f} g/cm2, Bethe mean {ia['bethe_mev_mean']:.1f} MeV")
    print("paired loss G4 minus fast by omitted tube column:")
    for r in out["by_tube_column"]:
        print(f"  {r['tube_gcm2'][0]:6.2f}-{min(r['tube_gcm2'][1], 1e4):7.2f} g/cm2  n {r['n']:5d}  dloss {r['dloss_median']:+7.1f}"
              f" +- {r['dloss_median_err']:.1f} MeV  tube Bethe {r['tube_bethe_mev_median']:6.1f} MeV  world column {r['world_gcm2_median']:.0f}")
    os.makedirs(outdir, exist_ok=True)
    ev[["run", "subrun", "event", "p_gen", "gcm2", "tube_gcm2", "tube_bethe_mev", "interior_above_gcm2",
        "interior_above_mev", "dloss"]].to_csv(
        os.path.join(outdir, "tubecheck_events.csv"), index=False)
    plots.write_json(outdir, "tubecheck.json", out)
