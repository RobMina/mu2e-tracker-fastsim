"""field-world -- cosmic muons from the stage-2 start states to the tracker and through it, fast against Geant4.

Reference: StrawGasStepTableDumper tables of a field-on cosmic stage-2 sample (CosmicSignal or CosmicCRYAll dts). Each
primary muon's stage-2 continuation starts on the stage-1 stopping surface (DS cryostat outer shell, TS-DS end) with
the Geant4 state there (particles: sx.., spx.., st). The fast arm transports it through the DS region in the field
(sim.fieldworld, phase 1) to the tracker mother surface, then through the tracker (sim.fieldsteps, phase 2).

Compared, paired per muon:
  * hand-off: whether each arm reaches the tracker, on which face (front, back, outer), and at the entry the momentum
    (fast - Geant4, and the loss from the start in each arm), position, direction and time, by start |p|;
  * tracker: the field-compare block (validation.fieldcompare.compare) on the Geant4 primary steps after its entry and
    the fast steps, for muons both arms bring into the tracker, all passages and first passage.
The Geant4 entry is the earliest tracker VD (13, 15, 22) or tracker SurfaceStep (0, 2, 4) crossing of the muon.
"""
import json
import os

import numpy as np
import pandas as pd

from trkfastsim import geometry, tables as T
from trkfastsim.sim import fieldsteps as fs, fieldworld as fw
from trkfastsim.validation import fieldcompare as fc

ENTRY_VDS = (13, 15, 22)
FACE_OF_ENTRY = {13: 0, -1: 0, 15: 2, -3: 2, 22: 4, -5: 4}      # VD id / -1 - SurfaceId -> SurfaceId of the face
START_P_EDGES = np.array([0.0, 300.0, 500.0, 1000.0, 2000.0, 5000.0, 20000.0, 1e9])


def stage2_starts(paths, part):
    """Start states of the stage-2 continuations of the primaries: KEY, sp, pdg, x, y, z, px, py, pz, t (Mu2e frame)."""
    cols = T.KEY + ["key", "pdg", "parent", "isprim", "ismcprim", "sx", "sy", "sz", "spx", "spy", "spz", "st"]
    p = T.read_many(paths, "particles", cols).rename(columns={"key": "sp"})
    prim = p[((p.isprim == 1) | (p.ismcprim == 1)) & (p.pdg.abs() == 13)][T.KEY + ["sp", "pdg"]]
    kids = p.merge(prim.rename(columns={"sp": "parent"}), on=T.KEY + ["parent", "pdg"])
    kids = kids.merge(part[T.KEY + ["sp"]], on=T.KEY + ["sp"])
    return kids.rename(columns={"sx": "x", "sy": "y", "sz": "z", "spx": "px", "spy": "py", "spz": "pz", "st": "t"})[
        T.KEY + ["sp", "pdg", "x", "y", "z", "px", "py", "pz", "t"]].reset_index(drop=True)


def g4_entry(vd, surf, org):
    cols = T.KEY + ["sp", "x", "y", "z", "px", "py", "pz", "t", "vdid"]
    ent = pd.concat([fc.entry_states(vd, ENTRY_VDS).reindex(columns=cols),
                     fc.entry_states_surface(surf, org).reindex(columns=cols)]).sort_values("t")
    ent = ent.groupby(T.KEY + ["sp"], sort=False).first().reset_index()
    ent["face"] = ent.vdid.map(FACE_OF_ENTRY).fillna(-1).astype(int)
    return ent


def _pmag(d, s=""):
    return np.sqrt(d[f"px{s}"] ** 2 + d[f"py{s}"] ** 2 + d[f"pz{s}"] ** 2).to_numpy()


def _q(x, qs=(0.1, 0.5, 0.9)):
    return [float(np.quantile(x, q)) for q in qs] if len(x) else []


def handoff_compare(start, ent, hand, final1):
    """Paired hand-off block (see the module docstring)."""
    key = T.KEY + ["sp"]
    s = start[key + ["px", "py", "pz", "t"]].assign(p0=lambda d: _pmag(d))[key + ["p0"]]
    st_ = final1[key + ["status"]].merge(s, on=key)
    g = ent[key].assign(g4=1)
    tab = st_.merge(g, on=key, how="left").fillna({"g4": 0})
    tab["fast"] = (tab.status == 3).astype(int)
    out = {"starts": int(len(tab)), "fast_status": tab.status.value_counts().sort_index().to_dict(),
           "reach": {"both": int(((tab.g4 == 1) & (tab.fast == 1)).sum()), "g4_only": int(((tab.g4 == 1) & (tab.fast == 0)).sum()),
                     "fast_only": int(((tab.g4 == 0) & (tab.fast == 1)).sum()), "neither": int(((tab.g4 == 0) & (tab.fast == 0)).sum())}}
    out["reach_by_start_p"] = []
    for lo, hi in zip(START_P_EDGES[:-1], START_P_EDGES[1:]):
        k = (tab.p0 >= lo) & (tab.p0 < hi)
        if k.sum() >= 5:
            g4r, fr = tab.g4[k].to_numpy(), tab.fast[k].to_numpy()
            out["reach_by_start_p"].append({"p_lo": float(lo), "p_hi": float(hi), "starts": int(k.sum()),
                                            "g4": float(g4r.mean()), "fast": float(fr.mean()),
                                            "fast_minus_g4": fc._mean_err(fr - g4r)})
    lost = tab[(tab.g4 == 1) & (tab.fast == 0)][key + ["p0", "status"]].merge(final1[key + ["x", "y", "z", "px", "py", "pz", "path"]], on=key)
    lr = np.hypot(lost.x - fw.DS_AXIS_XY[0], lost.y - fw.DS_AXIS_XY[1])
    out["g4_only_fast_end"] = {int(st): {"n": int(k.sum()), "start_p_q50": float(np.median(lost.p0[k])),
                                         "end_r_q10_q50_q90": _q(lr[k]), "end_z_q10_q50_q90": _q(lost.z[k]),
                                         "end_p_q10_q50_q90": _q(_pmag(lost[k])), "path_q50": float(np.median(lost.path[k]))}
                               for st in sorted(lost.status.unique()) for k in [(lost.status == st).to_numpy()]}
    m = ent.merge(hand, on=key, suffixes=("_g", "_f")).merge(s, on=key)
    pg, pf = _pmag(m, "_g"), _pmag(m, "_f")
    ff = fw.Handoff().face(m[["x_f", "y_f", "z_f"]].to_numpy())
    out["face_g4_by_fast"] = {int(a): {int(b): int(((m.face == a) & (ff == b)).sum()) for b in (0, 2, 4, -1)}
                              for a in (0, 2, 4, -1)}
    dx = np.sqrt((m.x_f - m.x_g) ** 2 + (m.y_f - m.y_g) ** 2 + (m.z_f - m.z_g) ** 2).to_numpy()
    cosang = np.clip((m.px_f * m.px_g + m.py_f * m.py_g + m.pz_f * m.pz_g).to_numpy() / np.maximum(pf * pg, 1e-9), -1, 1)
    ang = np.arccos(cosang)
    dt = (m.t_f - m.t_g).to_numpy()
    p0 = m.p0.to_numpy()
    same = (ff == m.face.to_numpy())
    rows = []
    for lo, hi in zip(START_P_EDGES[:-1], START_P_EDGES[1:]):
        k = (p0 >= lo) & (p0 < hi)
        if k.sum() < 5:
            continue
        rows.append({"p_lo": float(lo), "p_hi": float(hi), "n": int(k.sum()), "same_face": float(same[k].mean()),
                     "g4_loss_MeV": fc._mean_err((p0 - pg)[k]), "fast_loss_MeV": fc._mean_err((p0 - pf)[k]),
                     "dp_MeV": fc._mean_err((pf - pg)[k]), "dp_MeV_q10_q50_q90": _q((pf - pg)[k]),
                     "loss_q10_q50_q90_q99": {"g4": _q((p0 - pg)[k], (0.1, 0.5, 0.9, 0.99)),
                                              "fast": _q((p0 - pf)[k], (0.1, 0.5, 0.9, 0.99))},
                     "dpos_mm_q50_q90": _q(dx[k], (0.5, 0.9)), "dpos_x_pGeV_median": float(np.median(dx[k] * p0[k] / 1e3)),
                     "dangle_mrad_q50_q90": _q(1e3 * ang[k], (0.5, 0.9)),
                     "dt_ns": fc._mean_err(dt[k]), "dt_ns_q10_q50_q90": _q(dt[k])})
    out["at_entry_by_start_p"] = rows
    out["at_entry_by_face"] = {int(f): {"n": int(k.sum()), "dp_MeV": fc._mean_err((pf - pg)[k]),
                                        "g4_loss_MeV": fc._mean_err((p0 - pg)[k]), "fast_loss_MeV": fc._mean_err((p0 - pf)[k]),
                                        "dpos_mm_q50_q90": _q(dx[k], (0.5, 0.9))}
                               for f in (0, 2, 4) for k in [(m.face.to_numpy() == f) & same] if k.sum() >= 5}
    return out


def g4_crossing(traj, r_probe, axis=fw.DS_AXIS_XY):
    """First inward crossing of the radius r_probe by each track's Geant4 trajectory points (linear between points):
    KEY, sp, x, y, z, t, ke there."""
    key = T.KEY + ["sp"]
    t = traj.sort_values(key + ["ipt"])
    r = np.hypot(t.x.to_numpy() - axis[0], t.y.to_numpy() - axis[1])
    k = t[key].to_numpy()
    same = np.r_[(k[1:] == k[:-1]).all(1), False]
    cross = same & (r > r_probe) & (np.r_[r[1:], np.inf] <= r_probe)
    i = np.flatnonzero(cross)
    f = (r[i] - r_probe) / (r[i] - r[i + 1])
    cols = {c: t[c].to_numpy() for c in ("x", "y", "z", "t", "ke")}
    out = t.iloc[i][key].reset_index(drop=True)
    for c, v in cols.items():
        out[c] = v[i] + f * (v[i + 1] - v[i])
    return out.groupby(key, sort=False).first().reset_index()


def probe_compare(rng, start, traj, geom, field, world, r_probe, org, bnom, eopt, chord_mm, sagitta_mm):
    """Fast phase 1 stopped at the radius r_probe (closed cylinder over the DS) against the Geant4 trajectory there:
    reach, kinetic-energy difference and loss from the start, position offset, by start |p|."""
    from mu2efastsim import eloss
    key = T.KEY + ["sp"]
    stop = fw.Handoff(rmax=r_probe, z0=3000.0, z1=14200.0)
    _, fin = fs.transport(rng, start, geom, field, world, chord_mm=chord_mm, sagitta_mm=sagitta_mm,
                          envelope=fw.Region(), origin=org, bnom=bnom, stop=stop, straws=False, **eopt)
    g = g4_crossing(traj, r_probe)
    g = g[(g.z > 3000.0) & (g.z < 14200.0)]
    s = start[key + ["px", "py", "pz"]].assign(p0=lambda d: _pmag(d))[key + ["p0"]]
    m = g.merge(fin[fin.status == 3][key + ["x", "y", "z", "px", "py", "pz"]], on=key, suffixes=("_g", "_f")).merge(s, on=key)
    ke0 = np.sqrt(m.p0 ** 2 + eloss.MMU ** 2) - eloss.MMU
    kef = np.sqrt(_pmag(m, "") ** 2 + eloss.MMU ** 2) - eloss.MMU if "px" in m else None
    dx = np.sqrt((m.x_f - m.x_g) ** 2 + (m.y_f - m.y_g) ** 2 + (m.z_f - m.z_g) ** 2).to_numpy()
    n_g = s.merge(g[key], on=key).shape[0]
    out = {"r_probe": r_probe, "starts": int(len(s)), "g4_cross": int(n_g), "both": int(len(m)),
           "fast_status_of_g4_cross": fin.merge(g[key], on=key).status.value_counts().sort_index().to_dict(), "by_start_p": []}
    p0 = m.p0.to_numpy()
    for lo, hi in zip(START_P_EDGES[:-1], START_P_EDGES[1:]):
        k = (p0 >= lo) & (p0 < hi)
        if k.sum() < 5:
            continue
        out["by_start_p"].append({"p_lo": float(lo), "p_hi": float(hi), "n": int(k.sum()),
                                  "g4_dke_MeV": fc._mean_err((ke0 - m.ke)[k]), "fast_dke_MeV": fc._mean_err((ke0 - kef)[k]),
                                  "dke_q10_q50_q90": {"g4": _q((ke0 - m.ke)[k]), "fast": _q((ke0 - kef)[k])},
                                  "dpos_mm_q50_q90": _q(dx[k], (0.5, 0.9))})
    return out


def run(inputs, outdir, *, gdml, maps, seed=1, chord_mm=20.0, sagitta_mm=0.02, origin=None, max_events=None,
        mu_radiative=False, gas_state="solid", unbiased_only=False, self_pass_only=False, write_tables=False,
        probe_r=None, **kw):
    import time
    os.makedirs(outdir, exist_ok=True)
    geom = geometry.from_gdml(gdml)
    part, sgs, vd, surf, census = fc.primaries(inputs, 13)
    if max_events:
        ev = part[T.KEY].drop_duplicates().iloc[:max_events]
        part, sgs, vd, surf = (d.merge(ev, on=T.KEY) for d in (part, sgs, vd, surf))
    org = np.asarray(origin, float) if origin is not None else fc.fit_origin_z(sgs, geom)
    start = stage2_starts(inputs, part)
    ent = g4_entry(vd, surf, org)
    if unbiased_only:
        bad = fc.secondary_selected(inputs, part)
        n0 = len(start)
        start = start.merge(bad.assign(_b=1), on=T.KEY, how="left")
        start = start[start._b.isna()].drop(columns="_b")
        census["unbiased"] = {"starts": int(n0), "kept": int(len(start)), "events_secondary_selected": int(len(bad))}
    if self_pass_only:
        sp_ = fc.self_pass(sgs)
        start = start.merge(sp_[sp_.self][T.KEY + ["sp"]], on=T.KEY + ["sp"])
    field = fw.map_stack(maps)
    t0 = time.perf_counter()
    world = fw.WorldSupport(gdml)
    t_build = time.perf_counter() - t0
    rng = np.random.default_rng(seed)
    eopt = dict(radiative=mu_radiative, gas=fs.STRAW_GAS_G4STATE if gas_state == "gas" else None)
    bnom = field(org[None])[0]
    t0 = time.perf_counter()
    _, final1 = fs.transport(rng, start, geom, field, world, chord_mm=chord_mm, sagitta_mm=sagitta_mm,
                             envelope=fw.Region(), origin=org, bnom=bnom, stop=fw.Handoff(), straws=False, **eopt)
    t_world = time.perf_counter() - t0
    tm1 = dict(fs.transport.timing)
    hand = final1[final1.status == 3].drop(columns=["path", "chords", "status"]).merge(
        start[T.KEY + ["sp", "pdg"]], on=T.KEY + ["sp"])
    env = fs.Envelope(geom, axis_xy=org[:2])
    inter = fs.interior(gdml, geom, env, origin=org)
    t0 = time.perf_counter()
    fast, final2 = fs.transport(rng, hand, geom, field, inter, chord_mm=chord_mm, sagitta_mm=sagitta_mm,
                                envelope=env, origin=org, bnom=bnom, **eopt)
    t_trk = time.perf_counter() - t0
    res = {"inputs": len(inputs), "census": census, "origin_mm": org.tolist(), "bnom_T": bnom.tolist(), "seed": seed,
           "chord_mm": chord_mm, "sagitta_mm": sagitta_mm, "mu_radiative": bool(mu_radiative), "gas_state": gas_state,
           "starts": int(len(start)), "g4_entries": int(len(ent)), "g4_entry_vd_counts": ent.vdid.value_counts().sort_index().to_dict(),
           "start_radius_q01_q50_q99": _q(np.hypot(start.x - fw.DS_AXIS_XY[0], start.y - fw.DS_AXIS_XY[1]), (0.01, 0.5, 0.99)),
           "world": {"pieces": len(world.pieces), "materials": world.material_names,
                     "skipped": {f"{a}|{b}|{c}": n for (a, b, c), n in world.skipped.items()}, "build_s": t_build},
           "timing_s": {"world": t_world, "tracker": t_trk, "world_detail": tm1, "tracker_detail": dict(fs.transport.timing)},
           "handoff": handoff_compare(start, ent, hand, final1)}
    key = T.KEY + ["sp"]
    for r_p in (probe_r or []):
        traj = T.read_many(inputs, "trajpts").merge(start[key], on=key)
        res.setdefault("probe", []).append(probe_compare(np.random.default_rng(seed + 1000), start, traj, geom, field,
                                                         world, float(r_p), org, bnom, eopt, chord_mm, sagitta_mm))
    both = ent[key + ["t"]].merge(hand[key], on=key)
    g4 = sgs.merge(both.rename(columns={"t": "t_entry"}), on=key)
    g4 = g4[g4.t >= g4.t_entry - 1e-6].drop(columns="t_entry")
    entb = ent.merge(both[key], on=key)
    res["tracker"] = fc.compare(g4, fast, entb, final2, geom)
    out_t = final2[final2.status == 0][key + ["t"]].rename(columns={"t": "t_exit"})
    g1 = g4.merge(out_t, on=key)
    g1 = g1[g1.t <= g1.t_exit + fc.FIRST_PASS_NS].drop(columns="t_exit")
    f1 = fast.merge(out_t[key], on=key)
    res["tracker_first_pass"] = fc.compare(g1, f1, entb.merge(out_t[key], on=key), final2, geom)
    res = json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    with open(os.path.join(outdir, "fieldworld.json"), "w") as f:
        json.dump(res, f, indent=1)
    if write_tables:
        write_injector_table(os.path.join(outdir, "fastworld_field.root"), start, fast)
    return res


def write_injector_table(path, start, fast):
    """Primary-only step table for the art injector (StrawGasStepsFromTable): keys renumbered 1..n per event, as
    field-compare --write-tables."""
    from mu2efastsim import tables as core_tables
    rk = start[T.KEY + ["sp", "pdg"]].sort_values(T.KEY + ["sp"]).copy()
    rk["key"] = rk.groupby(T.KEY).cumcount() + 1
    parts = rk[T.KEY + ["key", "pdg"]].assign(parent=-1, isprim=1, ismcprim=1)
    cols = T.KEY + ["nsgs", "sid", "plane", "panel", "straw", "upanel", "shape", "ion", "sp", "eion", "len",
                    "width", "t", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"]
    df = fast.merge(rk[T.KEY + ["sp", "key"]], on=T.KEY + ["sp"]).drop(columns="sp").rename(columns={"key": "sp"})
    df = df.assign(nsgs=df.groupby(T.KEY).sid.transform("size"))
    core_tables.write(path, {"sgsteps": df[cols].reset_index(drop=True), "particles": parts}, as_ttree=True)
