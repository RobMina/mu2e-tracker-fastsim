"""refcheck -- content check of StrawGasStepTableDumper tables (G4 reference or fast sim), written as a stats JSON.

Per input set: tree sizes; StrawGasStep census (events, planes and panels hit, eIon and path length, step shape and
ionisation type, PDG); SimParticle linkage (every step's sp must exist in particles); the PrimaryFilter tracker term
(>= 12 good steps from one particle) per event; SurfaceStep sids; MCTrajectory points per particle; MC primaries per
event; live time. Batchable: --emit writes the counts of one batch, --merge sums emitted JSONs.
"""
import json

import numpy as np

from mu2efastsim import plots
from .. import tables as T

TREES = ["sgsteps", "surfsteps", "vdsteps", "trajpts", "particles", "gen", "subruns"]


def _counts(path):
    c = {"files": 1, "trees": {t: int(T.read(path, t, ["run"]).shape[0]) for t in TREES}}
    s = T.read(path, "sgsteps")
    p = T.read(path, "particles", T.KEY + ["key", "pdg", "ismcprim"])
    ss = T.read(path, "surfsteps", T.KEY + ["surf", "sp"])
    tp = T.read(path, "trajpts", T.KEY + ["sp"])
    m = s.merge(p.rename(columns={"key": "sp"}), on=T.KEY + ["sp"], how="left", indicator=True)
    f = T.trk_filter(s)
    c["sgs_events"] = int(s.groupby(T.KEY).ngroups)
    c["sgs_unlinked"] = int((m._merge != "both").sum())
    c["planes"] = {int(k): int(v) for k, v in s.plane.value_counts().items()}
    c["upanels_hit"] = sorted(int(u) for u in s.upanel.unique())
    c["shape"] = {int(k): int(v) for k, v in s["shape"].value_counts().items()}
    c["ion"] = {int(k): int(v) for k, v in s["ion"].value_counts().items()}
    c["pdg"] = {int(k): int(v) for k, v in m.pdg.value_counts().items()}
    c["sgs_from_mcprim"] = int(m.ismcprim.sum())
    c["eion_sum_MeV"] = float(s.eion.sum())
    c["len_sum_mm"] = float(s.len.sum())
    c["width_max_mm"] = float(s.width.max()) if len(s) else 0.0
    c["eion_hist"] = np.histogram(1e3 * s.eion, bins=60, range=(0, 6))[0].tolist()   # keV
    c["len_hist"] = np.histogram(s.len, bins=60, range=(0, 12))[0].tolist()          # mm
    c["events_pass_trk"] = int(f.pass_trk.sum())
    c["surf_sids"] = {int(k): int(v) for k, v in ss.surf.value_counts().items()}
    c["traj_particles"] = int(tp.groupby(T.KEY + ["sp"]).ngroups)
    c["traj_points"] = int(len(tp))
    c["mcprim_per_event"] = {int(k): int(v) for k, v in p.groupby(T.KEY).ismcprim.sum().value_counts().items()}
    c["livetime_s"] = T.livetime([path])
    return c


def _add(a, b):
    for k, v in b.items():
        if k == "upanels_hit":
            a[k] = sorted(set(a.get(k, [])) | set(v))
        elif isinstance(v, dict):
            a[k] = _add(a.get(k, {}), v)
        elif isinstance(v, list):
            a[k] = (np.asarray(a.get(k, np.zeros(len(v))), float) + np.asarray(v, float)).tolist()
        else:
            a[k] = a.get(k, 0) + v
    return a


def _summary(c):
    t = c["trees"]
    n = max(c["sgs_events"], 1)
    return {"files": c["files"], "trees": t, "sgs_events": c["sgs_events"],
            "steps_per_event": t["sgsteps"] / n, "sgs_unlinked": c["sgs_unlinked"],
            "planes_hit": len(c["planes"]), "panels_hit": len(c["upanels_hit"]),
            "eion_mean_keV": 1e3 * c["eion_sum_MeV"] / max(t["sgsteps"], 1),
            "len_mean_mm": c["len_sum_mm"] / max(t["sgsteps"], 1), "width_max_mm": c["width_max_mm"],
            "shape": c["shape"], "ion": c["ion"], "pdg_top": dict(sorted(c["pdg"].items(), key=lambda x: -x[1])[:6]),
            "sgs_from_mcprim_frac": c["sgs_from_mcprim"] / max(t["sgsteps"], 1),
            "events_pass_trk": c["events_pass_trk"], "pass_trk_frac_of_sgs_events": c["events_pass_trk"] / n,
            "surf_sids": dict(sorted(c["surf_sids"].items())), "traj_particles": c["traj_particles"],
            "traj_points_per_particle": c["traj_points"] / max(c["traj_particles"], 1),
            "mcprim_per_event": c["mcprim_per_event"], "livetime_s": c["livetime_s"]}


def run(inputs, outdir, emit=None, merge=False, **kw):
    if merge:
        c = {}
        for path in T.expand(inputs):
            c = _add(c, json.load(open(path)))
    else:
        c = {}
        for path in T.expand(inputs):
            c = _add(c, _counts(path))
    if emit:
        plots.write_json(*_split(emit), c)
        return
    s = _summary(c)
    for k, v in s.items():
        print(f"{k:28s} {v}")
    plots.write_json(outdir, "refcheck.json", {"summary": s, "counts": c})


def _split(path):
    import os
    return os.path.dirname(path) or ".", os.path.basename(path)
