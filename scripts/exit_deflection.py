"""Truth deflection of the primary muon between its last straw and its exit crossing of TT_Outer (sid 4, lower half,
y < 0), fast tables vs G4 tables (selected events, as surfcheck): space angle between the last StrawGasStep's momentum
and the exit SurfaceStep's momentum, q50/q84/q99 [mrad]; and the same for the entry crossing (upper half) against the
first straw. The reco residual at the exit crossing is reco - truth, so a fast truth that does not scatter behind the
last straw shows up as a too-narrow residual there.

    python3 tracker/scripts/exit_deflection.py "FAST/fasttrkC*.root" "G4/sgsteps.*.root"
"""
import sys

import numpy as np

from trkfastsim import tables as T


def deflections(pat):
    paths = T.expand([pat])
    s = T.read_many(paths, "sgsteps")
    p = T.read_many(paths, "particles")
    if "ismcprim" in p:
        pm = p[(p.ismcprim == 1) & (p.pdg.abs() == 13)]
    else:
        pm = p[(p.key == 1) & (p.pdg.abs() == 13)]
    pm = pm.rename(columns={"key": "sp"})[T.KEY + ["sp"]]
    s = s.merge(pm, on=T.KEY + ["sp"])
    f = T.trk_filter(s)
    sel = f[f.pass_trk].reset_index()[T.KEY]
    s = s.merge(sel, on=T.KEY).sort_values(T.KEY + ["t"])
    first, last = s.groupby(T.KEY).first().reset_index(), s.groupby(T.KEY).last().reset_index()
    ss = T.read_many(paths, "surfsteps").merge(pm, on=T.KEY + ["sp"]).merge(sel, on=T.KEY)
    ss = ss[ss.surf == 4]
    ss["ym"] = 0.5 * (ss.y0 + ss.y1)
    out = {}
    for lab, part, ref in (("entry", ss[ss.ym > 0].sort_values("t").groupby(T.KEY).first().reset_index(), first),
                           ("exit", ss[ss.ym <= 0].sort_values("t").groupby(T.KEY).last().reset_index(), last)):
        m = part.merge(ref[T.KEY + ["px", "py", "pz"]], on=T.KEY, suffixes=("", "_s"))
        a = m[["px", "py", "pz"]].to_numpy(float)
        b = m[["px_s", "py_s", "pz_s"]].to_numpy(float)
        c = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
        out[lab] = 1e3 * np.arccos(np.clip(c, -1, 1))
        if lab == "exit":   # exit crossing's distance from the last straw's straight line (truth displacement)
            last_pos = m.merge(ref[T.KEY + ["x1", "y1", "z1"]], on=T.KEY, suffixes=("", "_e"))
            q = last_pos[["x1_e", "y1_e", "z1_e"]].to_numpy(float)
            x = 0.5 * (last_pos[["x0", "y0", "z0"]].to_numpy(float) + last_pos[["x1", "y1", "z1"]].to_numpy(float))
            ub = b / np.linalg.norm(b, axis=1)[:, None]
            w = x - q
            out["exit_offset"] = np.linalg.norm(w - (w * ub).sum(1)[:, None] * ub, axis=1)
    return out, len(sel)


F, nf = deflections(sys.argv[1])
G, ng = deflections(sys.argv[2])
print(f"selected events fast {nf}, G4 {ng}")
for lab in ("entry", "exit"):
    for arm, d in (("fast", F[lab]), ("G4", G[lab])):
        print(f"  {lab:5s} {arm:4s} n {len(d):7d}  deflection straw <-> TT_Outer q50/q84/q99 "
              f"{np.quantile(d, 0.5):6.2f} / {np.quantile(d, 0.84):6.2f} / {np.quantile(d, 0.99):7.2f} mrad")
print(f"  exit offset from the last straw's line q50/q84/q99 mm: fast " + " / ".join(f"{np.quantile(F['exit_offset'], q):.2f}" for q in (0.5, 0.84, 0.99))
      + "  G4 " + " / ".join(f"{np.quantile(G['exit_offset'], q):.2f}" for q in (0.5, 0.84, 0.99)))
for lab in ("entry", "exit"):
    print(f"  {lab:5s} share > 300 mrad: fast {np.mean(F[lab] > 300):.4f}, G4 {np.mean(G[lab] > 300):.4f}")
