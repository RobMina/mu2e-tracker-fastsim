"""Tracker virtual-detector crossings (SurfaceSteps, sids 0-4) of the fast primary muons, for EventNtuple trksegsmc.

Surfaces from the v04 GDML (VirtualDetector_TT_*), in the detector frame where the tracker axis is z (as the G4
StrawGasStepTableDumper surfsteps tree): planes TT_FrontHollow (sid 0), TT_Mid + TT_MidInner (sid 1), TT_Back (sid 2);
cylinders TT_InSurf (sid 3) and TT_OutSurf (sid 4). Each VD is 0.02 mm thick: a SurfaceStep runs across that thickness.

The muon path is the polyline through its own StrawGasStep entry/exit points (time order), extended straight before the
first and after the last straw along the first/last step direction. No random numbers: the steps are untouched.
|p| at a crossing is the nearest step's |p|; t is that step's time shifted by the path length at the muon's speed.
Only primary-muon (sp == 1) crossings: G4 also records secondaries (about 5 % of its surfsteps, census in surfcheck).
"""
import numpy as np
import pandas as pd

from mu2efastsim.tables import KEY

C_MM_PER_NS = 299.792458
MMU = 105.6583755
THICK = 0.02                      # VD thickness [mm]
Z_FRONT, Z_MID, Z_BACK = -1631.51, 10.10, 1639.51
R_PLANE_FB, R_PLANE_MID = 850.1, 810.0
R_IN, R_OUT = 376.89, 850.11      # cylinder mid-radii (VD rmin..rmax 376.88-376.9, 850.1-850.12)
Z_CYL = (-1631.5, 1639.5)         # TT_InSurf/OutSurf z extent (centre 4.0, half length 1635.5)
EXTEND = 4000.0                   # straight extension before the first / after the last straw [mm]
PLANES = ((0, Z_FRONT, R_PLANE_FB), (1, Z_MID, R_PLANE_MID), (2, Z_BACK, R_PLANE_FB))
CYLS = ((3, R_IN), (4, R_OUT))
COLS = KEY + ["surf", "sindex", "sp", "edep", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz", "t"]


def _polyline(st):
    """Muon steps (one primary per event, time-ordered) -> segments A->B with the step index whose state they carry."""
    n = len(st)
    p0, p1 = st[["x0", "y0", "z0"]].to_numpy(float), st[["x1", "y1", "z1"]].to_numpy(float)
    u = st[["px", "py", "pz"]].to_numpy(float)
    u = u / np.maximum(np.linalg.norm(u, axis=1), 1e-30)[:, None]
    ev = st["_ev"].to_numpy()
    first = np.r_[True, ev[1:] != ev[:-1]]
    last = np.r_[ev[1:] != ev[:-1], True]
    # per step: [prev exit (or extension start) -> entry], [entry -> exit]; per event: [last exit -> extension end]
    a_in = np.where(first[:, None], p0 - EXTEND * u, np.roll(p1, 1, axis=0))
    A = np.concatenate([a_in, p0, p1[last]])
    B = np.concatenate([p0, p1, p1[last] + EXTEND * u[last]])
    idx = np.concatenate([np.arange(n), np.arange(n), np.flatnonzero(last)])
    # the gap segment before step k carries step k's state; the tail segment the last step's
    return A, B, idx


def _crossings(A, B):
    """All plane and cylinder crossings of segments A->B: (segment index, fraction t in [0, 1), sid)."""
    D = B - A
    out = []
    for sid, z, rmax in PLANES:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(D[:, 2] != 0, (z - A[:, 2]) / D[:, 2], -1.0)
        ok = (t >= 0) & (t < 1)
        x = A[:, 0] + t * D[:, 0]
        y = A[:, 1] + t * D[:, 1]
        ok &= np.hypot(x, y) < rmax
        k = np.flatnonzero(ok)
        out.append((k, t[k], np.full(len(k), sid)))
    a = D[:, 0] ** 2 + D[:, 1] ** 2
    b = 2 * (A[:, 0] * D[:, 0] + A[:, 1] * D[:, 1])
    for sid, r in CYLS:
        c = A[:, 0] ** 2 + A[:, 1] ** 2 - r * r
        disc = b * b - 4 * a * c
        good = (a > 0) & (disc >= 0)
        sq = np.sqrt(np.where(good, disc, 0.0))
        for sgn in (-1.0, 1.0):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (-b + sgn * sq) / (2 * a)
            z = A[:, 2] + t * D[:, 2]
            ok = good & (t >= 0) & (t < 1) & (z >= Z_CYL[0]) & (z <= Z_CYL[1])
            k = np.flatnonzero(ok)
            out.append((k, t[k], np.full(len(k), sid)))
    k = np.concatenate([o[0] for o in out])
    return k, np.concatenate([o[1] for o in out]), np.concatenate([o[2] for o in out])


def surface_steps(sgsteps):
    """Fast StrawGasSteps -> surfsteps rows (columns as StrawGasStepTableDumper's surfsteps tree) for the primary muon."""
    st = sgsteps[sgsteps.sp == 1].sort_values(KEY + ["t"], kind="stable").reset_index(drop=True)
    if not len(st):
        return pd.DataFrame({c: np.zeros(0) for c in COLS})
    st["_ev"] = st.groupby(KEY, sort=False).ngroup()
    A, B, idx = _polyline(st)
    k, t, sid = _crossings(A, B)
    seg = B[k] - A[k]
    L = np.linalg.norm(seg, axis=1)
    u = seg / np.maximum(L, 1e-30)[:, None]
    c = A[k] + t[:, None] * seg
    js = idx[k]
    pv = st[["px", "py", "pz"]].to_numpy(float)[js]
    p = np.linalg.norm(pv, axis=1)
    # normal: z for planes, radial for cylinders; half-crossing length capped for grazing tracks
    nrm = np.where((sid <= 2)[:, None], np.array([0.0, 0.0, 1.0]),
                   np.c_[c[:, 0], c[:, 1], np.zeros(len(c))] / np.maximum(np.hypot(c[:, 0], c[:, 1]), 1e-30)[:, None])
    h = np.minimum(0.5 * THICK / np.maximum(np.abs((u * nrm).sum(1)), 1e-6), 5.0)
    x0, x1 = c - h[:, None] * u, c + h[:, None] * u
    # time: from the carrying step's entry point along the path, at the muon's speed (negative before the entry)
    ent = st[["x0", "y0", "z0"]].to_numpy(float)[js]
    ds = ((c - ent) * u).sum(1)
    beta = p / np.sqrt(p * p + MMU * MMU)
    tt = st["t"].to_numpy()[js] + ds / (beta * C_MM_PER_NS)
    out = pd.DataFrame({kk: st[kk].to_numpy()[js] for kk in KEY})
    out["surf"] = sid.astype(np.int32)
    out["sindex"] = np.zeros(len(js), np.int32)
    out["sp"] = np.ones(len(js), np.int32)
    out["edep"] = np.zeros(len(js), np.float32)
    for j, ax in enumerate("xyz"):
        out[f"{ax}0"] = x0[:, j].astype(np.float32)
        out[f"{ax}1"] = x1[:, j].astype(np.float32)
        out[f"p{ax}"] = (u[:, j] * p).astype(np.float32)
    out["t"] = tt
    return out.sort_values(KEY + ["t"], kind="stable").reset_index(drop=True)[COLS]
