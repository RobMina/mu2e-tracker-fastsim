"""rangecheck -- energy loss near range-out in thick slabs: Geant4 (art/run_slab.sh) against the interior loss model.

For every G4 slab file (mu- at normal incidence through one NIST box; tag t<g/cm2>_p<MeV/c>) the fast arm is
steps.slab_group, the code path TrackerInterior.transport takes for one support segment (transport.slab_loss: sliced
Urban sampling, CSDA range-out). Per point: transmission (muon leaves through the exit face) G4 vs fast, and the exit
kinetic energy quantiles q10/q50/q90 of transmitted muons; rows where either arm transmits fewer than 99.9 % first.
The fraction of the CSDA range the slab takes (mean-loss integration of transport.slab_loss with fluctuations off) is
reported as `range_frac`, so the points near range-out (0.6-1.1) are easy to pick out.

    rangecheck "slab/G4_Cu/slab.*.root" "slab/G4_Pb/slab.*.root" "slab/G4_Al/slab.*.root" --outdir DIR [--nmu 30000]
"""
import json
import os
import re

import numpy as np

from mu2efastsim import eloss, materials
from mu2efastsim import transport as tr
from .. import steps as st, tables as T

_TAG = re.compile(r"t(\d+(?:\.\d+)?)_p(\d+)")


def _ke(p):
    return np.sqrt(np.asarray(p, float) ** 2 + eloss.MMU ** 2) - eloss.MMU


def g4_exit(path):
    """G4 primary mu-: number thrown and |p| at the end of the muons that left through the exit face (+z)."""
    pa = T.read(path, "particles")
    m = pa[(pa.isprim == 1) & (pa.pdg.abs() == 13)]
    pe = m[["epx", "epy", "epz"]].to_numpy(float)
    return int(len(m)), np.linalg.norm(pe[pe[:, 2] > 0], axis=1)


def csda_frac(mat, length_mm, p):
    """Fraction of the mean (CSDA) range taken by length_mm: mean-loss integration in 1 mm steps (no fluctuations)."""
    tk = float(_ke(p))
    s = 0.0
    while tk > tr.TK_STOP and s < 1e6:
        pp = np.sqrt((tk + eloss.MMU) ** 2 - eloss.MMU ** 2)
        tk -= float(eloss.dedx_restricted(mat, np.array([pp]), 1e9)[0]) * 1.0
        s += 1.0
    return length_mm / max(s, 1e-9)


def run(inputs, outdir, nmu=30000, seed=1, **kw):
    rows = []
    for path in T.expand(inputs):
        mt = _TAG.search(os.path.basename(path))
        if not mt:
            continue
        name = os.path.basename(os.path.dirname(os.path.abspath(path)))
        mat = materials.get(name)
        tg, pm = float(mt.group(1)), int(mt.group(2))
        L = tg / mat.rho * 10.0
        ng, pg = g4_exit(path)
        rng = np.random.default_rng([int(seed), int(round(tg * 1000)), pm])
        ok, pk, _, _ = st.slab_group(rng, mat, L, np.full(int(nmu), float(pm)), tail=False)
        kg, kf = _ke(pg), _ke(pk[ok])
        q = lambda x: [round(float(v), 1) for v in np.quantile(x, (0.1, 0.5, 0.9))] if len(x) > 20 else None
        tg4, tfa = len(pg) / max(ng, 1), float(ok.mean())
        err = np.sqrt(tg4 * (1 - tg4) / max(ng, 1) + tfa * (1 - tfa) / nmu)
        rows.append({"material": name, "g_cm2": tg, "p_MeV": pm, "range_frac": round(csda_frac(mat, L, pm), 3),
                     "n_g4": ng, "trans_g4": round(tg4, 4), "trans_fast": round(tfa, 4),
                     "trans_diff": round(tfa - tg4, 4), "trans_diff_err": round(float(err), 4),
                     "ke_exit_q_g4": q(kg), "ke_exit_q_fast": q(kf),
                     "below_filter_mom_g4": round(float(np.mean(pg < T.FILTER_MIN_MOM)), 4) if len(pg) else None,
                     "below_filter_mom_fast": round(float(np.mean(pk[ok] < T.FILTER_MIN_MOM)), 4) if ok.any() else None})
    rows.sort(key=lambda r: (r["material"], r["range_frac"]))
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "rangecheck.json"), "w") as fh:
        json.dump({"rows": rows}, fh, indent=1)
    print(f"{'material':9s} {'g/cm2':>6s} {'p':>5s} {'R frac':>6s} | {'trans G4':>8s} {'fast':>6s} {'diff':>7s} {'+-':>6s} | "
          f"KE exit q10/q50/q90 G4 | fast")
    for r in rows:
        if r["trans_g4"] < 0.999 or r["trans_fast"] < 0.999 or 0.5 < r["range_frac"] < 1.2:
            print(f"{r['material']:9s} {r['g_cm2']:6.1f} {r['p_MeV']:5d} {r['range_frac']:6.3f} | {r['trans_g4']:8.4f} "
                  f"{r['trans_fast']:6.4f} {r['trans_diff']:7.4f} {r['trans_diff_err']:6.4f} | {r['ke_exit_q_g4']} | {r['ke_exit_q_fast']}")
