"""Primary-muon StrawGasSteps per straw in G4 (and fast) tables: how often one muon has more than one step in the same
straw, and whether that changes the PrimaryFilter tracker-term decision (>= 12 good steps vs >= 12 distinct straws).

    python3 tracker/scripts/steps_per_straw.py "g4ref_v1_tables/sgsteps.*.root" [more tables ...]
"""
import sys

import numpy as np

from trkfastsim import tables as T

for pat in sys.argv[1:]:
    s = T.read_many([pat], "sgsteps")
    p = T.read_many([pat], "particles", T.KEY + ["key", "pdg"] + (["ismcprim"] if "ismcprim" in
                                                                  T.read_many([pat], "particles").columns else []))
    if "ismcprim" in p:
        pm = p[(p.ismcprim == 1) & (p.pdg.abs() == 13)]
    else:
        pm = p[(p.key == 1) & (p.pdg.abs() == 13)]
    s = s.merge(pm[T.KEY + ["key"]].rename(columns={"key": "sp"}), on=T.KEY + ["sp"])
    g = s[T.good_steps(s)]
    per = g.groupby(T.KEY + ["sp", "sid"]).size()
    nsteps = g.groupby(T.KEY + ["sp"]).size()
    nstraw = per.groupby(T.KEY + ["sp"]).size()
    print(f"{pat}: good primary-muon steps {len(g):,}; straws with > 1 step {np.mean(per > 1):.4f} "
          f"(max {per.max()}); muons with steps >= 12: {int((nsteps >= 12).sum()):,}, with straws >= 12: "
          f"{int((nstraw >= 12).sum()):,}; mean steps/muon {nsteps.mean():.3f}, straws/muon {nstraw.mean():.3f}")
