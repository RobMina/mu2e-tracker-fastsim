"""Material-free CE entry/hybrid geometric-acceptance diagnostics, not candidate physics predictions.

Propagate saved VD13 states to their first gas crossing with constant momentum and no scattering. Independent full
panel search checks the optimized chord-index first crossing on the same deterministic path. Hybrid position,
direction and momentum substitutions break correlations and only locate sensitivity, not corrections or efficiency.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from mu2efastsim import ehelix, tables
from trkfastsim import crossings, geometry
from trkfastsim.sim import fieldsteps as fs
from trkfastsim.validation import ceaudit

KEY = tables.KEY
XYZ = ["x", "y", "z"]
MOM = ["px", "py", "pz"]


def first_gas(start, geom, field, origin, chord_mm=10.0, max_path_mm=30000.0, max_chords=20000):
    """Vectorized lane stepping to first gas, with full-panel and optimized first-crossing agreement checks."""
    start = ceaudit.unique(start, "vacuum starts")
    env = fs.Envelope(geom, axis_xy=np.asarray(origin)[:2])
    optimized, full = fs.ChordIndex(geom), crossings.panel_index(geom)
    x = start[XYZ].to_numpy(float).copy()
    p = np.linalg.norm(start[MOM].to_numpy(float), axis=1)
    u = start[MOM].to_numpy(float) / p[:, None]
    if not (np.isfinite(x).all() and np.isfinite(u).all() and (p > 0).all()):
        raise ValueError("nonfinite or nonpositive vacuum start")
    n = len(start)
    hit = np.zeros(n, bool)
    status = np.full(n, 2, int)
    path = np.zeros(n)
    rmin = rmax = np.hypot(x[:, 0] - env.axis[0], x[:, 1] - env.axis[1])
    rmin, rmax = rmin.copy(), rmax.copy()
    act = np.flatnonzero(~env.leaving(x, u))
    status[np.setdiff1d(np.arange(n), act)] = 0
    tested_chords, disagreements = 0, 0
    for _ in range(max_chords):
        if not len(act):
            break
        xa, ua, pa = x[act], u[act], p[act]
        b0 = ehelix.field_at(field, xa)
        rho = pa / np.maximum(ehelix.K * np.linalg.norm(b0, axis=1), 1e-30)
        h = np.maximum(1e-3, np.minimum(np.minimum(chord_mm, np.sqrt(8 * rho * 0.02)), max_path_mm - path[act]))
        b = ehelix.field_at(field, ehelix.helix(xa, ua, pa, -1.0, b0, h / 2)[0])
        xn, un = ehelix.helix(xa, ua, pa, -1.0, b, h)
        cv = xn - xa
        length = np.linalg.norm(cv, axis=1)
        direction = cv / length[:, None]
        a = optimized.crossings(xa, direction, length, geom)
        b = crossings.straw_crossings(xa, direction, geom, index=full)
        if len(b):
            b = b[b[:, 2] < length[b[:, 0].astype(int)]]
            b[:, 3] = np.minimum(b[:, 3], length[b[:, 0].astype(int)])
            b = b[b[:, 3] > b[:, 2]]
        ha, hb = (np.unique(v[:, 0].astype(int)) for v in (a, b))
        tested_chords += len(act)
        disagreements += len(np.setxor1d(ha, hb))
        found = np.isin(np.arange(len(act)), hb)
        hit[act[found]] = True
        status[act[found]] = 3
        x[act], u[act] = xn, un
        path[act] += h
        r = np.hypot(xn[:, 0] - env.axis[0], xn[:, 1] - env.axis[1])
        rmin[act], rmax[act] = np.minimum(rmin[act], r), np.maximum(rmax[act], r)
        left = env.leaving(xn, un)
        status[act[left & ~found]] = 0
        capped = path[act] >= max_path_mm - 1e-9
        act = act[~found & ~left & ~capped]
    d = start[KEY].copy()
    d["vacuum_hit"], d["status"], d["path"], d["rmin"], d["rmax"] = hit, status, path, rmin, rmax
    for i, c in enumerate(XYZ):
        d[f"end_{c}"] = x[:, i]
    return d, {"tested_chords": tested_chords, "chord_index_full_search_hit_disagreements": disagreements,
               "note": "checks first-hit existence on identical vacuum chords, not the saved stochastic material path"}


def hybrids(fast, reference):
    """Fixed substitution menu on exactly common primary events; no proximity matching or response fitting."""
    fast, reference = (ceaudit.unique(d, name) for d, name in ((fast, "fast entries"), (reference, "G4 entries")))
    key = fast[KEY].merge(reference[KEY], on=KEY, validate="one_to_one")
    f, g = (key.merge(d, on=KEY, validate="one_to_one").sort_values(KEY).reset_index(drop=True) for d in (fast, reference))
    pf, pg = (np.linalg.norm(d[MOM].to_numpy(float), axis=1) for d in (f, g))
    uf, ug = f[MOM].to_numpy(float) / pf[:, None], g[MOM].to_numpy(float) / pg[:, None]

    def make(pos, direction, momentum):
        out = f.copy()
        out[XYZ] = pos[XYZ].to_numpy(float)
        out[MOM] = direction * momentum[:, None]
        return out

    return {"fast": f, "g4": g, "g4_xu_fast_p": make(g, ug, pf), "fast_xu_g4_p": make(f, uf, pg),
            "fast_x_g4_u_fast_p": make(f, ug, pf), "g4_x_fast_u_fast_p": make(g, uf, pf)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True, help="cerootcause saved events.csv.gz")
    ap.add_argument("--chain-dir", required=True)
    ap.add_argument("--gdml", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--origin", nargs=3, type=float, default=(-3904.0, 0.0, 10171.0))
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    events = ceaudit.unique(pd.read_csv(a.events), "root-cause events")
    for col in ("chain_has_vd13", "g4_track"):
        events[col] = events[col].fillna(False).astype(bool)
    select = events.chain_has_vd13 & (events.chain_steps == 0)
    keys = events.loc[select, KEY]
    fast = ceaudit.unique(tables.read(Path(a.chain_dir) / "states.root", "entries"), "fast entries")
    fast = keys.merge(fast, on=KEY, validate="one_to_one")
    reference = tables.read(Path(a.chain_dir) / "states.root", "g4entries")
    geom = geometry.from_gdml(a.gdml)
    field = ehelix.grid_field(*ehelix.read_g4bl(a.map))
    common = hybrids(fast, reference)
    result = {"original_events": len(events), "zero_step_fast_entries": len(fast),
              "common_g4_fast_zero_step_entries": len(common["fast"]),
              "scope": "truth-state material-free diagnostic on selected fast zero-step events, not a predicted yield",
              "arms": {}}
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for label, start in {"all_fast_zero_step": fast, **common}.items():
        tick = time.perf_counter()
        d, check = first_gas(start, geom, field, a.origin)
        d = d.merge(events[KEY + ["g4_track", "chain_endpoint", "chain_p", "entry_p"]], on=KEY, validate="one_to_one")
        result["arms"][label] = {"events": len(d), "vacuum_hits": int(d.vacuum_hit.sum()),
                                  "g4_good_tracks": int(d.g4_track.sum()),
                                  "vacuum_hits_with_g4_good_track": int((d.vacuum_hit & d.g4_track).sum()),
                                  "caps": int((d.status == 2).sum()), "wall_s": time.perf_counter() - tick, **check}
        d.to_csv(out / f"{label}.csv.gz", index=False)
    (out / "ceacceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
