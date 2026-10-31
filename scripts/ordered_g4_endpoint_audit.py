"""Audit saved truth-conditioned replay endpoints against identity-matched G4 particle ends."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mu2efastsim import eloss
from scripts.transport_capture_audit import digest
from trkfastsim import geometry, tables

KEY = tables.KEY + ["sp"]


def compare(initial, final, reference):
    if initial.duplicated(KEY).any() or reference.duplicated(KEY).any():
        raise ValueError("duplicate endpoint identity")
    f = initial[KEY].copy()
    if len(f) != len(final["ke"]):
        raise ValueError("final state/identity coverage mismatch")
    for name in ("ke", "stopped", "dt", "path"):
        f[f"fast_{name}"] = final[name]
    for index, axis in enumerate("xyz"):
        f[f"fast_{axis}"] = final["o"][:, index]
    j = f.merge(reference, on=KEY, how="left", validate="one_to_one", indicator=True)
    if not (j._merge == "both").all():
        raise ValueError("missing reference particle endpoint")
    j["g4_final_ke_MeV"] = j.ee - eloss.MMU
    j["g4_below_1MeV"] = j.g4_final_ke_MeV < 1.
    fast_xyz = j[[f"fast_{axis}" for axis in "xyz"]].to_numpy(float)
    g4_xyz = j[["ex", "ey", "ez"]].to_numpy(float)
    j["endpoint_distance_mm"] = np.linalg.norm(fast_xyz - g4_xyz, axis=1)
    j["fast_time_ns"] = initial.t.to_numpy() + final["dt"]
    j["terminal_time_fast_minus_g4_ns"] = j.fast_time_ns - j.et
    j["g4_chord_from_initial_mm"] = np.linalg.norm(
        geometry.to_mu2e(initial[["x0", "y0", "z0"]].to_numpy(float)) - g4_xyz, axis=1)
    stopped = j[j.fast_stopped].drop(columns="_merge")
    result = dict(lanes=len(j), fast_stopped=int(j.fast_stopped.sum()), g4_below_1MeV=int(j.g4_below_1MeV.sum()),
        both_stopped=int((j.fast_stopped & j.g4_below_1MeV).sum()),
        fast_stop_g4_not_stopped=int((j.fast_stopped & ~j.g4_below_1MeV).sum()),
        finite_path_but_g4_stopped=int((~j.fast_stopped & j.g4_below_1MeV).sum()),
        stopped_rows=stopped.astype(object).where(pd.notna(stopped), None).to_dict("records"),
        all_rows=j.drop(columns="_merge").astype(object).where(pd.notna(j.drop(columns="_merge")), None).to_dict("records"),
        interpretation="conditional first-straw mean-momentum starts; independent scattering/loss and unequal cutoff; chord is not G4 arc")
    return result, j.drop(columns="_merge")


def run(directory, g4, outdir):
    directory, g4, outdir = Path(directory), Path(g4), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    paths = [directory / "complete.json", directory / "initial_g4.csv", directory / "traversed.npz", g4]
    before = {str(path): digest(path) for path in paths}
    receipt = json.loads(paths[0].read_text())
    if receipt["status"] != "bounded_g4_entry_complete":
        raise ValueError("completed G4 entry replay required")
    initial = pd.read_csv(paths[1])
    particles = tables.read(g4, "particles")
    reference = particles[(particles.ismcprim == 1) & (particles.pdg.abs() == 13)].rename(columns={"key": "sp"})
    with np.load(paths[2], allow_pickle=False) as saved:
        final = {name[6:]: saved[name] for name in saved.files if name.startswith("final_")}
    result, rows = compare(initial, final, reference)
    result.update(input_sha256=before, scope="saved conditional terminal diagnostic; not stopping/range certification")
    if before != {str(path): digest(path) for path in paths}:
        raise ValueError("saved replay/reference changed")
    rows.to_csv(outdir / "endpoints.csv", index=False)
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--g4", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run(args.directory, args.g4, args.outdir)
