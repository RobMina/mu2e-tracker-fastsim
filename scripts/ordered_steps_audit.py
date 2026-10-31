"""Audit completed ordered full-chain products without rerunning transport."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest
from trkfastsim import ledger, tables
from trkfastsim.flighttime import linear_kinetic_dt_ns

KEY = ["run", "subrun", "event", "sp"]


def audit(mu, segments, vertices):
    if mu.duplicated(KEY).any():
        raise ValueError("duplicate full muon identity")
    active = mu[mu.path_interior_mm.notna()].copy()
    balance = (ledger.kinetic(active.p_plane_MeV_c.to_numpy()) - active.ke_final_MeV.to_numpy()
               - active.lost_support_MeV.to_numpy() - active.lost_thin_MeV.to_numpy())
    if not np.isfinite(balance).all() or np.max(np.abs(balance), initial=0.) > 1e-8:
        raise ValueError("primary terminal balance failed")
    if segments.duplicated(KEY + ["sid"]).any():
        raise ValueError("primary integration slices inflated step multiplicity")
    if len(segments):
        gas = segments.deposited_gas_MeV.to_numpy() + segments.unassigned_gas_MeV.to_numpy()
        if not np.allclose(gas, segments.gas_primary_debit_MeV, rtol=0, atol=1e-12):
            raise ValueError("gas debit partition failed")
        if not segments.escaping_gas_MeV.isna().all():
            raise ValueError("unassigned energy mislabeled escaping")
    if vertices.empty:
        if len(segments):
            raise ValueError("emitted steps lack path vertices")
        return dict(active_muons=len(active), entries=0, max_primary_balance_abs_MeV=0.)
    v = vertices.sort_values(KEY + ["path"], kind="stable").reset_index(drop=True)
    span = v.groupby(KEY, sort=False).path.diff().fillna(0).to_numpy()
    delta_t = v.groupby(KEY, sort=False).time_ns.diff().fillna(0).to_numpy()
    if (span < 0).any() or (delta_t < 0).any():
        raise ValueError("path or clock regresses")
    previous_ke = v.groupby(KEY, sort=False).kinetic_MeV.shift().fillna(v.kinetic_MeV).to_numpy()
    dt = linear_kinetic_dt_ns(span, previous_ke, v.kinetic_MeV.to_numpy())
    if not np.allclose(dt, delta_t, rtol=0, atol=1e-9):
        raise ValueError("vertex clock differs from physical slice integral")
    endpoints = v.groupby(KEY, sort=False).tail(1)[KEY + ["path", "kinetic_MeV"]]
    terminal = active.merge(endpoints, on=KEY, how="left", validate="one_to_one")
    if (terminal.path.isna() & (terminal.path_interior_mm > 0)).any():
        raise ValueError("terminal state lacks endpoint")
    if not np.allclose(terminal.path.fillna(0), terminal.path_interior_mm, rtol=0, atol=1e-9):
        raise ValueError("untraversed terminal extension")
    emitted = segments.merge(v[KEY + ["path", "time_ns"]], left_on=KEY + ["path_mm"],
                             right_on=KEY + ["path"], how="left", validate="one_to_one")
    if emitted.time_ns.isna().any() or not np.allclose(emitted.time_ns, emitted.t_integrated_ns, rtol=0, atol=1e-9):
        raise ValueError("step clock differs from recorded physical state")
    return dict(active_muons=len(active), entries=len(segments), vertices=len(v),
        max_primary_balance_abs_MeV=float(np.max(np.abs(balance), initial=0.)),
        max_vertex_clock_error_ns=float(np.max(np.abs(dt - delta_t), initial=0.)),
        stopped=int((active.stopped_interior == 1).sum()),
        gas_deposited_MeV=float(segments.deposited_gas_MeV.sum()),
        gas_unassigned_MeV=float(segments.unassigned_gas_MeV.sum()))


def run(directory, out):
    directory = Path(directory)
    paths = [directory / name for name in ("complete.json", "muons.csv", "segments.csv", "vertices.csv", "ordered_steps.root")]
    before = {str(p): digest(p) for p in paths}
    complete = json.loads(paths[0].read_text())
    if complete["status"] != "bounded_full_chain_complete":
        raise ValueError("full chain did not complete")
    frames = [pd.read_csv(p) for p in paths[1:4]]
    result = audit(*frames)
    steps = tables.read(paths[4], "sgsteps")
    if len(steps) != result["entries"] or not np.isclose(steps.eion.sum(), result["gas_deposited_MeV"], atol=1e-12):
        raise ValueError("ROOT steps differ from ledger")
    result.update(scope="stored-policy accounting/clock audit; not G4 physics closure", input_sha256=before)
    if before != {str(p): digest(p) for p in paths}:
        raise ValueError("product changed during audit")
    with Path(out).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.directory, args.out)
