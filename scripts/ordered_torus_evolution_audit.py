"""Saved-only conservation/clock/geometry-evidence audit of conditional torus sensitivity arms."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import io
import zipfile
import pickle

from scripts.transport_capture_audit import digest
from trkfastsim import ledger, steps
from trkfastsim.flighttime import linear_kinetic_dt_ns


class LabelUnpickler(pickle.Unpickler):
    """Allow only NumPy object-array reconstruction, not arbitrary pickle globals."""

    def find_class(self, module, name):
        allowed = {("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
                   ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        if (module, name) not in allowed:
            raise ValueError("unsupported global in legacy label array")
        return allowed[module, name]


def load_saved(path):
    """Read legacy object string labels through a restricted decoder; numeric arrays forbid pickle."""
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                raise ValueError("unexpected archive member")
            stream = io.BytesIO(archive.read(name))
            version = np.lib.format.read_magic(stream)
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, _, dtype = reader(stream)
            key = name[:-4]
            if dtype.hasobject:
                if key not in ("slice_kind", "slice_material") or len(shape) != 1:
                    raise ValueError("unexpected object array in saved evolution")
                value = LabelUnpickler(stream).load()
                if not isinstance(value, np.ndarray) or value.shape != shape or pd.api.types.infer_dtype(value, skipna=False) != "string":
                    raise ValueError("legacy label payload must contain only strings")
                arrays[key] = value.astype(str)
            else:
                stream.seek(0)
                arrays[key] = np.load(stream, allow_pickle=False)
    return arrays


def audit_slices(saved, momentum):
    names = ("start", "end", "ke0", "ke1", "t0", "t1", "pos0", "pos1", "dir")
    if not all(np.isfinite(saved["slice_" + n]).all() for n in names):
        raise ValueError("nonfinite conditional physical slices")
    start, end = saved["slice_start"], saved["slice_end"]
    ke0, ke1 = saved["slice_ke0"], saved["slice_ke1"]
    t0, t1 = saved["slice_t0"], saved["slice_t1"]
    if (saved["slice_it"] != 0).any() or not len(start) or start[0] != 0. or t0[0] != 0.:
        raise ValueError("single complete incoming-witness trajectory required")
    if (end <= start).any() or (ke1 > ke0).any() or (ke1 < .5).any() or (t1 <= t0).any():
        raise ValueError("conditional slice path/energy/time order failed")
    for left, right in ((end[:-1], start[1:]), (ke1[:-1], ke0[1:]), (t1[:-1], t0[1:])):
        if not np.array_equal(left, right):
            raise ValueError("conditional slice scalar continuity failed")
    if not np.allclose(saved["slice_pos1"][:-1], saved["slice_pos0"][1:], rtol=0, atol=1e-10):
        raise ValueError("conditional slice position discontinuity")
    direction = saved["slice_dir"]
    if not np.allclose(np.linalg.norm(direction, axis=1), 1., rtol=0, atol=1e-10):
        raise ValueError("conditional slice direction not unit")
    predicted = saved["slice_pos0"] + direction * (end - start)[:, None]
    position_error = float(np.max(np.abs(predicted - saved["slice_pos1"])))
    clock_error = float(np.max(np.abs(linear_kinetic_dt_ns(end - start, ke0, ke1) - (t1 - t0))))
    if position_error > 1e-8 or clock_error > 1e-9:
        raise ValueError("conditional slice endpoint or exact clock law failed")
    initial = float(ledger.kinetic(momentum)[0])
    balance = abs(initial - saved["final_ke"][0] - saved["final_lost_support"][0] - saved["final_lost_thin"][0])
    if abs(initial - ke0[0]) > 1e-10 or balance > 1e-8:
        raise ValueError("conditional primary balance failed")
    for last, final in ((end[-1], saved["final_path"][0]), (ke1[-1], saved["final_ke"][0]), (t1[-1], saved["final_dt"][0])):
        if last != final:
            raise ValueError("conditional terminal tail or stale state")
    if not np.array_equal(saved["slice_pos1"][-1], saved["final_o"][0]):
        raise ValueError("conditional terminal position mismatch")
    support = saved["slice_kind"] == "support"
    if abs((ke0[support] - ke1[support]).sum() - saved["final_lost_support"][0]) > 1e-9:
        raise ValueError("support debit not charged at traversed slices")
    gas = saved["slice_material"] == steps.STRAW_GAS.name
    deposit, unassigned = saved["slice_deposited"][gas], saved["slice_unassigned"][gas]
    if (deposit < 0).any() or (unassigned < 0).any() or not np.allclose(deposit + unassigned, ke0[gas] - ke1[gas], rtol=0, atol=1e-12):
        raise ValueError("conditional gas primary/deposit partition failed")
    if not np.isclose(saved["gas_eion"].sum(), deposit.sum(), rtol=0, atol=1e-12):
        raise ValueError("conditional gas row deposit disagrees with slices")
    stopped = saved["slice_stopped"]
    if stopped[:-1].any() or bool(stopped[-1]) != bool(saved["final_stopped"][0]):
        raise ValueError("conditional absorbing stop extended")
    totals = pd.DataFrame(dict(material=saved["slice_material"], kind=saved["slice_kind"], length_mm=end-start, debit_MeV=ke0-ke1))
    grouped = totals.groupby(["material", "kind"], as_index=False).sum()
    first = int(np.flatnonzero(gas)[0]) if gas.any() else None
    return dict(slices=len(start), position_error_mm=position_error, clock_error_ns=clock_error, primary_balance_MeV=float(balance),
        material_totals=grouped.to_dict("records"), first_gas_position_mm=saved["slice_pos0"][first].tolist() if first is not None else None,
        first_gas_direction=direction[first].tolist() if first is not None else None,
        first_gas_ke_MeV=float(ke0[first]) if first is not None else None,
        first_gas_path_mm=float(start[first]) if first is not None else None,
        first_gas_dt_ns=float(t0[first]) if first is not None else None)


def run(directory, out):
    directory, out = Path(directory), Path(out)
    paths = [directory / "complete.json", directory / "initial.npz"]
    receipt = json.loads(paths[0].read_text())
    result = dict(scope="saved conditional primary/clock and retained geometry evidence, not G4 physics or uncertainty bounds", arms={})
    with np.load(paths[1], allow_pickle=False) as saved:
        momentum = saved["momentum"]
    for tag, summary in receipt["summaries"].items():
        arm = directory / tag
        paths.append(arm / "receipt.json")
        if summary["status"] != "conditional_evolution_complete":
            result["arms"][tag] = dict(status=summary["status"], reason=summary["reason"])
            continue
        paths.append(arm / "evolved.npz")
        audited = audit_slices(load_saved(paths[-1]), momentum)
        calls = sorted(arm.glob("navigation_*/receipt.json"))
        if tag.endswith(("inner", "outer")):
            if len(calls) != summary["navigation_calls"] or not len(calls):
                raise ValueError("incomplete guarded navigation query coverage")
            for path in calls:
                call = json.loads(path.read_text())
                if call["status"] != "complete" or not call["replacements_strictly_below_plane"] or not call["envelope_summary"]["all_pairs_admitted"]:
                    raise ValueError("conditional evolution advanced refused geometry")
                for name in ("result.control.decisions.csv", "result.control.refusals.csv", "result.parent_comparisons.csv",
                             "result.ownership_comparisons.csv", "result.unknown_components.csv", "refined.inner.csv", "refined.outer.csv",
                             "endpoints.endpoints.csv", "raw_hits.csv", "cells.csv", "rays.npz", "placements.json"):
                    product = path.parent / name
                    if not product.is_file():
                        raise ValueError("guarded query evidence missing: " + str(product))
                    paths.append(product)
            paths += calls
        audited["navigation_calls"] = len(calls)
        result["arms"][tag] = audited
    contrasts = []
    for mode in ("mean", "sampled"):
        control = result["arms"][mode + "_stored_control"]
        for name in ("inner", "outer"):
            arm = result["arms"][mode + "_" + name]
            if arm.get("first_gas_ke_MeV") is not None and control.get("first_gas_ke_MeV") is not None:
                contrasts.append(dict(mode=mode, arm=name, scope="independent evolved sensitivity, no inferential uncertainty",
                    first_gas_delta_ke_MeV=arm["first_gas_ke_MeV"] - control["first_gas_ke_MeV"],
                    first_gas_delta_path_mm=arm["first_gas_path_mm"] - control["first_gas_path_mm"],
                    first_gas_delta_dt_ns=arm["first_gas_dt_ns"] - control["first_gas_dt_ns"],
                    first_gas_position_separation_mm=float(np.linalg.norm(np.array(arm["first_gas_position_mm"]) - control["first_gas_position_mm"]))))
    result.update(contrasts=contrasts, input_sha256={str(p): digest(p) for p in paths},
        source_sha256={str(Path(__file__)): digest(__file__)}, admitted_for_production=False)
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.directory, args.out)
