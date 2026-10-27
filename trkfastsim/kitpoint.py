"""One fixed extracted, primary-only tracker point; no stock reconstruction inputs.

See docs/kit-task.md for the population, F2 limitations and output-retention contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import uproot

from mu2efastsim import tables
from . import digicond, geometry, steps
from .analyses import fastdigi
from .reco import hitreco, ntuple, tracktable
from .reco.cli import _drift_table

SCHEMA = "tracker-point-v1"
SAMPLE = "extracted_v04_primary_v1"


def file_identity(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"name": Path(path).name, "bytes": Path(path).stat().st_size, "sha256": h.hexdigest()}


def code_identity():
    import mu2efastsim
    result = {}
    for name, root in (("tracker", Path(__file__).parent), ("core", Path(mu2efastsim.__file__).parent)):
        result[name] = {str(p.relative_to(root)): file_identity(p)["sha256"]
                        for p in sorted(root.rglob("*.py"))}
    return result


def generated_keys(gen):
    """Retain first-occurrence event order, including non-muon and zero-step events."""
    if any(k not in gen for k in tables.KEY + ["pdg"]):
        raise ValueError("generator table lacks event keys or pdg")
    raw = gen[tables.KEY].to_numpy()
    if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError("generator keys must be finite integers")
    limits = np.array([2 ** 24, 2 ** 20, 2 ** 20], np.int64)
    if np.any(raw < 0) or np.any(raw >= limits):
        raise ValueError("generator keys exceed the digi table identity domain")
    keys = gen[tables.KEY].drop_duplicates().to_numpy(np.int64)
    if not len(keys):
        raise ValueError("empty generator population has no defined yield denominator")
    return keys


def validate_drift(drift):
    arrays = {k: np.asarray(drift[k], np.float64) for k in ("dt", "rdrift", "dvel", "sderr", "uderr")}
    if any(a.ndim != 1 or len(a) != len(arrays["dt"]) or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("drift calibration must contain finite equal-length vectors")
    if len(arrays["dt"]) < 2 or np.any(np.diff(arrays["dt"]) <= 0):
        raise ValueError("drift time grid must be strictly increasing")
    if any(np.any(arrays[k] < 0) for k in ("sderr", "uderr")):
        raise ValueError("drift errors must be nonnegative")
    return arrays


def simulate(gen, gdml, seed):
    geom = geometry.from_gdml(str(gdml))
    world = steps.load_world(str(gdml), geom)
    interior = steps.TrackerInterior(str(gdml), geom)
    return steps.make_steps(gen, geom, np.random.default_rng([seed, 0]), stage="C", world=world,
                            interior=interior, sliced_msc=True, all_muons=True, backend="numpy")


def write_step_table(path, sgs):
    """Include an empty typed sgsteps tree so zero-crossing inputs remain usable."""
    cols = {k: sgs[k].to_numpy() for k in fastdigi.STEP_COLS}
    with uproot.recreate(path) as f:
        f.mktree("dump/sgsteps", {k: v.dtype for k, v in cols.items()})
        if len(sgs):
            f["dump/sgsteps"].extend(cols)


def inspect_written(path, keys, expected_tracks):
    """Count the highest-cycle product and compare every persisted event identity."""
    import awkward as ak
    with uproot.open(path) as f:
        tree = f[ntuple.TREE]
        ev = tree["evtinfo"].array()
        got = np.column_stack([ak.to_numpy(ev[k]) for k in tables.KEY])
        if not np.array_equal(got, keys):
            raise ValueError("EventNtuple persisted event identities differ from the generator population")
        trk = tree["trk"].array()
        count = ak.to_numpy(ak.num(trk["trk.status"]))
        active = int(ak.sum(trk["trk.nactive"]))
        if int(count.sum()) != expected_tracks:
            raise ValueError("EventNtuple persisted track count differs from the F2 good-fit count")
        return {"written_events": int(tree.num_entries), "events_with_fit": int(np.count_nonzero(count)),
                "fit_successes": int(count.sum()), "active_hits": active}


def validate_metrics(metrics):
    from .kit_task import validate_metrics as validate
    return validate(metrics)


def run(gen, gdml, digi_cond, reco_cond, drift_source, seed, outdir, workdir=None, retain_ntuple=False):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    inputs = {"gen": gen, "gdml": gdml, "digi_cond": digi_cond, "reco_cond": reco_cond, "drift_source": drift_source}
    if any(not Path(p).is_file() for p in inputs.values()):
        raise ValueError("every tracker point input must be a readable regular local file")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("metrics.json", "provenance.json", "EventNtuple.root")):
        raise ValueError("refusing to overwrite a previous tracker point product")
    if workdir:
        Path(workdir).mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    identities = {k: file_identity(p) for k, p in inputs.items()}
    g = tables.read(gen, "gen")
    keys = generated_keys(g)
    dc = digicond.Conditions.load(digi_cond)
    rc = hitreco.load_conditions(reco_cond)
    drift = validate_drift(_drift_table(drift_source))
    times = {}
    with tempfile.TemporaryDirectory(prefix="tracker-point-", dir=workdir) as tmp:
        tmp = Path(tmp)
        t = time.perf_counter()
        sgs, particles = simulate(g, gdml, seed)
        skeys = sgs[tables.KEY].drop_duplicates()
        if not pd.MultiIndex.from_frame(skeys).isin(pd.MultiIndex.from_arrays(keys.T)).all():
            raise ValueError("step table contains events absent from the generator population")
        stepfile = tmp / "steps.root"
        write_step_table(stepfile, sgs)
        times["steps_s"] = time.perf_counter() - t
        t = time.perf_counter()
        stats = {}
        _, digis, _ = fastdigi.digitize_table(str(stepfile), dc, np.random.default_rng([seed, 0, 3]),
                                             keys=keys, chunk=1000, stats=stats)
        digifile = tmp / "digis.root"
        fastdigi.write_digis(str(digifile), keys, digis, dc.nADCSamples)
        times["digitization_s"] = time.perf_counter() - t
        t = time.perf_counter()
        tb = tracktable.from_f2(str(digifile), rc, settings="mc", niter=4, all_fits=True, drift=drift)
        times["reco_s"] = time.perf_counter() - t
        t = time.perf_counter()
        rootfile = out / "EventNtuple.root" if retain_ntuple else tmp / "EventNtuple.root"
        writer = ntuple.write(tb, str(rootfile))
        counts = inspect_written(rootfile, keys, tb["meta"]["n_good"])
        product = file_identity(rootfile)
        times["writer_s"] = time.perf_counter() - t
        if {k: file_identity(p) for k, p in inputs.items()} != identities:
            raise ValueError("tracker point inputs changed during consumption; refusing provenance and metrics")
        n = len(keys)
        fraction = counts["events_with_fit"] / n
        metrics = {"generated_events": n, "primary_muons": len(particles), "events_with_steps": len(skeys),
                   "straw_steps": len(sgs), "digis": len(digis["sid"]), "hits": tb["meta"]["n_hits"],
                   "seeds": tb["meta"]["n_seeds"], "fit_attempts": tb["meta"]["n_seeds"], **counts,
                   "fit_event_fraction": fraction, "fit_event_fraction_se": float(np.sqrt(fraction * (1 - fraction) / n)),
                   **times, "wall_s": time.perf_counter() - start}
        validate_metrics(metrics)
        provenance = {"schema": SCHEMA, "sample": SAMPLE, "seed": seed, "inputs": identities,
                      "code": code_identity(), "configuration": {"stage": "C", "sliced_msc": True,
                      "all_primary_muons": True, "secondary_library": None, "step_backend": "numpy",
                      "digitizer": "numpy", "reco": "hybrid F2", "fixed_momentum_MeV_c": 2000.0,
                      "fit_material": False, "niter": 4, "settings": "mc", "stock_conditioned": False},
                      "input_binding": "SHA256 checked before and after consumption; not immutable snapshots or protection against transient change-and-restore",
                      "denominator": "distinct run/subrun/event in the complete gen table; no PrimaryFilter selection",
                      "uncertainty": "binomial event-count standard error only; excludes transport, conditions and model discrepancy",
                      "timing": "cold in-process stages with I/O; wall includes input reads/hashing and persisted verification, not endpoint staging",
                      "digi_stats": stats, "writer": writer, "product": {**product, "retained": bool(retain_ntuple)},
                      "limitations": ["primary-only", "fixed extracted geometry", "F2 is not stock KKLine or F1",
                                      "no physics or BO certification", "writer RNTuple reader limitations apply"]}
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("gen", "gdml", "digi-cond", "reco-cond", "drift-source", "outdir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workdir")
    p.add_argument("--retain-ntuple", action="store_true", help="local use only: retain the internal ROOT product")
    return run(**vars(p.parse_args(argv)))


if __name__ == "__main__":
    main()
