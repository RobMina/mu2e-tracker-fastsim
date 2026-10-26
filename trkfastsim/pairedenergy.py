"""Experimental paired support-energy sensitivity on hash-pinned saved envelope paths.

No geometry re-navigation, loss formula, stochastic pairing, physics bound, detector
acceptance or local-deposition claim is supplied. Exact-touch equal media are merged
before invoking the existing ordered/core transport so probe subdivisions do not
restart its stepping schedule. Timing stays unknown; only primary KE is closed.
"""
import csv
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mu2efastsim import eloss, materials
from . import orderedtransport, transportledger as ledger
from .analyses.ledgercheck import sha256, source_manifest

MODE = "paired-support-envelope-mean-sensitivity-v1"
TRANSPORT_MODE = "ordered-support-deterministic-mean-v1"
KEY = ["run", "subrun", "event", "sp"]
COLS = ["lane", "s0_mm", "s1_mm", "material"]
PIN_SCHEMA = "trkfastsim.support-energy-input-pin.v1"
SCOPE = ("Deterministic mean primary-KE sensitivity between paired saved support geometries, not energy bounds, "
         "deposition closure, integrated time, G4 comparison or production transport.")


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _json(path):
    return json.loads(Path(path).read_text())


def _summary(summary, budget):
    if (summary.get("schema") != "trkfastsim.support-envelope-replay.v1" or summary.get("complete") is not True
            or summary.get("mode") != "support-navigation-torus-envelope-v1" or summary.get("admitted_for_transport") is not False):
        raise ValueError("completed non-production paired envelope replay required")
    candidates = [c for c in summary["candidates"] if c["endpoint_budget_mm"] == budget]
    if len(candidates) != 1 or candidates[0]["all_pairs_admitted"] is not True or candidates[0]["refused_rays"] != 0:
        raise ValueError("selected endpoint budget must admit all saved pairs; no intersecting subsets")
    if candidates[0]["bounded_pair_rays"] != summary["seed_rows"]:
        raise ValueError("pair count disagrees with saved seed count")
    return candidates[0]


def _input_files(budget):
    prefix = f"envelope_{budget:.17g}mm"
    return ["summary.json", "seeds.csv", f"{prefix}_inner_intervals.csv", f"{prefix}_outer_intervals.csv",
            f"{prefix}_decisions.csv", f"{prefix}_refusals.csv"]


def _verify_upstream(summary):
    for group in ("inputs", "control_inputs"):
        for item in summary[group].values():
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"upstream {group} hash mismatch: {item['path']}")
    if sha256(summary["gdml"]["path"]) != summary["gdml"]["sha256"]:
        raise ValueError("upstream GDML hash mismatch")


def pin_inputs(directory, pin_path, *, endpoint_budget_mm):
    """Explicit approval snapshot; its printed SHA must be supplied independently to run().

    This records outputs absent from the earlier replay's own input hashes. It
    cannot authenticate maliciously changed files before pinning; review the chosen
    saved artifact first. No existing pin file is replaced, and no transport runs.
    """
    directory, pin_path = Path(directory).resolve(), Path(pin_path)
    if pin_path.exists():
        raise FileExistsError(pin_path)
    if isinstance(endpoint_budget_mm, bool) or not np.isfinite(endpoint_budget_mm) or endpoint_budget_mm < 0:
        raise ValueError("finite nonnegative endpoint budget required")
    summary = _json(directory / "summary.json")
    _summary(summary, endpoint_budget_mm)
    _verify_upstream(summary)
    if sha256(directory / "seeds.csv") != summary["inputs"]["seeds.csv"]["sha256"]:
        raise ValueError("saved envelope seeds differ from original audit bytes")
    content = dict(schema=PIN_SCHEMA, directory=str(directory), endpoint_budget_mm=float(endpoint_budget_mm),
        files={name: sha256(directory / name) for name in _input_files(endpoint_budget_mm)},
        scope="approval snapshot of saved geometry output bytes; no transport admission or independent truth")
    pin_path.parent.mkdir(parents=True, exist_ok=True)
    with pin_path.open("x") as stream:
        stream.write(json.dumps(content, indent=2, allow_nan=False) + "\n")
    return dict(pin=str(pin_path.resolve()), sha256=sha256(pin_path), files=len(content["files"]))


def _read_csv(data, max_rows, integer_columns):
    header = next(csv.reader(io.StringIO(data.decode("utf-8-sig"))), [])
    if not header or len(set(header)) != len(header) or any(not name for name in header):
        raise ValueError("unique nonempty raw CSV headers required")
    table = pd.read_csv(io.BytesIO(data), nrows=max_rows+1, float_precision="round_trip",
                        dtype={name: "string" for name in integer_columns})
    if len(table) > max_rows:
        raise ValueError("saved table exceeds row bound; no truncation")
    if not table.columns.is_unique:
        raise ValueError("unique CSV columns required")
    for key in integer_columns:
        if key not in table:
            raise ValueError(f"missing integer identity column {key}")
        text = table[key]
        if not text.str.fullmatch(r"0|[1-9][0-9]*", na=False).all():
            raise ValueError(f"{key} must contain exact nonnegative decimal integers")
        ceiling = str(np.iinfo(np.int64).max)
        if ((text.str.len() > len(ceiling)) | ((text.str.len() == len(ceiling)) & (text > ceiling))).any():
            raise ValueError(f"{key} exceeds int64")
        table[key] = text.astype("int64")
    return table


def _identities(table, seeds):
    if not set(KEY + ["lane"]) <= set(table.columns):
        raise ValueError("full identity and lane columns required")
    identity = seeds[KEY].assign(lane=np.arange(len(seeds), dtype=np.int64))
    if ((table.lane < 0) | (table.lane >= len(seeds))).any():
        raise ValueError("invalid lane index")
    expected = identity[KEY].to_numpy()[table.lane.to_numpy(np.int64)]
    if not np.array_equal(table[KEY].to_numpy(), expected):
        raise ValueError("saved identity disagrees with seed lane")


def canonical_intervals(table, lengths):
    """Validate full coverage then merge exact-touch same-label rows, keeping source-row membership.

    The medium is a literal core key with the same fixed unrestricted cut/loss
    policy for all rows. No tolerance, alias, priority, reordering or gap fill.
    """
    if not set(COLS) <= set(table.columns):
        raise ValueError("interval schema mismatch")
    spans = table[COLS].copy()
    lane = spans.lane.to_numpy()
    lengths = np.asarray(lengths, float)
    if lengths.ndim != 1 or not np.isfinite(lengths).all() or (lengths < 0).any():
        raise ValueError("invalid requested lengths")
    if lane.dtype.kind not in "iu" or ((lane < 0) | (lane >= len(lengths))).any():
        raise ValueError("invalid interval lanes")
    start, end = spans.s0_mm.to_numpy(float), spans.s1_mm.to_numpy(float)
    if not np.isfinite(start).all() or not np.isfinite(end).all() or ((start < 0) | (end <= start) | (end > lengths[lane])).any():
        raise ValueError("finite positive in-domain intervals required")
    if not np.array_equal(np.lexsort((start, lane)), np.arange(len(spans))):
        raise ValueError("intervals must be physically ordered")
    if len(spans) and (pd.api.types.infer_dtype(spans.material, skipna=False) != "string" or not spans.material.str.len().gt(0).all()):
        raise ValueError("nonempty material keys required")
    if set(spans.material) - {"vacuum"} - set(materials.BY_NAME):
        raise ValueError("material missing from core registry; no aliases or fallback")
    if len(lane):
        first = np.r_[True, lane[1:] != lane[:-1]]
        last = np.r_[lane[1:] != lane[:-1], True]
        if (start[first] != 0.).any() or not np.array_equal(start[1:][~first[1:]], end[:-1][~first[1:]]):
            raise ValueError("gap or overlap in interval coverage")
        if not np.array_equal(end[last], lengths[lane[last]]):
            raise ValueError("truncated requested tail")
        if not np.array_equal(np.unique(lane), np.flatnonzero(lengths > 0)):
            raise ValueError("missing positive-length lane")
        begins = first | np.r_[True, spans.material.to_numpy()[1:] != spans.material.to_numpy()[:-1]]
        groups = np.cumsum(begins) - 1
        canon = spans.groupby(groups, sort=False).agg(lane=("lane", "first"), s0_mm=("s0_mm", "first"),
                                                     s1_mm=("s1_mm", "last"), material=("material", "first")).reset_index(drop=True)
    else:
        if (lengths > 0).any():
            raise ValueError("missing positive-length lane")
        canon, groups = spans.copy(), np.empty(0, np.int64)
    mapping = pd.DataFrame(dict(source_row=np.arange(len(spans)), canonical_row=groups, lane=lane))
    return canon.astype({"lane": "int64"}), mapping


def load_pinned(pin_path, expected_sha256, *, max_rows=512, max_interval_rows=1000000):
    """Hash before parsing; parse the same pinned bytes, with exact integer identities."""
    import hashlib
    _integer(max_rows, "max_rows")
    _integer(max_interval_rows, "max_interval_rows")
    raw_pin = Path(pin_path).read_bytes()
    if not ledger._hash(expected_sha256, 64) or hashlib.sha256(raw_pin).hexdigest() != expected_sha256:
        raise ValueError("input pin SHA256 mismatch")
    pin = json.loads(raw_pin)
    if pin.get("schema") != PIN_SCHEMA:
        raise ValueError("input pin schema mismatch")
    budget = pin["endpoint_budget_mm"]
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not np.isfinite(budget) or budget < 0:
        raise ValueError("invalid pinned endpoint budget")
    if set(pin["files"]) != set(_input_files(budget)):
        raise ValueError("exact paired input file set required")
    data = {}
    for name, digest in pin["files"].items():
        data[name] = (Path(pin["directory"]) / name).read_bytes()
        if hashlib.sha256(data[name]).hexdigest() != digest:
            raise ValueError(f"pinned input hash mismatch: {name}")
    summary = json.loads(data["summary.json"])
    chosen = _summary(summary, budget)
    _verify_upstream(summary)
    seeds = _read_csv(data["seeds.csv"], max_rows, KEY)
    if len(seeds) != summary["seed_rows"] or seeds.duplicated(KEY).any():
        raise ValueError("seed count/identity mismatch")
    if pin["files"]["seeds.csv"] != summary["inputs"]["seeds.csv"]["sha256"]:
        raise ValueError("seed bytes disagree with original audit")
    required = ["x", "y", "z", "px", "py", "pz", "length_mm"]
    if not set(required) <= set(seeds.columns) or not np.isfinite(seeds[required].to_numpy(float)).all():
        raise ValueError("finite saved seed states required")
    pv = seeds[["px", "py", "pz"]].to_numpy(float)
    p = np.linalg.norm(pv, axis=1)
    if not np.isfinite(p).all() or (p <= 0).any() or (seeds.length_mm < 0).any():
        raise ValueError("nonzero finite momenta and nonnegative lengths required")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        tmax = eloss.kinematics(p)[-1]
    if not np.isfinite(tmax).all() or (tmax > 1e9).any():
        raise ValueError("seed momentum outside effective unrestricted 1e9 MeV cut domain")
    prefix = f"envelope_{budget:.17g}mm"
    decisions = _read_csv(data[f"{prefix}_decisions.csv"], max_rows, KEY + ["lane"])
    refusals = _read_csv(data[f"{prefix}_refusals.csv"], max_interval_rows, KEY + ["lane"])
    _identities(decisions, seeds)
    if (len(decisions) != len(seeds) or not np.array_equal(decisions.lane.to_numpy(), np.arange(len(seeds)))
            or not decisions.bounded_pair_admitted.eq(True).all() or not decisions.admitted_for_transport.eq(False).all()
            or len(refusals)):
        raise ValueError("all original lanes need a non-production bounded pair and no refusal")
    if not np.array_equal(decisions.requested_length_mm.to_numpy(float), seeds.length_mm.to_numpy(float)):
        raise ValueError("decision length disagrees with seed")
    arms, maps = {}, {}
    for arm in ("inner", "outer"):
        table = _read_csv(data[f"{prefix}_{arm}_intervals.csv"], max_interval_rows, KEY + ["lane"])
        _identities(table, seeds)
        if len(table) != chosen[f"{arm}_interval_rows"]:
            raise ValueError("interval count disagrees with envelope summary")
        arms[arm], maps[arm] = canonical_intervals(table, seeds.length_mm.to_numpy(float))
    return dict(pin=pin, pin_sha256=expected_sha256, summary=summary, seeds=seeds, intervals=arms, mapping=maps,
                direction=pv / p[:, None], kinetic_MeV=p * (p / (np.hypot(p, eloss.MMU) + eloss.MMU)))


def _final(result, source, initial_ke, lengths):
    """Validate per-identity primary accounting against final states, without dropping stopped lanes."""
    history = result["history"]
    rows = pd.DataFrame({key: history[key] for key in KEY})
    rows = rows.assign(debit=history["applied_loss_MeV"], path=history["path_mm"],
                       initial=history["ke_in_MeV"], final=history["ke_out_MeV"])
    grouped = rows.groupby(KEY, as_index=False, sort=False).agg(debit=("debit", "sum"), path=("path", "sum"),
                                                              initial=("initial", "first"), final=("final", "last"))
    final = source[KEY].merge(grouped, on=KEY, how="left", validate="one_to_one")
    ke, traveled, status = result["final_ke_MeV"], result["traveled_mm"], result["status"]
    if len(grouped) != len(source) or not np.isfinite(final[["debit", "path", "initial", "final"]]).all().all():
        raise ValueError("history does not contain exactly every seed identity")
    if (not np.array_equal(final.initial.to_numpy(), initial_ke) or not np.array_equal(final.final.to_numpy(), ke)
            or not np.allclose(final.path, traveled, rtol=0, atol=1e-8)
            or not np.allclose(final.debit + ke, initial_ke, rtol=1e-12, atol=1e-9)):
        raise ValueError("history/final primary energy or path closure failed")
    if ((traveled < 0) | (traveled > lengths)).any() or ((status == "transmitted") & (traveled != lengths)).any():
        raise ValueError("invalid final range/transmission")
    final = source.copy(deep=True)
    final["initial_ke_MeV"], final["final_ke_MeV"], final["traveled_mm"], final["status"] = initial_ke, ke, traveled, status
    final[["final_x_mm", "final_y_mm", "final_z_mm"]] = result["position_mm"]
    final["primary_debit_MeV"] = initial_ke - ke
    return final


def transport_pair(bundle, *, tracker_ref, core_ref, max_step_mm=1., fractional_loss=.05, stop_ke_MeV=1., max_steps=10000):
    """Use deterministic core mean loss only; no RNG is created or passed to sampling."""
    source, pin = bundle["seeds"], bundle["pin"]
    names = sorted(set().union(*(set(t.material) for t in bundle["intervals"].values())) - {"vacuum"})
    media = [ledger.medium()] + [ledger.medium(materials.get(name), cut_MeV=1e9, loss_basis="unrestricted") for name in names]
    metadata = ledger.provenance(geometry_sha256=bundle["summary"]["gdml"]["sha256"], tracker_commit=tracker_ref,
        core_commit=core_ref, inputs=[dict(name=name, sha256=digest) for name, digest in pin["files"].items()], media=media,
        frame="Mu2e world Cartesian", settings=dict(scope=SCOPE, mode=MODE, input_pin_sha256=bundle["pin_sha256"],
        sources_sha256=source_manifest(), material_keys=names, material_mapping="literal core registry; no aliases",
        seed_domain="saved tracker-envelope SurfaceStep midpoint; no upstream history or generated denominator",
        geometry_settings=bundle["summary"]["settings"], geometry_source_hashes=bundle["summary"]["sources_sha256"],
        reference_counts_inherited=bundle["summary"]["seed_domain"], geometry_admitted_for_transport=False,
        rng="none; deterministic core mean proposal", coalescence="exact-touch same material/cut/loss identity"))
    results, finals = {}, {}
    for arm in ("inner", "outer"):
        meta = ledger._json_copy(metadata)
        meta["settings"]["geometry_envelope"] = arm
        results[arm] = orderedtransport.transport_intervals(None, source[KEY].to_numpy(np.int64),
            source[["x", "y", "z"]].to_numpy(float), bundle["direction"], bundle["kinetic_MeV"], bundle["intervals"][arm], meta,
            mode=TRANSPORT_MODE, timing="unknown", max_step_mm=max_step_mm, fractional_loss=fractional_loss,
            stop_ke_MeV=stop_ke_MeV, max_steps=max_steps)
        finals[arm] = _final(results[arm], source, bundle["kinetic_MeV"], source.length_mm.to_numpy(float))
    paired = source[KEY].copy()
    for arm in ("inner", "outer"):
        for key in ("final_ke_MeV", "traveled_mm", "status"):
            paired[f"{arm}_{key}"] = finals[arm][key].to_numpy()
    paired["outer_minus_inner_ke_MeV"] = paired.outer_final_ke_MeV - paired.inner_final_ke_MeV
    paired["outer_minus_inner_path_mm"] = paired.outer_traveled_mm - paired.inner_traveled_mm
    paired["status_changed"] = paired.outer_status != paired.inner_status
    return dict(results=results, finals=finals, paired=paired, metadata=metadata)
