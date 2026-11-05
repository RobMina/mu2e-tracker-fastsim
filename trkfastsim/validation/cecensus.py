"""Reconcile a closed inclusive CE reference with independent generation/status/filter observations.

All generated events remain in the ledger. Nonzero transport statuses, overflows and killed tracks are recorded
as validity failures, never ordinary zero-response events. This does not certify physics closure or POT exposure.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import uproot

from trkfastsim.validation import cecontract, cestopwindow

KEY = cecontract.KEY
HEADER = re.compile(r"^\s*#+\s+PrintModule Run/Subrun/Event\s+(\d+)\s+(\d+)\s+(\d+)\s*$", re.M)
BOUNDARY = re.compile(r"^[^\r\n]*\bPrintModule Run/Subrun(?:/Event)?\b[^\r\n]*$", re.M)
PRODUCT = re.compile(r"^[ \t]*ProductPrint[ \t]+([^\r\n]+)$", re.M)
STATUS = re.compile(r"^[ \t]*status:[ \t]*(\d+)[ \t]+nG4Tracks:[ \t]*(\d+)[ \t]+"
                    r"overflowSimP:[ \t]*(\d+)[ \t]+nStepLimit:[ \t]*(\d+)"
                    r"(?:[ \t]+CPU:[^\r\n]+)?[ \t]*$", re.M)
PATHS = {"PrimaryPath": 0, "StockSelectionPath": 1}
REQUIRED_PRODUCTS = {
    "status": "mu2e::StatusG4_g4run__CaloLibraryCE.",
    "particles": "mu2e::SimParticlemv_g4run__CaloLibraryCE.",
    "generated": "mu2e::StageParticles_generate__CaloLibraryCE.",
    "stops": "mu2e::SimParticlemv_TargetStopResampler__CaloLibraryCE.",
    "generators": "mu2e::GenParticles_TargetStopResampler__CaloLibraryCE.",
    "source_ids": "art::EventIDs_TargetStopResampler__CaloLibraryCE.",
    "triggers": "art::TriggerResults_TriggerResults__CaloLibraryCE.",
    **{name: f"mu2e::StepPointMCs_g4run_{name}_CaloLibraryCE."
       for name in ("tracker", "calorimeter", "CRV", "virtualdetector", "stoppingtarget", "protonabsorber")}}

STATUS_COLUMNS = ("status", "nG4Tracks", "overflowSimP", "nStepLimit")


def positions(matches):
    return np.fromiter((m.start() for m in matches), dtype=np.int64, count=len(matches))


def read_log(text, expected, triggers=True):
    """Vectorized record ownership prevents status/selection rows crossing event or product boundaries."""
    headers = list(HEADER.finditer(text))
    if len(headers) != expected or text.count("PrintModule Run/Subrun/Event") != expected:
        raise ValueError("exact complete PrintModule event headers required")
    d = pd.DataFrame([m.groups() for m in headers], columns=KEY).astype(np.uint64)
    cecontract.unique(d, "printed CE census")
    native_max = np.iinfo(np.uint32).max
    if (d[KEY] >= native_max).any().any() or (d.run == 0).any() or (d.event == 0).any():
        raise ValueError("invalid native art identity bounds in printed census")
    bounds = positions(list(BOUNDARY.finditer(text)))
    owners = np.full(len(bounds), -1, dtype=np.int64)
    owners[np.searchsorted(bounds, positions(headers))] = np.arange(expected)

    def event_owner(matches):
        idx = np.searchsorted(bounds, positions(matches), side="right") - 1
        if (idx < 0).any() or (owners[idx] < 0).any():
            raise ValueError("record outside an event header")
        return owners[idx]

    products = list(PRODUCT.finditer(text))
    ppos = positions(products)
    tags = np.array([m.group(1).strip() for m in products])

    def product_rows(matches, kind, allowed):
        selected = np.flatnonzero(np.char.find(tags, kind) >= 0)
        if len(selected) != expected or not np.isin(tags[selected], allowed).all():
            raise ValueError(f"exact product/process required for {kind}")
        powners = event_owner([products[i] for i in selected])
        if not np.all(np.bincount(powners, minlength=expected) == 1):
            raise ValueError(f"one {kind} product required per event")
        indices = np.searchsorted(ppos, positions(matches), side="right") - 1
        if not np.isin(indices, selected).all():
            raise ValueError(f"{kind} row outside expected product")
        eowners = event_owner(matches)
        mapped = powners[np.searchsorted(selected, indices)]
        if not np.array_equal(mapped, eowners) or not np.all(np.bincount(eowners, minlength=expected) == 1):
            raise ValueError(f"one {kind} row required per event")
        return eowners

    status = list(STATUS.finditer(text))
    if len(status) != expected or len(re.findall(r"^[ \t]*status\b", text, re.M)) != expected:
        raise ValueError("exact complete status records required")
    order = product_rows(status, "StatusG4", ["mu2e::StatusG4_g4run__CaloLibraryCE"])
    values = np.asarray([m.groups() for m in status], dtype=np.uint64)
    if not np.isin(values[:, 2], [0, 1]).all():
        raise ValueError("invalid overflow flag")
    for i, name in enumerate(STATUS_COLUMNS):
        d[name] = np.zeros(expected, np.uint64)
        d.loc[order, name] = values[:, i]
    if triggers:
        for path, path_id in PATHS.items():
            pattern = re.compile(rf"^[ \t]*(pass|fail)[ \t]+(\d+)[ \t]+{path}[ \t]*$", re.M)
            rows = list(pattern.finditer(text))
            candidates = list(re.finditer(rf"^[ \t]*\S+[ \t]+\S+[ \t]+{path}\b[^\r\n]*$", text, re.M))
            if len(rows) != expected or not np.array_equal(positions(rows), positions(candidates)):
                raise ValueError(f"missing, malformed or unknown trigger state: {path}")
            if any(m.group(2) != str(path_id) for m in rows):
                raise ValueError(f"trigger path ID differs from declared mapping: {path}")
            order = product_rows(rows, "TriggerResults", ["art::TriggerResults_TriggerResults__CaloLibraryCE"])
            d[path] = False
            d.loc[order, path] = np.array([m.group(1) == "pass" for m in rows])
    return cecontract.unique(d, "closed CE census")


def reconcile(generation, saved, metadata, expected_keys):
    """Require exact independently observed status tuples and complete declared integer identities."""
    generation = cecontract.unique(generation, "generation")
    saved = cecontract.unique(saved, "saved read")
    metadata = cecontract.unique(metadata, "art FileIndex")
    expected_keys = cecontract.unique(expected_keys, "declared CE identities")
    for frame in (generation, saved, metadata):
        pd.testing.assert_frame_equal(frame[KEY], expected_keys[KEY], check_exact=True)
    pd.testing.assert_frame_equal(generation[KEY + list(STATUS_COLUMNS)], saved[KEY + list(STATUS_COLUMNS)],
                                  check_exact=True)
    if ((saved.status == 0) & ~saved.PrimaryPath).any() or (saved.StockSelectionPath & ~saved.PrimaryPath).any():
        raise ValueError("contradictory inclusive/stock path decisions")
    codes, counts = np.unique(saved.status, return_counts=True)
    valid = (saved.status == 0) & (saved.overflowSimP == 0) & (saved.nStepLimit == 0)
    return dict(events=len(saved), birth_keys_sha256=cecontract.fingerprint(saved),
                status_counts={str(int(k)): int(n) for k, n in zip(codes, counts)},
                status_zero_events=int((saved.status == 0).sum()), validity_failure_events=int((~valid).sum()),
                stock_selected_events=int(saved.StockSelectionPath.sum()),
                stock_rejected_events=int((~saved.StockSelectionPath).sum()),
                inclusive_path_pass_events=int(saved.PrimaryPath.sum()),
                physics_eligible=bool(valid.all()), generation_saved_metadata_exact=True,
                scope="closed event/status/filter census only; primary/source and physics audits remain separate",
                validity_verdict="not evaluated; identity and successful statuses are not physics closure")


def products(branches):
    """Require raw product branches even for events with empty per-detector collections."""
    missing = sorted(set(REQUIRED_PRODUCTS.values()) - set(branches))
    if missing:
        raise ValueError("missing exact typed raw reference product branches: " + ", ".join(missing))
    return dict(required_product_branches=REQUIRED_PRODUCTS, all_required_products_present=True,
                warning="branch presence does not certify every particle/persistent link or valid empty response")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--art", required=True)
    ap.add_argument("--generation-log", required=True)
    ap.add_argument("--saved-log", required=True)
    ap.add_argument("--expected-events", type=int, required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--subrun", type=int, required=True)
    ap.add_argument("--fileindex-byteswap", action="store_true")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    if a.expected_events <= 0 or a.run < 0 or a.subrun < 0:
        ap.error("positive event count and nonnegative run/subrun required")
    expected = pd.DataFrame(dict(run=np.full(a.expected_events, a.run, np.uint64),
                                 subrun=np.full(a.expected_events, a.subrun, np.uint64),
                                 event=np.arange(1, a.expected_events + 1, dtype=np.uint64)))
    gen = read_log(Path(a.generation_log).read_text(), a.expected_events, triggers=False)
    saved = read_log(Path(a.saved_log).read_text(), a.expected_events)
    meta = cestopwindow.load(a.art, a.expected_events, a.fileindex_byteswap)
    report = reconcile(gen, saved, meta, expected)
    with uproot.open(a.art) as root:
        report["raw_products"] = products(root["Events"].keys())
        report["event_cycles"] = {k: int(root[k].num_entries) for k in root.keys() if k.split(";")[0] == "Events"}
    report["input_sha256"] = {k: cestopwindow.checksum(p) for k, p in
                               (("art", a.art), ("generation_log", a.generation_log), ("saved_log", a.saved_log))}
    report["explicit_uproot_metadata_byteswap"] = a.fileindex_byteswap
    out = Path(a.outdir)
    out.mkdir(exist_ok=False)
    saved.to_csv(out / "events.csv.gz", index=False)
    (out / "cecensus.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
