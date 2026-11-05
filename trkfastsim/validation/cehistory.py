"""Exact finished CE stage history on the full supplied birth denominator.

Identity/filtering evidence only: an entry miss or zero-step entry was not sent to
art by the historical >=1 tracker-step filter. Reco output is selected output,
not proof that every birth was processed. No physics closure or zero-event
validity certification is made. Inputs are read-only; output must be fresh.
"""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from trkfastsim.validation import cecontract, cestopwindow

KEY = cecontract.KEY
TREE = "EventNtuple/ntuple"
FLAGS = ("hasentry", "hassteps", "injected", "digitized", "reco_selected", "ntuple")
SCOPE = ("Exact historical event identities and filtering evidence on the supplied full birth denominator only; "
         "not evidence that all births were physically processed, not physics closure, "
         "not production/POT normalization, and no zero-event validity certification.")


def _equal(left, right, label):
    if cecontract.fingerprint(left) != cecontract.fingerprint(right):
        raise ValueError(f"event identity mismatch: {label}")


def reconcile(births, particles, steps, injected, digitized, reco, ntuple):
    """Return one uint64-keyed row per birth; reject inconsistent stage sets.

    Particles must contain exactly one key-1 injector primary per entry event.
    Multiple gas steps per event are normal, not duplicate event records. Optional
    step ``sp`` must reference primary 1. Optional ntuple ``ntrack`` counts good
    tracks (status > 0), remaining unknown outside the persisted ntuple.
    """
    den = cecontract.unique(births[KEY], "births")
    p = cecontract.subset(particles, den, "particles")
    if ("key" not in p or not pd.api.types.is_integer_dtype(p["key"].dtype)
            or not (p["key"] == 1).all()):
        raise ValueError("particles require one integer key-1 primary per entry event")
    s = cecontract.integer_keys(steps, "steps")
    if "sp" in s and (not pd.api.types.is_integer_dtype(s.sp.dtype) or not (s.sp == 1).all()):
        raise ValueError("foreign steps: particle reference must be primary 1")
    step_ids = cecontract.subset(s[KEY].drop_duplicates(), p, "steps (no foreign steps allowed)")
    inj = cecontract.subset(injected, den, "injected")
    dig = cecontract.subset(digitized, den, "digitized")
    rec = cecontract.subset(reco, dig, "reco (must be subset of digitized)")
    nts = cecontract.subset(ntuple, den, "ntuple")
    _equal(inj, step_ids, "injected != step-bearing table events")
    _equal(dig, inj, "digitized != injected")
    _equal(nts, rec, "ntuple != reco selected output")
    d = den.copy()
    for flag, frame in zip(FLAGS, (p, step_ids, inj, dig, rec, nts)):
        # All keys have already been normalized to uint64; never float-promote IDs.
        membership = pd.MultiIndex.from_frame(frame[KEY])
        d[flag] = pd.MultiIndex.from_frame(d[KEY]).isin(membership)
    d["expected_skipped_noentry"] = ~d.hasentry
    d["expected_skipped_zero_steps"] = d.hasentry & ~d.hassteps
    d["expected_skipped_reco_selection"] = d.digitized & ~d.reco_selected
    if "ntrack" in nts:
        n = nts.ntrack
        if (not pd.api.types.is_integer_dtype(n.dtype) or n.isna().any() or (n < 0).any()):
            raise ValueError("ntuple ntrack must be nonnegative integers")
        # Nullable integer before left join avoids a float round trip.
        counts = nts[KEY].assign(ntrack=n.astype("UInt64"))
        d = d.merge(counts, on=KEY, how="left", validate="one_to_one")
    return d


def summarize(d):
    """Disjoint count partitions, stage digests, and explicit interpretation."""
    partitions = dict(no_entry=int((~d.hasentry).sum()),
                      entry_without_steps=int((d.hasentry & ~d.hassteps).sum()),
                      digitized_not_reco_selected=int((d.digitized & ~d.reco_selected).sum()),
                      reco_selected_and_ntuplized=int(d.ntuple.sum()))
    return dict(births=len(d), stages={flag: int(d[flag].sum()) for flag in FLAGS},
                count_partitions=partitions,
                partition_total=sum(partitions.values()),
                expected_skipped={c: int(d[c].sum()) for c in d if c.startswith("expected_skipped_")},
                identity_sha256={"births": cecontract.fingerprint(d),
                                 **{flag: cecontract.fingerprint(d.loc[d[flag], KEY]) for flag in FLAGS}},
                historical_filter="injection output equals unique sgsteps event keys (>=1 tracker step); "
                                  "particles also retain entry events with zero steps",
                reco_interpretation="reco art is selected output, not a processing receipt for omitted births",
                ntrack_definition="number of EventNtuple tracks with status > 0; unknown outside ntuple",
                scope=SCOPE)


def _tree_info(root, name):
    return dict(cycle=int(root.key(name).fCycle), entries=int(root[name].num_entries))


def _empty(columns):
    return pd.DataFrame({c: pd.Series(dtype=np.uint64) for c in columns})


def load_table(path, names):
    """Read highest cycles; fieldce's tables.write omits empty table trees.

    Missing particles/sgsteps trees are explicitly recorded as writer-omitted
    empty tables, not silently interpreted as a nonempty input population.
    The birth gen tree is mandatory, including when explicitly empty.
    """
    frames, trees = {}, {}
    with uproot.open(path) as root:
        for name, columns in names.items():
            full = f"dump/{name}"
            if full not in root:
                if name not in ("particles", "sgsteps"):
                    raise ValueError(f"missing required birth tree: {full}")
                frames[name] = _empty(columns)
                trees[full] = dict(cycle=None, entries=0, absent_empty_writer_convention=True)
            else:
                arrays = root[full].arrays(columns, library="np")
                frames[name] = pd.DataFrame({c: arrays[c] for c in columns})
                trees[full] = _tree_info(root, full)
    return frames, trees


def load_art(path, byteswap=False):
    """Use actual highest-cycle Events count, including zero, then exact FileIndex.

    Zero Events still requires a readable FileIndex and exact empty coverage;
    there is no guessed count or fallback identity source.
    """
    with uproot.open(path) as root:
        trees = {name: _tree_info(root, name) for name in ("Events", "FileIndex")}
        actual = trees["Events"]["entries"]
    return cestopwindow.load(path, actual, byteswap), trees


def load_ntuple(path):
    """Read evtinfo and good-track counts only, without track physics branches."""
    with uproot.open(path) as root:
        tree = root[TREE]
        columns = [f"evtinfo/{k}" for k in KEY]
        arrays = tree.arrays(columns + ["trk/trk.status"], library="ak")
        frame = pd.DataFrame({k: ak.to_numpy(arrays[f"evtinfo/{k}"]) for k in KEY})
        frame["ntrack"] = ak.to_numpy(ak.sum(arrays["trk/trk.status"] > 0, axis=1)).astype(np.uint64)
        if len(frame) != tree.num_entries:
            raise ValueError("ntuple evtinfo/tree entry count mismatch")
        trees = {TREE: _tree_info(root, TREE)}
    return frame, trees


def verify_report(path):
    """Rehash every closed input recorded by a completed stage-history report."""
    report = json.loads(Path(path).read_text())
    required = {"births", "table", "injected", "digitized", "reco", "ntuple"}
    inputs = report.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != required:
        raise ValueError("history report requires every exact closed input receipt")
    for label, item in inputs.items():
        if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
            raise ValueError(f"missing history input receipt: {label}")
        if cestopwindow.checksum(item["path"]) != item["sha256"]:
            raise ValueError(f"history input changed: {label}")
    return dict(verified=True, history_report_sha256=cestopwindow.checksum(path),
                verified_inputs=sorted(inputs), scope="input integrity only; not physics closure")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for option in ("births", "table", "inject-art", "digi-art", "reco-art", "ntuple", "outdir"):
        ap.add_argument(f"--{option}")
    ap.add_argument("--verify-report", help="verify every closed input in an existing history report only")
    ap.add_argument("--fileindex-byteswap", action="store_true")
    a = ap.parse_args(argv)
    if a.verify_report:
        if any(getattr(a, k) is not None for k in ("births", "table", "inject_art", "digi_art", "reco_art", "ntuple", "outdir")):
            ap.error("verify-report cannot be combined with history input/output options")
        print(json.dumps(verify_report(a.verify_report), indent=2))
        return
    if any(getattr(a, k) is None for k in ("births", "table", "inject_art", "digi_art", "reco_art", "ntuple", "outdir")):
        ap.error("all history inputs and outdir are required")
    out = Path(a.outdir)
    if out.exists():
        raise FileExistsError(f"output directory must be fresh: {out}")
    inputs = {}

    def receipt(label, path, trees):
        inputs[label] = dict(path=str(Path(path).resolve()), sha256=cestopwindow.checksum(path), trees=trees)

    birth, trees = load_table(a.births, {"gen": KEY})
    receipt("births", a.births, trees)
    table, trees = load_table(a.table, {"particles": KEY + ["key"], "sgsteps": KEY + ["sp"]})
    receipt("table", a.table, trees)
    stages = []
    for label, path in (("injected", a.inject_art), ("digitized", a.digi_art), ("reco", a.reco_art)):
        frame, trees = load_art(path, a.fileindex_byteswap)
        stages.append(frame)
        receipt(label, path, trees)
    nts, trees = load_ntuple(a.ntuple)
    receipt("ntuple", a.ntuple, trees)
    ledger = reconcile(birth["gen"], table["particles"], table["sgsteps"], *stages, nts)
    report = summarize(ledger)
    report.update(inputs=inputs, explicit_uproot_metadata_byteswap=a.fileindex_byteswap,
                  zero_event_stages=[label for label, frame in zip(("injected", "digitized", "reco", "ntuple"),
                                                                 [*stages, nts]) if frame.empty])
    out.mkdir(parents=True, exist_ok=False)
    ledger.to_csv(out / "events.csv.gz", index=False)
    (out / "cehistory.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
