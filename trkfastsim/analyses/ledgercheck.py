"""Controlled writer-only entry-ledger diagnostic, not a historical rerun or production acceptance test.

Run with python3 -m trkfastsim.analyses.ledgercheck INPUTS --gdml GDML --outdir NEW_DIR
--seed N --tracker-ref REF --core-ref REF. Reuses existing gen tables and GDML, never regenerates G4.
JSONL contains per-file exact quantiles; CSV contains paired per-muon counts, including zero-crossing muons.
Totals count file-local identities; overlapping event keys in different input files are not deduplicated.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from mu2efastsim import tables as core_tables
from .. import entryledger, geometry, steps as st, tables

IDENTITY = tables.KEY + ["sp"]
CROSSING = IDENTITY + ["sid", "occurrence"]
GEOMETRY = ["x0", "y0", "z0", "x1", "y1", "z1", "len"]
QUANTILES = [0., .01, .05, .5, .95, .99, 1.]
SCOPE = ("Controlled new internal writer-only diagnostic on frozen trajectories; not a rerun of historical "
         "before results, not full production acceptance, and not surface/reco-ready transport. "
         "The cumulative ledger excludes gas/wall/wire transport loss and flight time.")


def sha256(path):
    """Hash a local file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_gen(path):
    """Read only the highest gen cycle and report the exact selected object."""
    with uproot.open(path) as root:
        directory = root["dump"]
        cycles = [int(k.rsplit(";", 1)[1]) for k in directory.keys(cycle=True, recursive=False)
                  if k.rsplit(";", 1)[0] == "gen"]
        if not cycles:
            raise ValueError(f"{path}: no dump/gen tree")
        cycle = max(cycles)
        gen = directory[f"gen;{cycle}"].arrays(library="pd")
    return gen, {"object": f"dump/gen;{cycle}", "cycle": cycle,
                 "rows": len(gen), "events": len(gen[tables.KEY].drop_duplicates())}


def muon_counts(sgs, particles):
    """Count crossings and tracker-selection steps by KEY+sp, retaining unhit muons."""
    identities = particles.rename(columns={"key": "sp"})[IDENTITY].drop_duplicates()
    counts = identities.set_index(IDENTITY)
    counts["crossings"] = sgs.groupby(IDENTITY).size()
    good = sgs.loc[tables.good_steps(sgs)] if len(sgs) else sgs
    counts["good_steps"] = good.groupby(IDENTITY).size()
    counts = counts.fillna(0).astype({"crossings": "int64", "good_steps": "int64"})
    counts["crossed"] = counts.crossings > 0
    counts["selected"] = counts.good_steps >= tables.FILTER_MIN_TRK_STEPS
    return counts


def summarize_counts(counts):
    """Summarize per-muon and per-event tracker-term selection, not the full PrimaryFilter."""
    events = counts[["crossed", "selected"]].groupby(tables.KEY).any()
    return {"muons": len(counts), "crossings": int(counts.crossings.sum()),
            "good_steps": int(counts.good_steps.sum()), "crossing_muons": int(counts.crossed.sum()),
            "selected_muons": int(counts.selected.sum()), "crossing_events": int(events.crossed.sum()),
            "selected_events": int(events.selected.sum())}


def pair_counts(legacy, current):
    """Outer-pair identities, without conflating different muons of the same event."""
    pair = legacy.add_suffix("_legacy").join(current.add_suffix("_arm"), how="outer")
    numeric = ["crossings_legacy", "good_steps_legacy", "crossings_arm", "good_steps_arm"]
    flags = ["crossed_legacy", "selected_legacy", "crossed_arm", "selected_arm"]
    pair[numeric] = pair[numeric].fillna(0).astype("int64")
    pair[flags] = pair[flags].eq(True)
    pair["crossings_delta"] = pair.crossings_arm - pair.crossings_legacy
    pair["good_steps_delta"] = pair.good_steps_arm - pair.good_steps_legacy
    pair["selection_lost"] = pair.selected_legacy & ~pair.selected_arm
    pair["selection_gained"] = ~pair.selected_legacy & pair.selected_arm
    return pair


def summarize_pair(pair):
    """Report paired gains/losses of crossing and selected identities and events."""
    events = pair[["crossed_legacy", "crossed_arm", "selected_legacy", "selected_arm"]].groupby(tables.KEY).any()
    result = {"crossings_delta": int(pair.crossings_delta.sum()),
              "good_steps_delta": int(pair.good_steps_delta.sum()),
              "crossings_lost": int(-pair.crossings_delta.clip(upper=0).sum()),
              "crossings_gained": int(pair.crossings_delta.clip(lower=0).sum())}
    for label, frame in (("muons", pair), ("events", events)):
        for flag in ("crossed", "selected"):
            before, after = frame[f"{flag}_legacy"], frame[f"{flag}_arm"]
            result[f"{flag}_{label}_lost"] = int((before & ~after).sum())
            result[f"{flag}_{label}_gained"] = int((~before & after).sum())
            result[f"{flag}_{label}_both"] = int((before & after).sum())
    return result


def defect_quantiles(ledger):
    """Exact per-file cumulative entry defects; entries are not independent loss increments."""
    result = {}
    for label, frame in (("all_entries", ledger), ("emitted_entries", ledger.loc[ledger.emitted])):
        values = frame.energy_defect_MeV.to_numpy(float)
        finite = values[np.isfinite(values)]
        result[label] = {"entries": len(values), "nonfinite": int(len(values) - len(finite)),
                         "quantiles_MeV": dict(zip(map(str, QUANTILES),
                             np.quantile(finite, QUANTILES).tolist() if len(finite) else [None] * len(QUANTILES)))}
    result["removed_entries"] = int((~ledger.emitted).sum())
    return result


def crossing_frame(frame, columns):
    """Assign repeated straw crossings a stable within-muon occurrence index."""
    out = frame[IDENTITY + ["sid"] + columns].copy()
    out["occurrence"] = out.groupby(IDENTITY + ["sid"], sort=False).cumcount()
    return out.set_index(CROSSING)


def compare_frames(before, after, columns):
    """Compare matched geometry exactly; preserve multiplicity and expose unmatched rows."""
    left, right = crossing_frame(before, columns), crossing_frame(after, columns)
    common = left.index.intersection(right.index)
    a, b = left.loc[common, columns].to_numpy(float), right.loc[common, columns].to_numpy(float)
    equal = np.all(a == b, axis=1)
    return {"matched": len(common), "lost": len(left) - len(common), "gained": len(right) - len(common),
            "changed": int((~equal).sum()),
            "max_abs_difference": float(np.max(np.abs(a - b))) if a.size else None,
            "exact_match": bool(len(left) == len(right) == len(common) and equal.all())}


def compare_crossings(legacy_steps, arm_steps, legacy_ledger, arm_ledger):
    """Check frozen entries always and emitted geometry with allowance for stop removal."""
    frozen = compare_frames(legacy_ledger, arm_ledger, ["path_mm", "support_loss_MeV", "p_start_MeV_c"])
    emitted = compare_frames(legacy_steps, arm_steps, GEOMETRY)
    removed = int((~arm_ledger.emitted).sum())
    return {"frozen_entries": frozen, "emitted_geometry": emitted, "stop_removed_entries": removed,
            "expect_all_emitted_geometry_equal": removed == 0,
            "consistent_with_writer_only": bool(frozen["exact_match"] and emitted["changed"] == 0
                and emitted["gained"] == 0 and emitted["lost"] == removed)}


def source_manifest():
    """Hash Python sources actually present, in addition to caller-supplied pinned refs."""
    import mu2efastsim
    import trkfastsim
    return {name: {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*.py"))}
            for name, root in (("core", Path(mu2efastsim.__file__).parent),
                               ("tracker", Path(trkfastsim.__file__).parent))}


def run(inputs, *, gdml, outdir, seed, tracker_ref, core_ref):
    """Run three equal-seed arms file by file; refuse to reuse an existing output directory."""
    files = [Path(p).resolve() for p in tables.expand(inputs)]
    if not files:
        raise ValueError("at least one input is required")
    if len(set(files)) != len(files):
        raise ValueError("duplicate input files")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not tracker_ref.strip() or not core_ref.strip():
        raise ValueError("caller-supplied pinned refs are required")
    gdml = Path(gdml).resolve()
    if not gdml.is_file() or not all(p.is_file() for p in files):
        raise FileNotFoundError("GDML and every input must be existing local files")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=False)
    settings = {"stage": "C", "sliced_msc": True, "exit_scatter": False, "sampler": None, "offaxis": None,
                "capture_entry_ledger": True, "all_muons": True, "air_loss": True, "interior_msc": True,
                "msc_tail": True, "tail_k": None, "near_margin": st.FOOTPRINT_MARGIN,
                "kink_xx0": st.KINK_XX0, "kink_sup_xx0": st.KINK_SUP_XX0, "band_spread": False,
                "interior_msc_cal": None, "lever_moments": False}
    manifest = {"scope": SCOPE, "tracker_ref": tracker_ref, "core_ref": core_ref,
                "refs_are_caller_supplied": True, "sources_sha256": source_manifest(),
                "gdml": {"path": str(gdml), "sha256": sha256(gdml)}, "seed": seed,
                "seed_rule": "numpy.default_rng([seed, zero_based_file_index]), reset for each arm",
                "settings": settings, "interior": {"enabled": True, "calo": True},
                "modes": entryledger.MODES, "input_order": list(map(str, files)),
                "event_count_rule": "unique actual gen KEY per highest-cycle tree; totals sum file-local counts",
                "selection": {"scope": "tracker term only, per KEY+sp then event any",
                    "min_steps": tables.FILTER_MIN_TRK_STEPS, "eion_strict_min_MeV": tables.FILTER_MIN_STEP_EION,
                    "p_strict_min_MeV_c": tables.FILTER_MIN_MOM, "p_strict_max_MeV_c": tables.FILTER_MAX_MOM},
                "quantile_rule": "exact per file, entry-weighted cumulative energy defect; not pooled across files",
                "pairing_caveat": "Equal initial seeds freeze transport; gas-loss draws are not guaranteed row-paired "
                    "after stop removal changes array lengths. Selection changes include this sampling effect."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    geom = geometry.from_gdml(str(gdml))
    world = st.load_world(str(gdml), geom)
    interior = st.TrackerInterior(str(gdml), geom)
    totals = {mode: {} for mode in entryledger.MODES}
    paired_totals = {mode: {} for mode in entryledger.MODES}
    geometry_failures = {mode: 0 for mode in entryledger.MODES}
    total_events = 0
    with (out / "files.jsonl").open("w") as report, (out / "muons.csv").open("w") as csv:
        header = True
        for i, path in enumerate(files):
            gen, census = read_gen(path)
            total_events += census["events"]
            record = {"file_index": i, "input": str(path), "sha256": sha256(path), "gen": census, "arms": {}}
            for mode in entryledger.MODES:
                sgs, parts = st.make_steps(gen, geom, np.random.default_rng([seed, i]), world=world,
                                          interior=interior, entry_ledger=mode, **settings)
                ledger = st.make_steps.entry_ledger
                if ledger is None:
                    raise RuntimeError("make_steps did not capture the entry ledger")
                counts = muon_counts(sgs, parts)
                if mode == "legacy":
                    legacy_steps, legacy_ledger, legacy_counts = sgs, ledger, counts
                paired = pair_counts(legacy_counts, counts)
                summary = summarize_counts(counts)
                pairing = summarize_pair(paired)
                record["arms"][mode] = {"counts": summary, "paired_vs_legacy": pairing,
                    "energy_defect": defect_quantiles(ledger),
                    "crossing_comparison": compare_crossings(legacy_steps, sgs, legacy_ledger, ledger)}
                geometry_failures[mode] += int(not record["arms"][mode]["crossing_comparison"]["consistent_with_writer_only"])
                for key, value in summary.items():
                    totals[mode][key] = totals[mode].get(key, 0) + value
                for key, value in pairing.items():
                    paired_totals[mode][key] = paired_totals[mode].get(key, 0) + value
                paired.reset_index().assign(file_index=i, mode=mode).to_csv(csv, index=False, header=header)
                header = False
            report.write(json.dumps(record, allow_nan=False) + "\n")
            report.flush()
            del gen, sgs, parts, ledger, counts, paired, legacy_steps, legacy_ledger, legacy_counts
    result = {"scope": SCOPE, "complete": True, "files": len(files), "file_local_gen_events": total_events,
              "file_local_totals": totals, "file_local_paired_totals_vs_legacy": paired_totals,
              "files_inconsistent_with_writer_only": geometry_failures}
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tracker-ref", required=True)
    parser.add_argument("--core-ref", required=True)
    args = vars(parser.parse_args(argv))
    print(json.dumps(run(**args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
