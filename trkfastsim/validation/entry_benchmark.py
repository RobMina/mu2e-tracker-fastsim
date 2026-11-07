"""Census and paired reconstructed-track yields for the matched entry benchmark."""
import argparse
import json
from pathlib import Path
import re

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from trkfastsim.validation import active_straws, cosmic_saved, entry_replay as er
from trkfastsim.validation.entry_pilot import endpoints, frame


def process_time(path):
    text = Path(path).read_text()
    def number(label):
        return float(re.search(re.escape(label) + r":\s*([\d.]+)", text).group(1))
    elapsed = re.search(r"Elapsed \(wall clock\) time .*?:\s*([\d:.]+)", text).group(1)
    wall = 0.
    for part in elapsed.split(":"):
        wall = wall * 60 + float(part)
    code = number("Exit status")
    if code != 0:
        raise ValueError(f"unsuccessful process {path}: {code}")
    return dict(wall_s=wall, user_s=number("User time (seconds)"),
                system_s=number("System time (seconds)"), rss_kib=number("Maximum resident set size (kbytes)"))


def transport_check(entries, transport, metadata, label):
    events, particles = er.read(entries)
    sims = frame(transport, "dump/particles")
    if sims.duplicated(er.KEY + ["genidx"]).any():
        raise ValueError("duplicate transported primary identities")
    joined = particles.merge(sims, left_on=er.KEY + ["index"], right_on=er.KEY + ["genidx"],
                             how="outer", validate="one_to_one", indicator=True, suffixes=("", "_sim"))
    if not joined._merge.eq("both").all() or not (sims.isprim == 1).all():
        raise ValueError("transported primary census mismatch")
    np.testing.assert_array_equal(joined.pdg, joined.pdg_sim)
    np.testing.assert_array_equal(joined[["x", "y", "z"]], joined[["sx", "sy", "sz"]])
    steps = frame(transport, "dump/sgsteps")
    refs = steps[er.KEY + ["sp"]].merge(sims[er.KEY + ["key"]], left_on=er.KEY + ["sp"],
                                        right_on=er.KEY + ["key"], how="left", indicator=True,
                                        validate="many_to_one")
    if not refs._merge.eq("both").all():
        raise ValueError("step references unknown transported particle")
    result = dict(source_events=len(events), particles=len(particles), gas_steps=len(steps),
                  step_events=len(steps[er.KEY].drop_duplicates()),
                  step_primaries=len(steps[er.KEY + ["sp"]].drop_duplicates()))
    if label == "g4":
        result["readback"] = er.check(entries, transport)
        codes = sims.scode.value_counts().sort_index()
        result["stopping_codes"] = {str(k): int(v) for k, v in codes.items()}
        cut = json.loads(Path(metadata).read_text())["envelope"]
        leaving, excess = endpoints(sims, cut)
        exited = sims.scode.to_numpy() == 54
        if not leaving[exited].all():
            raise ValueError("cylinder endpoint is not outward")
        result["cylinder_exits"] = int(exited.sum())
        result["max_steps_stops"] = int((sims.scode == 55).sum())
        result["endpoint_excess_max_mm"] = float(excess[exited].max()) if exited.any() else None
    else:
        final = frame(transport, "dump/final")
        result["status"] = {str(k): int(v) for k, v in final.status.value_counts().sort_index().items()}
    return result


def arm(directory, events):
    directory = Path(directory)
    if not (directory / "chain.done").exists():
        raise ValueError("chain is not finished")
    counts = {}
    for name in ("inj.art", "dig.art", "mcs-CH.art"):
        with uproot.open(directory / name) as root:
            counts[name] = root["Events"].num_entries
    if counts["inj.art"] != len(events) or counts["dig.art"] != len(events):
        raise ValueError("input/injected/digitized census mismatch")
    tracks, info = active_straws.read(directory / "nts.root")
    if info["ntuple_entries"] != counts["mcs-CH.art"]:
        raise ValueError("ntuple entries differ from selected reconstruction output")
    if not tracks[er.KEY].merge(events[er.KEY], on=er.KEY, how="left", indicator=True)._merge.eq("both").all():
        raise ValueError("ntuple event absent from source census")
    with uproot.open(directory / "nts.root") as root:
        t = root["EventNtuple/ntuple"]
        values = t.arrays(["trk/trk.status", "trk/trk.chisq", "trk/trk.ndof"], library="ak")
    good = values["trk/trk.status"] > 0
    first = ak.argmax(good, axis=1, keepdims=True)
    chi = ak.to_numpy(ak.fill_none(ak.firsts(values["trk/trk.chisq"][first]), np.nan))
    ndf = ak.to_numpy(ak.fill_none(ak.firsts(values["trk/trk.ndof"][first]), np.nan))
    tracks["chi2ndf"] = chi / np.maximum(ndf, 1)
    timings = {stage: process_time(directory / f"{stage}.time")
               for stage in ("transport", "inject", "digi", "reco", "ntuple")}
    total = sum(t["wall_s"] for t in timings.values())
    result = dict(source_events=len(events), stages=counts, ntuple=info, timings=timings,
                  sum_process_wall_s=total, source_events_per_process_wall_s=len(events) / total,
                  source_scope="S2-selected tracker-entry experiment, not generated CRY or production showers")
    return tracks, result


def compare(entries, fast, g4, output):
    events, _ = er.read(entries)
    merged = events[er.KEY].copy()
    reports = {}
    for label, directory in (("fast", fast), ("g4", g4)):
        tracks, reports[label] = arm(directory, events)
        merged = merged.merge(tracks.rename(columns={c: f"{label}_{c}" for c in tracks if c not in er.KEY}),
                              on=er.KEY, how="left", validate="one_to_one")
        merged[f"{label}_track"] = merged[f"{label}_track"].fillna(False).astype(bool)
    both = merged.fast_track & merged.g4_track
    summary = dict(arms=reports, efficiency=cosmic_saved.efficiency(merged.fast_track, merged.g4_track),
                   paired_good_events=int(both.sum()))
    for field in ("nactive", "active_straws", "chi2ndf"):
        summary[field] = {label: active_straws.mean_error(merged.loc[both, f"{label}_{field}"])
                          for label in ("fast", "g4")}
        summary[field]["paired_difference"] = active_straws.mean_error(
            merged.loc[both, f"fast_{field}"] - merged.loc[both, f"g4_{field}"])
    summary["timing_scope"] = "Measured implemented primary-only routes including adapters/diagnostic output; not production-equivalent speedup."
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    merged.to_csv(output.with_suffix(".csv.gz"), index=False)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entries", required=True)
    ap.add_argument("--fast")
    ap.add_argument("--g4")
    ap.add_argument("--transport")
    ap.add_argument("--label", choices=["fast", "g4"])
    ap.add_argument("--metadata")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    if args.transport:
        if not args.label or not args.metadata:
            ap.error("--transport needs --label and --metadata")
        report = transport_check(args.entries, args.transport, args.metadata, args.label)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    else:
        if not args.fast or not args.g4:
            ap.error("comparison needs --fast and --g4")
        report = compare(args.entries, args.fast, args.g4, args.output)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
