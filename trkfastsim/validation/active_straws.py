"""Recount distinct active straws from saved first-good-track EventNtuples."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

KEY = ["run", "subrun", "event"]


def mean_error(values):
    values = np.asarray(values, float)
    return {"value": float(values.mean()) if len(values) else None,
            "error": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
            "events": len(values)}


def count_straws(hits, selected):
    """Count hit rows and unique (plane, panel, straw) per event without conflating events."""
    chosen = hits[selected]
    counts = ak.to_numpy(ak.num(chosen, axis=1))
    event = np.repeat(np.arange(len(counts)), counts)
    fields = [event] + [ak.to_numpy(ak.flatten(chosen[name])) for name in ("plane", "panel", "straw")]
    unique = np.unique(np.column_stack(fields), axis=0)
    distinct = np.bincount(unique[:, 0].astype(int), minlength=len(counts))
    return counts, distinct


def summarize_hits(hits, has_track, persisted_nactive):
    active, distinct = count_straws(hits, hits.state > -2)
    right, right_distinct = count_straws(hits, hits.state > 0)
    if not np.array_equal(active[has_track], np.asarray(persisted_nactive)[has_track]):
        raise ValueError("state > -2 hit rows differ from persisted nactive")
    return dict(active_rows=np.where(has_track, active, np.nan),
                active_straws=np.where(has_track, distinct, np.nan),
                right_only_rows=np.where(has_track, right, np.nan),
                right_only_straws=np.where(has_track, right_distinct, np.nan))


def read(path):
    with uproot.open(path) as root:
        tree = root["EventNtuple/ntuple"]
        values = tree.arrays([f"evtinfo/{key}" for key in KEY] + ["trk/trk.status", "trk/trk.nactive"], library="ak")
        good = values["trk/trk.status"] > 0
        has = ak.to_numpy(ak.any(good, axis=1))
        first = ak.argmax(good, axis=1, keepdims=True)
        hits = ak.fill_none(ak.firsts(tree["trkhits"].array()[first], axis=1), [], axis=0)
        nactive = ak.to_numpy(ak.fill_none(ak.firsts(values["trk/trk.nactive"][first]), 0))
        frame = pd.DataFrame({key: ak.to_numpy(values[f"evtinfo/{key}"]) for key in KEY})
        frame["track"] = has
        frame["nactive"] = np.where(has, nactive, np.nan)
        for key, value in summarize_hits(hits, has, nactive).items():
            frame[key] = value
        if frame.duplicated(KEY).any():
            raise ValueError("duplicate event identities")
        metadata = dict(ntuple_entries=int(tree.num_entries), good_tracks=int(has.sum()),
                        all_active_rows_match_nactive=True)
    return frame, metadata


def recount(baseline, fast, stock):
    """Preserve the saved birth denominator and first-good-track cohort; compare only the count definition."""
    if baseline.duplicated(KEY).any():
        raise ValueError("duplicate baseline identities")
    data = baseline.copy()
    for name, frame in (("fast", fast), ("g4", stock)):
        if not frame[KEY].merge(data[KEY], how="left", indicator=True, on=KEY)._merge.eq("both").all():
            raise ValueError("ntuple event absent from baseline")
        renamed = frame.rename(columns={key: f"{name}_new_{key}" for key in frame if key not in KEY})
        data = data.merge(renamed, on=KEY, how="left", validate="one_to_one")
        for key in ("track", "nactive"):
            old = data[f"{name}_{key}"].to_numpy()
            new = data[f"{name}_new_{key}"].to_numpy()
            if key == "track":
                old, new = pd.Series(old).fillna(False), pd.Series(new).fillna(False)
            if not np.array_equal(np.asarray(old), np.asarray(new), equal_nan=(key != "track")):
                raise ValueError(f"saved {name} {key} differs")
        if not np.array_equal(data[f"{name}_nstraw"].to_numpy(),
                              data[f"{name}_new_right_only_straws"].to_numpy(), equal_nan=True):
            raise ValueError("historical straw count is not reproduced by right-only state > 0")
    paired = data.fast_track.fillna(False).astype(bool) & data.g4_track.fillna(False).astype(bool)
    result = dict(birth_events=len(data), paired_good_tracks=int(paired.sum()),
                  definition="active KinKal states > -2; distinct (plane,panel,straw) per first good track",
                  superseded="historical state > 0 counts right-side drift only, not all active straws",
                  saved_tracks_nactive_and_right_only_counts_reproduced=True)
    for column in ("active_straws", "active_rows", "right_only_straws"):
        a = data.loc[paired, f"fast_new_{column}"].to_numpy(float)
        b = data.loc[paired, f"g4_new_{column}"].to_numpy(float)
        result[column] = dict(fast=mean_error(a), stock=mean_error(b), difference=mean_error(a - b))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--entry-events", required=True)
    parser.add_argument("--chain-events", required=True)
    parser.add_argument("--numbers", required=True, help="numbers.json to create or extend with CE counts")
    args = parser.parse_args(argv)
    stock, metadata = read(args.stock)
    result = {"stock": metadata}
    for arm in ("entry", "chain"):
        fast, metadata = read(getattr(args, arm))
        result[arm] = recount(pd.read_csv(getattr(args, arm + "_events")), fast, stock)
        result[arm]["ntuple"] = metadata
    path = Path(args.numbers)
    numbers = json.loads(path.read_text()) if path.exists() else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    numbers["ce_active_straws"] = result
    path.write_text(json.dumps(numbers, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
