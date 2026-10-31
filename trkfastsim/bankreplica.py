"""Opt-in source-event clustered donor replicas; production samplers are unchanged."""
import numpy as np
import pandas as pd

from . import tables

KEY = tables.KEY
TREES = ("events", "instraw", "instraw_steps", "offaxis")


def validate(bank):
    if set(bank) != set(TREES):
        raise ValueError("replicas require exactly the four donor-bank tables")
    events, ins, secondary, off = (bank[name] for name in TREES)
    if events.empty or events.duplicated(KEY).any() or events[KEY].isna().any().any():
        raise ValueError("nonempty unique donor source-event identities required")
    if ins.xing.duplicated().any() or not np.array_equal(np.sort(ins.xing.to_numpy()), np.arange(len(ins))):
        raise ValueError("donor crossing identifiers must be unique and dense")
    universe = pd.MultiIndex.from_frame(events[KEY])
    for name, frame in (("instraw", ins), ("offaxis", off)):
        if not pd.MultiIndex.from_frame(frame[KEY]).isin(universe).all():
            raise ValueError(f"{name} rows outside donor source-event universe")
    if not secondary.xing.isin(ins.xing).all():
        raise ValueError("secondary crossing lacks donor")
    counts = secondary.groupby("xing").size().reindex(ins.xing, fill_value=0).to_numpy()
    if "nsec" in ins and not np.array_equal(counts, ins.nsec.to_numpy()):
        raise ValueError("in-straw secondary counts disagree with bank")
    counts = off.groupby(KEY).size().reindex(universe, fill_value=0).to_numpy()
    if "noff" in events and not np.array_equal(counts, events.noff.to_numpy()):
        raise ValueError("off-axis counts disagree with donor universe")
    return events


def replica(bank, rng, *, selection=None, stratify="pbin"):
    """Resample eligible source events, including empty secondary donors, jointly across components.

    Selection is an explicit event-row-index vector for deterministic tests/replay;
    otherwise source events are drawn with replacement within their original pbin.
    Crossing/path/angle/plane matching and fallback remain the production samplers'.
    A source event's multiplicity is shared by all its crossings and off-axis rows.
    New dense synthetic event/crossing IDs prevent donor-copy joins from multiplying.
    The returned separate ancestry tables preserve original full event identities.
    These replicas estimate bank variation only after response/reco replay; they
    neither resample recipients nor restore correlations absent from the sampler.
    """
    events = validate(bank)
    n = len(events)
    if selection is None:
        if stratify not in ("pbin", "none"):
            raise ValueError("explicit supported donor bootstrap stratification required")
        selection = np.empty(n, np.int64)
        groups = [np.arange(n)] if stratify == "none" else events.groupby("pbin", sort=True).indices.values()
        for indices in groups:
            selection[indices] = rng.choice(indices, size=len(indices), replace=True)
    selection = np.asarray(selection)
    if selection.ndim != 1 or selection.dtype.kind not in "iu" or len(selection) != n:
        raise ValueError("one integer source-event selection per replica event required")
    if (selection < 0).any() or (selection >= n).any():
        raise ValueError("donor source-event selection is outside bank")
    ancestry = events.iloc[selection][KEY].reset_index(drop=True).rename(columns={k: f"source_{k}" for k in KEY})
    ancestry["source_event_row"] = selection
    ancestry["replica_event_row"] = np.arange(n)
    selected = events.iloc[selection].reset_index(drop=True).copy()
    joins = selected[KEY].copy().assign(_copy=np.arange(n))
    output = {}
    for name in ("instraw", "offaxis"):
        frame = joins.merge(bank[name], on=KEY, how="inner", validate="many_to_many")
        frame["event"] = frame._copy.to_numpy() + 1
        frame["run"], frame["subrun"] = 0, 0
        output[name] = frame
    ins = output["instraw"].sort_values(["_copy", "xing"], kind="stable").reset_index(drop=True)
    crossing = ins[["_copy", "xing"]].rename(columns={"xing": "source_xing"}).copy()
    crossing["replica_xing"] = np.arange(len(ins))
    secondary = crossing.merge(bank["instraw_steps"].rename(columns={"xing": "source_xing"}),
                               on="source_xing", how="inner", validate="many_to_many")
    secondary["xing"] = secondary.replica_xing
    secondary = secondary.drop(columns=["_copy", "source_xing", "replica_xing"])
    secondary = secondary.loc[:, bank["instraw_steps"].columns].reset_index(drop=True)
    ins["xing"] = np.arange(len(ins))
    output["instraw"] = ins.drop(columns="_copy").loc[:, bank["instraw"].columns]
    output["offaxis"] = output["offaxis"].drop(columns="_copy").loc[:, bank["offaxis"].columns].reset_index(drop=True)
    selected["run"], selected["subrun"], selected["event"] = 0, 0, np.arange(n) + 1
    output["events"], output["instraw_steps"] = selected, secondary
    validate(output)
    return output, dict(events=ancestry, crossings=crossing.rename(columns={"_copy": "replica_event_row"}))
