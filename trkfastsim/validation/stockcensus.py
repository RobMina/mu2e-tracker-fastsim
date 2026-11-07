"""Validation-only stock census; no quality, algorithm, truth or straw-uniqueness cuts."""
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


TRACK_INTS = ("status", "goodfit", "fitalg", "nhits", "nactive", "ndof")
TRACK_FLOATS = ("chisq", "fitcon")
HIT_FIELDS = ("plane", "panel", "straw", "state")
KEY_FIELDS = ("run", "subrun", "event")
_I64_MAX = np.iinfo(np.int64).max
_KEY_DTYPE = np.dtype([(name, np.int64) for name in KEY_FIELDS])


def validate_keys(keys):
    """Return a fresh int64 (N, 3) array of unique, nonnegative run/subrun/event keys."""
    if np.ma.isMaskedArray(keys) and np.any(np.ma.getmaskarray(keys)):
        raise ValueError("event keys must not be missing")
    a = np.asarray(keys)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("event keys must have shape (N, 3)")
    if a.dtype.kind not in "iu":
        raise ValueError("event keys must have integer dtype (no nonfinite or floating keys)")
    if np.any(a < 0) or np.any(a > _I64_MAX):
        raise ValueError("event keys out of range [0, int64 max]")
    a = np.array(a, dtype=np.int64, order="C", copy=True)
    rows = a.view(_KEY_DTYPE).reshape(-1)
    if len(np.unique(rows)) != len(a):
        raise ValueError("duplicate event keys")
    return a


def event_index(reference_keys, query_keys):
    """Exact full-key join: reference row for each query, or -1; duplicates in either fail."""
    ref = validate_keys(reference_keys).view(_KEY_DTYPE).reshape(-1)
    query = validate_keys(query_keys).view(_KEY_DTYPE).reshape(-1)
    result = np.full(len(query), -1, dtype=np.int64)
    if not len(ref):
        return result
    order = np.argsort(ref)
    sorted_ref = ref[order]
    pos = np.searchsorted(sorted_ref, query)
    inside = pos < len(ref)
    candidate = np.flatnonzero(inside)
    match = candidate[sorted_ref[pos[inside]] == query[inside]]
    result[match] = order[pos[match]]
    return result


def _numpy(array, name, kind):
    try:
        a = ak.to_numpy(array, allow_missing=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed or missing values in {name}") from exc
    if a.ndim != 1 or a.dtype.kind not in kind:
        raise ValueError(f"{name} must be a one-dimensional {'integer' if kind == 'iu' else 'floating'} leaf")
    if kind == "iu":
        if np.any(a > _I64_MAX):
            raise ValueError(f"{name} exceeds int64 range")
        a = a.astype(np.int64, copy=True)
    else:
        a = a.copy()
    return a


def _field(array, name, branch):
    if name not in ak.fields(array):
        raise ValueError(f"missing required field {branch}/{name}")
    return array[name]


def _counts(array, depth, name, nentries):
    if len(array) != nentries or array.ndim != depth:
        raise ValueError(f"malformed jagged alignment in {name}")
    try:
        counts = _numpy(ak.num(array, axis=1), name, "iu")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed jagged alignment in {name}") from exc
    return counts


def _flat(array, depth, name, kind):
    try:
        for axis in range(depth - 1, 0, -1):
            array = ak.flatten(array, axis=axis)
        return _numpy(array, name, kind)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed {name}: {exc}") from exc


def _limit(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < int(positive) or value > _I64_MAX:
        raise ValueError(f"{name} out of range")
    return int(value)


def _cap(value, limit, name):
    if value > limit:
        raise ValueError(f"{name} exceeded: {value} > {limit}; no truncation")


def read_stock(path, *, tree="EventNtuple/ntuple", chunk_entries=4096,
               max_events=1000000, max_tracks=1000000, max_hits=10000000):
    """Return (flat payload, metadata) from the highest existing TTree/RNTuple cycle.

    Input is a local file. Event keys are unique nonnegative int64 triples. Required
    top branches are evtinfo, trk and trkhits; missing fields never get substitutes.
    Every stored track/hit is retained in input order, including repeated straws.
    Caps raise, never truncate; each branch read is bounded by chunk_entries, with
    track/hit caps checked before flat output allocation (ROOT decoding comes first).
    Stock fillHitInfo and fillTrkInfoHits both visit every KalSeed hit; fillHits can
    omit the whole required branch, but cannot justify inconsistent recorded rows.
    """
    chunk_entries = _limit(chunk_entries, "chunk_entries", positive=True)
    limits = {"max_events": _limit(max_events, "max_events"),
              "max_tracks": _limit(max_tracks, "max_tracks"),
              "max_hits": _limit(max_hits, "max_hits")}
    if not isinstance(tree, str) or not tree or ";" in tree:
        raise ValueError("tree must be a nonempty uncycled ROOT path")
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"stock input must be an existing local file: {path}")
    fields = {"evtinfo": list(KEY_FIELDS),
              "trk": [f"trk.{name}" for name in TRACK_INTS + TRACK_FLOATS],
              "trkhits": list(HIT_FIELDS)}
    parts = {name: [] for name in ("evkey", "track_ev", "track_index", "hit_track", "hit_index", "hit_sid", "hit_state")}
    parts.update({f"track_{name}": [] for name in TRACK_INTS + TRACK_FLOATS})
    ntracks = nhits = 0
    with uproot.open(path, array_cache=None) as root:
        try:
            key = root.key(tree)
            obj = root[tree]
        except KeyError as exc:
            raise ValueError(f"missing required tree {tree}") from exc
        if obj.classname not in ("TTree", "ROOT::RNTuple"):
            raise ValueError(f"{tree} is not a TTree or RNTuple: {obj.classname}")
        nentries = int(obj.num_entries)
        _cap(nentries, limits["max_events"], "max_events")
        branches = {}
        for name in fields:
            try:
                branches[name] = obj[name]
            except KeyError as exc:
                raise ValueError(f"missing required branch {name}; hit census cannot be complete without trkhits") from exc
        metadata = {"path": str(path), "tree_key": f"{tree};{key.fCycle}",
                    "tree_class": obj.classname, "num_entries": nentries, "cycle": int(key.fCycle),
                    "read_paths": {name: branch.name for name, branch in branches.items()},
                    "fields": fields, "limits": limits, "chunk_entries": chunk_entries}
        # Read zero entries too so an empty tree still has to satisfy the required schema.
        for start in range(0, max(1, nentries), chunk_entries):
            stop = min(start + chunk_entries, nentries)
            size = stop - start
            evt = branches["evtinfo"].array(entry_start=start, entry_stop=stop, library="ak")
            evcols = [_numpy(_field(evt, name, "evtinfo"), f"evtinfo/{name}", "iu") for name in KEY_FIELDS]
            if any(len(col) != size for col in evcols):
                raise ValueError("malformed evtinfo alignment")
            evkey = validate_keys(np.column_stack(evcols))
            trk = branches["trk"].array(entry_start=start, entry_stop=stop, library="ak")
            status = _field(trk, "trk.status", "trk")
            counts = _counts(status, 2, "trk/trk.status", size)
            nt = int(np.sum(counts))
            _cap(ntracks + nt, limits["max_tracks"], "max_tracks")
            chunk = {"evkey": evkey,
                     "track_ev": np.repeat(np.arange(start, stop, dtype=np.int64), counts),
                     "track_index": _flat(ak.local_index(status, axis=1), 2, "track_index", "iu")}
            for name in TRACK_INTS + TRACK_FLOATS:
                leaf = _field(trk, f"trk.{name}", "trk")
                if not np.array_equal(_counts(leaf, 2, f"trk/trk.{name}", size), counts):
                    raise ValueError(f"malformed jagged alignment in trk/trk.{name}")
                chunk[f"track_{name}"] = _flat(leaf, 2, f"trk/trk.{name}", "iu" if name in TRACK_INTS else "f")
                if len(chunk[f"track_{name}"]) != nt:
                    raise ValueError(f"malformed flattened track alignment in trk/trk.{name}")
            hits = branches["trkhits"].array(entry_start=start, entry_stop=stop, library="ak")
            state = _field(hits, "state", "trkhits")
            if not np.array_equal(_counts(state, 3, "trkhits/state", size), counts):
                raise ValueError("malformed trkhits track alignment")
            hit_counts = _flat(ak.num(state, axis=2), 2, "trkhits counts", "iu")
            if len(hit_counts) != nt:
                raise ValueError("malformed trkhits hit-count alignment")
            nh = int(np.sum(hit_counts))
            _cap(nhits + nh, limits["max_hits"], "max_hits")
            # EventNtuple InfoStructHelper::fillHitInfo and fillTrkInfoHits share kseed.hits().
            if not np.array_equal(chunk["track_nhits"], hit_counts):
                raise ValueError("trk.nhits does not match recorded trkhits counts; incomplete hit census")
            hitcols = {}
            for name in HIT_FIELDS:
                leaf = _field(hits, name, "trkhits")
                outer = _counts(leaf, 3, f"trkhits/{name}", size)
                inner = _flat(ak.num(leaf, axis=2), 2, f"trkhits/{name} counts", "iu")
                if not np.array_equal(outer, counts) or not np.array_equal(inner, hit_counts):
                    raise ValueError(f"malformed jagged alignment in trkhits/{name}")
                hitcols[name] = _flat(leaf, 3, f"trkhits/{name}", "iu")
                if len(hitcols[name]) != nh:
                    raise ValueError(f"malformed flattened hit alignment in trkhits/{name}")
            for name, upper in (("plane", 36), ("panel", 6), ("straw", 96)):
                if np.any((hitcols[name] < 0) | (hitcols[name] >= upper)):
                    raise ValueError(f"invalid hitid {name}: expected 0..{upper - 1}")
            chunk["hit_track"] = np.repeat(np.arange(ntracks, ntracks + nt, dtype=np.int64), hit_counts)
            chunk["hit_index"] = _flat(ak.local_index(state, axis=2), 3, "hit_index", "iu")
            # Offline/DataProducts/inc/StrawId.hh: plane shift 10, panel shift 7, straw mask 0x7f.
            chunk["hit_sid"] = (hitcols["plane"] << 10) | (hitcols["panel"] << 7) | hitcols["straw"]
            chunk["hit_state"] = hitcols["state"]
            for name in parts:
                parts[name].append(chunk[name])
            ntracks += nt
            nhits += nh
    payload = {name: np.concatenate(chunks, axis=0) for name, chunks in parts.items()}
    payload["evkey"] = validate_keys(payload["evkey"])
    metadata.update(raw_tracks=ntracks, raw_hits=nhits, hit_counts_match_nhits=True,
                    recorded_hit_census_complete=True)
    return payload, metadata
