"""Explicit host boundary for resident reconstruction; no hit selection or grouping."""
import numpy as np


def _array(a, name, kind, shape=None):
    if np.ma.isMaskedArray(a) and np.any(np.ma.getmaskarray(a)):
        raise ValueError(f"{name} contains masked values")
    a = np.asarray(a)
    if a.dtype.kind not in kind or (shape is not None and a.shape != shape):
        raise ValueError(f"invalid dtype/shape for {name}")
    return a


def _integer(a, name, shape=None):
    a = _array(a, name, "iu", shape)
    if np.any(a < 0) or np.any(a > np.iinfo(np.int64).max):
        raise ValueError(f"{name} outside nonnegative int64")
    return a.astype(np.int64, copy=True)


def prepare(event_keys, sh, conditions):
    """Copy complete measured SH tables and gather matching geometry on the host.

    Accepts the evkey and s_* (prefix removed) columns of run1a_reco's saved
    output, or equivalent measured hits. No association/stock/truth is read.
    This is O(hits) preparation, not part of resident compute timing. Runtime
    invalid numeric geometry/times are retained for device failure statuses;
    malformed keys/indices/schemas raise before geometry indexing.
    """
    keys = _integer(event_keys, "event_keys")
    if keys.ndim != 2 or keys.shape[1] != 3 or len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("event_keys must be unique [events,3]")
    if "valid" in sh:
        raise ValueError("prepare accepts unpadded SH tables only; supply padded arrays to the device API")
    required = ("ev", "sid", "flag", "time", "wdist", "uvar")
    if any(k not in sh for k in required):
        raise ValueError(f"measured hits require {required}")
    ev = _integer(sh["ev"], "ev")
    if ev.ndim != 1:
        raise ValueError("ev must be flat")
    n = len(ev)
    sid, flag = (_integer(sh[k], k, (n,)) for k in ("sid", "flag"))
    if np.any(ev >= len(keys)):
        raise ValueError("hit event index outside complete event census")
    if np.any(((sid >> 10) >= 36) | ((sid & 127) >= 96) | (((sid >> 7) & 7) >= 6)):
        raise ValueError("invalid packed straw ID")
    if np.any(flag > 2**32 - 1):
        raise ValueError("invalid hit flag")
    us = ((sid >> 10) * 6 + ((sid >> 7) & 7)) * 96 + (sid & 127)
    st = conditions["straw"]
    geo = {k: _array(st[k], k, "fiu") for k in ("midx", "midy", "midz", "dirx", "diry", "dirz", "hlen")}
    size = geo["hlen"].size
    if any(v.shape != (size,) for v in geo.values()) or np.any(us >= size):
        raise ValueError("invalid geometry shape or unreferenced straw ID")
    hits = dict(ev=ev, sid=sid, flag=flag, valid=np.ones(n, bool),
                mid=np.column_stack([geo["mid" + k][us] for k in "xyz"]).astype(np.float64),
                direction=np.column_stack([geo["dir" + k][us] for k in "xyz"]).astype(np.float64),
                hlen=geo["hlen"][us].astype(np.float64))
    for k in ("time", "wdist", "uvar"):
        hits[k] = _array(sh[k], k, "fiu", (n,)).astype(np.float64, copy=True)
    return keys, hits
