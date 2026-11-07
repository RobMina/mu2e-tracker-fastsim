"""Strict host I/O and provenance for the Run1A validation driver, not a response kernel."""
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

KEYS = ("run", "subrun", "event")
FIELDS = ("sid", "tdc0", "tdc1", "tot0", "tot1", "pmp", "nadc", "adc", "flags")


def array(value, name, kind="iu", shape=None, nonnegative=True):
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ValueError(f"{name} contains missing values")
    a = np.asarray(value)
    if a.dtype.kind not in kind or (shape is not None and a.shape != shape):
        raise ValueError(f"{name} must contain {'integers' if kind == 'iu' else 'real numbers'} with shape {shape}")
    if not np.isfinite(a).all() or (nonnegative and np.any(a < 0)):
        raise ValueError(f"{name} contains nonfinite or negative values")
    if a.dtype.kind in "iu" and np.any(a > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds int64")
    return a


def read_table(path, tree="trkdigi/events", *, require_flags=True):
    with uproot.open(path) as root:
        rows = root[tree].arrays(library="ak")
    columns = [array(ak.to_numpy(rows[k]), k) for k in KEYS]
    keys = np.column_stack(columns)
    if keys.ndim != 2 or keys.shape[1] != 3 or np.any(keys > np.iinfo(np.uint32).max):
        raise ValueError("event keys must be uint32-domain integer run/subrun/event triples")
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("duplicate reference event keys")
    if not len(keys):
        raise ValueError("reference census has no events")
    for name in ("spill", "pbt", "pbtime"):
        if name in rows.fields and np.any(array(ak.to_numpy(rows[name]), name, "fiu", (len(keys),)) != 0):
            raise ValueError("off-spill reference requires spill and bunch time zero")
    counts = array(ak.to_numpy(ak.num(rows["d_sid"])), "digi counts", shape=(len(keys),)).astype(np.int64)
    fields = FIELDS if require_flags else tuple(k for k in FIELDS if k != "flags")
    for name in fields:
        values = rows[f"d_{name}"]
        if ak.any(ak.is_none(values, axis=0)) or ak.any(ak.is_none(values, axis=1)):
            raise ValueError(f"d_{name} contains missing values")
    flat = {}
    for name in fields:
        values = rows[f"d_{name}"]
        flat[name] = array(ak.to_numpy(ak.flatten(values)), f"d_{name}")
        if name != "adc" and not np.array_equal(ak.to_numpy(ak.num(values)), counts):
            raise ValueError(f"d_{name} jagged lengths differ from d_sid")
    sid = flat["sid"]
    if np.any(((sid >> 10) >= 36) | ((sid & 127) >= 96) | (((sid >> 7) & 7) >= 6)):
        raise ValueError("invalid packed straw ID")
    if not np.array_equal(ak.to_numpy(ak.sum(rows["d_nadc"], axis=1)), ak.to_numpy(ak.num(rows["d_adc"]))):
        raise ValueError("d_nadc does not account for every ADC sample per event")
    flat["ev"] = np.repeat(np.arange(len(keys)), counts)
    return keys, flat


def rectangular(raw, nsamp):
    if isinstance(nsamp, bool) or not isinstance(nsamp, (int, np.integer)) or nsamp <= 0:
        raise ValueError("waveform length must be a positive integer")
    if np.any(raw["nadc"] != nsamp):
        raise ValueError("saved waveform lengths differ from conditions")
    result = dict(ev=raw["ev"], sid=raw["sid"], pmp=raw["pmp"], adc=raw["adc"].reshape(len(raw["ev"]), nsamp))
    result.update({name: np.column_stack([raw[f"{name}{i}"] for i in range(2)]) for name in ("tdc", "tot")})
    return result


def write_report(path, report):
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(text)


def validate_comparison(stock, digi, nevents, window_ns, tdc_lsb_ns):
    if isinstance(nevents, bool) or not isinstance(nevents, (int, np.integer)) or nevents <= 0:
        raise ValueError("nevents must be a positive integer")
    if not np.isfinite(window_ns) or window_ns <= 0 or not np.isfinite(tdc_lsb_ns) or tdc_lsb_ns <= 0:
        raise ValueError("window and TDC LSB must be finite positive")
    for d in (stock, digi):
        ev = array(d["ev"], "ev")
        if ev.ndim != 1 or np.any(ev >= nevents):
            raise ValueError("event index outside complete census")
        n = len(ev)
        for k in ("sid", "pmp"):
            array(d[k], k, shape=(n,), nonnegative=False)
    n = len(stock["ev"])
    for k in ("tdc0", "tdc1", "tot0", "tot1", "nadc"):
        array(stock[k], k, shape=(n,), nonnegative=k == "nadc")
    adc = array(stock["adc"], "adc", nonnegative=False)
    if adc.ndim != 1 or np.sum(stock["nadc"], dtype=np.float64) != len(adc):
        raise ValueError("stock ADC count mismatch")
    n = len(digi["ev"])
    for k in ("tdc", "tot"):
        array(digi[k], k, shape=(n, 2), nonnegative=False)
    adc = array(digi["adc"], "adc", nonnegative=False)
    if adc.ndim != 2 or adc.shape[0] != n:
        raise ValueError("fast ADC shape mismatch")
