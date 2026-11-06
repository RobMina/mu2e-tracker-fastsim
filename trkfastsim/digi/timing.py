"""Shared, deterministic event-marker timing and stock microbunch ghost construction."""
from collections.abc import Mapping
from numbers import Integral

import numpy as np


def step_times(st, nev, c, event_timing):
    """Return (float64 marker-relative step times, float64 event lengths [nev]).

    st contains flat, finite step columns including integer ev and sid and time t.
    event_timing is None off spill, or a mapping with pbtime and event_length arrays
    of shape (nev,); supplied values are interpreted as float64 nanoseconds. No
    timing is sampled here. Missing off-spill timing means zero pbtime and the
    conditions' eventWindowLength, preserving the legacy random stream and times.
    On spill timing is mandatory, including empty populations. Add pbtime before
    applying stock fmod and correcting negative remainders; never fold arrivals.
    Invalid populations, StrawIds, nonfinite columns or timing, nonpositive event
    lengths, and a configured digitization end beyond the next marker raise ValueError.
    """
    if not isinstance(nev, Integral) or isinstance(nev, (bool, np.bool_)) or nev < 0:
        raise ValueError("nev must be a nonnegative integer")
    ev = np.asarray(st["ev"])
    if ev.ndim != 1 or ev.dtype.kind not in "iu" or np.any(ev < 0) or np.any(ev >= nev):
        raise ValueError("step event index outside event population or not integer")
    for name, value in st.items():
        value = np.asarray(value)
        if value.shape != ev.shape:
            raise ValueError(f"step {name} must be a flat array matching ev")
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"nonfinite step {name} or nonnumeric column")
    sid = np.asarray(st["sid"])
    if (sid.dtype.kind not in "iu" or np.any(sid < 0) or np.any(sid >= 36 * 1024)
            or np.any(((sid >> 7) & 7) >= 6) or np.any((sid & 127) >= 96)):
        raise ValueError("invalid StrawId")
    mbtime = float(c.mbtime)
    if not np.isfinite(mbtime) or mbtime <= 0:
        raise ValueError("mbtime must be finite and positive")
    time = np.asarray(st["t"], np.float64)
    if event_timing is None:
        if c.onspill:
            raise ValueError("event_timing with pbtime and event_length is required on spill")
        lengths = np.full(nev, c.eventWindowLength, np.float64)
    else:
        if not isinstance(event_timing, Mapping) or not {"pbtime", "event_length"} <= event_timing.keys():
            raise ValueError("event_timing requires pbtime and event_length arrays")
        values = {}
        for name in ("pbtime", "event_length"):
            raw = np.asarray(event_timing[name])
            if raw.shape != (nev,) or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
                raise ValueError(f"event_timing {name} must be a finite array of shape (nev,)")
            values[name] = np.asarray(raw, np.float64)
        lengths = values["event_length"]
        with np.errstate(over="ignore", invalid="ignore"):
            time = time + values["pbtime"][ev]
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("event_length must be finite and positive")
    dend = c.digitizationEnd - c.timeFromProtonsToDRMarker
    if np.any(lengths < dend):
        raise ValueError("digitization window extends past next event window marker")
    if not np.isfinite(time).all():
        raise ValueError("nonfinite marker-relative step time")
    if c.onspill:
        time = np.fmod(time, mbtime)
        time = np.where(time < 0, time + mbtime, time)
    return time, lengths


def ghost_indices(times, mbtime, buffer):
    """Return (source indices, float32 arrivals), ordered originals, early, late.

    times is a flat array of per-end arrivals, rounded to StrawCluster float32.
    Early arrivals strictly below buffer get a +mbtime copy; late arrivals
    strictly above mbtime-buffer get a -mbtime copy. Both tests use the originals,
    never recursively generated ghosts. Copy charge, wire distance, end and step
    provenance with the returned indices. Comparisons and shifts use float64,
    followed by float32 rounding, matching StrawCluster(primary, double deltat).
    """
    raw = np.asarray(times)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError("ghost times must be a finite flat array")
    mbtime, buffer = float(mbtime), float(buffer)
    if not np.isfinite(mbtime) or mbtime <= 0 or not np.isfinite(buffer) or buffer < 0:
        raise ValueError("ghost mbtime must be positive and buffer nonnegative, both finite")
    with np.errstate(over="ignore", invalid="ignore"):
        original = raw.astype(np.float32)
    if not np.isfinite(original).all():
        raise ValueError("ghost times overflow float32")
    t = original.astype(np.float64)
    early = np.flatnonzero(t < buffer)
    late = np.flatnonzero(t > mbtime - buffer)
    indices = np.concatenate((np.arange(len(t), dtype=np.int64), early, late))
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = np.concatenate((original, (t[early] + mbtime).astype(np.float32),
                                  (t[late] - mbtime).astype(np.float32)))
    if not np.isfinite(shifted).all():
        raise ValueError("ghost times overflow float32")
    return indices, shifted
