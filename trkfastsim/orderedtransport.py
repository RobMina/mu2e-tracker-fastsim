"""Opt-in support-only straight transport on explicit ordered intervals; no production replacement."""
import hashlib
from pathlib import Path

import numpy as np

from mu2efastsim import eloss, materials
from . import flighttime, segmenttransport, transportledger as ledger
from .steps import STRAW_GAS


def _material(name):
    return STRAW_GAS if name == STRAW_GAS.name else materials.get(name)


MODES = {"ordered-support-mean-replace-v1": "mean-replace", "ordered-support-sampled-stop-v1": "sampled",
         "ordered-support-deterministic-mean-v1": "sampled"}


def _validate_inputs(identity, origin, direction, kinetic, intervals):
    keys = np.asarray(identity)
    if keys.ndim != 2 or keys.shape[1] != 4 or keys.dtype.kind not in "iu" or (keys < 0).any():
        raise ValueError("identity must be a nonnegative integer (N,4) array")
    if (keys > np.iinfo(np.int64).max).any() or len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("identity must be unique and int64-compatible")
    o, d, ke = np.asarray(origin, float), np.asarray(direction, float), np.asarray(kinetic, float)
    n = len(keys)
    if o.shape != (n, 3) or d.shape != o.shape or ke.shape != (n,):
        raise ValueError("states need (N,3) positions/directions and (N,) kinetic energies")
    if not all(np.isfinite(v).all() for v in (o, d, ke)) or (ke < 0).any():
        raise ValueError("finite states and nonnegative kinetic energies required")
    if not np.allclose(np.linalg.norm(d, axis=1), 1., rtol=0, atol=1e-10):
        raise ValueError("unit directions required")
    if set(intervals.columns) != {"lane", "s0_mm", "s1_mm", "material"}:
        raise ValueError("interval schema mismatch")
    lane = intervals.lane.to_numpy()
    s0, s1 = intervals.s0_mm.to_numpy(float), intervals.s1_mm.to_numpy(float)
    names = intervals.material.to_numpy(str)
    if lane.dtype.kind not in "iu" or ((lane < 0) | (lane >= n)).any():
        raise ValueError("invalid interval lanes")
    if not np.isfinite(s0).all() or not np.isfinite(s1).all() or (s0 < 0).any() or (s1 <= s0).any():
        raise ValueError("intervals require finite positive extent")
    if not np.array_equal(np.lexsort((s0, lane)), np.arange(len(lane))):
        raise ValueError("intervals must be ordered by lane and physical distance")
    if len(lane):
        first = np.r_[True, lane[1:] != lane[:-1]]
        if (s0[first] != 0).any() or not np.array_equal(s0[~first], s1[:-1][~first[1:]]):
            raise ValueError("intervals must partition each ray from zero without gaps or overlaps")
    return keys.astype(np.int64), o, d, ke.copy(), lane.astype(np.int64), s0, s1, names


def transport_intervals(rng, identity, origin_mm, direction, kinetic_MeV, intervals, metadata, *,
                        mode, timing="unknown", max_step_mm=1., fractional_loss=.05,
                        stop_ke_MeV=1., max_steps=10000):
    """Transport support intervals in physical order with one active interval per muon.

    Sampled modes differ only in core overshoot treatment. The explicit deterministic-mean mode uses
    core mean proposals without RNG; sampled_loss_MeV then stores the full deterministic proposal.
    No mode reproduces legacy grouped transport.
    Intervals must partition each supplied ray from zero, including vacuum. Missing lanes have zero extent.
    The cutoff retains residual KE in an absorbing model_stop, including initially stopped lanes in vacuum.
    Positive material slices have unmodeled deposit/escape partitions. Timing is unknown unless explicitly
    linear-in-path; this assumption is persisted. No straws, scattering, field or upstream world are added.
    Returns history, final kinetic energy, position, traveled path and status, aligned with input identities.
    """
    if mode not in MODES or timing not in ("unknown", "linear-in-path"):
        raise ValueError("explicit supported transport mode and timing policy required")
    for value in (max_step_mm, fractional_loss, stop_ke_MeV):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("step, fraction and cutoff must be finite positive values")
    if fractional_loss > 1 or not isinstance(max_steps, (int, np.integer)) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("invalid fractional loss or step cap")
    meta = ledger._json_copy(metadata)
    ledger._validate_manifest(meta)
    keys, o, d, ke, lanes, s0, s1, names = _validate_inputs(identity, origin_mm, direction, kinetic_MeV, intervals)
    if any(name != "vacuum" and name not in {*materials.BY_NAME, STRAW_GAS.name} for name in np.unique(names)):
        raise ValueError("interval material missing from core registry")
    media = {"vacuum": 0}
    for name in np.unique(names[names != "vacuum"]):
        description = ledger.medium(_material(name), cut_MeV=1e9, loss_basis="unrestricted")
        matches = [i for i, item in enumerate(meta["media"]) if item == description]
        if len(matches) != 1:
            raise ValueError(f"manifest needs exactly one matching unrestricted core medium for {name}")
        media[name] = matches[0]
    loss_model = "mean" if mode == "ordered-support-deterministic-mean-v1" else "sampled"
    meta["settings"]["ordered_transport"] = dict(mode=mode, timing=timing, max_step_mm=max_step_mm,
        loss_model=loss_model, proposal_field="sampled_loss_MeV contains full proposed-step loss; deterministic for mean",
        fractional_loss=fractional_loss, stop_ke_MeV=stop_ke_MeV, max_steps=max_steps,
        accounting="primary kinetic debit only; deposition and escape unmodeled",
        path="supplied straight ordered intervals; no scattering",
        source_sha256={"tracker_segmenttransport": hashlib.sha256(Path(segmenttransport.__file__).read_bytes()).hexdigest(),
                       "tracker_orderedtransport": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    n = len(keys)
    traveled = np.zeros(n)
    status = np.where(ke == 0, "exhausted", np.where(ke <= stop_ke_MeV, "model_stop", "transmitted"))
    batches = []

    def append(lane, start, end, kin, kout, sampled, terminal, medium_id):
        length = end - start
        inert = (medium_id == 0) | (length == 0)
        columns = {name: keys[lane, i] for i, name in enumerate(ledger.KEY[:-1])}
        columns.update(segment=np.zeros(len(lane), dtype=np.int64), medium=np.full(len(lane), medium_id, np.int64),
            kind=np.full(len(lane), "physical"), terminal=terminal, path_mm=length,
            entry_mm=o[lane] + d[lane] * start[:, None], exit_mm=o[lane] + d[lane] * end[:, None],
            ke_in_MeV=kin, ke_out_MeV=kout, sampled_loss_MeV=sampled, sample_status=np.full(len(lane), "known"),
            applied_loss_MeV=kin - kout, deposited_MeV=np.where(inert, 0., np.nan),
            escaping_MeV=np.where(inert, 0., np.nan), deposit_status=np.where(inert, "known", "unmodeled"),
            escape_status=np.where(inert, "known", "unmodeled"))
        columns["dt_ns"] = (flighttime.linear_kinetic_dt_ns(length, kin, kout) if timing == "linear-in-path"
                            else np.where(length == 0, 0., np.nan))
        columns["dt_status"] = np.where((length == 0) | (timing == "linear-in-path"), "known", "unknown")
        batches.append(columns)

    no_extent = ~np.isin(np.arange(n), lanes)
    initial = np.flatnonzero(no_extent | (ke <= stop_ke_MeV))
    append(initial, np.zeros(len(initial)), np.zeros(len(initial)), ke[initial], ke[initial], np.zeros(len(initial)),
           np.where(ke[initial] == 0, "exhausted", np.where(ke[initial] <= stop_ke_MeV, "model_stop", "none")), 0)
    if len(lanes):
        first = np.r_[True, lanes[1:] != lanes[:-1]]
        starts = np.flatnonzero(first)
        ends = np.r_[starts[1:], len(lanes)]
        cursor = np.full(n, -1, dtype=np.int64)
        stop_row = np.zeros(n, dtype=np.int64)
        cursor[lanes[starts]], stop_row[lanes[starts]] = starts, ends
        while True:
            active = np.flatnonzero((cursor >= 0) & (cursor < stop_row) & (status == "transmitted"))
            if not len(active):
                break
            index = cursor[active]
            for name in np.unique(names[index]):
                use = active[names[index] == name]
                row = cursor[use]
                if name == "vacuum":
                    append(use, s0[row], s1[row], ke[use], ke[use], np.zeros(len(use)), np.full(len(use), "none"), 0)
                    traveled[use] = s1[row]
                else:
                    result = segmenttransport.transport_segment(rng, _material(name), ke[use], s1[row] - s0[row],
                        tcut_MeV=1e9, stopping=MODES[mode], max_step_mm=max_step_mm, fractional_loss=fractional_loss,
                        stop_ke_MeV=stop_ke_MeV, max_steps=max_steps, loss_model=loss_model)
                    sl = result["slices"]
                    local = sl["lane_index"]
                    append(use[local], s0[row[local]] + sl["s0_mm"], s0[row[local]] + sl["s1_mm"],
                           sl["ke_in_MeV"], sl["ke_out_MeV"], sl["sampled_loss_MeV"], sl["terminal"], media[name])
                    final = result["final"]
                    ke[use], traveled[use], status[use] = final["final_ke_MeV"], s0[row] + final["traveled_mm"], final["status"]
                cursor[use] += 1
    if batches:
        columns = {name: np.concatenate([batch[name] for batch in batches]) for name in ledger.FIELDS}
        order = np.lexsort(tuple(columns[name] for name in reversed(ledger.KEY[:-1])))
        columns = {name: value[order] for name, value in columns.items()}
        count = len(order)
        if count:
            same = np.logical_and.reduce([columns[name][1:] == columns[name][:-1] for name in ledger.KEY[:-1]])
            first = np.r_[True, ~same]
            columns["segment"] = np.arange(count) - np.repeat(np.flatnonzero(first), np.diff(np.r_[np.flatnonzero(first), count]))
        history = ledger.SegmentHistory(columns, meta)
    else:
        history = ledger.SegmentHistory.empty(meta)
    return dict(history=history, final_ke_MeV=ke, position_mm=o + d * traveled[:, None],
                traveled_mm=traveled, status=status)
