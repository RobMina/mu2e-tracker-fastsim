"""Load only immutable ParameterHit times from a passively validated native capture."""
import hashlib
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

KEYS = ("run", "subrun", "event", "iseed")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def join_keys(wanted, available):
    """Unique full-key join; missing or ambiguous seeds fail closed."""
    wanted, available = np.asarray(wanted), np.asarray(available)
    if any(x.ndim != 2 or x.shape[1] != 4 or x.dtype.kind not in "iu" for x in (wanted, available)):
        raise ValueError("expected integer run/subrun/event/seed keys")
    if any(np.any(x < 0) or len(np.unique(x, axis=0)) != len(x) for x in (wanted, available)):
        raise ValueError("negative or duplicate seed identities")
    dtype = np.dtype([(k, np.int64) for k in KEYS])
    w, a = (np.ascontiguousarray(x, dtype=np.int64).view(dtype).ravel() for x in (wanted, available))
    if not len(a):
        raise ValueError("empty native seed table")
    order = np.argsort(a)
    pos = np.searchsorted(a[order], w)
    safe = np.minimum(pos, len(a) - 1)
    if np.any(pos >= len(a)) or np.any(a[order[safe]] != w):
        raise ValueError("missing native seed identity")
    return order[safe]


def load_times(check_path, baseline_path, keys):
    """Verify the full capture gate and evidence hashes, then identity-join native constraint times."""
    check_path = Path(check_path).resolve()
    gate = json.loads(check_path.read_text())
    flags = ("passed", "final_capture_gate", "complete_event_coverage", "full_target_coverage", "recorded_capture_equivalent")
    if not all(gate.get(k) is True for k in flags) or gate.get("smoke_only") is not False:
        raise ValueError("native times require a successful full capture passivity gate")
    evidence = gate["inputs"]
    for name in ("control", "capture", "baseline", "targets"):
        if digest(evidence[name]["path"]) != evidence[name]["sha256"]:
            raise ValueError("capture evidence hash changed: " + name)
    if digest(baseline_path) != evidence["baseline"]["sha256"]:
        raise ValueError("fit input is not the certified frozen baseline")
    with uproot.open(evidence["capture"]["path"]) as root:
        parent = root["kktrace"]
        cycles = [k for k in parent.keys(cycle=True, recursive=False) if k.split(";")[0] == "seed_contract"]
        cycle = max(cycles, key=lambda k: int(k.rsplit(";", 1)[1]))
        seed = parent[cycle].arrays(list(KEYS) + ["valid", "error", "range_begin", "range_end", "ph_time", "ph_mask"])
    native_keys = np.column_stack([ak.to_numpy(seed[k]) for k in KEYS]).astype(np.int64)
    rows = join_keys(keys, native_keys)
    if len(seed) != gate["n_seeds"]:
        raise ValueError("native seed count differs from certified gate")
    selected = seed[rows]
    if not bool(ak.all(selected.valid & (selected.error == "") & (ak.num(selected.ph_time) == 1))):
        raise ValueError("invalid seed or not exactly one native constraint")
    if not bool(ak.all(ak.num(selected.ph_mask) == 6)):
        raise ValueError("native constraint mask shape differs")
    mask = ak.to_numpy(selected.ph_mask)
    if not np.all(mask == np.array([0, 0, 0, 0, 0, 1])):
        raise ValueError("native constraint is not momentum-only")
    times = np.asarray(ak.to_numpy(ak.flatten(selected.ph_time)), dtype=np.float64)
    begin, end = ak.to_numpy(selected.range_begin), ak.to_numpy(selected.range_end)
    if not np.all(np.isfinite(times) & np.isfinite(begin) & np.isfinite(end) & (begin <= end)):
        raise ValueError("nonfinite or reversed native seed times")
    if not np.array_equal(times, 0.5 * (begin + end)):
        raise ValueError("native immutable constraint differs from constructor midpoint")
    meta = {"check": str(check_path), "check_sha256": digest(check_path), "inputs": evidence,
            "seed_tree": "kktrace/" + cycle, "native_seeds": len(seed), "matched_seeds": len(keys),
            "changed_operand": "immutable constraint time only", "unchanged": ["seed parameters", "seed covariance",
                "initial owner range", "hit and crossing candidates", "priors", "status reference policy",
                "material", "ordering algorithm", "thresholds"], "stock_exact_fit": False}
    return times, meta
