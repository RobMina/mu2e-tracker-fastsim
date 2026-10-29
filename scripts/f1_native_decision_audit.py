"""Independent captured native decision quadratics; NOT a fit/status replay (no JAX)."""
import argparse
import hashlib
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

KEYS = ("run", "subrun", "event", "iseed")
ITER_KEYS = KEYS + ("iteration_ordinal",)
RTOL, ATOL = 1e-5, 1e-7
LIMITS = [
    "Only decision_valid captured iterations are numerically audited; unreached decisions are excluded.",
    "The documented full capture has 3449 iterations for only 147 selected targets, not all fits/seeds.",
    "Native (d0,phi0,z0,theta,t0,mom) coordinates and undeweighted old-reference covariance only; no regularization.",
    "Maximum endpoint quadratic > threshold is tested, not complete status/acceptance replay: gap, chi2, covariance and exception precedence are not replayed.",
    "No independent filtering, material, measurement, final KalSeed or full-fit equivalence claim.",
]


def require(ok, message):
    if not np.all(ok):
        raise ValueError(message)


def array(value):
    out = ak.to_numpy(value)
    require(not np.ma.isMaskedArray(out) or not np.any(np.ma.getmaskarray(out)), "missing values")
    return np.asarray(out)


def column(table, name, kind=None):
    out = array(table[name])
    require(out.ndim == 1 and len(out) == len(table), f"{name}: scalar shape")
    if kind:
        require(out.dtype.kind in kind, f"{name}: invalid dtype")
    return out


def keys(table, names=KEYS, unique=True):
    cols = [column(table, name, "iu") for name in names]
    require(all(np.all((x >= 0) & (x <= np.iinfo(np.int64).max)) for x in cols), "invalid integer identity")
    out = np.column_stack(cols).astype(np.int64)
    if unique:
        require(len(np.unique(out, axis=0)) == len(out), "duplicate identities")
    return out


def match_rows(wanted, available):
    """Full-key join, rejecting duplicate available identities and missing matches."""
    require(wanted.ndim == available.ndim == 2 and wanted.shape[1] == available.shape[1], "identity shapes differ")
    require(len(np.unique(available, axis=0)) == len(available), "duplicate available identities")
    if not len(wanted):
        return np.empty(0, dtype=np.int64)
    require(len(available) > 0, "missing identity population")
    dtype = np.dtype([(f"k{i}", np.int64) for i in range(wanted.shape[1])])
    w = np.ascontiguousarray(wanted, dtype=np.int64).view(dtype).ravel()
    a = np.ascontiguousarray(available, dtype=np.int64).view(dtype).ravel()
    order = np.argsort(a)
    loc = np.searchsorted(a[order], w)
    safe = np.minimum(loc, len(a)-1)
    require((loc < len(a)) & (a[order[safe]] == w), "missing full identity join")
    return order[safe]


def dense(table, name, width, finite=True):
    require(table[name].ndim == 2, f"{name}: shape")
    require(array(ak.num(table[name], axis=1)) == width, f"{name}: shape")
    out = array(ak.flatten(table[name], axis=1)).reshape(len(table), width)
    require(out.dtype.kind in "fiu", f"{name}: numeric dtype")
    if finite:
        require(np.isfinite(out), f"nonfinite {name}")
    return out


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quadratic(steps, covariance):
    """Batched s^T inv(C) s; singular/nonfinite operands fail, never add a ridge."""
    steps, covariance = np.asarray(steps), np.asarray(covariance)
    require(steps.shape[-1:] == (6,) and covariance.shape == steps.shape[:-1] + (6, 6), "quadratic operand shapes")
    require(np.isfinite(steps).all() and np.isfinite(covariance).all(), "nonfinite quadratic operands")
    try:
        inverse = np.linalg.inv(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("singular old reference covariance; no regularization") from error
    result = np.einsum("...i,...ij,...j->...", steps, inverse, steps)
    require(np.isfinite(inverse).all() and np.isfinite(result).all(), "nonfinite inverse/quadratic")
    return result


def residual(actual, stored, ids, mask):
    a, s, k = actual[mask], stored[mask], ids[mask]
    delta = np.abs(a-s)
    # Relative-to-stored residual, floored at ATOL so zero references remain JSON-finite.
    relative = delta / np.maximum(np.abs(s), ATOL)
    require(np.isfinite(delta).all() and np.isfinite(relative).all(), "nonfinite residual")
    close = np.isclose(a, s, rtol=RTOL, atol=ATOL)
    return {"n": len(a), "passed": bool(np.all(close)), "mismatches": int((~close).sum()),
            "max_absolute_residual": float(delta.max()) if len(delta) else None,
            "max_relative_residual": float(relative.max()) if len(relative) else None,
            "mismatch_keys": k[~close].tolist()}


def audit(iterations, pieces):
    """Return numeric agreement report; malformed capture raises ValueError (fail closed)."""
    it = iterations
    ik = keys(it, ITER_KEYS)
    require(len(it) > 0, "empty iteration capture")
    for table in (it, pieces):
        require(column(table, "schema_version", "iu") == 1, "unsupported schema")
    require(column(it, "capture_valid", "b"), "invalid iteration capture")
    require(column(it, "capture_error") == "", "iteration capture error")
    valid = column(it, "decision_valid", "b")
    status = column(it, "status", "iu")
    owner = column(pieces, "owner", "iu")
    require((owner == 0) | (owner == 1), "unknown piece owner")
    keys(pieces, ITER_KEYS + ("owner", "piece_index"))
    parent = match_rows(keys(pieces, ITER_KEYS, unique=False), ik)
    for name in ("miter", "iter"):
        require(column(pieces, name, "iu") == column(it, name, "iu")[parent], f"{name}: piece identity mismatch")
    counts = np.column_stack([column(it, name, "iu") for name in ("nold", "ncandidate")])
    require(counts >= 0, "negative piece counts")
    require(column(it, "candidate_present", "b") == (counts[:, 1] > 0), "candidate presence mismatch")
    require(~valid | (column(it, "prior_valid", "b") & (counts[:, 0] > 0) & (counts[:, 1] > 0)), "invalid decision flags")
    index = column(pieces, "piece_index", "iu")
    require(index < counts[parent, owner], "piece_index outside owner")
    for which in (0, 1):
        require(np.bincount(parent[owner == which], minlength=len(it)) == counts[:, which], "piece count mismatch")
    # All shapes are checked; excluded native failure payloads may contain nonfinite operands.
    dense(pieces, "parameters", 6, finite=False)
    dense(pieces, "covariance", 36, finite=False)
    ranges = dense(pieces, "range", 2)
    require(ranges[:, 0] <= ranges[:, 1], "piece range reversed")
    for name in ("front_step", "back_step"):
        require(it[name].ndim == 2, f"{name}: shape")
        require(array(ak.num(it[name])) == 6*valid, f"{name}: shape")
    references = dense(it, "reference_indices", 2)
    require(references.dtype.kind in "iu", "reference_indices: integer dtype")
    t, selected = it[valid], ik[valid]
    n = len(t)
    require(n > 0, "no decision_valid iterations")
    refs = references[valid]
    require((refs >= 0) & (refs < counts[valid, 0, None]), "reference_indices out of bounds")
    pkeys = keys(pieces, ITER_KEYS + ("owner", "piece_index"))
    repeated = np.repeat(selected, 2, axis=0)
    old_rows = match_rows(np.column_stack([repeated, np.zeros(2*n, dtype=int), refs.ravel()]), pkeys)
    ends = np.column_stack([np.zeros(n, dtype=int), counts[valid, 1]-1])
    new_rows = match_rows(np.column_stack([repeated, np.ones(2*n, dtype=int), ends.ravel()]), pkeys)
    old, new = pieces[old_rows], pieces[new_rows]
    raw = (dense(new, "parameters", 6) - dense(old, "parameters", 6)).reshape(n, 2, 6)
    require(np.isfinite(raw), "nonfinite raw step")
    stored_step = np.stack([dense(t, "front_step", 6), dense(t, "back_step", 6)], axis=1)
    step_agree = np.all(raw == stored_step, axis=(1, 2))
    calculated = quadratic(raw, dense(old, "covariance", 36).reshape(n, 2, 6, 6))
    stored = np.column_stack([column(t, name, "fiu") for name in ("front_chi2", "back_chi2")])
    threshold = column(t, "threshold", "fiu")
    require(np.isfinite(stored).all() and np.isfinite(threshold).all(), "nonfinite decision metric/threshold")
    exceeds = calculated.max(axis=1) > threshold
    stored_exceeds = stored.max(axis=1) > threshold
    agree = exceeds == stored_exceeds
    def summary(mask):
        return {"n_decisions": int(mask.sum()),
                "front": residual(calculated[:, 0], stored[:, 0], selected, mask),
                "back": residual(calculated[:, 1], stored[:, 1], selected, mask),
                "threshold_exceedance": {"agreements": int((agree & mask).sum()),
                    "mismatches": int((~agree & mask).sum()),
                    "recomputed_exceeds": int((exceeds & mask).sum()),
                    "captured_exceeds": int((stored_exceeds & mask).sum()),
                    "mismatch_keys": selected[mask & ~agree].tolist()}}
    all_decisions = summary(np.ones(n, dtype=bool))
    by_status = {}
    # Status groups, never event/iteration loops. Codes are recorded, not independently predicted.
    for code in np.unique(status):
        group = summary(status[valid] == code)
        group.update(n_iterations=int((status == code).sum()), n_excluded=int(((status == code) & ~valid).sum()))
        by_status[str(int(code))] = group
    success = np.isin(status, [0, 1])  # native converged/unconverged usable statuses
    groups = {}
    for name, mask in (("usable_status_0_1", success), ("failure_status_other", ~success)):
        groups[name] = summary(mask[valid])
        groups[name].update(n_iterations=int(mask.sum()), n_excluded=int((mask & ~valid).sum()))
    return {"passed": bool(step_agree.all() and all_decisions["front"]["passed"] and all_decisions["back"]["passed"] and agree.all()),
            "n_iterations": len(it), "n_targets": len(np.unique(ik[:, :4], axis=0)),
            "n_decision_valid": n, "n_excluded_decision_invalid": int((~valid).sum()),
            "rtol": RTOL, "atol": ATOL, "relative_residual_definition": "abs(recomputed-captured)/max(abs(captured),atol)",
            "regularization": "none", "raw_steps": {"exact": bool(step_agree.all()), "mismatch_keys": selected[~step_agree].tolist()},
            "all_decisions": all_decisions, "by_recorded_status": by_status, "recorded_status_groups": groups,
            "mismatch_key_fields": list(ITER_KEYS), "full_status_replay": False,
            "full_numerical_validation": False, "stock_exact_fit": False, "limits": LIMITS}


def highest(parent, name):
    candidates = [key for key in parent.keys(cycle=True, recursive=False) if key.rsplit(";", 1)[0] == name]
    require(bool(candidates), f"missing tree {name}")
    return max(candidates, key=lambda key: int(key.rsplit(";", 1)[1]))


def read_tree(root, directory, name, cycles, fields=None):
    dkey = highest(root, directory)
    parent = root[dkey]
    tkey = highest(parent, name)
    cycles[f"{directory}/{name}"] = f"{dkey}/{tkey}"
    return parent[tkey].arrays(fields)


def load_evidence(check, metadata):
    """Verify all four hashes, full passivity flags, highest-cycle census and full target join."""
    metadata.update(check=str(Path(check).resolve()), check_sha256=digest(check))
    gate = json.loads(Path(check).read_text())
    flags = ("passed", "final_capture_gate", "complete_event_coverage", "full_target_coverage", "recorded_capture_equivalent")
    require(all(gate.get(k) is True for k in flags) and gate.get("smoke_only") is False, "requires full capture passivity gate")
    metadata["inputs"] = gate["inputs"]
    for name in ("control", "capture", "baseline", "targets"):
        source = gate["inputs"][name]
        require(digest(source["path"]) == source["sha256"], "evidence hash changed: " + name)
    metadata["all_four_evidence_hashes_verified"] = True
    target = np.asarray(json.loads(Path(gate["inputs"]["targets"]["path"]).read_text())["keys"])
    require(target.ndim == 2 and target.shape[1] == 4 and len(target) > 0 and target.dtype.kind in "iu", "malformed target keys")
    require(np.all(target >= 0) and len(np.unique(target, axis=0)) == len(target), "invalid/duplicate target keys")
    tables, cycles = {}, {}
    for name in ("control", "capture", "baseline"):
        cycles[name] = {}
        with uproot.open(gate["inputs"][name]["path"]) as root:
            tables[name] = {"events": read_tree(root, "trkdump", "events", cycles[name], list(KEYS[:3])),
                            "trace": read_tree(root, "kktrace", "trace", cycles[name], list(KEYS) + (["iteration_capture_expected", "iteration_capture_written", "iteration_capture_requested", "iteration_capture_valid", "iteration_capture_errors"] if name == "capture" else []))}
            if name == "capture":
                for tree in ("iterations", "pieces"):
                    tables[name][tree] = read_tree(root, "kktrace", tree, cycles[name])
    metadata["highest_cycles"] = cycles
    captured = tables["capture"]
    event_keys, seed_keys = keys(captured["events"], KEYS[:3]), keys(captured["trace"])
    require(len(event_keys) == gate["n_events"] and len(seed_keys) == gate["n_seeds"], "highest-cycle event/seed census differs")
    for name in ("control", "baseline"):
        require(np.array_equal(event_keys, keys(tables[name]["events"], KEYS[:3])), "highest-cycle event identities differ")
        other = keys(tables[name]["trace"])
        require(len(other) == len(seed_keys), "highest-cycle trace census differs")
        match_rows(seed_keys, other)
    it, pieces = captured["iterations"], captured["pieces"]
    ik = keys(it, ITER_KEYS)
    require(len(it) == gate["n_iterations"], "highest-cycle iteration census differs")
    actual_targets = np.unique(ik[:, :4], axis=0)
    require(len(actual_targets) == len(target) == gate["n_targets"] == gate["n_targets_found"], "target census differs")
    match_rows(target, actual_targets)
    parent = match_rows(ik[:, :4], seed_keys)
    match_rows(seed_keys[:, :3], event_keys)
    counts = np.bincount(parent, minlength=len(seed_keys))
    trace = captured["trace"]
    for field in ("iteration_capture_expected", "iteration_capture_written"):
        require(counts == column(trace, field, "iu"), f"{field} mismatch")
    requested = np.zeros(len(trace), dtype=bool)
    requested[match_rows(target, seed_keys)] = True
    require(column(trace, "iteration_capture_requested", "b") == requested, "target capture requests differ")
    require(column(trace, "iteration_capture_valid", "b") == requested, "capture validity differs")
    require(column(trace, "iteration_capture_errors") == "", "trace capture error")
    order = np.lexsort((ik[:, 4], parent))
    require(ik[order, 4] == np.arange(len(it)) - np.repeat(np.cumsum(counts)-counts, counts), "iteration ordinal gaps")
    metadata.update(n_events=len(event_keys), n_seeds=len(seed_keys), target_limited=True,
                    all_captured_iterations_included=True)
    return it, pieces


def run(args):
    """Write exclusive fresh JSON, including explicit failures; return report."""
    with Path(args.out).open("x", encoding="utf-8") as stream:
        metadata = {}
        try:
            it, pieces = load_evidence(args.check, metadata)
            report = audit(it, pieces)
        except (ValueError, KeyError, OSError, TypeError, IndexError, AttributeError, OverflowError) as error:
            report = {"passed": False, "error": f"{type(error).__name__}: {error}",
                      "full_status_replay": False, "full_numerical_validation": False,
                      "stock_exact_fit": False, "limits": LIMITS}
        report.update(metadata)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="fresh exclusive JSON output (also on failure)")
    report = run(parser.parse_args())
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
