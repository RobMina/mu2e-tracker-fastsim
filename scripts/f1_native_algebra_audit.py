"""Batched native-input weight/effect replay; not F1 effect algebra or fit closure."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from f1_native_decision_audit import (ATOL, RTOL, ITER_KEYS, array, column, dense,
                                    keys, load_evidence, match_rows, read_tree, require)

INVERSE = "lu"
INVERSE_STATS = {"n_cholesky_success": 0, "n_cholesky_fallback": 0}
CHOL_FACTOR_LIMIT = ("chol-factor uses actual finite positive Cholesky pivots on mirrored lower storage; "
                     "failed lanes use NumPy LU on the original symmetric operand, without regularization. "
                     "This is arithmetic sensitivity, not ROOT bit parity or Bunch-Kaufman.")

LIMITS = ["Native (d0,phi0,z0,theta,t0,mom) captured inputs only; no coordinate wrapping.",
          "Only decision_valid candidate-present iterations replayed; exclusions counted explicitly.",
          "Not independent F1 measurement/material algebra, full-fit closure, status or acceptance replay.",
          "No regularization; target-limited capture, not all fits/seeds.",
          "NumPy LU and optional eigentest/Cholesky sensitivity path are not ROOT InvertChol/Bunch-Kaufman bit parity.",
          "Strict component tolerance and error-normalized residuals are separate diagnostics; neither is silently relaxed."]


def _cholesky_factor6(sym):
    """Actual batched 6x6 lower factor; success follows computed pivots, not eigenvalues.

    Only the six component indices are looped over, never the batch lanes.
    Failed lanes use harmless scratch values; the original operand is untouched.
    """
    lower = np.zeros_like(sym)
    success = np.ones(sym.shape[:-2], dtype=bool)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for j in range(6):
            pivot = sym[..., j, j] - np.sum(lower[..., j, :j] ** 2, axis=-1)
            success &= np.isfinite(pivot) & (pivot > 0)
            lower[..., j, j] = np.sqrt(np.where(success, pivot, 1.))
            column = (sym[..., j+1:, j] - np.einsum(
                "...ik,...k->...i", lower[..., j+1:, :j], lower[..., j, :j])) / lower[..., j, j, None]
            lower[..., j+1:, j] = np.where(success[..., None], column, 0.)
    return lower, success


def _inverse_chol_factor(matrix):
    """Lower-storage arithmetic diagnostic, not ROOT InvertChol/Bunch-Kaufman parity.

    Return mirrored lower storage for both the factor and unregularized LU paths.
    Arbitrary batch dimensions (including no batch or empty batches) are supported.
    """
    matrix = np.asarray(matrix, dtype=float)
    require(matrix.shape[-2:] == (6, 6), "chol-factor requires (...,6,6) inverse operand")
    sym = np.tril(matrix) + np.swapaxes(np.tril(matrix, -1), -1, -2)
    flat = sym.reshape(-1, 6, 6)
    lower, success = _cholesky_factor6(flat)
    INVERSE_STATS["n_cholesky_success"] += int(success.sum())
    INVERSE_STATS["n_cholesky_fallback"] += int((~success).sum())
    result = np.empty_like(flat)
    # Forward substitution for the triangular inverse, vectorized over successful lanes.
    lower = lower[success]
    li = np.zeros_like(lower)
    eye = np.eye(6)
    for j in range(6):
        li[..., j, :] = (eye[j] - np.einsum(
            "...k,...ki->...i", lower[..., j, :j], li[..., :j, :])) / lower[..., j, j, None]
    result[success] = np.swapaxes(li, -1, -2) @ li
    if (~success).any():
        try:
            fallback = np.linalg.inv(flat[~success])
        except np.linalg.LinAlgError as error:
            raise ValueError("Cholesky factor failed; fallback singular matrix; no regularization") from error
        require(np.isfinite(fallback), "Cholesky factor failed; nonfinite fallback inverse; no regularization")
        result[~success] = fallback
    result = np.tril(result) + np.swapaxes(np.tril(result, -1), -1, -2)
    require(np.isfinite(result), "nonfinite inverse")
    return result.reshape(matrix.shape)


def inverse(matrix):
    require(np.isfinite(matrix), "nonfinite inverse operand")
    if INVERSE == "chol-factor":
        return _inverse_chol_factor(matrix)
    try:
        if INVERSE == "chol":
            sym = np.tril(matrix) + np.swapaxes(np.tril(matrix, -1), -1, -2)
            positive = np.linalg.eigvalsh(sym)[..., 0] > 0
            safe = np.where(positive[..., None, None], sym, np.eye(6))
            lower = np.linalg.cholesky(safe)
            li = np.linalg.solve(lower, np.broadcast_to(np.eye(6), lower.shape))
            result = np.swapaxes(li, -1, -2) @ li
            fallback = ~positive
            if fallback.any():
                result[fallback] = np.linalg.inv(sym[fallback])
        else:
            result = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("singular matrix; no regularization") from error
    require(np.isfinite(result), "nonfinite inverse")
    return result


def parameters(w, b):
    c = inverse(w)
    x = np.einsum("...ij,...j->...i", c, b)
    require(np.isfinite(x), "nonfinite parameters")
    return x, c


def replay(prior_w, prior_b, kind, weights, vectors, changes, noise):
    """Batched effect-slot recurrence with demand-driven weight/parameter conversions."""
    n, slots = kind.shape
    material = kind == 2
    pw = np.zeros((n, slots+1, 6, 6))
    pb = np.zeros((n, slots+1, 6))
    piece = np.cumsum(material, axis=1)
    for direction in (1, -1):
        w, b = prior_w[:, int(direction == -1)].copy(), prior_b[:, int(direction == -1)].copy()
        cached_x, cached_c = np.zeros((n, 6)), np.zeros((n, 6, 6))
        have_parameters = np.zeros(n, bool)
        for j in (range(slots) if direction == 1 else range(slots-1, -1, -1)):
            m = material[:, j]
            rows = np.flatnonzero(m)
            if direction == -1:  # backward-before: downstream side of material
                pw[rows, piece[rows, j]] += w[rows]
                pb[rows, piece[rows, j]] += b[rows]
            need_parameters = m & ~have_parameters
            if need_parameters.any():
                cached_x[need_parameters], cached_c[need_parameters] = parameters(w[need_parameters], b[need_parameters])
            cached_x[m] += direction * changes[m, j]
            cached_c[m] += noise[m, j]
            have_parameters = np.where(kind[:, j] == 0, have_parameters, m)
            if m.any():
                w[m] = inverse(cached_c[m])
                b[m] = np.einsum("nij,nj->ni", w[m], cached_x[m])
            w = w + weights[:, j]
            b = b + vectors[:, j]
            require(np.isfinite(w).all() and np.isfinite(b).all(), "nonfinite recurrence")
            if direction == 1:  # forward-after: downstream side of material
                pw[rows, piece[rows, j]] = w[rows]
                pb[rows, piece[rows, j]] = b[rows]
        if direction == -1:
            pw[:, 0], pb[:, 0] = w + prior_w[:, 0], b + prior_b[:, 0]
    valid = np.arange(slots+1)[None, :] <= material.sum(axis=1)[:, None]
    x, c = parameters(np.where(valid[..., None, None], pw, np.eye(6)), np.where(valid[..., None], pb, 0.))
    return x, c, valid


def residual(actual, stored, ids):
    delta = np.abs(actual-stored).reshape(len(actual), -1)
    relative = delta / np.maximum(np.abs(stored).reshape(len(actual), -1), ATOL)
    require(np.isfinite(delta).all() and np.isfinite(relative).all(), "nonfinite residual")
    bad = ~np.isclose(actual, stored, rtol=RTOL, atol=ATOL).reshape(len(actual), -1)
    mismatch = bad.any(axis=1)
    _, first = np.unique(ids[mismatch, :-1], axis=0, return_index=True)
    return {"passed": bool(not mismatch.any()), "n_pieces": len(actual),
            "n_components_above_tolerance": int(bad.sum()), "n_pieces_above_tolerance": int(mismatch.sum()),
            "max_absolute_residual": float(delta.max()), "max_relative_residual": float(relative.max()),
            "absolute_quantiles": np.quantile(delta, [0., .5, .9, .99, 1.]).tolist(),
            "relative_quantiles": np.quantile(relative, [0., .5, .9, .99, 1.]).tolist(),
            "piece_keys": ids.tolist(), "per_piece_absolute_quantiles": np.quantile(delta, [0., .5, .9, .99, 1.], axis=1).T.tolist(),
            "per_piece_max_relative_residual": relative.max(axis=1).tolist(),
            "per_piece_max_absolute_residual": delta.max(axis=1).tolist(),
            "per_piece_relative_quantiles": np.quantile(relative, [0., .5, .9, .99, 1.], axis=1).T.tolist(),
            "per_piece_components_above_tolerance": bad.sum(axis=1).tolist(),
            "first_mismatch_keys_per_iteration": ids[mismatch][first].tolist()}


def audit(iterations, pieces, effects):
    it = iterations
    ik = keys(it, ITER_KEYS)
    require(len(it) > 0, "empty iteration capture")
    for table in (it, pieces, effects):
        require(column(table, "schema_version", "iu") == 1, "unsupported schema")
    require(column(it, "capture_valid", "b"), "invalid iteration capture")
    require(column(it, "capture_error") == "", "iteration capture error")
    counts = np.column_stack([column(it, f, "iu") for f in ("nold", "ncandidate", "neffects", "npriors", "nprior_weights")])
    require(counts >= 0, "negative counts")
    present, decision = column(it, "candidate_present", "b"), column(it, "decision_valid", "b")
    require(present == (counts[:, 1] > 0), "candidate presence mismatch")
    require(~decision | present, "decision_valid without candidate")
    selected = decision & present
    require(selected.any(), "no eligible decision_valid candidate-present iterations")
    for flag in ("prior_valid", "bounds_valid", "effects_captured"):
        require(column(it, flag, "b")[selected], "invalid " + flag)
    require(counts[selected, 3:] == 2, "initial prior counts must both equal two")
    for field, size, count in (("prior_weight_matrices", 36, 4), ("prior_weight_vectors", 6, 4)):
        require(it[field].ndim == 2, field + ": shape")
        require(array(ak.num(it[field])) == counts[:, count]*size, field + ": count/shape")
    pk = keys(pieces, ITER_KEYS + ("owner", "piece_index"))
    ek = keys(effects, ITER_KEYS + ("effect_index",))
    pp, ep = match_rows(pk[:, :5], ik), match_rows(ek[:, :5], ik)
    for table, parent in ((pieces, pp), (effects, ep)):
        for field in ("miter", "iter"):
            require(column(table, field, "iu") == column(it, field, "iu")[parent], field + ": identity mismatch")
    owner, index = pk[:, 5], pk[:, 6]
    require(np.isin(owner, [0, 1]), "unknown piece owner")
    require(index < counts[pp, owner], "piece_index outside count")
    for which in (0, 1):
        require(np.bincount(pp[owner == which], minlength=len(it)) == counts[:, which], "piece count mismatch")
    require(ek[:, 5] < counts[ep, 2], "effect_index outside count")
    require(np.bincount(ep, minlength=len(it)) == counts[:, 2], "effect count mismatch")
    for field, size in (("parameters", 6), ("covariance", 36)):
        dense(pieces, field, size, finite=False)
    # No per-event loop: scatter all selected effects into padded time-ordered slots.
    e = effects[selected[ep]]
    parent = match_rows(keys(e, ITER_KEYS, unique=False), ik[selected])
    slot = column(e, "effect_index", "iu").astype(np.int64)
    n, slots = int(selected.sum()), int(counts[selected, 2].max())
    time = column(e, "time", "fiu")
    require(np.isfinite(time), "nonfinite effect time")
    order = np.lexsort((slot, parent))
    require((np.diff(parent[order]) != 0) | (np.diff(time[order]) >= 0), "effects not time sorted")
    active = column(e, "active", "b") & column(e, "in_fit", "b")
    kind = column(e, "kind", "iu")
    require(np.isin(kind[active], [1, 2, 4]), "unsupported active effect kind")
    require(column(e, "payload_supported", "b")[active], "unsupported effect payload")
    kinds = np.zeros((n, slots), dtype=int)
    kinds[parent, slot] = np.where(active, kind, 0)
    operands = []
    for field, size, flag, use in (("weight_matrix", 36, "has_weight", active & (kind != 2)),
                                   ("weight_vector", 6, "has_weight", active & (kind != 2)),
                                   ("change_parameters", 6, "has_change", active & (kind == 2)),
                                   ("change_covariance", 36, "has_change", active & (kind == 2))):
        require(column(e, flag, "b")[use], "missing " + flag)
        require(e[field].ndim == 2, field + ": shape")
        require(array(ak.num(e[field])) == size*column(e, flag, "b"), field + ": count/shape")
        shape = (6, 6) if size == 36 else (6,)
        operand = np.zeros((n, slots) + shape)
        operand[parent[use], slot[use]] = dense(e[use], field, size).reshape((-1,) + shape)
        operands.append(operand)
    require(counts[selected, 1] == 1 + (kinds == 2).sum(axis=1), "candidate count must equal 1 + active in-fit materials")
    w = dense(it[selected], "prior_weight_matrices", 72).reshape(n, 2, 6, 6)
    b = dense(it[selected], "prior_weight_vectors", 12).reshape(n, 2, 6)
    x, c, valid = replay(w, b, kinds, *operands)
    rows, indices = np.nonzero(valid)
    ids = np.column_stack([ik[selected][rows], indices])
    captured = pieces[match_rows(np.column_stack([ids[:, :5], np.ones(len(ids), dtype=int), indices]), pk)]
    params = residual(x[valid], dense(captured, "parameters", 6), ids)
    stored_cov = dense(captured, "covariance", 36).reshape(-1, 6, 6)
    cov = residual(c[valid], stored_cov, ids)
    delta = x[valid] - dense(captured, "parameters", 6)
    variance = np.diagonal(stored_cov, axis1=-2, axis2=-1)
    supported = np.all(variance > 0, axis=1)
    normalized = np.abs(delta[supported]) / np.sqrt(variance[supported])
    params["component_names"] = ["d0", "phi0", "z0", "theta", "t0", "momentum"]
    params["max_absolute_by_component"] = np.max(np.abs(delta), axis=0).tolist()
    params["positive_diagonal_pieces"] = int(supported.sum())
    params["max_abs_difference_over_captured_sigma"] = normalized.max(axis=0).tolist()
    params["fraction_within_0p1_sigma"] = (normalized < .1).mean(axis=0).tolist()
    return {"passed": params["passed"] and cov["passed"], "n_iterations": len(it), "n_decision_valid": int(decision.sum()),
            "n_candidate_present": int(present.sum()), "n_replayed": n, "n_excluded_decision_invalid": int((~decision).sum()),
            "n_excluded_candidate_present_decision_invalid": int((present & ~decision).sum()),
            "n_initial_priors_selected": int(counts[selected, 3].sum()), "n_initial_prior_weights_selected": int(counts[selected, 4].sum()),
            "n_effects_total": len(effects), "n_effects_selected": len(e), "n_active_in_fit_effects": int(active.sum()),
            "n_candidate_pieces_replayed": len(ids), "parameters": params, "covariance": cov,
            "full_numerical_validation": False, "full_status_replay": False, "regularization": "none"}


def run(args):
    INVERSE_STATS.update(n_cholesky_success=0, n_cholesky_fallback=0)
    with Path(args.out).open("x", encoding="utf-8") as stream:
        metadata = {"full_numerical_validation": False, "full_status_replay": False, "stock_exact_fit": False,
                    "scope": "native captured-input effect algebra replay, not F1 full effect algebra or closure",
                    "regularization": "none", "inverse": INVERSE, "rtol": RTOL, "atol": ATOL, "limits": LIMITS,
                    "mismatch_key_fields": list(ITER_KEYS) + ["piece_index"], "quantile_levels": [0., .5, .9, .99, 1.],
                    "relative_residual_definition": "abs(recomputed-captured)/max(abs(captured),atol)"}
        try:
            it, pieces = load_evidence(args.check, metadata)
            with uproot.open(metadata["inputs"]["capture"]["path"]) as root:
                effects = read_tree(root, "kktrace", "effects", metadata["highest_cycles"]["capture"])
            report = audit(it, pieces, effects)
        except (ValueError, KeyError, OSError, TypeError, IndexError, AttributeError, OverflowError) as error:
            report = {"passed": False, "error": f"{type(error).__name__}: {error}"}
        if INVERSE == "chol-factor":
            metadata["limits"] = LIMITS + [CHOL_FACTOR_LIMIT]
            metadata["inverse_stats"] = INVERSE_STATS.copy()
        report.update(metadata)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="exclusive fresh JSON, including failures")
    parser.add_argument("--inverse", choices=["lu", "chol", "chol-factor"], default="lu")
    args = parser.parse_args()
    global INVERSE
    INVERSE = args.inverse
    report = run(args)
    summary = {k: v for k, v in report.items() if k not in ("parameters", "covariance")}
    for key in ("parameters", "covariance"):
        if key in report:
            summary[key] = {k: v for k, v in report[key].items() if not k.startswith("per_piece") and k not in ("piece_keys", "first_mismatch_keys_per_iteration")}
    print(json.dumps(summary, indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
