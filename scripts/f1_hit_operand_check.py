"""Validate passive straw CA/residual snapshots and native information; NOT fit closure."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from f1_native_decision_audit import (ATOL, RTOL, KEYS, ITER_KEYS, array, column,
                                    dense, digest, keys, load_evidence, match_rows,
                                    read_tree, require)

# KKFitConfig::tpocaPrec is fhicl::Atom<float>, promoted to double in KKFit::tprec_.
NATIVE_CA_PRECISION = float(np.float32(1e-4))
HIT_KEYS = ITER_KEYS + ("effect_index",)
ORIGINAL_KEYS = {"seed_contract": KEYS, "iterations": ITER_KEYS,
                 "pieces": ITER_KEYS + ("owner", "piece_index"), "effects": HIT_KEYS}
LIMITS = ["Captured native prediction derivatives and reference parameters only; not independent CA or residual physics.",
          "All captured straw effects included, including inactive/unusable and native failure iterations.",
          "Exact original-tree passivity requires identical full-key row order; reordered captures are rejected.",
          "No tolerance floors, regularization, global correction, full-fit or full numerical closure."]


def boolean(table, name):
    try:
        value = column(table, name, "biu")
    except TypeError as error:
        raise ValueError(name + ": expected bool/0/1 scalar column") from error
    require((value == 0) | (value == 1), name + ": expected bool/0/1")
    return value.astype(bool)


def metric(actual, stored, identities):
    """Keep every hit/component in the denominator, including arithmetic failures."""
    require(actual.shape == stored.shape, "information shapes differ")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.abs(actual - stored)
        finite = np.isfinite(actual) & np.isfinite(stored) & np.isfinite(delta)
        close = finite & np.isclose(actual, stored, rtol=RTOL, atol=ATOL)
    bad = (~close).reshape(len(identities), -1)
    per_hit = bad.sum(axis=1)
    good_delta = delta[finite]
    return {"passed": bool(close.all()), "n_hits": len(identities),
            "n_components": int(actual.size), "n_failed_components": int(bad.sum()),
            "n_nonfinite_components": int((~finite).sum()),
            "n_failed_hits": int((per_hit > 0).sum()),
            "max_finite_absolute_residual": float(good_delta.max()) if good_delta.size else None,
            "failed_hit_keys": identities[per_hit > 0].tolist(),
            "per_hit_failed_components": per_hit.tolist()}


def validate(rows, effects, iterations):
    """Vectorized full-population audit. Malformed snapshots raise ValueError.

    Numerical disagreement/nonfinite reconstructed or saved information is reported,
    never removed from the comparison denominator. No per-hit Python loops.
    """
    rk, ek, ik = keys(rows, HIT_KEYS), keys(effects, HIT_KEYS), keys(iterations, ITER_KEYS)
    require(len(iterations) > 0 and len(rows) > 0, "empty operand/iteration capture")
    for table in (rows, effects, iterations):
        require(column(table, "schema_version", "iu") == 1, "unsupported schema")
    require(column(iterations, "capture_valid", "b"), "invalid iteration capture")
    require(column(iterations, "capture_error") == "", "iteration capture error")
    require(column(iterations, "effects_captured", "b"), "effects not captured")
    ep = match_rows(ek[:, :5], ik)
    counts = column(iterations, "neffects", "iu")
    require(counts >= 0, "negative effect count")
    require(ek[:, 5] < counts[ep], "effect_index outside count")
    require(np.bincount(ep, minlength=len(iterations)) == counts, "effect count mismatch")
    for field in ("miter", "iter"):
        require(column(effects, field, "iu") == column(iterations, field, "iu")[ep], field + ": effect identity mismatch")
    straw = column(effects, "identity_kind", "iu") == 1
    expected = effects[straw]
    require(column(expected, "kind", "iu") == 1, "straw kind must be 1")
    require(len(rows) == len(expected), "operand population count mismatch")
    join = match_rows(rk, ek[straw])
    e = expected[join]
    for field in ("input_index", "shindex", "strawid", "miter", "iter"):
        left, right = column(rows, field, "iu"), column(e, field, "iu")
        require((left >= 0) & (left == right), field + ": operand identity mismatch")
    for field in ("active", "in_fit"):
        require(boolean(rows, field) == boolean(e, field), field + ": operand flag mismatch")
    for field in ("identity_valid", "payload_supported", "has_reference", "has_weight"):
        require(column(e, field, "b"), "invalid straw " + field)
    for field in ("constrain_tot", "constrain_long", "constrain_absdrift"):
        boolean(rows, field)
    state = column(rows, "hit_state", "iu")
    require((state >= -3) & (state <= 1), "invalid hit_state")
    require(boolean(rows, "active") == (state > -2), "active/hit_state mismatch")
    status = column(rows, "ca_status", "iu")
    require((status >= 0) & (status <= 6), "invalid ca_status")
    usable = boolean(rows, "ca_usable")
    require(usable == (status < 3), "ca_usable/status mismatch")
    precision = column(rows, "precision", "fiu")
    require(np.isfinite(precision) & (precision == NATIVE_CA_PRECISION),
            "unsupported CA precision (requires float32(1e-4) promoted to double)")
    for field, width in (("particle_poca", 4), ("sensor_poca", 4),
                         ("particle_direction", 3), ("sensor_direction", 3),
                         ("dDdP", 6), ("dTdP", 6)):
        require(rows[field].ndim == 2, field + ": shape")
        require(array(ak.num(rows[field], axis=1)) == width * usable, field + ": usable/empty shape")
        dense(rows[usable], field, width)
    for field in ("doca", "doca_variance", "toca_variance", "lsign"):
        value = column(rows, field, "fiu")
        require(np.isfinite(value[usable]), "nonfinite usable " + field)
        require(np.isnan(value[~usable]), "unusable " + field + " must be NaN")
    require(np.isin(column(rows, "lsign")[usable], [-1, 1]), "invalid lsign")
    require(dense(rows[usable], "particle_poca", 4)[:, 3] == column(e, "time", "fiu")[usable],
            "particle TOCA differs from exact effect time")
    scale = column(rows, "variance_scale", "fiu")
    require(np.isfinite(scale) & (scale > 0), "invalid variance_scale")
    rp = match_rows(rk[:, :5], ik)
    low, high = np.full(len(iterations), np.inf), np.full(len(iterations), -np.inf)
    np.minimum.at(low, rp, scale)
    np.maximum.at(high, rp, scale)
    represented = np.bincount(rp, minlength=len(iterations)) > 0
    require(low[represented] == high[represented], "variance_scale differs within iteration")
    require(rows["row_active"].ndim == 2 and np.all(array(ak.num(rows["row_active"], axis=1)) == 3), "row_active: shape")
    active = array(ak.flatten(rows["row_active"], axis=1)).reshape(len(rows), 3)
    require(active.dtype.kind in "biu" and np.all((active == 0) | (active == 1)), "invalid row_active")
    active = active.astype(bool)
    value, mvar, pvar = (dense(rows, field, 3) for field in ("row_value", "row_mvar", "row_pvar"))
    require(mvar[active] > 0, "nonpositive active measurement variance")
    require(pvar[active] >= 0, "negative active parameter variance")
    g = dense(rows, "row_derivative", 18, finite=False).reshape(-1, 3, 6)
    require(np.isfinite(g[active]), "nonfinite active derivative")
    require(np.isnan(g[~active]), "inactive derivative must be all NaN")
    # Mask BEFORE arithmetic: 0*NaN is not zero. pvar is deliberately NOT in the weight.
    g = np.where(active[..., None], g, 0.)
    value = np.where(active, value, 0.)
    safe_mvar = np.where(active, mvar, 1.)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        denominator = safe_mvar * scale[:, None]
        invvar = np.divide(1., denominator, out=np.zeros_like(safe_mvar, dtype=float), where=active)
        arithmetic_bad = np.any(active & (~np.isfinite(denominator) | (denominator <= 0) | ~np.isfinite(invvar)), axis=1)
        weight = np.einsum("nri,nr,nrj->nij", g, invvar, g)
        q = dense(e, "reference_parameters", 6, finite=False)
        vector = np.einsum("nij,nj->ni", weight, q) + np.einsum("nri,nr,nr->ni", g, value, invvar)
    weight = np.where(arithmetic_bad[:, None, None], np.nan, weight)
    vector = np.where(arithmetic_bad[:, None], np.nan, vector)
    wreport = metric(weight, dense(e, "weight_matrix", 36, finite=False).reshape(-1, 6, 6), rk)
    breport = metric(vector, dense(e, "weight_vector", 6, finite=False), rk)
    return {"schema_version": 1, "passed": wreport["passed"] and breport["passed"],
            "n_hits": len(rows), "n_expected_hits": len(expected), "n_iterations": len(iterations),
            "ca_precision": NATIVE_CA_PRECISION, "ca_precision_exact": True,
            "n_unusable_ca": int((~usable).sum()), "n_inactive_hits": int((~boolean(rows, "active")).sum()),
            "n_active_residual_rows": int(active.sum()), "n_excluded_hits": 0,
            "n_variance_arithmetic_failures": int(arithmetic_bad.sum()),
            "rtol": RTOL, "atol": ATOL, "weight_matrix": wreport, "weight_vector": breport,
            "hit_keys": rk.tolist(), "hit_key_fields": list(HIT_KEYS),
            "information_definition": "W=sum(g*g^T/(mvar*scale)); b=Wq+sum(g*value/(mvar*scale)); native prediction g",
            "full_closure": False, "full_numerical_validation": False, "limits": LIMITS}


def exact_originals(current, reference):
    """All fields, highest-cycle array snapshots, exact full keys and exact NaN-aware values."""
    summary = {}
    for name, identity in ORIGINAL_KEYS.items():
        a, b = current[name], reference[name]
        require(np.array_equal(keys(a, identity), keys(b, identity)), name + ": original full keys/order differ")
        require(set(ak.fields(a)) == set(ak.fields(b)), name + ": original field set differs")
        for field in ak.fields(a):
            x, y = a[field], b[field]
            require(ak.type(x) == ak.type(y), name + "/" + field + ": type differs")
            for axis in range(x.ndim):
                require(not ak.any(ak.is_none(x, axis=axis), axis=None) and not ak.any(ak.is_none(y, axis=axis), axis=None),
                        name + "/" + field + ": missing original value")
            for axis in range(1, x.ndim):
                require(ak.almost_equal(ak.num(x, axis=axis), ak.num(y, axis=axis), rtol=0, atol=0),
                        name + "/" + field + ": shape differs")
            xx, yy = array(ak.flatten(x, axis=None)), array(ak.flatten(y, axis=None))
            require(xx.dtype == yy.dtype and np.array_equal(xx, yy, equal_nan=xx.dtype.kind in "fc"),
                    name + "/" + field + ": original values differ")
        summary[name] = {"passed": True, "n_rows": len(a), "n_fields": len(ak.fields(a))}
    return summary


def run(args):
    """Fail closed with an exclusive report; never overwrite evidence or prior reports."""
    with Path(args.out).open("x", encoding="utf-8") as stream:
        metadata = {"full_closure": False, "full_numerical_validation": False, "limits": LIMITS}
        watched = {}
        report = {"passed": False}
        try:
            sources = (Path(__file__).resolve(), Path(__file__).with_name("f1_native_decision_audit.py").resolve(),
                       Path(__file__).resolve().parents[1]/"art/operand_capture/HitOperandCapture.hh")
            metadata["source_sha256"] = {str(path): digest(path) for path in sources}
            watched.update(metadata["source_sha256"])
            reference_path = str(Path(args.reference_capture).resolve())
            reference_hash = digest(reference_path)
            watched[reference_path] = reference_hash
            metadata["reference_capture"] = {"path": reference_path, "sha256": reference_hash}
            # Normal passivity/coverage/hash gate must pass BEFORE operand analysis.
            it, pieces = load_evidence(args.check, metadata)
            require(reference_hash != metadata["inputs"]["capture"]["sha256"], "reference capture must be independent of current capture")
            watched[str(Path(args.check).resolve())] = metadata["check_sha256"]
            for source in metadata["inputs"].values():
                watched[str(Path(source["path"]).resolve())] = source["sha256"]
            current = {"iterations": it, "pieces": pieces}
            with uproot.open(metadata["inputs"]["capture"]["path"]) as root:
                for name in ("seed_contract", "effects", "hit_operands"):
                    current[name] = read_tree(root, "kktrace", name, metadata["highest_cycles"]["capture"])
            with uproot.open(metadata["inputs"]["control"]["path"]) as root:
                disabled = read_tree(root, "kktrace", "hit_operands", metadata["highest_cycles"]["control"])
            require(len(disabled) == 0, "capture-off hit operand sidecar is not empty")
            metadata["capture_off_hit_operands_empty"] = True
            metadata["highest_cycles"]["reference_capture"] = {}
            with uproot.open(reference_path) as root:
                reference = {name: read_tree(root, "kktrace", name, metadata["highest_cycles"]["reference_capture"])
                             for name in ORIGINAL_KEYS}
            metadata["original_tree_passivity"] = exact_originals(current, reference)
            report = validate(current["hit_operands"], current["effects"], it)
        except (ValueError, KeyError, OSError, TypeError, IndexError, AttributeError, OverflowError) as error:
            report = {"passed": False, "error": f"{type(error).__name__}: {error}"}
        finally:
            # Rehash on success AND failure. Check the normal gate's inputs even if it failed partway.
            if "check_sha256" in metadata:
                watched[str(Path(args.check).resolve())] = metadata["check_sha256"]
            for source in metadata.get("inputs", {}).values():
                if isinstance(source, dict) and "path" in source and "sha256" in source:
                    watched[str(Path(source["path"]).resolve())] = source["sha256"]
            try:
                end = {path: digest(path) for path in watched}
                metadata["end_sha256"] = end
                metadata["hashes_unchanged"] = end == watched
                require(end == watched, "evidence/source changed during audit")
            except (OSError, ValueError) as error:
                report["passed"] = False
                report["integrity_error"] = f"{type(error).__name__}: {error}"
                metadata["hashes_unchanged"] = False
        report.update(metadata)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-capture", type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
