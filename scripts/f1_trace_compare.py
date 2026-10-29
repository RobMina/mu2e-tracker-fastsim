"""Compare actual FIRST F1/native iteration operands, never align fit histories.

Malformed records raise ValueError. Ambiguous semantic identities and unavailable
payloads are explicitly unsupported, not silently equal. Native miter is only an
identity label: its numbering is deliberately not prescribed here.
"""
import awkward as ak
import numpy as np

from trkfastsim.reco import fittrace, nativecompare

RTOL, ATOL = 1e-5, 1e-7
KEYS = ("run", "subrun", "event", "iseed", "iteration_ordinal", "miter", "iter")
LIMITS = [
    "First iteration only; phase order is report order, not proof of causal divergence.",
    "Final metadata is not an iteration-local cluster-membership snapshot.",
    "Crossings are identified by raw strawid only when unique; partners are source shindex.",
    "Material coordinate conversion is first-order, not native finite-shift arithmetic.",
    "No full-fit, candidate, decision, acceptance, geometry or native-exact closure claim.",
]


def _require(ok, message):
    if not np.all(ok):
        raise ValueError(message)


def _array(value):
    a = ak.to_numpy(value) if isinstance(value, (ak.Array, ak.Record)) else np.asarray(value)
    _require(not np.ma.isMaskedArray(a) or not np.any(np.ma.getmaskarray(a)), "missing values")
    return np.asarray(a)


def _col(table, name, kind=None):
    a = _array(table[name])
    _require(a.shape == (len(table),), name + ": scalar shape")
    if kind:
        _require(a.dtype.kind in kind, name + ": invalid dtype")
    return a


def _dense(table, name, shape):
    # Slicing a ragged capture to zero rows loses its trailing payload width.
    if len(table) == 0:
        _require(name in ak.fields(table), "missing " + name)
        return np.empty((0,) + shape, dtype=float)
    a = _array(table[name])
    # Native capture stores matrices flattened, but permit already-shaped synthetic arrays.
    _require(a.shape in ((len(table), int(np.prod(shape))), (len(table),) + shape), name + ": shape")
    return a.reshape((len(table),) + shape)


def _metric(a, b, exact=False):
    a, b = np.asarray(a), np.asarray(b)
    _require(a.shape == b.shape, "comparison shape mismatch")
    finite = np.isfinite(a) & np.isfinite(b)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta = np.abs(a.astype(float) - b.astype(float))
        relative = delta / np.maximum(np.abs(b.astype(float)), ATOL)
        close = (a == b) if exact else np.isclose(a, b, rtol=RTOL, atol=ATOL)
    good = finite & np.isfinite(delta) & np.isfinite(relative)
    bad = ~(good & close)
    locations = np.argwhere(bad)[:12]
    flat = np.flatnonzero(bad)[:12]
    values = lambda v: [float(x) if np.isfinite(x) else None for x in v.ravel()[flat]]
    return {"passed": bool(not bad.any()), "n_components": int(a.size),
            "mismatches": int(bad.sum()), "nonfinite": int((~good).sum()),
            "max_absolute_residual": float(delta[good].max()) if good.any() else None,
            "max_relative_residual": float(relative[good].max()) if good.any() else None,
            "first_mismatch_indices": locations.tolist(), "first_actual_values": values(a),
            "first_native_values": values(b)}


def _match(wanted, available):
    """One-dimensional unique-identity join, returning -1 for absent identities."""
    if not len(available):
        return np.full(len(wanted), -1, dtype=int)
    order = np.argsort(available)
    loc = np.searchsorted(available[order], wanted)
    safe = np.minimum(loc, len(available) - 1)
    return np.where((loc < len(available)) & (available[order[safe]] == wanted), order[safe], -1)


def compare_first(trace, native_iteration, native_old_pieces, native_effects,
                  hit_ids, crossing_ids, seed_hit_mask, crossing_partner):
    """Return JSON-safe phase metrics for one seed's first actual iteration.

    ``trace`` has exactly the per-record shapes from fittrace.shapes, no batch,
    stage or iteration axes. ``owner_count`` is F1's *last piece index*, so the
    required one-piece initial owner has value zero. Native tables must already
    be selected to this iteration (old pieces owner=0); identities are rechecked.
    Hit IDs/partners are source shindex, NOT tensor positions. Crossing IDs are
    raw strawid. Comparison tolerances are fixed at rtol=1e-5, atol=1e-7.
    """
    hit_ids, crossing_ids, seed_hit_mask, crossing_partner = map(
        np.asarray, (hit_ids, crossing_ids, seed_hit_mask, crossing_partner))
    nh, nx = len(hit_ids), len(crossing_ids)
    for name, a, shape, kind in (("hit_ids", hit_ids, (nh,), "iu"),
                                ("crossing_ids", crossing_ids, (nx,), "iu"),
                                ("seed_hit_mask", seed_hit_mask, (nh,), "b"),
                                ("crossing_partner", crossing_partner, (nx,), "iu")):
        _require(a.shape == shape and a.dtype.kind in kind, name + ": shape/dtype")
    _require(np.all(hit_ids >= 0) and np.all(crossing_ids >= 0) and np.all(crossing_partner >= -1),
             "invalid source identities")
    t = {name: _array(trace[name]) for name in fittrace.shapes(nh, nx)}
    for name, shape in fittrace.shapes(nh, nx).items():
        _require(t[name].shape == shape, "trace " + name + ": first-record shape")
    _require(t["valid"] and t["iteration"] == 0, "require valid first F1 iteration")
    _require(t["owner_count"] == 0, "require one old initial F1 piece (owner_count=0)")
    order = t["order"]
    _require(order.dtype.kind in "iu" and np.array_equal(np.sort(order), np.arange(nh + nx + 1)),
             "trace order must be a permutation")
    natural = {k: v[np.argsort(order)] for k, v in t.items() if k.startswith("eff_") or k == "is_material"}
    it, old, e = native_iteration, native_old_pieces, native_effects
    _require(len(it) == 1, "require one native iteration row")
    _require(_col(it, "iter", "iu") == 0, "require native iter==0")
    _require(_col(it, "iteration_ordinal", "iu") == 0, "require native iteration_ordinal==0")
    _require(len(old) == 1 and _col(it, "nold", "iu")[0] == 1, "require one old initial native piece")
    _require(_col(old, "owner", "iu") == 0, "require native old owner=0")
    _require(_col(old, "piece_index", "iu") == 0, "require old piece_index=0")
    for table in (it, old, e):
        _require(_col(table, "schema_version", "iu") == 1, "unsupported native schema")
        for key in KEYS:
            _require(_col(table, key, "iu") == _col(it, key, "iu")[0], "native full identity mismatch: " + key)
    _require(_col(it, "capture_valid", "b")[0], "invalid native capture")
    _require(_col(it, "capture_error")[0] == "", "native capture error")
    _require(_col(it, "effects_captured", "b")[0], "native effects not captured")
    _require(_col(it, "neffects", "iu")[0] == len(e), "native effect count mismatch")
    ei = _col(e, "effect_index", "iu")
    _require(np.array_equal(np.sort(ei), np.arange(len(e))), "duplicate/noncontiguous native effect_index")
    phases = {}

    def phase(name):
        p = {"metrics": {}, "unsupported": []}
        phases[name] = p
        return p

    def compare(p, name, a, b, exact=False):
        p["metrics"][name] = _metric(a, b, exact)

    def transformed(p, fun, args, fields):
        try:
            values = fun(*args)
        except (ValueError, np.linalg.LinAlgError) as error:
            p["metrics"]["transform"] = {"passed": False, "error": str(error)}
            return
        for (name, target), value in zip(fields, values):
            compare(p, name, value, target)

    p = phase("seed_old_owner")
    transformed(p, nativecompare.transform_parameters,
                (t["owner_parameters"][:1], t["owner_covariance"][:1]),
                (("parameters", _dense(old, "parameters", (6,))), ("covariance", _dense(old, "covariance", (6, 6)))))
    compare(p, "range", t["owner_ranges"][:1], _dense(old, "range", (2,)))

    p = phase("effect_membership")
    kind = _col(e, "kind", "iu")
    hits, crossings, constraints = np.flatnonzero(kind == 1), np.flatnonzero(kind == 2), np.flatnonzero(kind == 4)
    unknown = ~np.isin(kind, [1, 2, 4])
    if unknown.any():
        p["unsupported"].append({"reason": "native cluster/unknown effects have no F1 identity mapping",
                                  "effect_indices": ei[unknown].tolist(), "kinds": kind[unknown].tolist()})
    hids = _col(e, "shindex", "iu")[hits]
    xids = _col(e, "strawid", "iu")[crossings]
    ambiguous = False
    for label, ids in (("F1 hit shindex", hit_ids), ("F1 crossing strawid", crossing_ids),
                       ("native hit shindex", hids), ("native crossing strawid", xids)):
        unique, counts = np.unique(ids, return_counts=True)
        if np.any(counts > 1) or np.any(ids < 0):
            ambiguous = True
            p["unsupported"].append({"reason": "ambiguous/invalid " + label,
                                      "duplicate_ids": unique[counts > 1].tolist()})
    if len(constraints) > 1:
        ambiguous = True
        p["unsupported"].append("multiple native constraints cannot map to F1's single constraint slot")
    if "identity_kind" in ak.fields(e):
        ik = _col(e, "identity_kind", "iu")
        if np.any(ik[hits] != 1) or np.any(ik[crossings] != 5) or np.any(ik[constraints] != 2):
            ambiguous = True
            p["unsupported"].append("unsupported hit/crossing/constraint identity_kind")
    if "identity_valid" in ak.fields(e) and not _col(e, "identity_valid", "b").all():
        ambiguous = True
        p["unsupported"].append("native identity_valid=false")
    if "input_index" in ak.fields(e):
        inputs = _col(e, "input_index", "iu")
        for family_rows in (np.r_[hits, constraints], crossings):
            if len(np.unique(inputs[family_rows])) != len(family_rows) or np.any(inputs[family_rows] < 0):
                ambiguous = True
                p["unsupported"].append("duplicate/invalid native generic input identity")
    partner_slot = _match(crossing_partner, hit_ids) if not ambiguous else np.full(nx, -1)
    if np.any((crossing_partner >= 0) & (partner_slot < 0)):
        ambiguous = True
        p["unsupported"].append("crossing partner absent from source hit identities")
    row = np.full(nh + nx + 1, -1, dtype=int)
    expected = np.r_[seed_hit_mask, np.zeros(nx, bool), True]
    if not ambiguous:
        expected[nh:nh + nx] = (partner_slot >= 0) & seed_hit_mask[np.maximum(partner_slot, 0)] if nh else False
        hm, xm = _match(hit_ids, hids), _match(crossing_ids, xids)
        row[:nh] = np.where(hm >= 0, hits[np.maximum(hm, 0)], -1) if len(hits) else -1
        row[nh:nh + nx] = np.where(xm >= 0, crossings[np.maximum(xm, 0)], -1) if len(crossings) else -1
        if len(constraints):
            row[-1] = constraints[0]
        compare(p, "hit_membership", row[:nh] >= 0, seed_hit_mask, True)
        compare(p, "crossing_membership", row[nh:nh + nx] >= 0, expected[nh:nh + nx], True)
        compare(p, "constraint_membership", np.array([len(constraints)]), np.array([1]), True)
        native_unmatched = np.setdiff1d(np.r_[hits, crossings, constraints], row[row >= 0])
        compare(p, "native_unmatched_count", np.array(len(native_unmatched)), np.array(0), True)
        p["native_unmatched_effect_indices"] = ei[native_unmatched].tolist()
        xr = np.flatnonzero(row[nh:nh + nx] >= 0)
        compare(p, "crossing_partner", crossing_partner[xr], _col(e, "partner_shindex", "iu")[row[nh + xr]], True)
    p["n_expected"] = int(expected.sum()) if not ambiguous else None
    p["n_native"] = len(e)
    slots = np.flatnonzero((row >= 0) & expected)
    refs = slots[slots < nh + nx]

    p = phase("reference")
    if ambiguous:
        p["unsupported"].append("ambiguous membership prevents reference matching")
    elif not {"has_reference", "reference_parameters", "reference_covariance"}.issubset(ak.fields(e)):
        p["unsupported"].append("native reference payload unavailable")
    else:
        available = _col(e, "has_reference", "b")[row[refs]]
        if not available.all():
            p["unsupported"].append({"reason": "missing native reference", "slots": refs[~available].tolist()})
        use = refs[available]
        transformed(p, nativecompare.transform_parameters,
                    (t["reference_parameters"][use], t["reference_covariance"][use]),
                    (("parameters", _dense(e[row[use]], "reference_parameters", (6,))),
                     ("covariance", _dense(e[row[use]], "reference_covariance", (6, 6)))))
        if "reference_range" in ak.fields(e):
            compare(p, "range", np.broadcast_to(t["owner_ranges"][0], (len(use), 2)),
                    _dense(e[row[use]], "reference_range", (2,)))
        if "reference_index" in ak.fields(e):
            compare(p, "index", t["reference_index"][use], _col(e, "reference_index", "iu")[row[use]], True)

    p = phase("hit_active")
    if ambiguous:
        p["unsupported"].append("ambiguous hit membership")
    else:
        use = slots[slots < nh]
        compare(p, "active", t["hit_active"][use], _col(e, "active", "b")[row[use]], True)
        # Non-seed activity is an F1 membership leak, not padding to discard.
        compare(p, "nonseed_active", t["hit_active"][~seed_hit_mask], np.zeros((~seed_hit_mask).sum(), bool), True)

    p = phase("bounds_priors")
    if {"range_begin", "range_end", "bounds_valid"}.issubset(ak.fields(it)) and _col(it, "bounds_valid", "b")[0]:
        compare(p, "bounds", t["bounds"], np.array([_col(it, f)[0] for f in ("range_begin", "range_end")]))
    else:
        p["unsupported"].append("native bounds unavailable/invalid")
    if {"prior_parameters", "prior_covariances", "prior_valid"}.issubset(ak.fields(it)) and _col(it, "prior_valid", "b")[0]:
        transformed(p, nativecompare.transform_parameters, (t["prior_parameters"], t["prior_covariance"]),
                    (("prior_parameters", _dense(it, "prior_parameters", (2, 6))[0]),
                     ("prior_covariance", _dense(it, "prior_covariances", (2, 6, 6))[0])))
    if "deweight" in ak.fields(it):
        compare(p, "deweight", t["deweight"], _col(it, "deweight")[0])
    if {"prior_weight_matrices", "prior_weight_vectors", "prior_valid"}.issubset(ak.fields(it)) and _col(it, "prior_valid", "b")[0]:
        try:
            x, c = nativecompare.transform_parameters(t["prior_parameters"], t["prior_covariance"] * t["deweight"])
            w = np.linalg.inv(c)
            compare(p, "prior_weight_matrix", w, _dense(it, "prior_weight_matrices", (2, 6, 6))[0])
            compare(p, "prior_weight_vector", np.einsum("...ij,...j->...i", w, x), _dense(it, "prior_weight_vectors", (2, 6))[0])
        except (ValueError, np.linalg.LinAlgError) as error:
            p["metrics"]["prior_transform"] = {"passed": False, "error": str(error)}
    else:
        p["unsupported"].append("native prior weights unavailable/invalid")

    for name, flag, is_material, transform, operands, outputs in (
        ("measurement", "has_weight", False, nativecompare.transform_measurement,
         ("eff_xref", "eff_H", "eff_r", "eff_w"), (("weight_matrix", (6, 6)), ("weight_vector", (6,)))),
        ("material", "has_change", True, nativecompare.transform_material,
         ("eff_xref", "eff_D", "eff_sv", "eff_dm", "eff_dmv"),
         (("change_parameters", (6,)), ("change_covariance", (6, 6))))):
        p = phase(name)
        if ambiguous:
            p["unsupported"].append("ambiguous effect identities")
            continue
        if not {flag, *(field for field, _ in outputs)}.issubset(ak.fields(e)):
            p["unsupported"].append("native " + flag + " payload unavailable")
            continue
        flags = _col(e, flag, "b")
        family = (np.arange(nh + nx + 1) >= nh) & (np.arange(nh + nx + 1) < nh + nx)
        use = slots[family[slots] == is_material]
        available = flags[row[use]]
        missing = use[~available]
        # Flags describe capture availability, not numerical nonzero weights.
        operative = natural["is_material"] if is_material else np.any(natural["eff_w"] != 0, axis=-1)
        missing = missing[operative[missing]]
        unmatched = np.flatnonzero(flags & ~np.isin(np.arange(len(e)), row[use]))
        p["unmatched_native_effect_indices"] = ei[unmatched].tolist()
        p["missing_native_payload_slots"] = missing.tolist()
        if len(unmatched) or len(missing):
            p["unsupported"].append("unmatched native payload or unavailable operative F1 payload")
        use = use[available]
        if is_material:
            if {"active", "in_fit"}.issubset(ak.fields(e)):
                native_operative = _col(e, "active", "b")[row[use]] & _col(e, "in_fit", "b")[row[use]]
                compare(p, "participation", operative[use], native_operative, True)
                p["excluded_nonparticipating_native_changes"] = int((~native_operative).sum())
                use = use[native_operative & operative[use]]
            else:
                p["unsupported"].append("native material participation unavailable; cached changes cannot match masked F1 kernels")
        p["compared_f1_slots"] = use.tolist()
        p["compared_native_effect_indices"] = ei[row[use]].tolist()
        if "payload_supported" in ak.fields(e) and not _col(e, "payload_supported", "b")[row[use]].all():
            p["unsupported"].append("native payload_supported=false")
            continue
        p["n_compared_effects"] = len(use)
        transformed(p, transform, tuple(natural[f][use] for f in operands),
                    tuple((field, _dense(e[row[use]], field, shape)) for field, shape in outputs))

    p = phase("uncompared")
    p["unsupported"].append("ordering/CA geometry, candidate trajectory, decision and later iterations not compared")
    for p in phases.values():
        p["passed"] = all(m["passed"] for m in p["metrics"].values()) and not p["unsupported"]
        p["status"] = "different" if any(not m["passed"] for m in p["metrics"].values()) else "unsupported" if p["unsupported"] else "equal"
    differing = next((name for name, p in phases.items() if p["status"] == "different"), None)
    unresolved = next((name for name, p in phases.items() if p["status"] != "equal"), None)
    return {"phases": phases, "phase_order": list(phases), "first_differing_phase": differing,
            "first_unresolved_phase": unresolved, "compared_metrics_agree": differing is None,
            "passed": unresolved is None, "full_closure": False, "stock_exact": False,
            "rtol": RTOL, "atol": ATOL, "limits": LIMITS.copy()}


def compare_saved(trace_dir, check, out):
    """Recompare immutable actual-F1 trace operands without rerunning any fit."""
    import json
    from pathlib import Path
    import uproot
    from f1_native_decision_audit import load_evidence, read_tree, keys, digest
    trace_dir, check, out = Path(trace_dir), Path(check), Path(out)
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    _require(manifest.get("trace_passivity_exact") is True and manifest.get("completed") is True,
             "requires complete passive actual-F1 trace")
    _require(manifest["trace_sha256"] == digest(trace_dir / "trace.npz"), "trace hash changed")
    metadata = {}
    iterations, pieces = load_evidence(check, metadata)
    _require(manifest["check_sha256"] == metadata["check_sha256"], "trace/native gate differs")
    metadata["all_captured_iterations_included"] = False
    with np.load(trace_dir / "trace.npz", allow_pickle=False) as data:
        k = data["key"]
        _require(np.array_equal(k, manifest["key"]), "trace manifest key differs")
        _require(data["stage_index"][0] == 0, "missing initial F1 stage")
        h, x = data["hit_ids"], data["crossing_ids"]
        trace = {name: data[name][0] for name in fittrace.shapes(len(h), len(x))}
        with uproot.open(metadata["inputs"]["capture"]["path"]) as f:
            effects = read_tree(f, "kktrace", "effects", metadata["highest_cycles"]["capture"])
        select = lambda t: np.all(keys(t, unique=False) == k, axis=1) & (_col(t, "iteration_ordinal") == 0)
        result = compare_first(trace, iterations[select(iterations)],
            pieces[select(pieces) & (_col(pieces, "owner") == 0)], effects[select(effects)],
            h, x, data["seed_hit_mask"], data["crossing_partner"])
    result.update(metadata, trace_manifest=str(trace_dir / "manifest.json"),
                  trace_manifest_sha256=digest(trace_dir / "manifest.json"),
                  trace_sha256=manifest["trace_sha256"], comparator_sha256=digest(__file__))
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"first_differing_phase": result["first_differing_phase"],
        "phases": {name: {"status": phase["status"], "unsupported": phase["unsupported"],
            "metrics": {key: {k: v for k, v in metric.items() if not k.startswith("first_")}
                        for key, metric in phase["metrics"].items()}}
            for name, phase in result["phases"].items()}}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--check", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    compare_saved(arguments.trace_dir, arguments.check, arguments.out)
