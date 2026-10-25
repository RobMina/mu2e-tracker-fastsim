"""Require capture-on/off full-schedule trace identity before inspecting rejected steps."""
import argparse
import json

import awkward as ak
import numpy as np
import uproot

KEYS = ("run", "subrun", "event", "iseed")
SAME = ("stage", "maxit", "status", "miter", "iter", "goodseed", "chi2", "ndof", "pf", "pb", "cf", "cb",
        "hmiter", "hstatus", "hchi2", "shind", "state", "frozen", "algo", "qbkg", "qsign", "qdrift", "doca",
        "nxing", "nxact")


def compare(control, captured):
    """Exact comparison of stored histories/states; not proof of unrecorded effect identity."""
    ordered = []
    for table, enabled in ((control, False), (captured, True)):
        if len(table) == 0:
            raise ValueError("empty trace cannot establish capture equivalence")
        if not bool(ak.all(table.full_schedule & (table.stage == -1) & (table.maxit == -1))):
            raise ValueError("only full-schedule, uncapped rows are accepted")
        if not bool(ak.all(table.capture_enabled == enabled)):
            raise ValueError("expected capture disabled in control and enabled in candidate")
        keys = np.column_stack([ak.to_numpy(table[k]) for k in KEYS])
        if len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError("duplicate run/subrun/event/seed identity")
        order = np.lexsort(tuple(keys[:, i] for i in range(len(KEYS) - 1, -1, -1)))
        ordered.append(table[order])
    a, b = ordered
    if len(a) != len(b) or any(not ak.almost_equal(a[k], b[k], rtol=0, atol=0) for k in KEYS):
        raise ValueError("capture runs have different seed populations")
    different = [key for key in SAME if not ak.almost_equal(a[key], b[key], rtol=0, atol=0)]
    if different:
        raise ValueError("capture changed recorded history/state: " + ", ".join(different))
    if bool(ak.any(a.rejected != 0)):
        raise ValueError("capture-disabled control contains a rejection snapshot")
    expected = (b.status == 4) & (b.goodseed != 0)
    if not bool(ak.all((b.rejected != 0) == expected)):
        raise ValueError("snapshot coverage differs from extension ParamsDiverged population")
    r = b[b.rejected != 0]
    if bool(ak.any(r.status != 4)):
        raise ValueError("snapshot does not identify ParamsDiverged")
    n = len(r)
    status, counts = np.unique(ak.to_numpy(b.status), return_counts=True)
    result = {"n_seeds": len(b), "recorded_capture_equivalent": True, "n_rejected": n,
              "status_counts": {str(int(s)): int(c) for s, c in zip(status, counts)},
              "scope": "stored full-schedule histories and states; unrecorded effects not compared"}
    if not n:
        return result
    p = ak.to_numpy(r.rejected_parameters).reshape(n, 4, 6)
    cov = ak.to_numpy(r.rejected_covariances).reshape(n, 4, 6, 6)
    dp = ak.to_numpy(r.rejected_steps).reshape(n, 2, 6)
    metrics = ak.to_numpy(r.rejected_metrics).reshape(n, 5)
    if not all(np.isfinite(x).all() for x in (p, cov, dp, metrics)):
        raise ValueError("nonfinite rejection snapshot")
    np.testing.assert_allclose(dp, p[:, [0, 2]] - p[:, [1, 3]], rtol=0, atol=0)
    refcov = cov[:, [1, 3]]
    chi = np.einsum("nei,nei->ne", dp, np.linalg.solve(refcov, dp[..., None])[..., 0])
    np.testing.assert_allclose(chi, metrics[:, :2], rtol=1e-5, atol=1e-7)
    if not np.all((metrics[:, :2] > metrics[:, 2, None]).any(axis=1)):
        raise ValueError("snapshot does not cross its recorded parameter-change threshold")
    if np.any(np.diagonal(refcov, axis1=-2, axis2=-1) <= 0):
        raise ValueError("nonpositive reference diagonal; weak-direction projection is undefined")
    sigma = np.sqrt(np.diagonal(refcov, axis1=-2, axis2=-1))
    correlation = refcov / sigma[..., :, None] / sigma[..., None, :]
    if not np.allclose(correlation, np.swapaxes(correlation, -1, -2), rtol=0, atol=1e-10):
        raise ValueError("asymmetric reference; covariance-mode interpretation is invalid")
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    if np.any(eigenvalues[..., 0] <= 0):
        raise ValueError("nonpositive reference eigenvalue; covariance-mode interpretation is invalid")
    projection = np.einsum("nei,nei->ne", dp / sigma, eigenvectors[..., 0])
    result["reference_min_eigenvalue_q10_50_90"] = np.quantile(eigenvalues[..., 0], [.1, .5, .9], axis=0).tolist()
    result["weak_step_abs_q10_50_90"] = np.quantile(np.abs(projection), [.1, .5, .9], axis=0).tolist()
    result["endpoint_order"] = ["front", "back"]
    result["mode_diagnosis"] = diagnose_modes(dp, refcov, metrics, r)
    result["reference_selection"] = reference_selection(p, cov, r)
    return result


def reference_selection(parameters, covariances, rows):
    """Compare the exact chosen references with the retained trajectory endpoints."""
    n = len(rows)
    retained = np.stack([ak.to_numpy(rows.pf), ak.to_numpy(rows.pb)], axis=1).reshape(n, 2, 6)
    retained_cov = np.stack([ak.to_numpy(rows.cf), ak.to_numpy(rows.cb)], axis=1).reshape(n, 2, 6, 6)
    references = parameters[:, [1, 3]]
    reference_cov = covariances[:, [1, 3]]
    ranges = ak.to_numpy(rows.rejected_ranges).reshape(n, 4, 2)
    ref_index = ak.to_numpy(rows.rejected_reference_indices).reshape(n, 2)
    midpoint = ranges[:, [0, 2]].mean(axis=-1)
    ref_ranges = ranges[:, [1, 3]]
    inside = (midpoint >= ref_ranges[..., 0]) & (midpoint <= ref_ranges[..., 1])
    different = np.any(references != retained, axis=-1) | np.any(reference_cov != retained_cov, axis=(-2, -1))
    return {"endpoint_order": ["front", "back"],
            "n_reference_differs_from_retained_endpoint": different.sum(axis=0).tolist(),
            "n_candidate_midpoint_inside_chosen_reference_range": inside.sum(axis=0).tolist(),
            "chosen_reference_index_q10_50_90": np.quantile(ref_index, [.1, .5, .9], axis=0).tolist(),
            "chosen_reference_momentum_sigma_q10_50_90": np.quantile(np.sqrt(reference_cov[..., 5, 5]), [.1, .5, .9], axis=0).tolist(),
            "retained_endpoint_momentum_sigma_q10_50_90": np.quantile(np.sqrt(retained_cov[..., 5, 5]), [.1, .5, .9], axis=0).tolist()}


def diagnose_modes(dp, refcov, metrics, rows):
    """Decompose positive-definite endpoint quadratic forms without regularization."""
    sigma = np.sqrt(np.diagonal(refcov, axis1=-2, axis2=-1))
    correlation = refcov / sigma[..., :, None] / sigma[..., None, :]
    symmetric = np.max(np.abs(correlation - np.swapaxes(correlation, -1, -2)), axis=(-2, -1)) <= 1e-10
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    pd = symmetric & (eigenvalues[..., 0] > 0)
    failed = metrics[:, :2] > metrics[:, 2, None]
    selected = pd & failed
    result = {"n_endpoints": int(pd.size), "n_positive_definite": int(pd.sum()),
              "n_above_threshold": int(failed.sum()), "n_pd_above_threshold": int(selected.sum()),
              "n_asymmetric": int((~symmetric).sum()), "n_nonpositive": int((eigenvalues[..., 0] <= 0).sum())}
    if not selected.any():
        return result
    z = dp[selected] / sigma[selected]
    w, v = eigenvalues[selected], eigenvectors[selected]
    coeff = np.einsum("ni,nij->nj", z, v)
    terms = coeff ** 2 / w
    quadratic = terms.sum(axis=-1)
    fraction = terms / quadratic[:, None]
    dominant = np.argmax(terms, axis=-1)
    param = ("d0", "phi0", "z0", "theta", "t0", "mom")
    result.update(
        reference_momentum_sigma_q10_50_90=np.quantile(sigma[selected][:, 5], [.1, .5, .9]).tolist(),
        n_momentum_diagonal_above_threshold=int(((z[:, 5] ** 2) > np.broadcast_to(metrics[:, 2, None], selected.shape)[selected]).sum()),
        momentum_diagonal_over_full_q10_50_90=np.quantile(z[:, 5] ** 2 / quadratic, [.1, .5, .9]).tolist(),
        chi2_q10_50_90=np.quantile(quadratic, [.1, .5, .9]).tolist(),
        condition_number_q10_50_90=np.quantile(w[:, -1] / w[:, 0], [.1, .5, .9]).tolist(),
        weakest_mode_fraction_q10_50_90=np.quantile(fraction[:, 0], [.1, .5, .9]).tolist(),
        n_weakest_mode_dominates=int((dominant == 0).sum()),
        mode_dominance_counts=np.bincount(dominant, minlength=6).tolist(),
        raw_step_abs_q50_90_99={k: np.quantile(np.abs(dp[selected][:, i]), [.5, .9, .99]).tolist() for i, k in enumerate(param)},
        diagonal_chi2_q50_90_99={k: np.quantile(z[:, i] ** 2, [.5, .9, .99]).tolist() for i, k in enumerate(param)},
        weakest_direction_abs_q50={k: float(np.median(np.abs(v[:, i, 0]))) for i, k in enumerate(param)},
        n_raw_phi_step_gt_pi=int((np.abs(dp[selected][:, 1]) > np.pi).sum()))
    order = np.argsort(quadratic)[-5:][::-1]
    track, endpoint = np.nonzero(selected)
    result["largest_examples"] = [dict(
        **{key: int(ak.to_numpy(rows[key])[track[j]]) for key in KEYS},
        miter=int(ak.to_numpy(rows.miter)[track[j]]), iteration=int(ak.to_numpy(rows.iter)[track[j]]),
        endpoint=("front", "back")[endpoint[j]], chi2=float(quadratic[j]),
        weakest_fraction=float(fraction[j, 0]), min_eigenvalue=float(w[j, 0]),
        raw_step=dp[selected][j].tolist()) for j in order]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control")
    parser.add_argument("captured")
    parser.add_argument("--expected-events", type=int, required=True)
    args = parser.parse_args()
    with uproot.open(args.control) as a, uproot.open(args.captured) as b:
        events_a, events_b = a["trkdump/events"], b["trkdump/events"]
        if events_a.num_entries != args.expected_events or events_b.num_entries != args.expected_events:
            raise ValueError("finished highest-cycle event entries do not match requested coverage")
        for key in ("run", "subrun", "event"):
            np.testing.assert_array_equal(events_a[key].array(library="np"), events_b[key].array(library="np"))
        result = compare(a["kktrace/trace"].arrays(), b["kktrace/trace"].arrays())
        result["n_events"] = events_a.num_entries
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
