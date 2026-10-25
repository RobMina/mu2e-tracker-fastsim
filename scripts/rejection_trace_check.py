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
    r = b[b.rejected != 0]
    if bool(ak.any(r.status != 4)):
        raise ValueError("snapshot does not identify ParamsDiverged")
    n = len(r)
    result = {"n_seeds": len(b), "recorded_capture_equivalent": True, "n_rejected": n,
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
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    projection = np.einsum("nei,nei->ne", dp / sigma, eigenvectors[..., 0])
    result["reference_min_eigenvalue_q10_50_90"] = np.quantile(eigenvalues[..., 0], [.1, .5, .9], axis=0).tolist()
    result["weak_step_abs_q10_50_90"] = np.quantile(np.abs(projection), [.1, .5, .9], axis=0).tolist()
    result["endpoint_order"] = ["front", "back"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control")
    parser.add_argument("captured")
    args = parser.parse_args()
    with uproot.open(args.control) as a, uproot.open(args.captured) as b:
        result = compare(a["kktrace/trace"].arrays(), b["kktrace/trace"].arrays())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
