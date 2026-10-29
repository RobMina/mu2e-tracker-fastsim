"""Independent analytic first-record fixture; no fitter/native replay inputs."""
import copy
import json
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import f1_trace_compare as C
from trkfastsim.reco import fittrace


def analytic(x):
    """Independent coordinate/Jacobian formulas (including momentum flight time)."""
    a, b, c, d, t, p = x
    q, h, n = 1 + c*c, np.sqrt(1 + c*c), np.sqrt(1 + c*c + d*d)
    m, light = 105.6583755, 299.792458
    en = np.hypot(p, m)
    iv = en / (light*p)
    s = -a*c*n/q
    y = np.array([a/h, np.arctan2(-1., c), b-a*c*d/q, np.arctan2(h, d), t+s*iv, p])
    j = np.zeros((6, 6))
    j[0, 0], j[0, 2] = 1/h, -a*c/(q*h)
    j[1, 2] = 1/q
    j[2, :4] = [-c*d/q, 1., -a*d*(1-c*c)/q**2, -a*c/q]
    j[3, 2], j[3, 3] = c*d/(h*n*n), -h/(n*n)
    j[4, 0], j[4, 2] = -c*n/q*iv, -a*(n/q+c*c/(n*q)-2*c*c*n/q**2)*iv
    j[4, 3], j[4, 4], j[4, 5] = -a*c*d/(n*q)*iv, 1., -s*m*m/(light*en*p*p)
    j[5, 5] = 1.
    return y, j


def fixture():
    nh, nx = 3, 2
    t = {k: np.zeros(v, dtype=fittrace.dtype(k)) for k, v in fittrace.shapes(nh, nx).items()}
    x = np.array([125., -730., .35, .8, 42., 380.])
    cov = np.diag([1., 2., .001, .002, 3., 4.])
    y, j = analytic(x)
    cy = j @ cov @ j.T
    t.update(valid=np.array(True), iteration=np.array(0), owner_count=np.array(0),
             owner_parameters=np.tile(x, (3, 1)), owner_covariance=np.tile(cov, (3, 1, 1)),
             owner_ranges=np.tile([10., 60.], (3, 1)),
             reference_parameters=np.tile(x, (5, 1)), reference_covariance=np.tile(cov, (5, 1, 1)),
             prior_parameters=np.tile(x, (2, 1)), prior_covariance=np.tile(cov, (2, 1, 1)),
             deweight=np.array(2.), bounds=np.array([20., 55.]),
             order=np.array([3, 1, 5, 4, 0, 2]), hit_active=np.array([True, True, False]))
    t["eff_xref"][:] = x
    h = np.zeros((6, 3, 6))
    h[0, 0, 0], h[1, 1, 1], h[5, 0, 5] = 1., 1., 1.
    r, w = np.zeros((6, 3)), np.zeros((6, 3))
    r[0, 0], r[1, 1], r[5, 0] = .25, -.5, .125
    w[0, 0], w[1, 1], w[5, 0] = .5, 2., 4.
    d = np.zeros((6, 6, 3))
    d[3, 2, 0], d[3, 3, 1], d[3, 5, 2] = .5, .75, 1.
    sv, dm, dmv = np.zeros(6), np.zeros(6), np.zeros(6)
    sv[3], dm[3], dmv[3] = .02, -.3, .04
    for k, value in (("eff_H", h), ("eff_r", r), ("eff_w", w), ("eff_D", d),
                     ("eff_sv", sv), ("eff_dm", dm), ("eff_dmv", dmv)):
        t[k] = value[t["order"]]
    t["is_material"] = np.array([False, False, False, True, False, False])[t["order"]]
    ident = dict(run=1, subrun=2, event=3, iseed=0, iteration_ordinal=0, miter=7, iter=0, schema_version=1)
    pw = np.linalg.inv(cy*2)
    it = ak.Array([dict(ident, nold=1, neffects=4, capture_valid=True, capture_error="", effects_captured=True,
                        bounds_valid=True, range_begin=20., range_end=55., prior_valid=True, deweight=2.,
                        prior_parameters=np.tile(y, (2, 1)).ravel().tolist(),
                        prior_covariances=np.tile(cy, (2, 1, 1)).ravel().tolist(),
                        prior_weight_matrices=np.tile(pw, (2, 1, 1)).ravel().tolist(),
                        prior_weight_vectors=np.tile(pw@y, (2, 1)).ravel().tolist())])
    old = ak.Array([dict(ident, owner=0, piece_index=0, parameters=y.tolist(), covariance=cy.ravel().tolist(), range=[10., 60.])])
    effects = []
    for index, (slot, kind, hid, sid, partner) in enumerate(((0, 1, 101, 900, -1), (3, 2, -1, 900, 101),
                                                          (1, 1, 53, 800, -1), (5, 4, -1, -1, -1))):
        hy = h[slot] @ np.linalg.inv(j)
        wm = hy.T @ np.diag(w[slot]) @ hy
        wb = hy.T @ (w[slot] * (hy@y-r[slot]))
        jd = j @ d[slot]
        change = jd[:, 2] * dm[slot]
        change_cov = jd @ np.diag([sv[slot], sv[slot], dmv[slot]]) @ jd.T
        effects.append(dict(ident, effect_index=index, kind=kind, identity_kind={1: 1, 2: 5, 4: 2}[kind],
                            shindex=hid, strawid=sid, partner_shindex=partner, active=True, in_fit=True,
                            has_reference=kind != 4, reference_parameters=y.tolist() if kind != 4 else [],
                            reference_covariance=cy.ravel().tolist() if kind != 4 else [], reference_index=0,
                            payload_supported=True, has_weight=kind != 2, has_change=kind == 2,
                            weight_matrix=wm.ravel().tolist() if kind != 2 else [],
                            weight_vector=wb.tolist() if kind != 2 else [],
                            change_parameters=change.tolist() if kind == 2 else [],
                            change_covariance=change_cov.ravel().tolist() if kind == 2 else []))
    return [t, it, old, ak.Array(effects)[::-1], np.array([101, 53, 77]), np.array([900, 1200]),
            np.array([True, True, False]), np.array([101, 77])]


def replace(table, row, field, value):
    a = ak.to_list(table)
    a[row][field] = value
    return ak.Array(a)


def test_independent_fixture_permutation_and_limits():
    args = fixture()
    r = C.compare_first(*args)
    assert r["compared_metrics_agree"], r
    assert r["first_differing_phase"] is None
    assert r["first_unresolved_phase"] == "uncompared"
    assert not r["passed"] and not r["full_closure"]
    assert r["phases"]["measurement"]["n_compared_effects"] == 3
    assert r["phases"]["material"]["n_compared_effects"] == 1
    assert all(p["status"] == "equal" for name, p in r["phases"].items() if name != "uncompared")
    json.dumps(r, allow_nan=False)
    # An unsorted native table is joined by identities, not row/order position.
    args[3] = args[3][::-1]
    assert C.compare_first(*args) == r


@pytest.mark.parametrize("field,index,value,phase", [
    ("owner_parameters", (0, 0), 130., "seed_old_owner"),
    ("owner_covariance", (0, 0, 0), 2., "seed_old_owner"),
    ("reference_parameters", (1, 0), 130., "reference"),
    ("reference_covariance", (1, 0, 0), 2., "reference"),
    ("hit_active", (1,), False, "hit_active"),
    ("bounds", (0,), 19., "bounds_priors"),
    ("prior_covariance", (1, 0, 0), 2., "bounds_priors"),
    ("eff_r", (4, 0), 2., "measurement"),
    ("eff_dm", (0,), -.9, "material"),
    ("eff_dmv", (0,), .2, "material"),
    ("owner_parameters", (0, 0), np.nan, "seed_old_owner"),
])
def test_phase_injection(field, index, value, phase):
    args = fixture()
    args[0][field][index] = value
    r = C.compare_first(*args)
    assert r["first_differing_phase"] == phase, r
    assert not r["passed"]
    json.dumps(r, allow_nan=False)


def test_membership_and_source_partner_not_positions():
    args = fixture()
    args[6][1] = False
    assert C.compare_first(*args)["first_differing_phase"] == "effect_membership"
    args = fixture()
    args[7][0] = 53
    r = C.compare_first(*args)
    assert r["first_differing_phase"] == "effect_membership"
    assert r["phases"]["effect_membership"]["metrics"]["crossing_partner"]["mismatches"] == 1


@pytest.mark.parametrize("which,field,value,match", [
    (1, "iter", 1, "iter==0"), (1, "iteration_ordinal", 1, "iteration_ordinal==0"),
    (1, "nold", 2, "one old"), (1, "capture_valid", False, "invalid native"),
    (1, "neffects", 5, "count mismatch"), (2, "owner", 1, "owner=0"),
    (2, "piece_index", 1, "piece_index"), (2, "event", 4, "full identity"),
    (3, "miter", 0, "full identity"), (3, "effect_index", 0, "duplicate"),
])
def test_native_guards(which, field, value, match):
    args = fixture()
    args[which] = replace(args[which], 0, field, value)
    with pytest.raises(ValueError, match=match):
        C.compare_first(*args)


@pytest.mark.parametrize("field,value,match", [
    ("iteration", np.array(1), "first F1"), ("valid", np.array(False), "first F1"),
    ("owner_count", np.array(1), "one old"), ("order", np.zeros(6, int), "permutation"),
    ("owner_parameters", np.zeros((1, 3, 6)), "first-record shape")])
def test_trace_guards(field, value, match):
    args = fixture()
    args[0][field] = value
    with pytest.raises(ValueError, match=match):
        C.compare_first(*args)


@pytest.mark.parametrize("which", [4, 5])
def test_duplicate_f1_identities_unsupported(which):
    args = fixture()
    args[which][1] = args[which][0]
    r = C.compare_first(*args)
    assert r["phases"]["effect_membership"]["status"] == "unsupported"
    assert r["phases"]["measurement"]["status"] == "unsupported"
    assert not r["full_closure"]


@pytest.mark.parametrize("mode", ["hits", "crossings", "constraint", "cluster"])
def test_native_ambiguity_and_clusters_never_equal(mode):
    args = fixture()
    records = ak.to_list(args[3])
    if mode == "hits":
        records[1]["shindex"] = records[3]["shindex"]
    else:
        source = {"crossings": 2, "constraint": 0, "cluster": 1}[mode]
        extra = copy.deepcopy(records[source])
        extra["effect_index"] = 4
        if mode == "cluster":
            extra["kind"] = 3
        records.append(extra)
        args[1] = replace(args[1], 0, "neffects", 5)
    args[3] = ak.Array(records)
    r = C.compare_first(*args)
    assert r["phases"]["effect_membership"]["unsupported"]
    assert not r["passed"]


def test_missing_payload_and_finite_failure():
    args = fixture()
    args[3] = replace(args[3], 1, "has_weight", False)
    r = C.compare_first(*args)
    assert r["phases"]["measurement"]["missing_native_payload_slots"] == [1]
    assert r["phases"]["measurement"]["status"] == "unsupported"
    args = fixture()
    args[3] = replace(args[3], 1, "reference_parameters", [float("inf")]*6)
    r = C.compare_first(*args)
    assert r["first_differing_phase"] == "reference"
    assert r["phases"]["reference"]["metrics"]["parameters"]["nonfinite"] == 6
    json.dumps(r, allow_nan=False)


def test_missing_optional_phase_not_closure():
    args = fixture()
    args[3] = ak.without_field(args[3], "reference_covariance")
    r = C.compare_first(*args)
    assert r["first_differing_phase"] is None
    assert r["first_unresolved_phase"] == "reference"
    assert not r["passed"]


def test_invalid_native_semantic_identities():
    for field, value in (("identity_kind", 3), ("identity_valid", False)):
        args = fixture()
        if field == "identity_valid":
            args[3] = ak.with_field(args[3], np.ones(4, bool), field)
        args[3] = replace(args[3], 0, field, value)
        r = C.compare_first(*args)
        assert r["phases"]["effect_membership"]["unsupported"]
        assert not r["passed"]


def test_duplicate_generic_input_identity():
    args = fixture()
    args[3] = ak.with_field(args[3], np.array([0, 0, 0, 1]), "input_index")
    r = C.compare_first(*args)
    assert r["phases"]["effect_membership"]["status"] == "unsupported"


def test_absent_partner_and_missing_native_member():
    args = fixture()
    args[7][0] = 999
    assert C.compare_first(*args)["phases"]["effect_membership"]["status"] == "unsupported"
    args = fixture()
    args[3] = replace(args[3], 1, "shindex", 999)
    r = C.compare_first(*args)
    assert r["first_differing_phase"] == "effect_membership"
    assert r["phases"]["effect_membership"]["native_unmatched_effect_indices"] == [2]
    assert r["phases"]["measurement"]["unmatched_native_effect_indices"] == [2]


@pytest.mark.parametrize("flag,phase", [("has_weight", "measurement"), ("has_change", "material"),
                                        ("has_reference", "reference")])
def test_no_available_payload_rows(flag, phase):
    args = fixture()
    args[3] = ak.with_field(args[3], np.zeros(4, bool), flag)
    r = C.compare_first(*args)
    assert r["phases"][phase]["status"] == "unsupported"
    assert not r["passed"]
    json.dumps(r, allow_nan=False)


def test_excluded_material_cache_not_compared_to_zero_kernel():
    args = fixture()
    args[0]["eff_dm"][0] = 0
    args[0]["eff_sv"][0] = 0
    args[0]["eff_dmv"][0] = 0
    args[0]["is_material"][0] = False
    args[3] = replace(args[3], 2, "in_fit", False)
    result = C.compare_first(*args)
    phase = result["phases"]["material"]
    assert phase["status"] == "equal"
    assert phase["n_compared_effects"] == 0
    assert phase["excluded_nonparticipating_native_changes"] == 1
    args[3] = replace(args[3], 2, "in_fit", True)
    assert not C.compare_first(*args)["phases"]["material"]["metrics"]["participation"]["passed"]
