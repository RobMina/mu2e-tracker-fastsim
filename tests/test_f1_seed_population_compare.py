"""Analytic population tests; no JAX, native inputs, or closure claims."""
from copy import deepcopy
import json
import sys

import numpy as np
import pytest

import f1_seed_population_compare as C


def payload(n=8):
    keys = np.array([[1400, 0, 7, 0], [1401, 0, 7, 0], [1400, 1, 7, 0],
                     [1400, 0, 7, 1], [1401, 1, 7, 1], [1400, 0, 8, 0],
                     [1400, 0, 8, 1], [1400, 2, 7, 0]], dtype=np.int64)[:n]
    return {"keys": keys.copy(), "good": np.ones(n, dtype=bool),
            "stockok": np.ones(n, dtype=bool), "kl": np.zeros((n, 6)),
            "stock_params": np.zeros((n, 6)), "stock_errors": np.ones((n, 6))}


def permute(p, order):
    return {name: value[order].copy() for name, value in p.items()}


def test_full_keys_distinguish_runs_subruns_events_and_seeds():
    p = payload()
    assert C.validate(p) == 8
    for name, index in C.IDENTITY.items():
        p[name] = p["keys"][:, index].copy()
    assert C.validate(p) == 8
    result = C.summary(p)
    assert not set(C.IDENTITY).intersection(result["summary_all_output_nonfinite"])
    assert "keys" not in result["summary_all_output_nonfinite"]


def test_analytic_acceptance_all_eligible_and_binomial_error():
    p = payload()
    p["stockok"][:] = [1, 1, 1, 1, 1, 0, 0, 0]
    p["good"][:] = [1, 1, 1, 0, 0, 1, 0, 0]
    result = C.summary(p)
    a = result["acceptance"]
    for name, value in {"n": 8, "stock_good": 5, "stock_rejected": 3, "f1_good": 4,
                        "f1_rejected": 4, "both_good": 3, "gains": 1, "losses": 2,
                        "neither": 2, "net": -1, "agreement": 5}.items():
        assert a[name] == value
    assert a["acceptance_agreement"] == 5 / 8
    assert a["binomial_se"] == pytest.approx(np.sqrt((5 / 8) * (3 / 8) / 8))
    assert a["acceptance_agreement_target"] == 0.995
    for name in C.KL:
        f = result["fidelity"][name]
        assert f["n"] == f["n_finite"] == f["within_0p1"] == 3
        assert f["n_nonfinite"] == 0
        assert f["frac_within_0p1"] == 1
        assert f["binomial_se"] == 0
        assert f["target_frac_within_0p1"] == 0.99
    assert result["full_fit_closure"] is False
    assert result["full_f1_closure"] is False


def test_per_parameter_stock_sigma_and_strict_threshold():
    p = payload(4)
    p["stock_errors"][:] = np.array([1, 2, 4, 8, 16, 32])
    p["kl"][:] = np.array([0, 0.05, 0.1, -0.1])[:, None] * p["stock_errors"]
    f = C.summary(p)["fidelity"]
    for name in ("d0", "z0", "theta", "t0", "mom"):
        assert f[name]["n_finite"] == 4
        assert f[name]["within_0p1"] == 2
        assert f[name]["frac_within_0p1"] == 0.5
        assert f[name]["binomial_se"] == 0.25


def test_phi_wrap_crosses_both_branches_and_multiple_turns():
    p = payload(4)
    p["stock_params"][:, 1] = [np.pi - .01, -np.pi + .01, 0, 0]
    p["kl"][:, 1] = [-np.pi + .01, np.pi - .01, 10 * np.pi + .03, np.pi]
    f = C.summary(p)["fidelity"]["phi0"]
    assert f["within_0p1"] == 3
    assert f["n_finite"] == 4
    assert f["frac_within_0p1"] == .75


@pytest.mark.parametrize("field", ["kl", "stock_params", "stock_errors"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_remain_denominator_failures(field, value):
    p = payload(4)
    p[field][1, :] = value
    assert C.validate(p) == 4
    with np.errstate(all="raise"):
        result = C.summary(p)
    for name in C.KL:
        f = result["fidelity"][name]
        assert f["n"] == 4
        assert f["n_finite"] == f["within_0p1"] == 3
        assert f["n_nonfinite"] == 1
        assert f["frac_within_0p1"] == .75
        assert f["binomial_se"] == pytest.approx(np.sqrt(.75 * .25 / 4))
    assert result["acceptance"]["both_good"] == 4
    assert result["summary_all_output_nonfinite"][field]["n_nonfinite"] == 6
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("error", [0., -1., -np.inf, np.inf, np.nan])
def test_bad_errors_do_not_drop_rows_or_manufacture_zero_pulls(error):
    p = payload(4)
    p["stock_errors"][0, :] = error
    f = C.summary(p)["fidelity"]
    for name in C.KL:
        assert f[name]["n"] == 4
        assert f[name]["n_nonfinite"] == 1
        assert f[name]["within_0p1"] == 3


def test_rejected_nonfinites_reported_for_every_output_without_entering_fidelity():
    p = payload(4)
    p["stockok"][:] = [1, 1, 0, 0]
    p["good"][:] = [1, 0, 1, 0]
    p["kl"][1:, :] = np.nan
    p["returned_cov"] = np.zeros((4, 2, 2))
    p["returned_cov"][3, :, :] = np.inf
    p["complex_output"] = np.array([0, 0, complex(0, np.inf), 0])
    p["empty_output"] = np.empty((4, 0, 2))
    result = C.summary(p)
    assert result["fidelity"]["mom"]["n"] == 1
    assert result["fidelity"]["mom"]["frac_within_0p1"] == 1
    nf = result["summary_all_output_nonfinite"]
    assert nf["kl"]["n_elements"] == 24
    assert nf["kl"]["n_nonfinite"] == 18
    assert nf["kl"]["n_rows_nonfinite"] == 3
    assert nf["kl"]["by_population"]["stock_rejected"]["n_nonfinite"] == 12
    assert nf["kl"]["by_population"]["f1_rejected"]["n_nonfinite"] == 12
    assert nf["returned_cov"]["by_population"]["neither"]["n_nonfinite"] == 4
    assert nf["complex_output"]["n_nonfinite"] == 1
    assert nf["empty_output"]["n_nonfinite"] == 0
    assert nf["good"]["n_nonfinite"] == 0


def test_status_counts_by_stage_and_rejection_populations():
    p = payload(4)
    p["stockok"][:] = [1, 1, 0, 0]
    p["good"][:] = [1, 0, 1, 0]
    p["stat"] = np.array([[-1, 0], [0, 2], [0, 0], [2, 7]])
    status = C.summary(p)["status_counts"]
    assert status["codes"] == [-1, 0, 2, 7]
    assert status["n_stages"] == 2
    assert status["populations"]["all_eligible"]["by_stage"] == [[1, 2, 1, 0], [0, 2, 1, 1]]
    assert status["populations"]["stock_rejected"]["final"] == {"-1": 0, "0": 1, "2": 0, "7": 1}
    assert status["populations"]["losses"]["final"]["2"] == 1
    assert status["populations"]["gains"]["final"]["0"] == 1
    assert "status_counts" not in C.summary(payload())


@pytest.mark.parametrize("n,stages", [(0, 3), (4, 0), (0, 0)])
def test_empty_status_arrays(n, stages):
    p = payload(n)
    p["stat"] = np.empty((n, stages), dtype=int)
    s = C.summary(p)["status_counts"]
    assert s["codes"] == []
    assert s["populations"]["all_eligible"]["by_stage"] == [[]] * stages
    assert s["populations"]["all_eligible"]["final"] == {}


@pytest.mark.parametrize("n", [0, 4])
def test_empty_both_good_intersection_has_no_invented_fraction(n):
    p = payload(n)
    p["good"][:] = False
    p["kl"][:] = np.nan
    result = C.summary(p)
    for name in C.KL:
        f = result["fidelity"][name]
        assert f["n"] == f["within_0p1"] == f["n_nonfinite"] == 0
        assert f["frac_within_0p1"] is None
        assert f["binomial_se"] is None
    if not n:
        assert result["acceptance"]["acceptance_agreement"] is None
        assert result["acceptance"]["binomial_se"] is None
    json.dumps(result, allow_nan=False)


def paired():
    candidate = payload()
    candidate["stockok"][:] = [1, 1, 1, 1, 1, 0, 0, 0]
    candidate["good"][:] = [1, 1, 0, 1, 0, 1, 0, 0]
    baseline = deepcopy(candidate)
    baseline["good"][:] = [1, 1, 1, 0, 0, 0, 1, 0]
    candidate["kl"][0, :] = .05
    candidate["kl"][1, :] = .2
    baseline["kl"][0, :] = .2
    baseline["kl"][1, :] = .25
    return candidate, baseline


@pytest.mark.parametrize("seed", [0, 1, 17, 422])
def test_baseline_full_key_alignment_and_analytic_common_population(seed):
    candidate, baseline = paired()
    expected = C.summary(candidate, baseline)
    rng = np.random.default_rng(seed)
    actual = C.summary(permute(candidate, rng.permutation(8)), permute(baseline, rng.permutation(8)))
    assert actual == expected
    a = actual["baseline"]["acceptance"]
    assert a["gains"] == a["losses"] == a["both_good"] == a["neither"] == 2
    assert a["baseline_good"] == a["f1_good"] == 4
    assert a["acceptance_agreement"] == .5
    common = actual["baseline"]["common_both_good"]
    assert common["n_tracks"] == 2
    assert common["candidate_vs_stock"]["mom"]["within_0p1"] == 1
    assert common["baseline_vs_stock"]["mom"]["within_0p1"] == 0
    assert common["candidate_minus_baseline"]["mom"]["n"] == 2
    assert common["candidate_minus_baseline"]["mom"]["within_0p1"] == 1
    assert common["candidate_minus_baseline"]["mom"]["frac_within_0p1"] == .5
    assert actual["baseline"]["stock_reference_exact"] is True


def test_paired_kl_delta_in_stock_sigma_with_phi_wrapping():
    p = payload(4)
    b = deepcopy(p)
    p["stock_errors"][:] = 2
    b["stock_errors"][:] = 2
    p["kl"][:, :] = [.02, -np.pi + .01, .2, -.2, .3, 1.]
    b["kl"][:, :] = [0, np.pi - .01, .04, -.04, .5, .9]
    f = C.summary(p, b)["baseline"]["common_both_good"]["candidate_minus_baseline"]
    for name in ("d0", "phi0", "z0", "theta", "mom"):
        assert f[name]["within_0p1"] == 4
    assert f["t0"]["within_0p1"] == 0


@pytest.mark.parametrize("arm", ["candidate", "baseline"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_paired_nonfinite_outcomes_not_omitted(arm, value):
    p, b = payload(4), payload(4)
    (p if arm == "candidate" else b)["kl"][1, :] = value
    result = C.summary(p, permute(b, [3, 0, 2, 1]))
    common = result["baseline"]["common_both_good"]
    assert common["n_tracks"] == 4
    for name in C.KL:
        f = common["candidate_minus_baseline"][name]
        assert f["n"] == 4 and f["n_nonfinite"] == 1
        assert f["frac_within_0p1"] == .75
        assert common[arm + "_vs_stock"][name]["n_nonfinite"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("n", [0, 4])
def test_empty_paired_intersection(n):
    p, b = payload(n), payload(n)
    p["good"][:n // 2] = False
    b["good"][n // 2:] = False
    result = C.summary(p, b)
    common = result["baseline"]["common_both_good"]
    assert common["n_tracks"] == 0
    for field in ("candidate_vs_stock", "baseline_vs_stock", "candidate_minus_baseline"):
        assert common[field]["d0"]["frac_within_0p1"] is None


@pytest.mark.parametrize("field", ["stock_params", "stock_errors"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_baseline_exact_stock_allows_matching_nonfinites(field, value):
    p = payload(4)
    p[field][0, :] = value
    b = permute(p, [2, 0, 3, 1])
    result = C.summary(p, b)
    assert result["baseline"]["stock_reference_exact"] is True
    assert result["baseline"]["common_both_good"]["candidate_vs_stock"]["d0"]["n_nonfinite"] == 1


@pytest.mark.parametrize("field", ["stockok", "stock_params", "stock_errors"])
def test_changed_baseline_stock_reference_fails_even_on_rejected_rows(field):
    p = payload(4)
    p["good"][0] = False
    b = deepcopy(p)
    if field == "stockok":
        b[field][0] = False
    else:
        b[field][0, 0] = np.nextafter(b[field][0, 0], np.inf)
    with pytest.raises(ValueError, match="stock reference changed"):
        C.summary(p, b)


@pytest.mark.parametrize("replacement", [0., np.inf, -np.inf])
def test_baseline_nan_must_match_exactly(replacement):
    p = payload(4)
    p["stock_params"][2, 4] = np.nan
    b = deepcopy(p)
    b["stock_params"][2, 4] = replacement
    with pytest.raises(ValueError, match="stock reference changed"):
        C.summary(p, b)


@pytest.mark.parametrize("kind", ["missing", "extra", "run", "subrun", "event", "seed", "duplicate"])
def test_baseline_missing_extra_duplicate_and_different_full_keys_fail(kind):
    p, b = payload(4), payload(4)
    if kind == "missing":
        b = permute(b, [0, 1, 2])
    elif kind == "extra":
        b = payload(5)
    elif kind == "duplicate":
        b["keys"][3] = b["keys"][0]
    else:
        b["keys"][0, {"run": 0, "subrun": 1, "event": 2, "seed": 3}[kind]] += 100
    with pytest.raises(ValueError, match="keys"):
        C.summary(p, b)


@pytest.mark.parametrize("field", C.REQUIRED)
def test_required_fields_missing(field):
    p = payload()
    del p[field]
    with pytest.raises(ValueError, match="missing"):
        C.validate(p)


@pytest.mark.parametrize("value", [None, [], "payload", np.zeros(3)])
def test_nonmapping_payload(value):
    with pytest.raises(ValueError, match="mapping"):
        C.validate(value)


@pytest.mark.parametrize("field,value", [
    ("keys", np.zeros((4, 3), dtype=int)), ("keys", np.zeros((4, 4))),
    ("keys", np.full((4, 4), np.nan)), ("keys", np.zeros((4, 4), dtype=bool)),
    ("good", np.ones(4, dtype=int)), ("stockok", np.ones((4, 1), dtype=bool)),
    ("kl", np.zeros((4, 6), dtype=int)), ("kl", np.zeros((4, 6), dtype=complex)),
    ("stock_params", np.zeros((4, 5))), ("stock_errors", np.ones((3, 6))),
    ("stat", np.zeros((4, 3))), ("stat", np.zeros(4, dtype=int)),
    ("stat", np.zeros((4, 2), dtype=bool)), ("other", np.ones(3)),
    ("other", np.array(1.)), ("other", np.full(4, "bad")),
    ("other", np.zeros(4, dtype=object)), ("other", [0, 1, 2, 3]),
    ("other", np.ma.array(np.zeros(4), mask=[0, 1, 0, 0])),
])
def test_malformed_fields_fail_closed(field, value):
    p = payload(4)
    p[field] = value
    with pytest.raises(ValueError):
        C.validate(p)


@pytest.mark.parametrize("kind", ["duplicate", "negative", "inconsistent_identity", "identity_type", "field_name"])
def test_invalid_keys_and_field_names(kind):
    p = payload(4)
    if kind == "duplicate":
        p["keys"][1] = p["keys"][0]
    elif kind == "negative":
        p["keys"][0, 1] = -1
    elif kind == "inconsistent_identity":
        p["event"] = p["keys"][:, 2] + 1
    elif kind == "identity_type":
        p["run"] = p["keys"][:, 0].astype(float)
    else:
        p[3] = np.zeros(4)
    with pytest.raises(ValueError):
        C.validate(p)


def test_unsigned_keys_large_values_and_signed_unsigned_alignment():
    p = payload(4)
    b = deepcopy(p)
    b["keys"] = b["keys"].astype(np.uint64)
    assert C.summary(p, b)["baseline"]["stock_reference_exact"]
    p["keys"] = p["keys"].astype(np.uint64)
    p["keys"][:, 0] += np.uint64(2**63)
    b = permute(p, [1, 3, 0, 2])
    assert C.validate(p) == 4
    assert C.summary(p, b)["baseline"]["stock_reference_exact"]
    b["keys"][0, 0] -= np.uint64(2**63)
    with pytest.raises(ValueError, match="census"):
        C.summary(p, b)


def test_float32_and_float16_arithmetic_does_not_overflow_before_promotion():
    p = payload(4)
    for name in ("kl", "stock_params", "stock_errors"):
        p[name] = p[name].astype(np.float16)
    p["kl"][:, :] = 60000
    p["stock_params"][:, :] = -60000
    with np.errstate(all="raise"):
        f = C.summary(p)["fidelity"]
    assert f["mom"]["n_finite"] == 4
    assert f["mom"]["within_0p1"] == 0


def test_summary_does_not_mutate_or_require_writable_inputs():
    p, b = paired()
    before, bbefore = deepcopy(p), deepcopy(b)
    for value in (*p.values(), *b.values()):
        value.flags.writeable = False
    C.validate(p)
    C.summary(p, b)
    for name in p:
        np.testing.assert_array_equal(p[name], before[name])
        np.testing.assert_array_equal(b[name], bbefore[name])


def test_large_vectorized_population_and_no_jax_import():
    n = 20000
    p = payload(0)
    p["keys"] = np.column_stack((np.arange(n) // 1000, np.arange(n) % 3,
                                 np.arange(n), np.arange(n) % 4))
    p["good"] = np.arange(n) % 2 == 0
    p["stockok"] = np.arange(n) % 3 == 0
    p["kl"] = np.zeros((n, 6))
    p["stock_params"] = np.zeros((n, 6))
    p["stock_errors"] = np.ones((n, 6))
    p["stat"] = np.zeros((n, 3), dtype=int)
    imports_before = set(sys.modules)
    result = C.summary(p, permute(p, np.arange(n)[::-1]))
    assert result["n"] == n
    assert result["baseline"]["acceptance"]["agreement"] == n
    assert not any(name == "jax" or name.startswith("jax.") for name in set(sys.modules) - imports_before)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_matching_nonfinite_kl_is_not_successful_paired_fidelity(value):
    p = payload(4)
    p["kl"][0, :] = value
    result = C.summary(p, permute(p, [2, 1, 3, 0]))
    f = result["baseline"]["common_both_good"]["candidate_minus_baseline"]
    for name in C.KL:
        assert f[name]["n"] == 4
        assert f[name]["n_nonfinite"] == 1
        assert f[name]["within_0p1"] == 3


def test_arithmetic_overflow_is_retained_failure_not_input_rejection():
    p = payload(4)
    p["kl"][0, :] = np.finfo(float).max
    p["stock_params"][0, :] = -np.finfo(float).max
    with np.errstate(all="raise"):
        result = C.summary(p)
    assert result["summary_all_output_nonfinite"]["kl"]["n_nonfinite"] == 0
    for name in C.KL:
        assert result["fidelity"][name]["n"] == 4
        assert result["fidelity"][name]["n_nonfinite"] == 1
        assert result["fidelity"][name]["within_0p1"] == 3


def test_baseline_diagnostics_include_its_rejected_output_and_status():
    p, b = payload(4), payload(4)
    b["good"][0] = False
    b["kl"][0, :] = np.nan
    b["stat"] = np.array([[7], [0], [0], [0]])
    result = C.summary(p, permute(b, [2, 0, 3, 1]))["baseline"]
    assert result["common_both_good"]["n_tracks"] == 3
    assert result["summary_all_output_nonfinite"]["kl"]["by_population"]["f1_rejected"]["n_nonfinite"] == 6
    assert result["status_counts"]["populations"]["f1_rejected"]["final"]["7"] == 1


@pytest.mark.parametrize("field", ["good", "stockok", "kl", "stock_params", "stock_errors"])
def test_baseline_types_are_validated_before_comparison(field):
    p, b = payload(4), payload(4)
    b[field] = b[field].astype(int)
    with pytest.raises(ValueError, match=field):
        C.summary(p, b)
