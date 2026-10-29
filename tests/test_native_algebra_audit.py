"""Independent short Gaussian chains, ragged batches, guards and evidence gate."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import f1_native_algebra_audit as A
import f1_native_decision_audit as D


def sample():
    iterations, pieces, effects = [], [], []
    # Same event, distinct seed/subrun: a positional or partial-key join is incorrect.
    for seed, subrun, material in ((0, 0, False), (1, 0, True), (0, 1, True)):
        ident = dict(run=1, subrun=subrun, event=9, iseed=seed, iteration_ordinal=0,
                     miter=1, iter=0, schema_version=1)
        cf, cb = np.eye(6)*2, np.eye(6)*3
        xf, xb = np.arange(6.)*.2, np.arange(6.)*.3 + .1
        wf, wb = np.linalg.inv(cf), np.linalg.inv(cb)
        bf, bb = wf@xf, wb@xb
        h1, h2 = np.eye(6)*.7, np.eye(6)*.4
        z1, z2 = np.arange(6.)*.5, np.arange(6.)*.6
        q, dm = np.eye(6)*.2, np.arange(6.)*.05+.1
        # Independent global Gaussian posterior on upstream/downstream states.
        # x_down = x_up + dm + noise. With no material there is a single state.
        if material:
            qi = np.linalg.inv(q)
            precision = np.block([[wf+h1+qi, -qi], [-qi, wb+h2+qi]])
            rhs = np.r_[bf+h1@z1-qi@dm, bb+h2@z2+qi@dm]
            covariance = np.linalg.inv(precision)
            mean = np.linalg.solve(precision, rhs)
            result = [(mean[:6], covariance[:6, :6]), (mean[6:], covariance[6:, 6:])]
        else:
            precision = wf+wb+h1+h2
            result = [(np.linalg.solve(precision, bf+bb+h1@z1+h2@z2), np.linalg.inv(precision))]
        kinds = [1, 2, 4] if material else [1, 4]
        iterations.append(dict(ident, capture_valid=True, capture_error="", decision_valid=True,
                               candidate_present=True, prior_valid=True, bounds_valid=True, effects_captured=True,
                               nold=1, ncandidate=len(result), neffects=len(kinds), npriors=2, nprior_weights=2,
                               prior_weight_matrices=np.stack([wf, wb]).ravel().tolist(),
                               prior_weight_vectors=np.stack([bf, bb]).ravel().tolist()))
        for owner, values in ((0, [(xf, cf)]), (1, result)):
            for index, (x, c) in enumerate(values):
                pieces.append(dict(ident, owner=owner, piece_index=index, parameters=x.tolist(), covariance=c.ravel().tolist()))
        for slot, kind in enumerate(kinds):
            w, z = (h1, z1) if kind == 1 else (h2, z2)
            effects.append(dict(ident, effect_index=slot, kind=kind, time=float(slot), active=True, in_fit=True,
                                payload_supported=True, has_weight=kind != 2, has_change=kind == 2,
                                weight_matrix=w.ravel().tolist() if kind != 2 else [],
                                weight_vector=(w@z).tolist() if kind != 2 else [],
                                change_parameters=dm.tolist() if kind == 2 else [],
                                change_covariance=q.ravel().tolist() if kind == 2 else []))
    return ak.Array(iterations), ak.Array(pieces)[::-1], ak.Array(effects)[::-1]


def replace(table, row, field, value):
    records = ak.to_list(table)
    records[row][field] = value
    return ak.Array(records)


def test_batched_short_chains():
    it, p, e = sample()
    report = A.audit(it[::-1], p, e)
    assert report["passed"], report
    assert report["n_replayed"] == 3 and report["n_candidate_pieces_replayed"] == 5
    assert not report["full_numerical_validation"]
    assert report["parameters"]["max_absolute_residual"] < 1e-13
    json.dumps(report, allow_nan=False)


def test_inactive_padding_and_no_material():
    eye = np.broadcast_to(np.eye(6), (2, 2, 6, 6)).copy()
    b = np.ones((2, 2, 6))
    for slots in (0, 3):
        x, c, valid = A.replay(eye, b, np.zeros((2, slots), int), np.zeros((2, slots, 6, 6)),
                               np.zeros((2, slots, 6)), np.zeros((2, slots, 6)), np.zeros((2, slots, 6, 6)))
        assert valid.sum() == 2
        np.testing.assert_allclose(x[:, 0], 1.)
        np.testing.assert_allclose(c[:, 0], np.broadcast_to(np.eye(6)/2, (2, 6, 6)))


def test_mismatch_and_exclusions():
    it, p, e = sample()
    report = A.audit(it, replace(p, 0, "parameters", [100.]*6), e)
    assert not report["passed"]
    assert report["parameters"]["first_mismatch_keys_per_iteration"] == [[1, 1, 9, 0, 0, 1]]
    report = A.audit(replace(it, 0, "decision_valid", False), p, e)
    assert report["passed"] and report["n_excluded_candidate_present_decision_invalid"] == 1
    assert report["n_excluded_decision_invalid"] == 1


@pytest.mark.parametrize("field,value,match", [
    ("nprior_weights", 1, "prior counts"), ("npriors", 0, "prior counts"),
    ("neffects", -1, "negative"), ("neffects", 3, "effect count"),
    ("ncandidate", 2, "piece count"), ("candidate_present", False, "presence"),
    ("effects_captured", False, "effects_captured"), ("prior_valid", False, "prior_valid"),
    ("capture_valid", False, "capture"), ("bounds_valid", False, "bounds_valid"),
    ("prior_weight_vectors", [float("nan")]*12, "nonfinite")])
def test_iteration_guards(field, value, match):
    it, p, e = sample()
    with pytest.raises(ValueError, match=match):
        A.audit(replace(it, 0, field, value), p, e)


@pytest.mark.parametrize("field,value,match", [
    ("effect_index", -1, "identity"), ("effect_index", 99, "outside"),
    ("miter", 99, "identity mismatch"), ("event", 99, "full identity"),
    ("kind", 3, "unsupported"), ("has_weight", False, "missing"),
    ("weight_matrix", [0.]*5, "shape"), ("time", -1., "time sorted"),
    ("payload_supported", False, "payload")])
def test_effect_guards(field, value, match):
    it, p, e = sample()
    with pytest.raises(ValueError, match=match):
        A.audit(it, p, replace(e, 0, field, value))


def test_duplicates_and_material_count():
    it, p, e = sample()
    for tables in ((ak.concatenate([it, it[:1]]), p, e), (it, ak.concatenate([p, p[:1]]), e),
                   (it, p, ak.concatenate([e, e[:1]]))):
        with pytest.raises(ValueError, match="duplicate"):
            A.audit(*tables)
    with pytest.raises(ValueError, match="candidate count"):
        A.audit(it, p, replace(e, 1, "active", False))


@pytest.mark.parametrize("method", ["lu", "chol", "chol-factor"])
def test_gate_highest_effect_cycle_exclusive(tmp_path, monkeypatch, method):
    monkeypatch.setattr(A, "INVERSE", method)
    # Reuse gate fixture only; replace capture payload with this independent Gaussian fixture.
    spec = importlib.util.spec_from_file_location("decision_test", Path(__file__).with_name("test_native_decision_audit.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    check, gate = helper.evidence(tmp_path)
    it, p, e = sample()
    capture = Path(gate["inputs"]["capture"]["path"])
    with uproot.update(capture) as root:
        for name, table in (("iterations", it), ("pieces", p), ("effects", e[:1]), ("effects", e)):
            root["kktrace/"+name] = {field: table[field] for field in table.fields}
    gate["inputs"]["capture"]["sha256"] = D.digest(capture)
    check.write_text(json.dumps(gate))
    args = SimpleNamespace(check=check, out=tmp_path / "algebra.json")
    report = A.run(args)
    assert report["passed"], report
    assert report["highest_cycles"]["capture"]["kktrace/effects"].endswith("effects;2")
    assert report["all_four_evidence_hashes_verified"]
    assert json.loads(args.out.read_text()) == report
    if method == "chol-factor":
        assert report["inverse_stats"]["n_cholesky_success"] > 0
        assert report["inverse_stats"]["n_cholesky_fallback"] == 0
        assert A.CHOL_FACTOR_LIMIT in report["limits"]
        repeated = A.run(SimpleNamespace(check=check, out=tmp_path / "repeat.json"))
        assert repeated["inverse_stats"] == report["inverse_stats"]
    else:
        assert "inverse_stats" not in report
        assert report["limits"] == A.LIMITS
    with pytest.raises(FileExistsError):
        A.run(args)
    gate["smoke_only"] = True
    check.write_text(json.dumps(gate))
    report = A.run(SimpleNamespace(check=check, out=tmp_path / "fail.json"))
    assert not report["passed"] and "full capture passivity" in report["error"]
    if method == "chol-factor":
        assert report["inverse_stats"] == {"n_cholesky_success": 0, "n_cholesky_fallback": 0}


def test_singular_inverse_no_regularization():
    with pytest.raises(ValueError, match="singular"):
        A.inverse(np.zeros((2, 6, 6)))
    it, p, e = sample()
    # Singular priors alone are legal weights; only actual inverse operations fail.
    report = A.audit(replace(it, 0, "prior_weight_matrices", [0.]*72), p, e)
    assert not report["passed"]


def test_all_no_material_chain():
    it, p, e = sample()
    keep_p = (D.column(p, "subrun") == 0) & (D.column(p, "iseed") == 0)
    keep_e = (D.column(e, "subrun") == 0) & (D.column(e, "iseed") == 0)
    report = A.audit(it[:1], p[keep_p], e[keep_e])
    assert report["passed"] and report["n_candidate_pieces_replayed"] == 1


def correlated_batch():
    rng = np.random.default_rng(371)
    chains = [(2, 2, 0, 2), (0, 2, 0, 2, 0, 1, 2, 4), (1, 0, 2, 2), (4, 0, 1), ()]
    n, slots = len(chains), max(map(len, chains))
    kind = np.zeros((n, slots), int)
    weights, noise = np.zeros((2, n, slots, 6, 6))
    vectors, changes = np.zeros((2, n, slots, 6))
    prior_w, prior_b = np.empty((n, 2, 6, 6)), np.empty((n, 2, 6))
    expected = []

    def positive():
        factor = rng.normal(size=(6, 6))
        return factor @ factor.T * .2 + np.eye(6) * .5

    for row, chain in enumerate(chains):
        kind[row, :len(chain)] = chain
        count = chain.count(2) + 1
        precision, rhs = np.zeros((6*count, 6*count)), np.zeros(6*count)
        for side, block in ((0, slice(0, 6)), (1, slice(-6, None))):
            prior_w[row, side] = positive()
            prior_b[row, side] = prior_w[row, side] @ rng.normal(size=6)
            precision[block, block] += prior_w[row, side]
            rhs[block] += prior_b[row, side]
        piece = 0
        for slot, effect in enumerate(chain):
            block = slice(6*piece, 6*(piece+1))
            if effect == 2:
                noise[row, slot], changes[row, slot] = positive(), rng.normal(size=6)
                design = np.zeros((6, 6*count))
                design[:, block] = -np.eye(6)
                design[:, 6*(piece+1):6*(piece+2)] = np.eye(6)
                precision += design.T @ np.linalg.solve(noise[row, slot], design)
                rhs += design.T @ np.linalg.solve(noise[row, slot], changes[row, slot])
                piece += 1
            elif effect:
                weights[row, slot] = positive()
                vectors[row, slot] = weights[row, slot] @ rng.normal(size=6)
                precision[block, block] += weights[row, slot]
                rhs[block] += vectors[row, slot]
        mean = np.linalg.solve(precision, rhs).reshape(count, 6)
        covariance = np.linalg.inv(precision)
        marginal = np.stack([covariance[6*j:6*(j+1), 6*j:6*(j+1)] for j in range(count)])
        expected.append((mean, marginal))
    return (prior_w, prior_b, kind, weights, vectors, changes, noise), expected


@pytest.mark.parametrize("method", ["lu", "chol", "chol-factor"])
def test_correlated_multiple_material_ragged_posterior(monkeypatch, method):
    monkeypatch.setattr(A, "INVERSE", method)
    operands, expected = correlated_batch()
    x, c, valid = A.replay(*operands)
    for row, (mean, covariance) in enumerate(expected):
        np.testing.assert_array_equal(valid[row], np.arange(valid.shape[1]) < len(mean))
        np.testing.assert_allclose(x[row, valid[row]], mean, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(c[row, valid[row]], covariance, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("method", ["lu", "chol", "chol-factor"])
def test_demand_driven_conversion_spy(monkeypatch, method):
    monkeypatch.setattr(A, "INVERSE", method)
    operands, _ = correlated_batch()
    kind = operands[2]
    expected = []
    for direction in (1, -1):
        cached = np.zeros(len(kind), bool)
        for slot in (range(kind.shape[1]) if direction == 1 else range(kind.shape[1]-1, -1, -1)):
            material = kind[:, slot] == 2
            needed = int((material & ~cached).sum())
            if needed:
                expected.extend([("parameters", (needed, 6, 6)), ("inverse", (needed, 6, 6))])
            if material.any():
                expected.append(("inverse", (int(material.sum()), 6, 6)))
            cached[kind[:, slot] != 0] = material[kind[:, slot] != 0]
    final_shape = (len(kind), kind.shape[1]+1, 6, 6)
    expected.extend([("parameters", final_shape), ("inverse", final_shape)])
    calls = []
    original_parameters, original_inverse = A.parameters, A.inverse

    def parameters(w, b):
        calls.append(("parameters", w.shape))
        return original_parameters(w, b)

    def inverse(matrix):
        calls.append(("inverse", matrix.shape))
        return original_inverse(matrix)

    monkeypatch.setattr(A, "parameters", parameters)
    monkeypatch.setattr(A, "inverse", inverse)
    A.replay(*operands)
    assert calls == expected
    assert sum(shape[0] for name, shape in calls[:-2] if name == "parameters") == 8
    assert sum(shape[0] for name, shape in calls[:-2] if name == "inverse") == 24


@pytest.mark.parametrize("method", ["lu", "chol", "chol-factor"])
@pytest.mark.parametrize("stage", ["weight", "covariance", "posterior"])
def test_singular_used_replay_operands_fail(monkeypatch, method, stage):
    monkeypatch.setattr(A, "INVERSE", method)
    prior_w = np.broadcast_to(np.eye(6), (1, 2, 6, 6)).copy()
    prior_b = np.zeros((1, 2, 6))
    kind = np.array([[2 if stage != "posterior" else 0]])
    noise = np.zeros((1, 1, 6, 6))
    if stage == "weight":
        prior_w[0, 0] = 0.
    elif stage == "covariance":
        noise[0, 0] = -np.eye(6)
    else:
        prior_w[:] = 0.
    with pytest.raises(ValueError, match="singular matrix; no regularization"):
        A.replay(prior_w, prior_b, kind, np.zeros_like(noise), np.zeros((1, 1, 6)), np.zeros((1, 1, 6)), noise)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3), (2, 1, 3), (0,), (2, 0, 3)])
def test_chol_factor_correlated_spd_shapes(monkeypatch, batch_shape):
    monkeypatch.setattr(A, "INVERSE", "chol-factor")
    rng = np.random.default_rng(233771)
    factor = rng.normal(size=batch_shape + (6, 6))
    matrix = factor @ np.swapaxes(factor, -1, -2) + np.eye(6) * .5
    vector = rng.normal(size=batch_shape + (6,))
    expected = np.linalg.inv(matrix)
    expected_x = np.einsum("...ij,...j->...i", expected, vector)
    # DMAT's lower triangle is authoritative, including for parameters(weight, vector).
    stored = np.tril(matrix) + np.triu(np.full_like(matrix, 123.), 1)
    before = stored.copy()
    old_success = A.INVERSE_STATS["n_cholesky_success"]
    old_fallback = A.INVERSE_STATS["n_cholesky_fallback"]
    x, covariance = A.parameters(stored, vector)
    assert covariance.shape == batch_shape + (6, 6)
    assert x.shape == batch_shape + (6,)
    np.testing.assert_allclose(covariance, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(x, expected_x, rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(covariance, np.swapaxes(covariance, -1, -2))
    np.testing.assert_array_equal(stored, before)
    assert A.INVERSE_STATS["n_cholesky_success"] - old_success == int(np.prod(batch_shape))
    assert A.INVERSE_STATS["n_cholesky_fallback"] == old_fallback


def test_chol_factor_mixed_indefinite_fallback(monkeypatch):
    monkeypatch.setattr(A, "INVERSE", "chol-factor")
    rng = np.random.default_rng(463)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    indefinite = (q * np.array([-2., -1., .5, 1., 2., 3.])) @ q.T
    matrix = np.stack([np.eye(6), indefinite, -np.eye(6), np.eye(6)*2]).reshape(2, 2, 6, 6)
    sym = np.tril(matrix) + np.swapaxes(np.tril(matrix, -1), -1, -2)
    expected = np.linalg.inv(sym)
    original_inv = np.linalg.inv
    calls = []

    def fallback(operand):
        calls.append(operand.copy())
        return original_inv(operand)

    monkeypatch.setattr(np.linalg, "inv", fallback)
    old = A.INVERSE_STATS.copy()
    result = A.inverse(np.tril(matrix) + np.triu(np.full_like(matrix, -91.), 1))
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], sym.reshape(-1, 6, 6)[[1, 2]])
    np.testing.assert_allclose(result, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(result, np.swapaxes(result, -1, -2))
    assert A.INVERSE_STATS["n_cholesky_success"] - old["n_cholesky_success"] == 2
    assert A.INVERSE_STATS["n_cholesky_fallback"] - old["n_cholesky_fallback"] == 2


@pytest.mark.parametrize("injected_root", [.5, np.nan, 0.])
def test_chol_factor_positive_eigenvalues_do_not_override_failed_pivot(monkeypatch, injected_root):
    monkeypatch.setattr(A, "INVERSE", "chol-factor")
    matrix = np.broadcast_to(np.eye(6), (2, 6, 6)).copy()
    matrix[0, 0, 1] = matrix[0, 1, 0] = .75
    assert np.all(np.linalg.eigvalsh(matrix) > 0)
    expected = np.linalg.inv(matrix)
    original_sqrt, original_inv = np.sqrt, np.linalg.inv
    roots, fallbacks = [], []

    def sqrt(pivot):
        result = original_sqrt(pivot)
        # Inject a factor arithmetic failure in one SPD lane. The next actual
        # computed pivot is negative/nonfinite, despite positive eigenvalues.
        if not roots:
            result[0] = injected_root
        roots.append(pivot.copy())
        return result

    def inv(operand):
        fallbacks.append(operand.copy())
        return original_inv(operand)

    def forbidden(*args, **kwargs):
        pytest.fail("chol-factor must not use an eigentest or all-or-nothing NumPy Cholesky")

    monkeypatch.setattr(np, "sqrt", sqrt)
    monkeypatch.setattr(np.linalg, "inv", inv)
    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden)
    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    result = A.inverse(matrix)
    assert len(fallbacks) == 1
    np.testing.assert_array_equal(fallbacks[0], matrix[:1])
    np.testing.assert_allclose(result, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 1)])
def test_chol_factor_singular_fallback_error(monkeypatch, batch_shape):
    monkeypatch.setattr(A, "INVERSE", "chol-factor")
    with pytest.raises(ValueError, match="Cholesky factor failed; fallback singular matrix; no regularization"):
        A.inverse(np.zeros(batch_shape + (6, 6)))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_chol_factor_nonfinite_operand(monkeypatch, bad):
    monkeypatch.setattr(A, "INVERSE", "chol-factor")
    matrix = np.eye(6)
    matrix[0, 0] = bad
    with pytest.raises(ValueError, match="nonfinite inverse operand"):
        A.inverse(matrix)


def test_chol_factor_nonfinite_fallback_error(monkeypatch):
    monkeypatch.setattr(A, "INVERSE", "chol-factor")
    monkeypatch.setattr(np.linalg, "inv", lambda matrix: np.full_like(matrix, np.nan))
    with pytest.raises(ValueError, match="Cholesky factor failed; nonfinite fallback inverse; no regularization"):
        A.inverse(-np.eye(6))


@pytest.mark.parametrize("shape", [(5, 5), (2, 6, 5), (6,)])
def test_chol_factor_requires_six_components(monkeypatch, shape):
    monkeypatch.setattr(A, "INVERSE", "chol-factor")
    with pytest.raises(ValueError, match="requires .*6,6"):
        A.inverse(np.ones(shape))


def test_chol_factor_multiple_material_lower_storage(monkeypatch):
    monkeypatch.setattr(A, "INVERSE", "chol-factor")
    operands, expected = correlated_batch()
    # Exercise native lower-only weights and change covariances throughout replay.
    for index in (0, 3, 6):
        operands[index][...] = np.tril(operands[index])
    x, c, valid = A.replay(*operands)
    for row, (mean, covariance) in enumerate(expected):
        np.testing.assert_allclose(x[row, valid[row]], mean, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(c[row, valid[row]], covariance, rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(c, np.swapaxes(c, -1, -2))
