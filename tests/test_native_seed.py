"""Native seed coordinate, covariance, capture guards and operand isolation tests."""
import json

import awkward as ak
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import uproot

from trkfastsim.reco import kalman_jax as K, nativeseed as N, nativetime as T


def operands():
    x = np.array([[50., -900., .2, .7, 35., 2000.], [-70., 300., -.3, -.4, 95., 800.]])
    par = np.asarray(jax.vmap(K.kl_params)(jnp.asarray(x)))
    rng = np.random.default_rng(42)
    a = rng.normal(size=(2, 6, 6))
    a *= np.array([15., .05, 10., .02, 2., 1.])[None, :, None]
    cov = a @ a.swapaxes(1, 2)
    return x, par, cov


def fixture(tmp_path, change=None, cycle=False):
    _, par, cov = operands()
    capture, baseline, control, targets = [tmp_path / p for p in ("capture.root", "baseline", "control", "targets")]
    for path in (baseline, control, targets):
        path.write_text(path.name)
    ph_cov = np.zeros((2, 6, 6))
    ph_cov[:, 5, 5] = K.MOM_CONSTRAINT ** 2
    data = {"run": np.array([1, 1]), "subrun": np.array([0, 0]), "event": np.array([11, 22]),
            "iseed": np.array([0, 0]), "valid": np.array([True, True]), "error": ak.Array(["", ""]),
            "schema_version": np.array([1, 1]), "mass": np.full(2, K.MMU), "parameters": par.copy(),
            "covariance": cov.reshape(2, 36).copy(), "bnom": np.array([[0., 0., .001]] * 2),
            "range_begin": np.array([10., 30.]), "range_end": np.array([20., 40.]),
            "ph_time": np.array([[15.], [35.]]), "ph_mask": np.array([[0, 0, 0, 0, 0, 1]] * 2),
            "ph_parameters": par.copy(), "ph_covariance": ph_cov.reshape(2, 36)}
    if change:
        change(data)
    with uproot.recreate(capture) as root:
        if cycle:
            root["kktrace/seed_contract"] = dict(data, schema_version=np.array([999, 999]))
        root["kktrace/seed_contract"] = data
    gate = {k: True for k in ("passed", "final_capture_gate", "complete_event_coverage", "full_target_coverage", "recorded_capture_equivalent")}
    gate.update(smoke_only=False, n_seeds=2, inputs={name: {"path": str(path), "sha256": T.digest(path)}
        for name, path in (("capture", capture), ("baseline", baseline), ("control", control), ("targets", targets))})
    check = tmp_path / "check.json"
    check.write_text(json.dumps(gate))
    return check, baseline, np.array([[1, 0, 22, 0], [1, 0, 11, 0]])


def test_transform_roundtrip_and_full_covariance():
    x, par, cov = operands()
    before = par.copy(), cov.copy()
    actual, transformed = N.transform_seed(par, cov)
    np.testing.assert_allclose(actual, x, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(jax.vmap(K.kl_params)(jnp.asarray(actual)), par, rtol=1e-13, atol=1e-12)
    forward = np.asarray(jax.vmap(jax.jacfwd(K.kl_params))(jnp.asarray(x)))
    inverse = np.linalg.inv(forward)
    np.testing.assert_allclose(transformed, inverse @ cov @ inverse.swapaxes(1, 2), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(forward @ transformed @ forward.swapaxes(1, 2), cov, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(par, before[0])
    np.testing.assert_array_equal(cov, before[1])
    assert abs(transformed[0, 4, 5]) > 0


def test_native_seed_join_cycle_metadata_and_time_isolation(tmp_path, monkeypatch):
    check, baseline, keys = fixture(tmp_path, cycle=True)
    original = T.load_times
    calls = []
    def verified(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(T, "load_times", verified)
    before, _ = original(check, baseline, keys)
    x, cov, ranges, meta = N.load_seed(check, baseline, keys)
    expected_x, par, native_cov = operands()
    expected_cov = N.transform_seed(par, native_cov)[1]
    np.testing.assert_allclose(x, expected_x[::-1], rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(cov, expected_cov[::-1])
    np.testing.assert_array_equal(ranges, [[30., 40.], [10., 20.]])
    assert len(calls) == 1 and calls[0][:2] == (check, baseline)
    np.testing.assert_array_equal(calls[0][2], keys)
    np.testing.assert_array_equal(original(check, baseline, keys)[0], before)
    assert all(type(a) is np.ndarray for a in (x, cov, ranges))
    assert meta["seed_tree"] == "kktrace/seed_contract;2"
    assert meta["changed_operands"] == ["seed parameters", "seed covariance", "initial owner range",
                                        "initial prior operands", "seed-derived momentum constraint target"]
    assert "immutable constraint time" in meta["unchanged"]
    assert meta["stock_exact_fit"] is False
    assert "same native-time check" in meta["constraint_time_policy"]
    json.dumps(meta, allow_nan=False)


@pytest.mark.parametrize("field,index,value,error", [
    ("schema_version", 0, 2, "schema_version"),
    ("mass", 0, 0., "mass"),
    ("mass", 0, np.nan, "nonfinite"),
    ("bnom", (0, 2), np.inf, "nonfinite"),
    ("parameters", (0, 0), np.nan, "nonfinite"),
    ("parameters", (0, 5), 0., "momentum"),
    ("parameters", (0, 1), 1., "downward"),
    ("parameters", (0, 1), 0., "downward"),
    ("covariance", (0, 0), -1., "positive definite"),
    ("covariance", (0, 1), 99., "symmetric"),
    ("covariance", (0, 0), np.nan, "nonfinite"),
    ("ph_parameters", (0, 5), 1900., "momentum"),
    ("ph_covariance", (0, 35), 1., "sigma"),
    ("ph_covariance", (0, 35), -1., "sigma"),
    ("ph_covariance", (0, 0), np.nan, "nonfinite"),
    ("ph_mask", (0, 0), 1, "momentum-only"),
    ("range_end", 0, 0., "reversed"),
    ("range_begin", 0, np.nan, "nonfinite"),
    ("ph_time", (0, 0), 14., "midpoint"),
    ("valid", 0, False, "invalid seed"),
])
def test_invalid_native_seed_fails_closed(tmp_path, field, index, value, error):
    def change(data):
        data[field][index] = value
    check, baseline, keys = fixture(tmp_path, change)
    with pytest.raises(ValueError, match=error):
        N.load_seed(check, baseline, keys)


@pytest.mark.parametrize("field", ["schema_version", "mass", "bnom", "parameters", "covariance", "ph_parameters", "ph_covariance"])
def test_missing_native_operand_rejected(tmp_path, field):
    check, baseline, keys = fixture(tmp_path, lambda d: d.pop(field))
    with pytest.raises(ValueError, match="missing schema-1"):
        N.load_seed(check, baseline, keys)


def test_gate_and_identity_rejection(tmp_path):
    check, baseline, keys = fixture(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        N.load_seed(check, baseline, keys[[0, 0]])
    with pytest.raises(ValueError, match="missing"):
        N.load_seed(check, baseline, keys + [0, 0, 100, 0])
    with pytest.raises(ValueError, match="integer"):
        N.load_seed(check, baseline, keys.astype(float))
    gate = json.loads(check.read_text())
    gate["final_capture_gate"] = False
    check.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="full capture"):
        N.load_seed(check, baseline, keys)
    gate["final_capture_gate"] = True
    check.write_text(json.dumps(gate))
    baseline.write_text("changed")
    with pytest.raises(ValueError, match="hash changed"):
        N.load_seed(check, baseline, keys)


def test_shape_and_nonpositive_covariance_guards(tmp_path):
    check, baseline, keys = fixture(tmp_path, lambda d: d.update(parameters=np.zeros((2, 5))))
    with pytest.raises(ValueError, match="shape"):
        N.load_seed(check, baseline, keys)
    _, par, cov = operands()
    with pytest.raises(ValueError, match="positive definite"):
        N.transform_seed(par, np.zeros_like(cov))
    with pytest.raises(ValueError, match="momentum"):
        N.transform_seed(par * [1, 1, 1, 1, 1, -1], cov)


def test_finite_bnom_not_used_and_supported_muon_rounding(tmp_path):
    def change(data):
        data["bnom"][:] = [100., -70., 4.]
        data["mass"][:] -= 4e-6
    check, baseline, keys = fixture(tmp_path, change)
    x, _, _, _ = N.load_seed(check, baseline, keys)
    np.testing.assert_allclose(x, operands()[0][::-1], rtol=1e-13, atol=1e-12)


def test_ranges_and_verified_time_do_not_change_transformed_seed(tmp_path):
    def change(data):
        data["range_begin"] += 1000.
        data["range_end"] += 1000.
        data["ph_time"] += 1000.
    check, baseline, keys = fixture(tmp_path, change)
    before = {str(p): T.digest(p) for p in (check, baseline, tmp_path / "capture.root")}
    times, _ = T.load_times(check, baseline, keys)
    x, cov, ranges, _ = N.load_seed(check, baseline, keys)
    expected_x, par, native_cov = operands()
    np.testing.assert_allclose(x, expected_x[::-1], rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(cov, N.transform_seed(par, native_cov)[1][::-1])
    np.testing.assert_array_equal(ranges, [[1030., 1040.], [1010., 1020.]])
    np.testing.assert_array_equal(times, ranges.mean(axis=1))
    assert not np.any(x[:, 4] == times)
    assert before == {p: T.digest(p) for p in before}


@pytest.mark.parametrize("sigma", [0., -1., np.nan, np.inf, .001])
def test_configured_constraint_must_agree(tmp_path, monkeypatch, sigma):
    check, baseline, keys = fixture(tmp_path)
    monkeypatch.setattr(K, "MOM_CONSTRAINT", sigma)
    with pytest.raises(ValueError, match="sigma"):
        N.load_seed(check, baseline, keys)


def test_tiny_positive_covariance_not_regularized():
    _, par, _ = operands()
    cov = np.broadcast_to(np.diag([1e-20, 2e-20, 3e-20, 4e-20, 5e-20, 6e-20]), (2, 6, 6)).copy()
    _, actual = N.transform_seed(par, cov)
    jac = np.asarray(jax.vmap(jax.jacfwd(N.to_fitter6))(jnp.asarray(par)))
    np.testing.assert_allclose(actual, jac @ cov @ jac.swapaxes(1, 2), rtol=1e-13, atol=0)
    np.testing.assert_array_equal(actual[:, 5, 5], cov[:, 5, 5])
