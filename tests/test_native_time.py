"""Certified time input guards and isolated scalar use in the diagnostic fitter."""
import json

import awkward as ak
import numpy as np
import pytest
import uproot

from trkfastsim.reco import nativetime as N


def fixture(tmp_path):
    capture, baseline, control, targets = [tmp_path / p for p in ("capture.root", "baseline", "control", "targets")]
    for path in (baseline, control, targets):
        path.write_text(path.name)
    with uproot.recreate(capture) as root:
        root["kktrace/seed_contract"] = {"run": np.array([1, 1]), "subrun": np.array([0, 0]),
            "event": np.array([11, 22]), "iseed": np.array([0, 0]), "valid": np.array([True, True]),
            "error": ak.Array(["", ""]), "range_begin": np.array([10., 30.]), "range_end": np.array([20., 40.]),
            "ph_time": ak.Array([[15.], [35.]]), "ph_mask": ak.Array([[0, 0, 0, 0, 0, 1]] * 2)}
    gate = {k: True for k in ("passed", "final_capture_gate", "complete_event_coverage", "full_target_coverage", "recorded_capture_equivalent")}
    gate.update(smoke_only=False, n_seeds=2, inputs={name: {"path": str(path), "sha256": N.digest(path)}
        for name, path in (("capture", capture), ("baseline", baseline), ("control", control), ("targets", targets))})
    check = tmp_path / "check.json"
    check.write_text(json.dumps(gate))
    return check, baseline, gate


def test_native_time_join_and_hashes(tmp_path):
    check, baseline, gate = fixture(tmp_path)
    times, meta = N.load_times(check, baseline, np.array([[1, 0, 22, 0], [1, 0, 11, 0]]))
    np.testing.assert_array_equal(times, [35., 15.])
    assert type(times) is np.ndarray
    assert meta["stock_exact_fit"] is False
    assert meta["changed_operand"] == "immutable constraint time only"
    baseline.write_text("changed")
    with pytest.raises(ValueError, match="hash changed"):
        N.load_times(check, baseline, np.array([[1, 0, 11, 0]]))


@pytest.mark.parametrize("key", ["passed", "final_capture_gate", "complete_event_coverage", "full_target_coverage", "recorded_capture_equivalent"])
def test_incomplete_gate_rejected(tmp_path, key):
    check, baseline, gate = fixture(tmp_path)
    gate[key] = False
    check.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="full capture"):
        N.load_times(check, baseline, np.array([[1, 0, 11, 0]]))


def test_join_rejects_missing_and_duplicate_seeds():
    keys = np.array([[1, 0, 11, 0], [1, 0, 22, 0]])
    with pytest.raises(ValueError, match="missing"):
        N.join_keys(np.array([[1, 0, 99, 0]]), keys)
    with pytest.raises(ValueError, match="duplicate"):
        N.join_keys(keys[[0, 0]], keys)


def test_native_constraint_scalar_keeps_reconstructed_owner(monkeypatch):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from trkfastsim.reco import kalman_jax as K
    from test_kalman_jax import _toy_track, _anns_available
    if not _anns_available():
        pytest.skip("ANN weights unavailable")
    xt, h, hj, cj, d, dj = _toy_track(3)
    x0 = jnp.asarray(xt + np.array([3., -2., .01, .005, 2.]))
    sx = jnp.r_[x0, K.MOM]
    hs = K.straw_poca(sx, hj["mid"], hj["wdir"])[0]
    xs = K.straw_poca(sx, cj["mid"], cj["wdir"])[0] + K.kkmaterial.STRAW["wrad"]
    times = jnp.r_[sx[4] + hs / K.speed(sx[5]), sx[4] + xs / K.speed(sx[5])]
    expected = .5 * (times.min() + times.max())
    owners = []
    original = K.seed_owner
    def record(*args):
        owners.append(tuple(np.asarray(x) for x in args[:4]))
        return original(*args)
    monkeypatch.setattr(K, "seed_owner", record)
    f = lambda tc: K.fit_track(x0, hj, cj, dj, K.stock_stages(), K.load_anns(), time_model="stock", constraint_time=tc)
    old = f(None)
    same = f(expected)
    changed = f(expected + 1.)
    for a, b in zip(jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(same)):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
    for a, b in zip(owners[0], owners[2]):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(np.asarray(old[0]["xe"]), np.asarray(changed[0]["xe"]))
    with pytest.raises(ValueError, match="native constraint"):
        K.fit_track(x0, hj, cj, dj, K.stock_stages(), K.load_anns(), constraint_time=expected)
