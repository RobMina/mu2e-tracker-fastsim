"""Synthetic row-policy and fail-closed provenance tests; no native ROOT inputs."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import f1_measurement_audit as A


VARIANTS = {
    "saved": (False, False, False),
    "time_only": (False, True, False),
    "long_only": (False, False, True),
    "time_long": (False, True, True),
    "all_source_rows": (True, True, True),
}
SOURCE_NAMES = ("kalman_jax.py", "linefit_jax.py", "linefit.py", "hitreco.py", "cli.py", "combo.py",
                "strawaxis.py", "nativegeometry.py", "nativecompare.py", "vr_f1.py")


def independent_information(q, H, r, w):
    """Native residual convention: W=sum(w H^T H), b=Wq-sum(w H^T r)."""
    W = np.sum(w[..., None, None] * H[..., :, None] * H[..., None, :], axis=-3)
    b = (W @ q[..., None])[..., 0] - np.sum((w * r)[..., None] * H, axis=-2)
    return W, b


@pytest.fixture
def operands():
    q = np.array([[2., -3., 4., .5, -1., 6.], [-2., 1., 3., 5., .25, 2.],
                  [1., 3., -2., 4., 6., -.5], [3., 2., 1., -1., .5, 4.]])
    actual = (np.arange(72, dtype=float).reshape(4, 3, 6) - 30.) / 16.
    delta = np.array([[1., -.5, .25, 0., 2., -.25], [.5, 1., -.75, 2., .25, 0.],
                      [-.5, .25, 1., -.25, 0., 2.]])
    source = actual + delta[None] * np.array([1., 2., -.5, 3.])[:, None, None]
    r = np.array([[.5, -1., 2.], [2., .25, -3.], [-1., 4., .5], [3., -2., 1.]])
    w = np.array([[.5, 2., 4.], [0., 3., 1.], [2., 0., .25], [0., 0., 0.]])
    return q, actual, source, r, w


def selected_rows(actual, source, mask):
    return np.where(np.asarray(mask)[None, :, None], source, actual)


def test_substitutions_replace_only_requested_rows_without_mutating_inputs(operands, monkeypatch):
    q, actual, source, r, w = operands
    native_W, native_b = independent_information(q, source, r, w)
    inputs = (*operands, native_W, native_b)
    snapshots = [value.copy() for value in inputs]
    for value in inputs:
        value.setflags(write=False)
    calls = []
    information = A.N.information

    def record(parameters, H, residual, weight):
        assert parameters is q and residual is r and weight is w
        assert not np.shares_memory(H, actual)
        assert not np.shares_memory(H, source)
        W, b = information(parameters, H, residual, weight)
        calls.append((H.copy(), W.copy(), b.copy()))
        return W, b

    monkeypatch.setattr(A.N, "information", record)
    result = A.substitutions(*inputs)
    assert list(result) == list(VARIANTS)
    assert len(calls) == len(VARIANTS)
    for (H, W, b), mask in zip(calls, VARIANTS.values()):
        expected_H = selected_rows(actual, source, mask)
        np.testing.assert_array_equal(H, expected_H)
        expected_W, expected_b = independent_information(q, expected_H, r, w)
        np.testing.assert_allclose(W, expected_W, rtol=1e-13, atol=1e-12)
        np.testing.assert_allclose(b, expected_b, rtol=1e-13, atol=1e-12)
    for value, snapshot in zip(inputs, snapshots):
        np.testing.assert_array_equal(value, snapshot)


@pytest.mark.parametrize("target", VARIANTS)
def test_substitution_metrics_against_independent_native_operands(operands, target):
    q, actual, source, r, w = operands
    native_H = selected_rows(actual, source, VARIANTS[target])
    native_W, native_b = independent_information(q, native_H, r, w)
    result = A.substitutions(*operands, native_W, native_b)
    for label, mask in VARIANTS.items():
        H = selected_rows(actual, source, mask)
        W, b = independent_information(q, H, r, w)
        bad_W = ~np.isclose(W, native_W, rtol=1e-5, atol=1e-7)
        bad_b = ~np.isclose(b, native_b, rtol=1e-5, atol=1e-7)
        entry = result[label]
        assert entry["weight_matrix"]["passed"] == (label == target)
        assert entry["weight_vector"]["passed"] == (label == target)
        assert entry["weight_matrix"]["mismatches"] == int(bad_W.sum())
        assert entry["weight_vector"]["mismatches"] == int(bad_b.sum())
        assert entry["failing_hit_indices"] == np.flatnonzero(
            np.any(bad_W, axis=(-1, -2)) | np.any(bad_b, axis=-1)).tolist()
        assert 3 not in entry["failing_hit_indices"]
    json.dumps(result, allow_nan=False)


def test_zero_weight_support_distinguishes_time_long_and_doca():
    q = np.tile([2., 3., 4., 0., 0., 0.], (4, 1))
    actual = np.broadcast_to(np.eye(3, 6), (4, 3, 6)).copy()
    source = 2. * actual
    r = np.ones((4, 3))
    w = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.], [0., 0., 0.]])
    native_W = np.zeros((4, 6, 6))
    native_W[0, 0, 0], native_W[1, 1, 1], native_W[2, 2, 2] = 4., 8., 12.
    native_b = np.zeros((4, 6))
    native_b[0, 0], native_b[1, 1], native_b[2, 2] = 6., 20., 42.
    result = A.substitutions(q, actual, source, r, w, native_W, native_b)
    expected = {"saved": [0, 1, 2], "time_only": [0, 2], "long_only": [0, 1],
                "time_long": [0], "all_source_rows": []}
    for label, failing in expected.items():
        assert result[label]["failing_hit_indices"] == failing
        assert result[label]["weight_matrix"]["mismatches"] == len(failing)
        assert result[label]["weight_vector"]["mismatches"] == len(failing)


def test_failing_hit_indices_union_matrix_only_and_vector_only(operands):
    q, actual, _, r, w = operands
    native_W, native_b = independent_information(q, actual, r, w)
    native_W[2, 0, 0] += 1.
    native_b[1, 5] -= 2.
    result = A.substitutions(*operands, native_W, native_b)["saved"]
    assert result["weight_matrix"]["first_mismatch_indices"] == [[2, 0, 0]]
    assert result["weight_vector"]["first_mismatch_indices"] == [[1, 5]]
    assert result["failing_hit_indices"] == [1, 2]


def forbidden_downstream(*args, **kwargs):
    pytest.fail("early audit gate reached downstream data or row reconstruction")


@pytest.fixture
def saved_trace(tmp_path, monkeypatch):
    root = Path(A.__file__).resolve().parents[1]
    key = np.array([1, 2, 3, 0], dtype=np.int64)
    np.savez(tmp_path / "trace.npz", key=key, stage_index=np.array([0]))
    manifest = {
        "completed": True, "trace_passivity_exact": True, "environment": {},
        "check_sha256": "synthetic-native-check", "trace_sha256": A.digest(tmp_path / "trace.npz"),
        "time_model": "stock", "reference_model": "previous", "straw_geometry": "approximate",
        "crossing_time": "current", "key": key.tolist(),
        "inputs": {"baseline": {"path": "unused-baseline.root", "sha256": "synthetic-baseline"}},
        "source": {"trkfastsim/reco/" + name: A.digest(root / "trkfastsim/reco" / name)
                   for name in SOURCE_NAMES},
    }
    for label in ("conditions", "alignment"):
        path = tmp_path / (label + ".txt")
        path.write_text("synthetic " + label + "\n")
        manifest[label] = {"path": str(path), "sha256": A.digest(path)}
    metadata = {"check_sha256": manifest["check_sha256"],
                "inputs": {"baseline": dict(manifest["inputs"]["baseline"])}}
    monkeypatch.setattr(A, "compare_first", forbidden_downstream)
    monkeypatch.setattr(A.fittrace, "shapes", forbidden_downstream)
    monkeypatch.setattr(A.uproot, "open", forbidden_downstream)
    return tmp_path, manifest, metadata


def run_early_audit(saved_trace, **kwargs):
    directory, manifest, metadata = saved_trace
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return A.audit_one(directory, metadata, None, None, None, None, None, None, **kwargs)


@pytest.mark.parametrize("field", ["completed", "trace_passivity_exact"])
@pytest.mark.parametrize("value", [False, None, 1, "missing"])
def test_audit_requires_literal_complete_and_passive(saved_trace, monkeypatch, field, value):
    manifest = saved_trace[1]
    if value == "missing":
        manifest.pop(field)
    else:
        manifest[field] = value
    monkeypatch.setattr(A.np, "load", forbidden_downstream)
    with pytest.raises(ValueError, match="requires complete passive saved trace"):
        run_early_audit(saved_trace)


@pytest.mark.parametrize("field,value,message", [
    ("environment", {"F1_TEST": "1"}, "requires default F1 environment"),
    ("check_sha256", "changed", "native gate differs"),
    ("trace_sha256", "changed", "trace hash changed"),
    ("time_model", "changed", "unsupported saved fit policy"),
    ("reference_model", "changed", "unsupported saved fit policy"),
    ("straw_geometry", "changed", "unsupported saved fit policy"),
    ("crossing_time", "changed", "unsupported saved fit policy"),
])
def test_audit_manifest_gates_before_loading(saved_trace, monkeypatch, field, value, message):
    saved_trace[1][field] = value
    monkeypatch.setattr(A.np, "load", forbidden_downstream)
    with pytest.raises(ValueError, match=message):
        run_early_audit(saved_trace)


def test_audit_detects_trace_bytes_changed_after_manifest(saved_trace, monkeypatch):
    with (saved_trace[0] / "trace.npz").open("ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(A.np, "load", forbidden_downstream)
    with pytest.raises(ValueError, match="trace hash changed"):
        run_early_audit(saved_trace)


@pytest.mark.parametrize("name", SOURCE_NAMES[:-1])
@pytest.mark.parametrize("allow_loader", [False, True])
def test_audit_nonloader_source_guard_cannot_be_waived(saved_trace, monkeypatch, name, allow_loader):
    rel = "trkfastsim/reco/" + name
    saved_trace[1]["source"][rel] = "changed"
    monkeypatch.setattr(A.np, "load", forbidden_downstream)
    with pytest.raises(ValueError, match="trace input/row source changed: " + rel):
        run_early_audit(saved_trace, allow_loader_reconstruction=allow_loader)


def test_audit_loader_source_requires_explicit_permission(saved_trace, monkeypatch):
    saved_trace[1]["source"]["trkfastsim/reco/vr_f1.py"] = "changed"
    monkeypatch.setattr(A.np, "load", forbidden_downstream)
    with pytest.raises(ValueError, match="loader source differs; explicit reconstruction option required"):
        run_early_audit(saved_trace)


def test_audit_loader_permission_does_not_bypass_key_guard(saved_trace):
    saved_trace[1]["source"]["trkfastsim/reco/vr_f1.py"] = "changed"
    saved_trace[1]["key"][2] += 1
    with pytest.raises(ValueError, match="trace key differs"):
        run_early_audit(saved_trace, allow_loader_reconstruction=True)


def test_audit_baseline_guard_before_loading(saved_trace, monkeypatch):
    saved_trace[1]["inputs"]["baseline"]["sha256"] = "changed"
    monkeypatch.setattr(A.np, "load", forbidden_downstream)
    with pytest.raises(ValueError, match="baseline differs"):
        run_early_audit(saved_trace)


@pytest.mark.parametrize("label", ["conditions", "alignment"])
def test_audit_auxiliary_input_hash_guard(saved_trace, monkeypatch, label):
    Path(saved_trace[1][label]["path"]).write_text("changed input\n")
    monkeypatch.setattr(A.np, "load", forbidden_downstream)
    with pytest.raises(ValueError, match=label + " hash changed"):
        run_early_audit(saved_trace)


@pytest.mark.parametrize("component", range(4), ids=["run", "subrun", "event", "iseed"])
def test_audit_checks_full_saved_key_before_rows(saved_trace, component):
    saved_trace[1]["key"][component] += 1
    with pytest.raises(ValueError, match="trace key differs"):
        run_early_audit(saved_trace)


def test_audit_requires_stage_zero_before_rows(saved_trace):
    directory, manifest, _ = saved_trace
    np.savez(directory / "trace.npz", key=np.asarray(manifest["key"]), stage_index=np.array([1]))
    manifest["trace_sha256"] = A.digest(directory / "trace.npz")
    with pytest.raises(ValueError, match="missing first stage"):
        run_early_audit(saved_trace)
