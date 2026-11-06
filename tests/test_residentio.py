"""Host-boundary and reporting contracts independent of private detector inputs."""
import hashlib
import json

import numpy as np
import pytest

from trkfastsim.reco.residentio import prepare
from trkfastsim.validation.residentcheck import digest, validate_output


def fixture():
    keys = np.array([[1, 0, 2], [1, 0, 3]], np.int64)
    sh = dict(ev=np.array([0, 0]), sid=np.array([0, 1]), flag=np.array([1024, 1024]),
              time=np.array([3., 4.]), wdist=np.array([0., 1.]), uvar=np.array([900., 900.]))
    st = {k: np.zeros(2) for k in ("midx", "midy", "midz", "dirx", "diry", "dirz")}
    st.update(dirx=np.ones(2), hlen=np.full(2, 1000.))
    return keys, sh, {"straw": st}


def test_prepare_preserves_all_rows_and_empty_events_without_aliasing():
    keys, sh, c = fixture()
    outk, out = prepare(keys, sh, c)
    np.testing.assert_array_equal(outk, keys)
    for k in sh:
        np.testing.assert_array_equal(out[k], sh[k])
        assert not np.shares_memory(out[k], sh[k])
    assert not np.shares_memory(outk, keys)
    assert out["valid"].all() and out["mid"].shape == (2, 3)
    sh["time"][0] = np.nan
    assert out["time"][0] == 3


@pytest.mark.parametrize("column", ["ev", "sid", "flag", "time", "wdist", "uvar"])
def test_masked_measurement_rejected(column):
    keys, sh, c = fixture()
    sh[column] = np.ma.array(sh[column], mask=[True, False])
    with pytest.raises(ValueError, match="masked"):
        prepare(keys, sh, c)


@pytest.mark.parametrize("column", ["midx", "dirx", "hlen"])
def test_masked_geometry_rejected(column):
    keys, sh, c = fixture()
    c["straw"][column] = np.ma.array(c["straw"][column], mask=[False, True])
    with pytest.raises(ValueError, match="masked"):
        prepare(keys, sh, c)


@pytest.mark.parametrize("column,value", [("ev", [-1, 0]), ("ev", [2, 0]), ("ev", [0., 1.]),
                                           ("sid", [96, 0]), ("sid", [768, 0]), ("sid", [65536, 0]),
                                           ("sid", [2, 0]), ("flag", [2**32, 0]),
                                           ("uvar", [[1, 2]]), ("time", [1+1j, 1])])
def test_malformed_index_schema(column, value):
    keys, sh, c = fixture()
    sh[column] = np.array(value)
    with pytest.raises(ValueError):
        prepare(keys, sh, c)


@pytest.mark.parametrize("keys", [np.array([[1, 0, 2], [1, 0, 2]]), np.array([[1, -1, 2]]),
                                 np.array([[1, 0, 2]], float), np.array([[1, 2]]),
                                 np.array([[1, 0, 2**63]], np.uint64)])
def test_bad_keys(keys):
    _, sh, c = fixture()
    with pytest.raises(ValueError):
        prepare(keys, sh, c)


def test_large_lossless_keys_and_runtime_failures_kept():
    keys, sh, c = fixture()
    keys[:, 2] += 2**60
    sh["time"][0] = np.nan
    sh["uvar"][1] = -1
    c["straw"]["dirx"][0] = np.inf
    outk, out = prepare(keys, sh, c)
    np.testing.assert_array_equal(keys, outk)
    assert np.isnan(out["time"][0]) and out["uvar"][1] == -1 and np.isinf(out["direction"][0, 0])


def test_empty_tables_and_digest(tmp_path):
    keys, sh, c = fixture()
    outk, out = prepare(keys, {k: v[:0] for k, v in sh.items()}, c)
    assert outk.shape == (2, 3) and out["mid"].shape == (0, 3)
    path = tmp_path / "data"
    path.write_bytes(b"abc")
    assert digest(path) == hashlib.sha256(b"abc").hexdigest()


def test_real_resident_output_and_tamper_checks():
    jax = pytest.importorskip("jax")
    from trkfastsim.reco.resident import reconstruct
    keys, sh, c = fixture()
    keys, hits = prepare(keys, sh, c)
    out = jax.device_get(reconstruct(keys, hits, bz=1., max_hits=4, group_batch=1, max_iterations=0))
    validate_output(keys, hits, out)
    for name in ("event_keys", "input_valid", "event_n_hits", "group_valid", "hit_group", "group_n_hits", "group_success"):
        changed = {k: np.array(v, copy=True) for k, v in out.items()}
        a = changed[name]
        a.flat[0] = not a.flat[0] if a.dtype.kind == "b" else a.flat[0] + 1
        with pytest.raises(RuntimeError):
            validate_output(keys, hits, changed)


def test_cli_saved_hits_no_overwrite(tmp_path):
    pytest.importorskip("jax")
    from trkfastsim.validation.residentcheck import main
    keys, sh, c = fixture()
    source = tmp_path / "hits.npz"
    cond = tmp_path / "conditions.json"
    np.savez(source, evkey=keys, **{"s_"+k: v for k, v in sh.items()})
    cond.write_text(json.dumps({"straw": {k: v.tolist() for k, v in c["straw"].items()}}))
    out = tmp_path / "out"
    args = ["--hits", str(source), "--conditions", str(cond), "--outdir", str(out), "--bz", "1",
            "--max-hits", "4", "--group-batch", "1", "--max-iterations", "0"]
    main(args)
    report = json.loads((out / "report.json").read_text())
    assert report["events"] == 2 and report["hits"] == 2 and report["groups"] == 1
    assert report["repeat_arrays_exact"]
    assert report["output_sha256"] == digest(out / "resident.npz")
    with pytest.raises(SystemExit):
        main(args)


def test_host_adapter_refuses_padded_mask_instead_of_reactivating():
    keys, sh, c = fixture()
    sh["valid"] = np.array([True, False])
    with pytest.raises(ValueError, match="unpadded"):
        prepare(keys, sh, c)


def test_host_adapter_rejects_nonexistent_plane_with_oversized_geometry():
    keys, sh, c = fixture()
    sh["sid"][0] = 36 << 10
    c["straw"] = {k: np.resize(v, 40000) for k, v in c["straw"].items()}
    with pytest.raises(ValueError, match="straw ID"):
        prepare(keys, sh, c)


@pytest.mark.parametrize("mutation", ["missing_hit", "wrong_event", "nan_objective", "event_count"])
def test_independent_saved_census_tamper(mutation):
    jax = pytest.importorskip("jax")
    from trkfastsim.reco.resident import reconstruct
    keys, sh, c = fixture()
    keys, hits = prepare(keys, sh, c)
    out = jax.device_get(reconstruct(keys, hits, bz=1., max_hits=4, group_batch=1, max_iterations=0))
    out = {k: np.array(v, copy=True) for k, v in out.items()}
    if mutation == "missing_hit":
        out["hit_group"][-1] = -1
        out["group_n_hits"][0] -= 1
    elif mutation == "wrong_event":
        out["group_ev"][0] = 1
    elif mutation == "nan_objective":
        out["group_objective_valid"][0] = True
        out["group_final_objective"][0] = np.nan
    else:
        out["event_n_groups"][0] += 1
    with pytest.raises(RuntimeError):
        validate_output(keys, hits, out)


def test_comparison_is_structural_not_a_numerical_pass_label():
    jax = pytest.importorskip("jax")
    from trkfastsim.reco.resident import reconstruct
    from trkfastsim.validation.residentcheck import compare_outputs
    keys, sh, c = fixture()
    keys, hits = prepare(keys, sh, c)
    out = jax.device_get(reconstruct(keys, hits, bz=1., max_hits=4, group_batch=1, max_iterations=0))
    report = compare_outputs(out, out)
    assert report["structure_exact"] and report["status_changed_group_rows"] == []
    changed = {k: np.array(v, copy=True) for k, v in out.items()}
    changed["group_status"][0] = 8
    assert compare_outputs(out, changed)["status_changed_group_rows"] == [0]
    changed["hit_group"][0] = -1
    with pytest.raises(RuntimeError, match="structural"):
        compare_outputs(out, changed)
    changed = {k: np.array(v, copy=True) for k, v in out.items()}
    changed["input_time"][0] += 1
    with pytest.raises(RuntimeError, match="input"):
        compare_outputs(out, changed)
