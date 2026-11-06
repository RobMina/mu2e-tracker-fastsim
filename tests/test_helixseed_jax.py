"""Standalone synthetic oracles for the device seed; CPU parity is supplemental.

Tolerances are deliberate numerical comparisons, not bitwise/device promises.
No saved hits, conditions, truth grouping, or imports from other tests are used.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from trkfastsim.reco import helixseed_jax as seed

fit_jit = jax.jit(seed.fit_helix_seed)
DIAGNOSTICS = {"success", "status", "n_hits", "condition"}


def helix(phase, radius=710.0, pitch=190.0, center=(130.0, -270.0), z0=55.0):
    phase = np.asarray(phase)
    return np.column_stack((center[0] + radius * np.cos(phase),
                            center[1] + radius * np.sin(phase), z0 + pitch * phase))


def fit(points, *, compiled=True, **kwargs):
    points = np.asarray(points)[None]
    return (fit_jit if compiled else seed.fit_helix_seed)(
        points, np.ones(points.shape[:2], bool), bz=kwargs.pop("bz", 1.0), **kwargs)


def assert_failed(out, rows=slice(None), status=None):
    assert not np.asarray(out["success"])[rows].any()
    assert (np.asarray(out["status"])[rows] != seed.SUCCESS).all()
    for key in out.keys() - DIAGNOSTICS:
        assert np.isnan(np.asarray(out[key])[rows]).all(), key
    if status is not None:
        np.testing.assert_array_equal(np.asarray(out["status"])[rows], status)


def assert_matches(actual, expected, *, atol=2e-9, rtol=2e-10):
    for key in expected:
        if key in {"success", "status", "n_hits"}:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        else:
            np.testing.assert_allclose(actual[key], expected[key], atol=atol, rtol=rtol,
                                       equal_nan=True, err_msg=key)


@pytest.mark.parametrize("pitch", [-420.0, -7.0, 7.0, 420.0])
@pytest.mark.parametrize("sense", [-1, 1])
@pytest.mark.parametrize("start", [-3.6, 2.9, 13.0])
def test_analytic_signs_phase_wraps_and_momentum(pitch, sense, start):
    phase = start + sense * np.linspace(0, 1.7, 19)
    out = fit(helix(phase, pitch=pitch), bz=-1.35)
    assert out["success"][0]
    assert out["status"][0] == seed.SUCCESS
    np.testing.assert_allclose(out["center_xy"][0], [130, -270], atol=1e-9, rtol=0)
    np.testing.assert_allclose(out["radius"], 710, rtol=1e-12)
    np.testing.assert_allclose(out["pitch"], pitch, rtol=1e-12)
    np.testing.assert_allclose(out["phase_span"], 1.7, atol=1e-12, rtol=0)
    np.testing.assert_allclose(out["z_ref"], 55 + pitch * phase.mean(), atol=1e-10, rtol=0)
    np.testing.assert_allclose(out["pt_abs"], 0.299792458 * 1.35 * 710, rtol=1e-12)
    np.testing.assert_allclose(out["p_abs"], 0.299792458 * 1.35 * np.hypot(710, pitch), rtol=1e-12)
    branch_shift = (out["phase_ref"][0] - phase.mean()) / (2 * np.pi)
    np.testing.assert_allclose(branch_shift, np.round(branch_shift), atol=1e-12, rtol=0)
    assert out["circle_rms"][0] < 1e-9
    assert out["z_rms"][0] < 1e-9
    assert not ({"charge", "direction", "pz", "cov"} & out.keys())


def test_weighted_noisy_independent_lstsq_oracle():
    phase = np.linspace(-0.6, 1.7, 31)
    p = helix(phase)
    p += np.column_stack((0.7 * np.cos(np.arange(len(p))),
                          0.4 * np.sin(np.arange(len(p))),
                          0.3 * np.sin(np.arange(len(p)))))
    w = np.linspace(0.3, 4, len(p))
    out = fit(p, weights=w[None])
    assert out["success"][0]
    design = np.column_stack((2 * p[:, :2], np.ones(len(p))))
    coeff = np.linalg.lstsq(design * np.sqrt(w[:, None]),
                           np.sum(p[:, :2]**2, axis=1) * np.sqrt(w), rcond=None)[0]
    radius = np.sqrt(coeff[2] + coeff[:2] @ coeff[:2])
    phi = np.unwrap(np.arctan2(p[:, 1] - coeff[1], p[:, 0] - coeff[0]))
    line = np.linalg.lstsq(np.column_stack((phi, np.ones(len(p)))) * np.sqrt(w[:, None]),
                          p[:, 2] * np.sqrt(w), rcond=None)[0]
    np.testing.assert_allclose(out["center_xy"][0], coeff[:2], atol=1e-8, rtol=0)
    np.testing.assert_allclose(out["radius"], radius, rtol=1e-10)
    np.testing.assert_allclose(out["pitch"], line[0], rtol=1e-10)
    circle_residual = np.linalg.norm(p[:, :2] - coeff[:2], axis=1) - radius
    z_residual = p[:, 2] - (line[0] * phi + line[1])
    np.testing.assert_allclose(out["circle_rms"], np.sqrt(np.average(circle_residual**2, weights=w)),
                               atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(out["z_rms"], np.sqrt(np.average(z_residual**2, weights=w)),
                               atol=1e-10, rtol=1e-10)
    assert abs(fit(p)["radius"][0] - out["radius"][0]) > 1e-4


def test_permutation_translation_and_weight_scaling():
    phase = np.linspace(2.8, 4.3, 27)
    p = helix(phase, pitch=-35)
    w = np.geomspace(0.1, 100, len(p))[None]
    ref = fit(p, weights=w)
    permutation = np.random.default_rng(502).permutation(len(p))
    permuted = fit(p[permutation], weights=w[:, permutation] * 1e200)
    assert_matches(permuted, ref, atol=2e-10, rtol=2e-11)
    shift = np.array([1e9, -2e9, 3e9])
    moved = fit(p + shift, weights=w)
    assert moved["success"][0]
    np.testing.assert_allclose(moved["center_xy"] - shift[:2], ref["center_xy"], atol=2e-5, rtol=0)
    np.testing.assert_allclose(moved["z_ref"] - shift[2], ref["z_ref"], atol=1e-6, rtol=0)
    for key in ("radius", "pitch", "condition", "phase_span", "pt_abs", "p_abs"):
        np.testing.assert_allclose(moved[key], ref[key], rtol=2e-7)


def test_ragged_nonfinite_masks_and_invalid_active_positions():
    p = np.full((5, 20, 3), np.nan)
    p[:, ::2] = np.inf
    valid = np.zeros((5, 20), bool)
    weights = np.full((5, 20), -np.inf)
    weights[:, ::2] = np.nan
    ids = np.array([1, 4, 7, 10, 13, 16])
    points = helix(np.linspace(2.6, 3.6, len(ids)))
    for lane in (0, 3, 4):
        p[lane, ids] = points
        valid[lane, ids] = True
        weights[lane, ids] = np.arange(1, len(ids) + 1)
    p[1, :3] = points[:3]
    valid[1, :3] = True
    weights[1, :3] = 1
    p[3, ids[0], 0] = np.inf
    p[4, ids[0], 1] = np.nan
    out = fit_jit(p, valid, weights=weights, bz=1.2)
    np.testing.assert_array_equal(out["success"], [True, False, False, False, False])
    np.testing.assert_array_equal(out["n_hits"], [6, 3, 0, 6, 6])
    assert_failed(out, [1, 2], [seed.INSUFFICIENT_HITS] * 2)
    assert_failed(out, [3, 4], [seed.INVALID_INPUT] * 2)
    ref = fit(points, weights=weights[0, ids][None], bz=1.2)
    assert_matches({key: value[:1] for key, value in out.items()}, ref)


@pytest.mark.parametrize("weight", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_bad_weights_fail_only_their_lane(weight):
    p = np.tile(helix(np.linspace(0, 1, 8))[None], (3, 1, 1))
    w = np.ones(p.shape[:2])
    w[1, 2] = weight
    out = fit_jit(p, np.ones(p.shape[:2], bool), weights=w, bz=1.0)
    np.testing.assert_array_equal(out["success"], [True, False, True])
    np.testing.assert_array_equal(out["n_hits"], [8, 8, 8])
    assert_failed(out, [1], [seed.INVALID_INPUT])
    assert_matches({key: value[:1] for key, value in out.items()}, fit(p[0]))


@pytest.mark.parametrize("batch,hits", [(0, 0), (0, 7), (3, 0), (3, 1), (3, 3)])
@pytest.mark.parametrize("compiled", [False, True])
def test_empty_shapes_and_small_capacity(batch, hits, compiled):
    out = (fit_jit if compiled else seed.fit_helix_seed)(
        np.zeros((batch, hits, 3)), np.ones((batch, hits), bool), bz=1)
    assert out["center_xy"].shape == (batch, 2)
    assert all(value.shape == (batch,) for key, value in out.items() if key != "center_xy")
    for key, value in out.items():
        assert isinstance(value, jax.Array)
        expected = {"success": np.bool_, "status": np.int32, "n_hits": np.int64}.get(key, np.float64)
        assert value.dtype == expected
    assert_failed(out)
    assert (np.asarray(out["condition"]) == np.inf).all()
    np.testing.assert_array_equal(out["n_hits"], np.full(batch, hits))


def test_entirely_empty_mask_with_nonzero_capacity():
    out = fit_jit(np.full((3, 12, 3), np.nan), np.zeros((3, 12), bool),
                  weights=np.full((3, 12), np.inf), bz=1.0)
    assert_failed(out, status=np.full(3, seed.INSUFFICIENT_HITS))
    np.testing.assert_array_equal(out["n_hits"], 0)
    np.testing.assert_array_equal(out["condition"], np.inf)


def test_rank_repeated_z_and_loop_failures_are_lane_local():
    phase = np.linspace(-1, 1, 12)
    good = helix(phase)
    line = np.column_stack((phase, 2 * phase, 4 * phase))
    nearly_line = line.copy()
    nearly_line[:, 1] += 1e-8 * phase**2
    zero_z = good.copy()
    zero_z[:, 2] = 3
    duplicate = good.copy()
    duplicate[3] = duplicate[2]
    tied_z = good.copy()
    tied_z[3, 2] = tied_z[2, 2]
    repeated = np.tile(good[0], (len(good), 1))
    vertical = repeated.copy()
    vertical[:, 2] = phase
    loop = helix(np.linspace(0, 2.2 * np.pi, len(phase)))
    p = np.stack([good, line, nearly_line, zero_z, duplicate, tied_z, repeated, vertical, loop])
    out = fit_jit(p, np.ones(p.shape[:2], bool), bz=1.0)
    np.testing.assert_array_equal(out["success"], [True] + [False] * 8)
    assert_failed(out, slice(1, None))
    assert_failed(out, [1, 2, 6, 7], [seed.ILL_CONDITIONED] * 4)
    assert_failed(out, [3, 4, 5, 8], [seed.INVALID_GEOMETRY] * 4)
    relaxed = fit(nearly_line, condition_limit=1e30)
    assert_failed(relaxed, status=[seed.ILL_CONDITIONED])


def test_short_arc_and_dynamic_condition_limit():
    p = helix(np.linspace(-0.0005, 0.0005, 33), radius=1e6, pitch=8e4)
    out = fit(p, condition_limit=jnp.asarray(1e12))
    assert out["success"][0]
    assert 1e6 < out["condition"][0] < 1e12
    np.testing.assert_allclose(out["radius"], 1e6, rtol=1e-8)
    np.testing.assert_allclose(out["pitch"], 8e4, rtol=1e-8)
    tight = fit(p, condition_limit=jnp.asarray(1e5))
    assert_failed(tight, status=[seed.ILL_CONDITIONED])
    np.testing.assert_array_equal(tight["condition"], out["condition"])


@pytest.mark.parametrize("name,number", [
    ("bz", 0.0), ("bz", np.nan), ("bz", np.inf), ("bz", -np.inf),
    ("condition_limit", 0.0), ("condition_limit", 1.0), ("condition_limit", -1.0),
    ("condition_limit", np.nan), ("condition_limit", np.inf), ("condition_limit", -np.inf),
])
def test_invalid_runtime_configuration_fails_all_lanes(name, number):
    p = np.tile(helix(np.linspace(0, 1, 8))[None], (3, 1, 1))
    valid = np.ones(p.shape[:2], bool)
    valid[1] = False
    p[2, 0] = np.nan
    kwargs = {"bz": jnp.asarray(1.0), "condition_limit": jnp.asarray(1e12)}
    kwargs[name] = jnp.asarray(number)
    out = fit_jit(p, valid, **kwargs)
    assert_failed(out, status=[seed.INVALID_CONFIGURATION] * 3)
    np.testing.assert_array_equal(out["n_hits"], [8, 0, 8])


def test_field_sign_and_scale_and_finite_output_guard():
    p = helix(np.linspace(0, 2, 13), pitch=-210)
    positive, negative, double = (fit(p, bz=jnp.asarray(b)) for b in (0.8, -0.8, 1.6))
    assert_matches(positive, negative, atol=0, rtol=0)
    np.testing.assert_allclose(double["pt_abs"], 2 * positive["pt_abs"], rtol=1e-14)
    np.testing.assert_allclose(double["p_abs"], 2 * positive["p_abs"], rtol=1e-14)
    assert positive["pitch"][0] < 0 < positive["p_abs"][0]
    overflow = fit(p, bz=jnp.asarray(1e308))
    assert_failed(overflow, status=[seed.NONFINITE_RESULT])


def test_overflow_inputs_never_reach_eigh_nonfinite(monkeypatch):
    good = helix(np.linspace(-1, 1, 12))
    bad = good.copy()
    bad[0, 0], bad[1, 0] = -1.7e308, 1.7e308
    p = np.stack([good, bad, good])
    w = np.ones(p.shape[:2])
    w[2] = np.inf
    original_eigh = jnp.linalg.eigh
    seen = []

    def finite_eigh(matrix, *args, **kwargs):
        assert np.isfinite(np.asarray(matrix)).all()
        seen.append(matrix.shape)
        return original_eigh(matrix, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(jnp.linalg, "eigh", finite_eigh)
        eager = seed.fit_helix_seed(p, np.ones(p.shape[:2], bool), bz=1.0, weights=w)
    assert seen == [(3, 2, 2)]
    out = fit_jit(p, np.ones(p.shape[:2], bool), bz=1.0, weights=w)
    assert_matches(out, eager)
    assert out["success"][0]
    assert_failed(out, [1, 2], [seed.NONFINITE_RESULT, seed.INVALID_INPUT])


def test_four_hit_minimum_batch_reorder_and_single_lane_agreement():
    phase = np.linspace(2.8, 4.2, 4)
    p = np.stack([helix(phase, radius=80, pitch=-30),
                  helix(-phase, radius=1400, pitch=900, center=(-600, 120)),
                  helix(phase, radius=400, pitch=10), np.full((4, 3), np.nan)])
    valid = np.ones(p.shape[:2], bool)
    valid[-1] = False
    out = fit_jit(p, valid, bz=0.6)
    np.testing.assert_array_equal(out["success"], [True, True, True, False])
    np.testing.assert_allclose(out["radius"][:3], [80, 1400, 400], rtol=1e-12)
    np.testing.assert_allclose(out["pitch"][:3], [-30, 900, 10], rtol=1e-12)
    np.testing.assert_allclose(out["center_xy"][:3], [[130, -270], [-600, 120], [130, -270]], atol=1e-9)
    order = [3, 1, 0, 2]
    reordered = fit_jit(p[order], valid[order], bz=0.6)
    assert_matches(reordered, {key: value[np.array(order)] for key, value in out.items()})
    for lane in range(len(p)):
        single = fit_jit(p[lane:lane + 1], valid[lane:lane + 1], bz=0.6)
        assert_matches(single, {key: value[lane:lane + 1] for key, value in out.items()})


@pytest.mark.parametrize("positions,valid,weights", [
    (np.zeros((8, 3)), np.ones((1, 8), bool), None),
    (np.zeros((1, 8, 2)), np.ones((1, 8), bool), None),
    (np.zeros((1, 8, 3), complex), np.ones((1, 8), bool), None),
    (np.zeros((1, 8, 3), bool), np.ones((1, 8), bool), None),
    (np.zeros((1, 8, 3)), np.ones((1, 7), bool), None),
    (np.zeros((1, 8, 3)), np.ones((1, 8), int), None),
    (np.zeros((1, 8, 3)), np.ones((1, 8), bool), np.ones(8)),
    (np.zeros((1, 8, 3)), np.ones((1, 8), bool), np.ones((1, 8), complex)),
    (np.zeros((1, 8, 3)), np.ones((1, 8), bool), np.ones((1, 8), bool)),
])
def test_static_array_shape_and_dtype_checks(positions, valid, weights):
    for function in (seed.fit_helix_seed, fit_jit):
        with pytest.raises(ValueError):
            function(positions, valid, bz=1.0, weights=weights)


@pytest.mark.parametrize("name", ["bz", "condition_limit"])
@pytest.mark.parametrize("value", [[1.0], 1j, True])
def test_static_scalar_shape_and_dtype_checks(name, value):
    p = helix(np.linspace(0, 1, 8))[None]
    kwargs = {"bz": 1.0, name: value}
    for function in (seed.fit_helix_seed, fit_jit):
        with pytest.raises(ValueError):
            function(p, np.ones(p.shape[:2], bool), **kwargs)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.uint32])
def test_real_input_dtypes_and_no_mutation(dtype):
    p = helix(np.linspace(0, 1, 8), center=(1000, 1000)).astype(dtype)[None]
    valid = np.ones((1, 8), bool)
    w = np.arange(1, 9, dtype=dtype)[None]
    before = [x.copy() for x in (p, valid, w)]
    with pytest.raises(TypeError):
        seed.fit_helix_seed(p, valid)
    out = fit_jit(p, valid, bz=1, weights=w)
    assert out["success"][0]
    assert out["radius"].dtype == np.float64
    for value, expected in zip((p, valid, w), before):
        np.testing.assert_array_equal(value, expected)


def test_traceable_no_callbacks_and_device_execution():
    p = jnp.asarray(helix(np.linspace(0, 1, 8))[None])
    valid = jnp.ones((1, 8), dtype=bool)
    w = jnp.ones((1, 8), dtype=jnp.float64)
    bz, limit = jnp.asarray(1.0), jnp.asarray(1e12)
    traced = jax.make_jaxpr(seed.fit_helix_seed)(p, valid, bz=bz, weights=w, condition_limit=limit)
    text = str(traced).lower()
    assert "eigh" in text
    for forbidden in ("callback", "host_call", "io_callback", "pure_callback"):
        assert forbidden not in text
    eager = seed.fit_helix_seed(p, valid, bz=bz, weights=w, condition_limit=limit)
    # Precompile; a subsequent invocation cannot copy input/output to the host.
    compiled = fit_jit.lower(p, valid, bz=bz, weights=w, condition_limit=limit).compile()
    with jax.transfer_guard("disallow_explicit"):
        device = compiled(p, valid, bz=bz, weights=w, condition_limit=limit)
        jax.block_until_ready(device)
    assert_matches(device, eager)


def test_supplemental_numpy_parity_on_synthetic_measured_data():
    # Reference import is test-only. All analytic/failure tests above stand alone.
    fit_numpy = pytest.importorskip("trkfastsim.reco.helixseed").fit_helix_seed

    rng = np.random.default_rng(508)
    phase = np.linspace(2.8, 4.8, 24)
    p = np.stack([helix(phase, pitch=-210), helix(-phase, radius=80, pitch=33),
                  helix(phase, center=(1e7, -2e7), z0=3e7), helix(phase)])
    p[1] += rng.normal(0, 0.05, p[1].shape)
    valid = np.ones(p.shape[:2], bool)
    valid[0, 3::4] = False
    valid[-1] = False
    p[~valid] = np.nan
    w = rng.uniform(0.1, 4, valid.shape)
    w[~valid] = np.inf
    ref = fit_numpy(p, valid, weights=w, bz=-1.0117)
    out = fit_jit(p, valid, weights=w, bz=-1.0117)
    assert_matches(out, ref, rtol=2e-10, atol=2e-9)
    assert_matches(seed.fit_helix_seed(p, valid, weights=w, bz=-1.0117), ref,
                   rtol=2e-10, atol=2e-9)
