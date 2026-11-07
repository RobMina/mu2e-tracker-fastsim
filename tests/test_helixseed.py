"""Independent analytic geometry tests for the bounded experimental seed."""
import numpy as np
import pytest

from trkfastsim.reco.helixseed import fit_helix_seed


def helix(phase, radius=710.0, pitch=190.0, center=(130.0, -270.0), z0=55.0):
    phase = np.asarray(phase)
    return np.column_stack((center[0] + radius * np.cos(phase),
                            center[1] + radius * np.sin(phase), z0 + pitch * phase))


def fit(points, **kwargs):
    p = np.asarray(points)[None]
    return fit_helix_seed(p, np.ones(p.shape[:2], bool), bz=kwargs.pop("bz", 1.0), **kwargs)


def assert_failed(out, rows):
    assert not out["success"][rows].any()
    for key in out.keys() - {"success", "n_hits", "condition"}:
        assert np.isnan(out[key][rows]).all(), key


@pytest.mark.parametrize("pitch", [-420.0, -7.0, 7.0, 420.0])
@pytest.mark.parametrize("sense", [-1, 1])
@pytest.mark.parametrize("start", [-3.6, 2.9, 13.0])
def test_analytic_signs_and_global_wrap(pitch, sense, start):
    phase = start + sense * np.linspace(0, 1.7, 19)
    out = fit(helix(phase, pitch=pitch), bz=-1.35)
    assert out["success"][0]
    np.testing.assert_allclose(out["center_xy"][0], [130, -270], atol=1e-9)
    np.testing.assert_allclose(out["radius"], 710, rtol=1e-12)
    np.testing.assert_allclose(out["pitch"], pitch, rtol=1e-12)
    np.testing.assert_allclose(out["phase_span"], 1.7, atol=1e-12)
    np.testing.assert_allclose(out["z_ref"], 55 + pitch * phase.mean(), atol=1e-10)
    np.testing.assert_allclose(out["pt_abs"], 0.299792458 * 1.35 * 710, rtol=1e-12)
    np.testing.assert_allclose(out["p_abs"], 0.299792458 * 1.35 * np.hypot(710, pitch), rtol=1e-12)
    assert out["circle_rms"][0] < 1e-9
    assert out["z_rms"][0] < 1e-9
    branch_shift = (out["phase_ref"][0] - phase.mean()) / (2 * np.pi)
    assert abs(branch_shift - np.round(branch_shift)) < 1e-12
    assert not ({"charge", "q", "direction", "pz", "momentum", "cov"} & out.keys())


def test_permutation_translation_and_weight_scale():
    phase = np.linspace(2.8, 4.3, 27)
    p = helix(phase, pitch=-35)
    w = np.geomspace(0.1, 100, len(p))[None]
    ref = fit(p, weights=w)
    permutation = np.random.default_rng(502).permutation(len(p))
    permuted = fit(p[permutation], weights=w[:, permutation] * 1e200)
    for key in ref:
        np.testing.assert_allclose(permuted[key], ref[key], atol=2e-10, rtol=2e-11)
    shift = np.array([1e9, -2e9, 3e9])
    moved = fit(p + shift, weights=w)
    assert moved["success"][0]
    np.testing.assert_allclose(moved["center_xy"] - shift[:2], ref["center_xy"], atol=2e-5)
    np.testing.assert_allclose(moved["z_ref"] - shift[2], ref["z_ref"], atol=1e-6)
    for key in ("radius", "pitch", "condition", "phase_span", "pt_abs", "p_abs"):
        np.testing.assert_allclose(moved[key], ref[key], rtol=2e-7)


def test_weighted_noisy_fit_matches_independent_lstsq():
    phase = np.linspace(-0.6, 1.7, 31)
    p = helix(phase)
    p[:, 0] += 0.7 * np.cos(np.arange(len(p)))
    p[:, 1] += 0.4 * np.sin(np.arange(len(p)))
    p[:, 2] += 0.3 * np.sin(np.arange(len(p)))
    w = np.linspace(0.3, 4, len(p))
    out = fit(p, weights=w[None])
    assert out["success"][0]
    design = np.column_stack((2 * p[:, :2], np.ones(len(p))))
    coeff = np.linalg.lstsq(design * np.sqrt(w[:, None]),
                           np.sum(p[:, :2]**2, axis=1) * np.sqrt(w), rcond=None)[0]
    radius = np.sqrt(coeff[2] + coeff[:2] @ coeff[:2])
    phi = np.unwrap(np.arctan2(p[:, 1] - coeff[1], p[:, 0] - coeff[0]))
    line = np.linalg.lstsq(np.column_stack((phi, np.ones(len(phi)))) * np.sqrt(w[:, None]),
                          p[:, 2] * np.sqrt(w), rcond=None)[0]
    np.testing.assert_allclose(out["center_xy"][0], coeff[:2], atol=1e-8)
    np.testing.assert_allclose(out["radius"], radius, rtol=1e-10)
    np.testing.assert_allclose(out["pitch"], line[0], rtol=1e-10)
    assert out["circle_rms"][0] > 0
    assert out["z_rms"][0] > 0
    assert abs(fit(p)["radius"][0] - out["radius"][0]) > 1e-4


def test_ragged_masks_nonfinite_padding_and_bad_active_lane():
    p = np.full((5, 20, 3), np.nan)
    valid = np.zeros((5, 20), bool)
    weights = np.full((5, 20), -np.inf)
    ids = np.array([1, 4, 7, 10, 13, 16])
    points = helix(np.linspace(2.6, 3.6, len(ids)))
    p[0, ids] = points
    valid[0, ids] = True
    weights[0, ids] = np.arange(1, len(ids) + 1)
    p[1, :3] = points[:3]
    valid[1, :3] = True
    weights[1, :3] = 1
    p[3, ids] = points
    valid[3, ids] = True
    weights[3, ids] = 1
    p[3, ids[0], 0] = np.inf
    p[4, ids] = points
    valid[4, ids] = True
    weights[4, ids] = 1
    p[4, ids[0], 1] = np.nan
    with np.errstate(all="raise"):
        out = fit_helix_seed(p, valid, weights=weights, bz=1.2)
    np.testing.assert_array_equal(out["success"], [True, False, False, False, False])
    np.testing.assert_array_equal(out["n_hits"], [6, 3, 0, 6, 6])
    assert_failed(out, [1, 2, 3, 4])
    ref = fit(points, weights=weights[0, ids][None], bz=1.2)
    for key in ref:
        np.testing.assert_allclose(out[key][:1], ref[key], atol=1e-10)


@pytest.mark.parametrize("batch,hits", [(0, 0), (0, 7), (3, 0), (3, 1), (3, 3)])
def test_empty_shapes_and_insufficient_capacity(batch, hits):
    out = fit_helix_seed(np.zeros((batch, hits, 3)), np.ones((batch, hits), bool), bz=1)
    assert out["center_xy"].shape == (batch, 2)
    assert all(value.shape == (batch,) for key, value in out.items() if key != "center_xy")
    assert out["success"].dtype == np.bool_
    assert_failed(out, slice(None))


def test_degenerate_lanes_do_not_poison_good_lane():
    phase = np.linspace(-1, 1, 12)
    good = helix(phase)
    collinear = np.column_stack((phase, 2 * phase, 4 * phase))
    nearly_line = collinear.copy()
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
    loops = helix(np.linspace(0, 2.2 * np.pi, len(phase)))
    p = np.stack([good, collinear, nearly_line, zero_z, duplicate, tied_z, repeated, vertical, loops])
    with np.errstate(all="raise"):
        out = fit_helix_seed(p, np.ones(p.shape[:2], bool), bz=1)
    np.testing.assert_array_equal(out["success"], [True] + [False] * 8)
    assert_failed(out, slice(1, None))
    assert out["condition"][1] > 1e12
    assert out["condition"][2] > 1e12


def test_short_arc_condition_diagnostic_and_configurable_limit():
    p = helix(np.linspace(-0.0005, 0.0005, 33), radius=1e6, pitch=8e4)
    out = fit(p)
    assert out["success"][0]
    assert 1e6 < out["condition"][0] < 1e12
    np.testing.assert_allclose(out["radius"], 1e6, rtol=1e-8)
    np.testing.assert_allclose(out["pitch"], 8e4, rtol=1e-8)
    tight = fit(p, condition_limit=1e5)
    assert_failed(tight, slice(None))
    np.testing.assert_equal(tight["condition"], out["condition"])


def test_b_reversal_and_field_scaling_do_not_assign_sign():
    p = helix(np.linspace(0, 2, 13), pitch=-210)
    positive, negative, double = fit(p, bz=0.8), fit(p, bz=-0.8), fit(p, bz=1.6)
    for key in positive:
        np.testing.assert_array_equal(positive[key], negative[key])
    np.testing.assert_allclose(double["pt_abs"], 2 * positive["pt_abs"])
    np.testing.assert_allclose(double["p_abs"], 2 * positive["p_abs"])
    assert positive["pitch"][0] < 0 < positive["p_abs"][0]


@pytest.mark.parametrize("bz", [0, np.nan, np.inf, -np.inf, [1], 1j, "1", True])
def test_invalid_field(bz):
    with pytest.raises(ValueError):
        fit(helix(np.linspace(0, 1, 8)), bz=bz)


@pytest.mark.parametrize("limit", [0, 1, -1, np.nan, np.inf, [1e12], 1j])
def test_invalid_condition_limit(limit):
    with pytest.raises(ValueError):
        fit(helix(np.linspace(0, 1, 8)), condition_limit=limit)


@pytest.mark.parametrize("weight", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_active_weights(weight):
    w = np.ones((1, 8))
    w[0, 2] = weight
    with pytest.raises(ValueError, match="positive"):
        fit(helix(np.linspace(0, 1, 8)), weights=w)


@pytest.mark.parametrize("positions,valid,weights", [
    (np.zeros((8, 3)), np.ones((1, 8), bool), None),
    (np.zeros((1, 8, 2)), np.ones((1, 8), bool), None),
    (np.zeros((1, 8, 3), complex), np.ones((1, 8), bool), None),
    (np.zeros((1, 8, 3)), np.ones((1, 7), bool), None),
    (np.zeros((1, 8, 3)), np.ones((1, 8), int), None),
    (np.zeros((1, 8, 3)), np.ones((1, 8), bool), np.ones(8)),
    (np.zeros((1, 8, 3)), np.ones((1, 8), bool), np.ones((1, 8), complex)),
])
def test_invalid_shapes_and_types(positions, valid, weights):
    with pytest.raises(ValueError):
        fit_helix_seed(positions, valid, bz=1, weights=weights)


def test_field_is_explicit_and_inputs_are_not_mutated():
    p = helix(np.linspace(0, 1, 8))[None]
    valid = np.ones((1, 8), bool)
    w = np.linspace(1, 3, 8)[None]
    before = [x.copy() for x in (p, valid, w)]
    with pytest.raises(TypeError):
        fit_helix_seed(p, valid)
    fit_helix_seed(p, valid, bz=1, weights=w)
    for value, expected in zip((p, valid, w), before):
        np.testing.assert_array_equal(value, expected)


def test_mixed_successful_batch_and_four_hit_minimum():
    phase = np.linspace(2.8, 4.2, 4)
    p = np.stack([helix(phase, radius=80, pitch=-30),
                  helix(-phase, radius=1400, pitch=900, center=(-600, 120)),
                  helix(phase, radius=400, pitch=10)])
    out = fit_helix_seed(p, np.ones(p.shape[:2], bool), bz=0.6)
    assert out["success"].all()
    np.testing.assert_allclose(out["radius"], [80, 1400, 400], rtol=1e-12)
    np.testing.assert_allclose(out["pitch"], [-30, 900, 10], rtol=1e-12)
    np.testing.assert_allclose(out["center_xy"], [[130, -270], [-600, 120], [130, -270]], atol=1e-9)


def test_relaxed_limit_does_not_accept_numerical_rank_deficiency():
    x = np.linspace(-2, 2, 12)
    p = np.column_stack((x, 2 * x + 1e-8 * x**2, 3 * x))
    assert_failed(fit(p, condition_limit=1e30), slice(None))
