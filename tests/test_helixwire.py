"""Analytic 3D constructions and an independent scalar minimization oracle."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from trkfastsim.reco.helixseed import fit_helix_seed
from trkfastsim.reco.helixwire import wire_residuals


def seed(**changes):
    result = dict(center_xy=np.array([13., -21.]), radius=140., pitch=85.,
                  phase_ref=.31, z_ref=17.)
    result.update(changes)
    return result


def point_tangent(s, z):
    """Independent parametric helix, used only to construct test geometry."""
    z = np.asarray(z)
    phi = s["phase_ref"] + (z-s["z_ref"])/s["pitch"]
    p = np.stack((s["center_xy"][0]+s["radius"]*np.cos(phi),
                  s["center_xy"][1]+s["radius"]*np.sin(phi), z), axis=-1)
    t = np.stack((-s["radius"]/s["pitch"]*np.sin(phi),
                  s["radius"]/s["pitch"]*np.cos(phi), np.ones_like(z)), axis=-1)
    return p, t


def construction(s, z, direction, u=12., offset=0.):
    """A wire with an exactly known stationary separation, normal to both curves."""
    p, t = point_tangent(s, z)
    d = np.asarray(direction, dtype=float)
    d = d/np.linalg.norm(d, axis=-1, keepdims=True)
    normal = np.cross(t, d)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return p-offset*normal-np.asarray(u)[..., None]*d, d


def call(s=None, mid=None, direction=None, measured_u=12., uvar=4., hlen=100., **kw):
    if s is None:
        s = seed()
    if mid is None:
        mid, default_d = construction(s, 29., [1., .4, .3])
        if direction is None:
            direction = default_d
    if direction is None:
        direction = [1., .4, .3]
    return wire_residuals(s, mid, direction, measured_u, uvar, hlen, **kw)


def assert_finite(out):
    for name in ("doca", "signed_doca", "wire_u", "wire_pull", "z"):
        assert np.isfinite(out[name]).all(), name


@pytest.mark.parametrize("pitch", [-280., -15., 15., 280.])
@pytest.mark.parametrize("offset", [-.7, 0., .7])
@pytest.mark.parametrize("direction", [[1., 0., 0.], [.4, -.7, .5], [0., 0., 1.]])
def test_known_normal_offset_and_intersection(pitch, offset, direction):
    s = seed(pitch=pitch)
    z = np.array([-40., 0., 17., 65.])
    u = np.array([-24., 7., 0., 34.])
    mid, d = construction(s, z, direction, u=u, offset=offset)
    measured = u + np.array([.8, -.6, .2, -.3])
    out = call(s, mid, d, measured_u=measured, uvar=9., tolerance=1e-9)
    assert out["valid"].all() and out["converged"].all() and out["inside"].all()
    np.testing.assert_allclose(out["z"][0], z, atol=2e-9)
    np.testing.assert_allclose(out["doca"], abs(offset), atol=2e-9)
    np.testing.assert_allclose(out["signed_doca"], offset, atol=2e-9)
    np.testing.assert_allclose(out["wire_u"][0], u, atol=2e-9)
    np.testing.assert_allclose(out["wire_pull"][0], (u-measured)/3., atol=2e-9)
    assert_finite(out)


@pytest.mark.parametrize("scale", [1e-300, .03, 19., 1e300])
def test_direction_scaling_and_reversal(scale):
    s = seed(pitch=-95.)
    mid, d = construction(s, 40., [.3, .9, -.8], offset=.5)
    base = call(s, mid, d, measured_u=13.)
    scaled = call(s, mid, d*scale, measured_u=13.)
    reverse = call(s, mid, -d*scale, measured_u=-13.)
    for key in base:
        np.testing.assert_allclose(scaled[key], base[key], atol=3e-10, err_msg=key)
    for key in ("doca", "z", "converged", "inside", "valid", "sign_valid"):
        np.testing.assert_allclose(reverse[key], base[key], atol=3e-10)
    for key in ("signed_doca", "wire_u", "wire_pull"):
        np.testing.assert_allclose(reverse[key], -base[key], atol=3e-10)


def test_tilt_and_3d_along_wire_not_xy_parameter():
    s = seed()
    mid, d = construction(s, 37., [1., 2., 7.], u=80., offset=.8)
    out = call(s, mid, 5*d, measured_u=79., uvar=16.)
    assert out["converged"].item()
    assert out["wire_u"].item() == pytest.approx(80., abs=2e-7)
    assert out["wire_pull"].item() == pytest.approx(.25, abs=2e-7)
    flattened = call(s, mid, [d[0], d[1], 0.], measured_u=79., uvar=16.)
    assert abs(flattened["doca"].item() - out["doca"].item()) > 5.


def test_scalar_scipy_objective_oracle():
    rng = np.random.default_rng(4203)
    s = seed(radius=220., pitch=-130.)
    z = rng.uniform(-130., 130., 48)
    directions = rng.normal(size=(48, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    u = rng.uniform(-50., 50., 48)
    p, _ = point_tangent(s, z)
    # Generic perturbations, not the constructed-normal helper used above.
    mid = p-u[:, None]*directions+rng.normal(0., .4, (48, 3))
    measured = u+rng.normal(0., 3., 48)
    out = call(s, mid, directions, measured_u=measured, tolerance=1e-9,
               max_iterations=20)
    assert out["converged"].all()
    # Independent scalar oracle minimizes Euclidean distance without module Newton derivatives.
    for i in range(48):
        def objective(zz):
            position, _ = point_tangent(s, zz)
            v = position-mid[i]
            separation = v - np.dot(v, directions[i])*directions[i]
            return np.dot(separation, separation)

        start = mid[i, 2]+measured[i]*directions[i, 2]
        result = minimize_scalar(objective, bounds=(start-15., start+15.),
                                 method="bounded", options={"xatol": 1e-11})
        assert result.success
        assert out["z"][0, i] == pytest.approx(result.x, abs=4e-6)
        assert out["doca"][0, i]**2 == pytest.approx(result.fun, abs=1e-10)
        assert objective(out["z"][0, i]) <= objective(start)+1e-10


def test_fit_seed_direct_and_explicit_batch_columns():
    s = seed()
    p, _ = point_tangent(s, np.linspace(-90., 130., 15))
    fitted = fit_helix_seed(p[None], np.ones((1, 15), bool), bz=1.)
    assert fitted["success"].item()
    mid, d = construction(s, np.array([0., 25., 90.]), [.4, .7, .3], offset=.4)
    direct = call(fitted, mid, d)
    expected = call(s, mid, d)
    for name in ("doca", "signed_doca", "z", "wire_u", "wire_pull"):
        np.testing.assert_allclose(direct[name], expected[name], atol=1e-8)
    expanded = dict(fitted, center_xy=fitted["center_xy"][:, None, :])
    for name in ("radius", "pitch", "phase_ref", "z_ref", "success"):
        expanded[name] = expanded[name][:, None]
    explicit = call(expanded, mid, d)
    for name in direct:
        np.testing.assert_array_equal(explicit[name], direct[name])


def test_seed_batch_broadcast_mask_and_no_input_mutation():
    s = seed(center_xy=np.array([[13., -21.]]*4), radius=np.array([140., 140., 140., np.nan]),
             pitch=np.array([85., 85., 85., 85.]), success=np.array([True, False, True, True]))
    mid, d = construction(seed(), np.array([10., 40., 70.]), [.2, -.4, .8])
    before = {k: np.copy(v) for k, v in s.items()}
    out = call(s, mid, d)
    assert out["doca"].shape == (4, 3)
    assert out["converged"][[0, 2]].all()
    assert not out["valid"][[1, 3]].any()
    for k in ("doca", "signed_doca", "wire_u", "wire_pull", "z"):
        assert (out[k][[1, 3]] == 0).all()
    for k in s:
        np.testing.assert_array_equal(s[k], before[k])
    np.testing.assert_array_equal(out["doca"][0], out["doca"][2])


@pytest.mark.parametrize("batch,hits", [(0, 5), (3, 0), (0, 0)])
def test_empty(batch, hits):
    s = seed(radius=np.ones((batch, 1))*140.)
    out = call(s, np.empty((batch, hits, 3)), np.empty((batch, hits, 3)),
               measured_u=np.empty((batch, hits)), uvar=1.)
    for value in out.values():
        assert value.shape == (batch, hits)


@pytest.mark.parametrize("name,bad", [("radius", 0.), ("radius", -1.), ("pitch", 0.),
                                     ("pitch", np.inf), ("phase_ref", np.nan),
                                     ("z_ref", np.inf), ("center_xy", [np.nan, 0.])])
def test_invalid_seeds_are_flagged_with_finite_payload(name, bad):
    out = call(seed(**{name: bad}), mid=[1., 0., 3.])
    assert not out["valid"].any() and not out["converged"].any()
    assert not out["inside"].any() and not out["sign_valid"].any()
    assert_finite(out)


def test_stationary_maximum_never_converges():
    s = seed(center_xy=[0., 0.], radius=2., pitch=3., phase_ref=0., z_ref=0.)
    out = call(s, [-1., 0., 0.], [0., 0., 1.], measured_u=0.)
    assert out["valid"].item()
    assert out["doca"].item() == pytest.approx(3.)
    assert not out["converged"].any() and not out["iterations"].any()
    assert_finite(out)


def test_flat_objective_and_tangent_contact_are_not_convergence():
    s = seed(center_xy=[0., 0.], radius=2., pitch=3., phase_ref=0., z_ref=0.)
    flat = call(s, [0., 0., 0.], [0., 0., 1.], measured_u=0.)
    assert flat["doca"].item() == pytest.approx(2.)
    assert not flat["converged"].any()
    contact = call(s, [2., 0., 0.], [0., 2/3, 1.], measured_u=0.)
    assert contact["doca"].item() == 0.
    assert not contact["converged"].any()
    assert not contact["sign_valid"].any()
    assert_finite(flat)
    assert_finite(contact)


def test_nearly_flat_positive_curvature_fails_rank_guard():
    s = seed(center_xy=[0., 0.], radius=2., pitch=3., phase_ref=0., z_ref=0.)
    out = call(s, [1e-15, 0., 0.], [0., 0., 1.], measured_u=0.)
    assert out["valid"].item() and not out["converged"].item()


def test_zero_budget_and_final_update_convergence():
    s = seed()
    mid, d = construction(s, 29., [.2, .3, .8])
    at_start = call(s, mid, d, max_iterations=0)
    assert at_start["converged"].item() and at_start["iterations"].item() == 0
    not_yet = call(s, mid, d, measured_u=15., max_iterations=0)
    assert not not_yet["converged"].item()
    assert not_yet["z"].item() == pytest.approx(mid[2]+15*d[2])
    complete = call(s, mid, d, measured_u=15.)
    count = int(complete["iterations"].item())
    assert count > 0 and complete["converged"].item()
    last = call(s, mid, d, measured_u=15., max_iterations=count)
    before = call(s, mid, d, measured_u=15., max_iterations=count-1)
    assert last["converged"].item() and not before["converged"].item()


def test_small_clipped_step_is_not_false_convergence():
    s = seed()
    mid, d = construction(s, 29., [.2, .3, .8])
    out = call(s, mid, d, measured_u=20., max_iterations=3,
               max_step=1e-10, tolerance=1e-6)
    z0 = mid[2]+20*d[2]
    assert not out["converged"].item() and out["iterations"].item() == 3
    assert abs(out["z"].item()-z0) <= 3.01e-10
    assert_finite(out)


def test_phase_step_and_total_basin_bounds():
    s = seed(radius=4., pitch=2.)
    mid, d = construction(s, 20., [.3, .2, .9])
    measured = np.linspace(-50., 50., 101)
    z0 = mid[2]+measured*d[2]
    one = call(s, mid, d, measured_u=measured, max_iterations=1, max_step=1e6)
    many = call(s, mid, d, measured_u=measured, max_iterations=40, max_step=1e6)
    assert (np.abs(one["z"][0]-z0) <= np.pi*abs(s["pitch"])/8+1e-12).all()
    assert (np.abs(many["z"][0]-z0) <= np.pi*abs(s["pitch"])/2+1e-12).all()
    assert (many["iterations"] <= 40).all()
    assert not many["converged"].all()  # no global-search rescue of every lane


def test_finite_wire_is_separate_not_endpoint_minimization():
    s = seed(center_xy=[0., 0.], phase_ref=0., z_ref=0.)
    # Exactly representable u on x-oriented wire at z=0.
    mid = [s["radius"]-12., 0., 0.]
    out = call(s, mid, [1., 0., 0.], hlen=np.array([11., 12., 13.]))
    assert out["converged"].all()
    np.testing.assert_array_equal(out["inside"], [[False, True, True]])
    np.testing.assert_array_equal(out["wire_u"], [[12., 12., 12.]])
    np.testing.assert_array_equal(out["doca"], [[0., 0., 0.]])
    zero = call(s, [s["radius"], 0., 0.], [1., 0., 0.], measured_u=0., hlen=0.)
    assert zero["inside"].item()


@pytest.mark.parametrize("kwargs", [dict(max_iterations=-1), dict(max_iterations=1.5),
    dict(max_iterations=True), dict(max_iterations=2**32), dict(max_step=0.),
    dict(max_step=np.inf), dict(max_step=[1.]), dict(max_step=True),
    dict(tolerance=0.), dict(tolerance=np.nan), dict(tolerance=1j)])
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        call(**kwargs)


@pytest.mark.parametrize("kwargs", [dict(mid=[1., np.nan, 3.]), dict(mid=[1., 2.]),
    dict(mid=np.zeros((1, 1, 1, 3))), dict(mid=[1+2j, 0., 0.]),
    dict(direction=[0., 0., 0.]), dict(direction=[1., np.inf, 0.]),
    dict(direction=[True, False, False]), dict(direction=[1., 0.]),
    dict(measured_u=np.nan), dict(measured_u="bad"), dict(uvar=0.), dict(uvar=-1.),
    dict(uvar=np.inf), dict(hlen=-1.), dict(hlen=np.nan),
    dict(measured_u=np.ones((1, 1, 1))), dict(uvar=np.ones((3, 2)), hlen=np.ones((4, 2)))])
def test_bad_geometry_and_measurements(kwargs):
    with pytest.raises(ValueError):
        call(**kwargs)


@pytest.mark.parametrize("change", [dict(center_xy=[1., 2., 3.]),
    dict(center_xy=np.zeros((2, 3, 2))), dict(radius=np.ones((2, 3))),
    dict(pitch="bad"), dict(success=1), dict(success=np.ones((2, 3), bool)),
    dict(center_xy=np.zeros((2, 2)), radius=np.ones(3))])
def test_bad_seed_shapes_and_types(change):
    with pytest.raises(ValueError):
        call(seed(**change), mid=[1., 2., 3.])


def test_missing_seed_parameter():
    s = seed()
    del s["pitch"]
    with pytest.raises(ValueError, match="pitch"):
        call(s, mid=[1., 2., 3.])


def test_geometry_validated_even_when_seed_or_output_empty():
    with pytest.raises(ValueError, match="nonzero"):
        call(seed(radius=np.empty(0)), mid=np.empty((0, 3)), direction=[0., 0., 0.])
    with pytest.raises(ValueError, match="finite"):
        call(seed(success=False), mid=[np.nan, 0., 0.])


@pytest.mark.parametrize("changes,mid,measured", [
    (dict(radius=1e308, pitch=1e-308), [0., 0., 0.], 0.),
    (dict(z_ref=-1e308), [0., 0., 1e308], 1e308),
    (dict(center_xy=[1e308, -1e308]), [-1e308, 1e308, 0.], 0.),
])
def test_extreme_finite_geometry_does_not_leak_nonfinite_payload(changes, mid, measured):
    with np.errstate(all="raise"):
        out = call(seed(**changes), mid, [0., 0., 1.], measured_u=measured)
    assert not out["valid"].item() and not out["converged"].item()
    assert_finite(out)


def test_batched_wires_and_batch_only_measurement_columns():
    z = np.array([[10., 30., 50.], [-20., 15., 60.]])
    mid, d = construction(seed(), z, [.2, .4, .9], u=np.array([[12.], [21.]]))
    s = seed(radius=np.array([[140.], [140.]]))
    out = call(s, mid, d, measured_u=np.array([[13.], [20.]]),
               uvar=np.array([[4.], [9.]]), hlen=np.array([[15.], [20.]]))
    assert out["converged"].all()
    np.testing.assert_allclose(out["z"], z, atol=1e-6)
    np.testing.assert_allclose(out["wire_u"], [[12.]*3, [21.]*3], atol=1e-6)
    np.testing.assert_allclose(out["wire_pull"], [[-.5]*3, [1/3]*3], atol=1e-6)
    np.testing.assert_array_equal(out["inside"], [[True]*3, [False]*3])


def test_translation_rotation_and_phase_branch_invariance():
    s = seed()
    mid, d = construction(s, np.array([-30., 20., 50.]), [.3, -.8, .4], offset=.5)
    base = call(s, mid, d, measured_u=13.)
    angle = .83
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    shift = np.array([12000., -3000., 2000.])
    shifted_seed = dict(s, center_xy=rotation[:2, :2]@s["center_xy"]+shift[:2],
                        z_ref=s["z_ref"]+shift[2], phase_ref=s["phase_ref"]+angle+12*np.pi)
    moved = call(shifted_seed, mid@rotation.T+shift, d@rotation.T, measured_u=13.)
    assert moved["converged"].all()
    np.testing.assert_allclose(moved["z"]-shift[2], base["z"], atol=1e-8)
    for name in ("doca", "signed_doca", "wire_u", "wire_pull"):
        np.testing.assert_allclose(moved[name], base[name], atol=1e-8)


def test_local_boundary_stall_preserves_finite_unconverged_payload():
    s = seed(center_xy=[0., 0.], radius=2., pitch=3., phase_ref=0., z_ref=0.)
    mid = [-4.224074793767505, -8.345102866231356, -3.5903981552706687]
    d = np.array([.1985120251859436, .47263172809375736, -.8586106367007597])
    measured = -1.4667043493933873
    out = call(s, mid, d, measured_u=measured, max_iterations=30)
    z0 = mid[2]+measured*d[2]/np.linalg.norm(d)
    assert out["valid"].item() and not out["converged"].item()
    assert 0 < out["iterations"].item() < 30
    assert abs(out["z"].item()-z0) == pytest.approx(np.pi*abs(s["pitch"])/2, abs=1e-12)
    assert_finite(out)


def test_failed_positive_curvature_line_search_retains_last_iterate():
    s = seed(center_xy=[0., 0.], radius=2., pitch=3., phase_ref=0., z_ref=0.)
    mid = [-9.918445500280226, -3.821524645097476, 9.03508032927229]
    d = [-.8026886009346064, -.3012971076394988, .5146951164113728]
    measured = -.12493256903928662
    out = call(s, mid, d, measured_u=measured, max_iterations=30)
    assert out["valid"].item() and not out["converged"].item()
    attempts = out["iterations"].item()
    assert 1 < attempts < 30
    previous = call(s, mid, d, measured_u=measured, max_iterations=attempts-1)
    for name in ("doca", "signed_doca", "wire_u", "wire_pull", "z"):
        np.testing.assert_array_equal(previous[name], out[name])
    assert_finite(out)
