"""Independent analytic geometry and synthetic fixed-geometry timing tests.

Scalar formulas/Brent inversion below are test oracles, not kernel helpers.
These tests do not validate direction inference or joint spatial fitting.
"""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from trkfastsim.reco import helixtiming as timing_module
from trkfastsim.reco.fielddrift import build_model, drift_info
from trkfastsim.reco.helixtiming import prepare_timing, profile_t0


BZ = 1.0117056369781494
MASS = 105.6583755
C = 299.792458


def scalar_helix(p, zref, z):
    """Independent integral of a rotating transverse tangent (no sinc helper)."""
    x, y, tx, ty, w = p
    dz = z-zref
    if w == 0:
        return np.array([x+tx*dz, y+ty*dz, z]), np.array([tx, ty, 1.])
    angle = w*dz
    return (np.array([x+(tx*math.sin(angle)-ty*(1-math.cos(angle)))/w,
                      y+(ty*math.sin(angle)+tx*(1-math.cos(angle)))/w, z]),
            np.array([tx*math.cos(angle)-ty*math.sin(angle),
                      tx*math.sin(angle)+ty*math.cos(angle), 1.]))


def geometry(batch=2, hits=13, *, tilted=True, span=1300.):
    """Build stationary POCAs from the independent normal to track and wire."""
    p = np.tile([12., -7., .9, -.4, .0021], (batch, 1))
    p[:, 4] *= np.where(np.arange(batch) % 2 == 0, 1., -1.)
    zref = np.full(batch, 35.)
    z = np.broadcast_to(np.linspace(-span, span, hits)+35., (batch, hits)).copy()
    cal, hv = np.zeros((batch, hits, 3)), np.zeros((batch, hits, 3))
    points, qs = np.zeros_like(cal), np.zeros_like(cal)
    for b in range(batch):
        for h in range(hits):
            point, tangent = scalar_helix(p[b], zref[b], z[b, h])
            s = np.array([.3, 1., .65 if tilted else 0.])
            s /= np.linalg.norm(s)
            normal = np.cross(s, tangent)
            normal /= np.linalg.norm(normal)
            q = normal*(.3+1.7*(h+1)/(hits+1))
            sensor = point-q
            cal[b, h], hv[b, h] = sensor+70*s, sensor-130*s
            points[b, h], qs[b, h] = point, q
    inputs = dict(local_parameters=p, z_ref=zref, poca_z=z, wire_cal=cal, wire_hv=hv,
                  early_end=np.broadcast_to(np.arange(hits) % 2, (batch, hits)).copy(),
                  early_time=np.full((batch, hits), 1300.),
                  signal_speed=np.full((batch, hits), 180.),
                  valid=np.ones((batch, hits), bool), bz=BZ, seed_valid=np.ones(batch, bool))
    return inputs, points, qs


@pytest.fixture(scope='module')
def model():
    drift = dict(wireVoltage=1450., kVcm=[.4, .8, 1.5, 3., 6., 12., 24.],
                 cmus=[3.2, 4.5, 5.7, 6.5, 7., 7.1, 7.],
                 phiBins=9, deltaDistance=.037, deltaTime=.19)
    response = dict(driftIgnorePhi=False, dRdTScale=1.17,
                    driftOffBins=[0., 2.5], driftOffset=[.012]*8,
                    driftRMSBins=[-.1, 2.7], signedDriftRMS=[.11]*7,
                    unsignedDriftRMS=[.75]*7)
    return build_model(drift, response, wire_radius=.0125, outer_radius=2.5, bz=BZ)


def synthetic(model, batch=2, hits=13):
    inputs, _, _ = geometry(batch, hits)
    prepared = prepare_timing(**inputs)
    assert np.all(prepared['row_valid'])
    truth_t0 = 1100.+np.arange(batch)*17.
    flight = np.where(np.arange(batch) % 2 == 0, 1, -1)
    dt = np.zeros((batch, hits))
    for b in range(batch):
        for h in range(hits):
            radius, phi = prepared['doca'][b, h], prepared['phi'][b, h]
            dt[b, h] = brentq(lambda t: float(drift_info(model, t, phi)['r_drift'])-radius,
                             -20., 150., xtol=1e-12)
    inputs['early_time'] = (truth_t0[:, None]+flight[:, None]*prepared['tof_positive']+dt
                           +prepared['propagation_time'])
    return inputs, prepare_timing(**inputs), truth_t0, flight, dt


@pytest.mark.parametrize('end', [0, 1])
@pytest.mark.parametrize('bz', [BZ, -BZ])
def test_analytic_helix_geometry_both_ends_and_field(end, bz):
    inputs, points, qs = geometry()
    inputs['early_end'][:] = end
    inputs['bz'] = bz
    actual = prepare_timing(**inputs)
    assert actual['row_valid'].all()
    assert np.all(actual['status'] == 'ready')
    np.testing.assert_allclose(actual['particle_poca'], points, atol=4e-13, rtol=0)
    np.testing.assert_allclose(actual['sensor_poca'], points-qs, atol=4e-13, rtol=0)
    for b in range(2):
        p = inputs['local_parameters'][b]
        scale = math.sqrt(1+p[2]**2+p[3]**2)
        momentum = .299792458*abs(bz)*scale/abs(p[4])
        beta = momentum/math.hypot(momentum, MASS)
        assert actual['p_abs'][b] == pytest.approx(momentum, rel=3e-16)
        assert actual['beta'][b] == pytest.approx(beta, rel=3e-16)
        for h in range(13):
            early = inputs['wire_cal'][b, h] if end == 0 else inputs['wire_hv'][b, h]
            late = inputs['wire_hv'][b, h] if end == 0 else inputs['wire_cal'][b, h]
            s = (early-late)/np.linalg.norm(early-late)
            q = qs[b, h]
            bunit = np.array([0., 0., np.sign(bz)])
            radial = np.cross(s, bunit)
            if np.dot(q, radial) < 0:
                radial *= -1
            phi = math.atan2(np.dot(q, radial), np.dot(q, bunit))
            if phi > math.pi/2:
                phi = math.pi-phi
            _, tangent = scalar_helix(p, inputs['z_ref'][b], inputs['poca_z'][b, h])
            signed = np.sign(np.dot(np.cross(s, tangent/scale), -q))*np.linalg.norm(q)
            tof = (inputs['poca_z'][b, h]-inputs['z_ref'][b])*scale/(beta*C)
            # Sub-mm separations at metre-scale positions magnify angular roundoff.
            assert actual['phi'][b, h] == pytest.approx(phi, abs=1e-12)
            assert actual['signed_doca_positive'][b, h] == pytest.approx(signed, abs=4e-13)
            assert actual['tof_positive'][b, h] == pytest.approx(tof, abs=4e-15)
    np.testing.assert_allclose(actual['propagation_distance'], 70 if end == 0 else 130, atol=2e-13)
    np.testing.assert_allclose(actual['sensor_time'], 1300-(70 if end == 0 else 130)/180)


def test_tilted_wire_cross_must_not_be_normalized():
    inputs, _, qs = geometry(batch=1)
    actual = prepare_timing(**inputs)
    s = inputs['wire_cal']-inputs['wire_hv']
    s /= np.linalg.norm(s, axis=-1)[..., None]
    radial = np.cross(s, [0., 0., 1.])
    radial /= np.linalg.norm(radial, axis=-1)[..., None]
    wrong = np.arctan2(np.abs(np.sum(qs*radial, axis=-1)), qs[..., 2])
    wrong = np.where(wrong > np.pi/2, np.pi-wrong, wrong)
    assert np.max(np.abs(wrong-actual['phi'])) > .05


def test_independent_analytic_line_is_geometry_only_no_beta_fallback():
    p = np.array([[1., 2., .4, -.6, 0.]])
    point, tangent = scalar_helix(p[0], 10., 30.)
    s = np.array([1., 0., 0.])
    q = 1.2*np.cross(s, tangent)/np.linalg.norm(np.cross(s, tangent))
    sensor = point-q
    actual = prepare_timing(p, [10.], [[30.]], [[sensor+20*s]], [[sensor-40*s]],
                            [[0]], [[100.]], [[200.]], [[True]], bz=BZ, seed_valid=[True])
    assert actual['stationary'][0, 0]
    assert actual['inside'][0, 0]
    np.testing.assert_allclose(actual['sensor_poca'][0, 0], sensor)
    assert actual['doca'][0, 0] == pytest.approx(1.2)
    assert actual['sensor_time'][0, 0] == pytest.approx(99.9)
    assert actual['status'][0] == 'undefined_momentum'
    assert not actual['row_valid'][0] and np.isnan(actual['beta'][0])


@pytest.mark.parametrize('field,value,label', [
    ('early_end', 2, 'invalid_input'), ('early_end', np.nan, 'invalid_input'),
    ('signal_speed', -1., 'invalid_input'), ('signal_speed', 0., 'invalid_input'),
    ('signal_speed', np.inf, 'invalid_input'), ('early_time', np.nan, 'invalid_input'),
    ('poca_z', np.nan, 'invalid_input'), ('wire_cal', np.nan, 'invalid_input'),
])
def test_selected_invalid_hit_fails_whole_lane_and_retains_diagnostics(field, value, label):
    inputs, _, _ = geometry()
    inputs[field] = inputs[field].astype(float)
    inputs[field][0, 3] = value
    actual = prepare_timing(**inputs)
    assert not actual['row_valid'][0] and actual['row_valid'][1]
    assert actual['status'][0] == label
    assert actual['hit_status'][0, 3] == label
    assert np.isfinite(actual['sensor_time']).all()
    assert actual['doca'][0, 0] > 0


def test_masked_nan_empty_rows_and_zero_shape():
    inputs, _, _ = geometry(batch=3)
    inputs['valid'][0, 2] = False
    inputs['valid'][1] = False
    for field in ('poca_z', 'wire_cal', 'wire_hv', 'early_end', 'early_time', 'signal_speed'):
        inputs[field] = inputs[field].astype(float)
        inputs[field][~inputs['valid']] = np.nan
    actual = prepare_timing(**inputs)
    np.testing.assert_array_equal(actual['row_valid'], [True, False, True])
    assert actual['status'][1] == 'empty'
    for key in ('doca', 'phi', 'sensor_time', 'tof_positive', 'signed_doca_positive',
                'propagation_distance', 'propagation_time', 'particle_poca', 'sensor_poca',
                'inside', 'parallel', 'stationary', 'angle_valid', 'sign_valid', 'valid'):
        assert not np.any(actual[key][~inputs['valid']])
    for batch, hits in ((0, 4), (3, 0), (0, 0)):
        inp, _, _ = geometry(batch=batch, hits=hits)
        result = prepare_timing(**inp)
        assert result['row_valid'].shape == (batch,)
        assert not result['row_valid'].any()


@pytest.mark.parametrize('mode', ['none', 'false', 'nan', 'zero_slope'])
def test_requires_explicit_caller_certification_and_measured_curvature(mode):
    inputs, _, _ = geometry()
    if mode == 'none':
        del inputs['seed_valid']
    elif mode == 'false':
        inputs['seed_valid'][:] = False
    elif mode == 'nan':
        inputs['local_parameters'][:, 0] = np.nan
    else:
        inputs['local_parameters'][:, 2:4] = 0
    actual = prepare_timing(**inputs)
    assert not actual['row_valid'].any()
    assert np.isnan(actual['p_abs']).all()
    assert np.isnan(actual['beta']).all()


@pytest.mark.parametrize('position', ['endpoint', 'outside', 'degenerate'])
def test_finite_wire_rejection(position):
    inputs, points, qs = geometry(batch=1)
    sensor = points[0, 0]-qs[0, 0]
    s = inputs['wire_cal'][0, 0]-inputs['wire_hv'][0, 0]
    s /= np.linalg.norm(s)
    if position == 'degenerate':
        inputs['wire_cal'][0, 0] = inputs['wire_hv'][0, 0]
    else:
        # Put sensor at/beyond the late end to avoid endpoint subtraction roundoff.
        inputs['wire_hv'][0, 0] = sensor+(1 if position == 'outside' else 0)*s
        if position == 'endpoint':
            # Axis-aligned wire with exactly represented endpoint POCA.
            inputs['poca_z'][0, 0] = inputs['z_ref'][0]
            inputs['wire_hv'][0, 0] = [12., -8., 35.]
            inputs['wire_cal'][0, 0] = [82., -8., 35.]
    actual = prepare_timing(**inputs)
    assert not actual['row_valid'][0]
    assert not actual['inside'][0, 0] or position == 'degenerate'
    assert actual['status'][0] == ('invalid_input' if position == 'degenerate' else 'outside_wire')


def test_zero_doca_undefined_angle_and_sign():
    inputs, points, qs = geometry(batch=1)
    inputs['wire_cal'] += qs
    inputs['wire_hv'] += qs
    actual = prepare_timing(**inputs)
    assert not actual['angle_valid'].any()
    assert not actual['sign_valid'].any()
    assert actual['status'][0] == 'undefined_angle'
    assert np.max(actual['doca']) < 1e-9
    assert not actual['row_valid'][0]


def test_parallel_and_nonstationary_and_degenerate_triple():
    inputs, points, _ = geometry(batch=1)
    h = 6  # z=zref, exactly known tangent
    tangent = np.array([.9, -.4, 1.])
    tangent /= np.linalg.norm(tangent)
    point = points[0, h]
    q = np.cross(tangent, [0., 1., 0.])
    inputs['wire_cal'][0, h] = point-q+70*tangent
    inputs['wire_hv'][0, h] = point-q-130*tangent
    actual = prepare_timing(**inputs)
    assert actual['parallel'][0, h]
    assert not actual['sign_valid'][0, h]
    assert actual['status'][0] == 'parallel'
    inputs, _, _ = geometry(batch=1)
    inputs['poca_z'][0, h] += .05
    actual = prepare_timing(**inputs)
    assert actual['status'][0] == 'nonstationary'
    assert not actual['stationary'][0, h]
    # Orthogonal q and cross(s,pdir), though q nonzero: never invent a sign.
    inputs, _, _ = geometry(batch=1)
    point, tangent = scalar_helix(inputs['local_parameters'][0], 35., 35.)
    s = np.array([1., 0., 0.])
    q = tangent-np.dot(tangent, s)*s
    q /= np.linalg.norm(q)
    inputs['wire_cal'][0, h] = point-q+70*s
    inputs['wire_hv'][0, h] = point-q-130*s
    actual = prepare_timing(**inputs)
    assert not actual['sign_valid'][0, h]
    assert actual['signed_doca_positive'][0, h] == 0.
    assert actual['angle_valid'][0, h]


@pytest.mark.parametrize('option,value', [('max_rows', 1), ('max_cells', 25),
                                           ('max_rows', True), ('max_cells', -1),
                                           ('bz', 0.), ('bz', np.nan), ('mass', 0.)])
def test_geometry_bounds_and_config_fail(option, value):
    inputs, _, _ = geometry()
    inputs[option] = value
    with pytest.raises(ValueError):
        prepare_timing(**inputs)


@pytest.mark.parametrize('field,value', [('valid', np.ones((2, 13))),
                                         ('seed_valid', [1, 1]),
                                         ('poca_z', [1., 2.]), ('early_end', True)])
def test_geometry_shapes_are_strict(field, value):
    inputs, _, _ = geometry()
    inputs[field] = value
    with pytest.raises(ValueError):
        prepare_timing(**inputs)


def test_synthetic_t0_recovery_and_both_tof_directions(model):
    inputs, prepared, t0, flight, dt = synthetic(model)
    actual = profile_t0(model, prepared, inputs['valid'])
    np.testing.assert_array_equal(actual['direction'], [-1, 1])
    assert actual['profile_valid'].all()
    assert actual['resolution'] == .005
    for b in range(2):
        right = 1 if flight[b] == 1 else 0
        wrong = 1-right
        assert actual['t0'][b, right] == pytest.approx(t0[b], abs=.0026)
        assert actual['score'][b, wrong] > actual['score'][b, right]+1.
        np.testing.assert_allclose(actual['delta_t'][b, right], dt[b], atol=.0026, rtol=0)
        np.testing.assert_allclose(actual['drift_residual'][b, right], 0., atol=.0003)
        expected_anchor = np.median(prepared['sensor_time'][b]-flight[b]*prepared['tof_positive'][b])
        assert actual['anchor'][b, right] == expected_anchor
    assert not actual['direction_resolved'].any()
    np.testing.assert_allclose(actual['signed_error'], .11, atol=0, rtol=0)
    expected = .5*np.sum(actual['pull']**2, axis=-1)
    np.testing.assert_array_equal(actual['score'], expected)
    assert np.all(actual['status'] == 'profiled_grid')


def test_common_signal_shift_changes_sensor_time_and_t0_only(model):
    inputs, prepared, _, _, _ = synthetic(model)
    original = profile_t0(model, prepared, inputs['valid'])
    shift = 37.25
    inputs['early_time'] += shift
    shifted_timing = prepare_timing(**inputs)
    shifted = profile_t0(model, shifted_timing, inputs['valid'])
    np.testing.assert_allclose(shifted_timing['sensor_time']-prepared['sensor_time'], shift,
                               atol=2e-13, rtol=0)
    np.testing.assert_allclose(shifted['t0']-original['t0'], shift, atol=2e-13, rtol=0)
    for key in ('offset', 'score', 'delta_t', 'r_drift', 'pull'):
        np.testing.assert_array_equal(shifted[key], original[key])


@pytest.mark.parametrize('cells', [13, 26, 67, 104, 3000])
def test_profile_exact_query_chunk_invariance(model, cells):
    inputs, prepared, _, _, _ = synthetic(model)
    reference = profile_t0(model, prepared, inputs['valid'])
    actual = profile_t0(model, prepared, inputs['valid'], max_query_cells=cells)
    assert actual.keys() == reference.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)


def test_profile_exact_row_chunk_invariance(model):
    inputs, prepared, _, _, _ = synthetic(model)
    both = profile_t0(model, prepared, inputs['valid'])
    for b in range(2):
        one = {k: v[b:b+1] for k, v in prepared.items()}
        actual = profile_t0(model, one, inputs['valid'][b:b+1], max_query_cells=13)
        for key in ('t0', 'anchor', 'offset', 'score', 'profile_valid', 'delta_t', 'r_drift',
                    'drift_residual', 'signed_error', 'pull', 'status'):
            np.testing.assert_array_equal(actual[key], both[key][b:b+1])


def test_no_profile_for_bad_or_empty_geometry_and_no_queries(model, monkeypatch):
    inputs, _, _ = geometry(batch=3)
    inputs['valid'][0] = False
    inputs['seed_valid'][1] = False
    inputs['early_time'][2, 0] = np.nan
    prepared = prepare_timing(**inputs)
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid geometry must not query model')
    monkeypatch.setattr(timing_module, 'drift_info', forbidden)
    actual = profile_t0(model, prepared, inputs['valid'])
    assert np.all(actual['status'][0] == 'empty')
    assert np.all(actual['status'][1:] == 'invalid_geometry')
    assert np.isnan(actual['t0']).all()
    assert not actual['profile_valid'].any()
    assert not actual['delta_t'].any()
    for batch, hits in ((0, 4), (3, 0), (0, 0)):
        inp, _, _ = geometry(batch=batch, hits=hits)
        result = profile_t0(model, prepare_timing(**inp), inp['valid'])
        assert result['t0'].shape == (batch, 2)


def test_masked_nan_profile_diagnostics_and_fixed_membership(model):
    inputs, prepared, _, _, _ = synthetic(model)
    inputs['valid'][:, -1] = False
    inputs['early_time'][:, -1] = np.nan
    prepared = prepare_timing(**inputs)
    actual = profile_t0(model, prepared, inputs['valid'])
    assert actual['profile_valid'].all()
    for key in ('delta_t', 'r_drift', 'drift_residual', 'signed_error', 'pull'):
        assert not actual[key][:, :, -1].any()
    changed = inputs['valid'].copy()
    changed[0, 0] = False
    with pytest.raises(ValueError, match='membership'):
        profile_t0(model, prepared, changed)


@pytest.mark.parametrize('bad_error', [0., -1., np.nan, np.inf])
def test_nonpositive_or_nonfinite_selected_sigma_is_candidate_failure(model, monkeypatch, bad_error):
    inputs, prepared, _, _, _ = synthetic(model, batch=1)
    def bad_info(model, dt, phi):
        return dict(r_drift=np.ones_like(dt), signed_error=np.full_like(dt, bad_error),
                    unsigned_error=np.ones_like(dt))
    monkeypatch.setattr(timing_module, 'drift_info', bad_info)
    actual = profile_t0(model, prepared, inputs['valid'], grid_step=10.)
    assert np.all(actual['status'] == 'unavailable')
    assert np.isnan(actual['score']).all()
    assert not actual['profile_valid'].any()


def test_single_bad_hit_sigma_invalidates_candidate_no_drop(model, monkeypatch):
    inputs, prepared, _, _, _ = synthetic(model, batch=1)
    original_phi = prepared['phi'][0, 0]
    def bad_info(model, dt, phi):
        return dict(r_drift=np.ones_like(dt),
                    signed_error=np.where(phi == original_phi, -1., 1.))
    monkeypatch.setattr(timing_module, 'drift_info', bad_info)
    actual = profile_t0(model, prepared, inputs['valid'])
    assert np.all(actual['status'] == 'unavailable')


def test_negative_and_tail_queries_unclipped_and_bounded(model, monkeypatch):
    inputs, prepared, _, _, _ = synthetic(model, batch=1)
    # Enormous measured time spread is allowed, not a dynamic membership cut.
    prepared['sensor_time'][0, 0] -= 500.
    prepared['sensor_time'][0, -1] += 500.
    sizes, ranges = [], []
    real_info = timing_module.drift_info
    def record(model, dt, phi):
        sizes.append(dt.size)
        ranges.append((np.min(dt), np.max(dt)))
        return real_info(model, dt, phi)
    monkeypatch.setattr(timing_module, 'drift_info', record)
    actual = profile_t0(model, prepared, inputs['valid'], max_query_cells=37)
    assert max(sizes) <= 37
    assert min(x[0] for x in ranges) < -50
    assert max(x[1] for x in ranges) > 200
    assert np.all(actual['outside_oracle'] >= 2)
    for d in range(2):
        info = real_info(model, actual['delta_t'][0, d], prepared['phi'][0])
        np.testing.assert_array_equal(actual['r_drift'][0, d], info['r_drift'])
    assert np.min(actual['r_drift']) < 0
    assert np.max(actual['r_drift']) > 2.5


@pytest.mark.parametrize('bounds', [(20., 50.), (-200., -100.)])
def test_boundary_is_not_profile_success(model, bounds):
    inputs, prepared, _, _, _ = synthetic(model, batch=1)
    actual = profile_t0(model, prepared, inputs['valid'], offset_min=bounds[0], offset_max=bounds[1])
    assert actual['boundary'].all()
    assert not actual['profile_valid'].any()
    assert np.all(actual['status'] == 'profile_boundary')
    assert np.isfinite(actual['score']).all()
    assert np.all((actual['offset'] == bounds[0]) | (actual['offset'] == bounds[1]))


@pytest.mark.parametrize('options', [dict(max_query_cells=12), dict(max_query_cells=0),
                                     dict(max_query_cells=True), dict(refinements=-1),
                                     dict(refinements=13), dict(grid_step=0.),
                                     dict(offset_min=1., offset_max=0.),
                                     dict(grid_step=1e-9), dict(offset_max=np.inf)])
def test_profile_hard_bounds(model, options):
    inputs, prepared, _, _, _ = synthetic(model, batch=1)
    with pytest.raises(ValueError):
        profile_t0(model, prepared, inputs['valid'], **options)


def test_all_coarse_points_and_refinement_query_bound(model, monkeypatch):
    inputs, prepared, _, _, _ = synthetic(model, batch=1, hits=1)
    # Single hit anchor is ts-tof, so dt == -offset to floating precision.
    queries = []
    def quadratic(model, dt, phi):
        queries.append(dt.copy())
        return dict(r_drift=prepared['doca'][0, 0]+.01*(dt-17.123),
                    signed_error=np.ones_like(dt))
    monkeypatch.setattr(timing_module, 'drift_info', quadratic)
    actual = profile_t0(model, prepared, inputs['valid'], max_query_cells=17)
    seen = np.concatenate(queries)
    assert all(len(q) <= 17 for q in queries)
    assert len(seen) == 2*(501+21+21)
    coarse = np.arange(-50., 200.01, .5)
    assert np.all(np.min(np.abs(seen[:, None]-coarse), axis=0) < 1e-12)
    np.testing.assert_allclose(actual['offset'], -17.125, atol=2e-13, rtol=0)


def test_keep_coarse_best_across_refinement_with_calibration_switch(model, monkeypatch):
    inputs, prepared, _, _, _ = synthetic(model, batch=1, hits=1)
    def discontinuous(model, dt, phi):
        # Narrow isolated exactly sampled minimum: refinement may not replace it.
        extra = np.where(np.abs(dt-18.) < 1e-10, 0., 1.+np.abs(dt-17.9))
        return dict(r_drift=prepared['doca'][0, 0]+extra,
                    signed_error=np.ones_like(dt))
    monkeypatch.setattr(timing_module, 'drift_info', discontinuous)
    actual = profile_t0(model, prepared, inputs['valid'], max_query_cells=1)
    np.testing.assert_array_equal(actual['offset'], -18.*np.ones((1, 2)))
    np.testing.assert_array_equal(actual['score'], np.zeros((1, 2)))


def test_tilted_endpoint_roundoff_is_rejected():
    inputs, points, qs = geometry(batch=1)
    inputs['wire_cal'][:] = points-qs
    actual = prepare_timing(**inputs)
    assert not actual['inside'].any()
    assert actual['status'][0] == 'outside_wire'


def test_stationary_maximum_requires_positive_analytic_hessian():
    # At z=0, gradient=0 but Hessian=1-2=-1: a stationary point that is not a POCA.
    s, sensor = np.array([1., 0., 0.]), np.array([0., 20., 0.])
    actual = prepare_timing([[0., 0., 1., 0., .1]], [0.], [[0.]],
                            [[sensor+50*s]], [[sensor-50*s]], [[0]], [[100.]],
                            [[200.]], [[True]], bz=BZ, seed_valid=[True])
    assert actual['inside'][0, 0]
    assert actual['sign_valid'][0, 0]
    assert not actual['stationary'][0, 0]
    assert actual['status'][0] == 'nonstationary'


@pytest.mark.parametrize('field', ['local_parameters', 'wire_cal', 'poca_z'])
def test_extreme_finite_geometry_is_flagged_without_warning(field):
    inputs, _, _ = geometry()
    inputs[field][:] = 1e308
    actual = prepare_timing(**inputs)
    assert not actual['row_valid'].any()
    for key in ('particle_poca', 'sensor_poca', 'doca', 'phi', 'propagation_distance',
                'propagation_time', 'sensor_time', 'tof_positive', 'signed_doca_positive'):
        assert np.isfinite(actual[key]).all()


@pytest.mark.parametrize('sigma', [0., -.12])
def test_actual_native_model_nonpositive_sigma_fails(model, sigma):
    signed = replace(model.signed, intercept=np.full_like(model.signed.intercept, sigma),
                     slope=np.zeros_like(model.signed.slope))
    bad_model = replace(model, signed=signed)
    inputs, prepared, _, _, _ = synthetic(model, batch=1)
    actual = profile_t0(bad_model, prepared, inputs['valid'])
    assert np.all(actual['status'] == 'unavailable')
    assert not actual['profile_valid'].any()


def test_nonconstant_signed_error_score_matches_direct_query(model):
    varying = replace(model.signed, intercept=np.full_like(model.signed.intercept, .08),
                      slope=np.full_like(model.signed.slope, .035))
    varied_model = replace(model, signed=varying)
    inputs, prepared, _, _, _ = synthetic(model)
    inputs['early_time'] += np.linspace(-.7, .8, inputs['valid'].shape[1])
    prepared = prepare_timing(**inputs)
    actual = profile_t0(varied_model, prepared, inputs['valid'])
    for b in range(2):
        for d in range(2):
            info = drift_info(varied_model, actual['delta_t'][b, d], prepared['phi'][b])
            expected_pull = (prepared['doca'][b]-info['r_drift'])/info['signed_error']
            assert actual['score'][b, d] == .5*np.sum(expected_pull**2)
            np.testing.assert_array_equal(actual['signed_error'][b, d], info['signed_error'])
            assert np.ptp(info['signed_error']) > .001


def test_no_refinement_without_a_valid_coarse_point(model, monkeypatch):
    inputs, prepared, _, _, _ = synthetic(model, batch=1, hits=1)
    sizes = []
    def bad_info(model, dt, phi):
        sizes.append(dt.size)
        return dict(r_drift=np.zeros_like(dt), signed_error=-np.ones_like(dt))
    monkeypatch.setattr(timing_module, 'drift_info', bad_info)
    actual = profile_t0(model, prepared, inputs['valid'], max_query_cells=17)
    assert sum(sizes) == 2*501  # no invented zero-offset refinement bracket
    assert np.all(actual['status'] == 'unavailable')


def test_score_ties_keep_first_coarse_point_exactly_across_chunks(model, monkeypatch):
    inputs, prepared, _, _, _ = synthetic(model, batch=1, hits=1)
    def flat_info(model, dt, phi):
        return dict(r_drift=np.full_like(dt, prepared['doca'][0, 0]),
                    signed_error=np.ones_like(dt))
    monkeypatch.setattr(timing_module, 'drift_info', flat_info)
    large = profile_t0(model, prepared, inputs['valid'])
    small = profile_t0(model, prepared, inputs['valid'], max_query_cells=1)
    for key in large:
        np.testing.assert_array_equal(large[key], small[key], err_msg=key)
    assert np.all(small['offset'] == -200.)
    assert np.all(small['status'] == 'profile_boundary')


@pytest.mark.parametrize('distance,defined', [(0.5e-9, False), (2e-9, True)])
def test_fixed_engineering_zero_doca_threshold(distance, defined):
    tangent, s = np.array([.9, -.4, 1.]), np.array([1., 0., 0.])
    q = distance*np.cross(s, tangent)/np.linalg.norm(np.cross(s, tangent))
    actual = prepare_timing([[0., 0., .9, -.4, .0021]], [0.], [[0.]],
                            [[-q+100*s]], [[-q-100*s]], [[0]], [[100.]],
                            [[200.]], [[True]], bz=BZ, seed_valid=[True])
    assert bool(actual['angle_valid'][0, 0]) is defined
    assert bool(actual['sign_valid'][0, 0]) is defined


@pytest.mark.parametrize('sine,is_parallel', [(.5e-5, True), (2e-5, False)])
def test_parallel_threshold_uses_normalized_cross(sine, is_parallel):
    tangent = np.array([.9, -.4, 1.])
    tangent /= np.linalg.norm(tangent)
    perp = np.cross(tangent, [0., 1., 0.])
    perp /= np.linalg.norm(perp)
    s = math.sqrt(1-sine*sine)*tangent+sine*perp
    q = np.cross(s, tangent)
    q /= np.linalg.norm(q)
    actual = prepare_timing([[0., 0., .9, -.4, .0021]], [0.], [[0.]],
                            [[-q+100*s]], [[-q-100*s]], [[0]], [[100.]],
                            [[200.]], [[True]], bz=BZ, seed_valid=[True])
    assert bool(actual['parallel'][0, 0]) is is_parallel
    assert bool(actual['sign_valid'][0, 0]) is not is_parallel


def test_multiple_row_tiles_remain_exact_under_query_budget(model, monkeypatch):
    inputs, _, _ = geometry(batch=35, hits=1)
    prepared = prepare_timing(**inputs)
    def constant_radius(model, dt, phi):
        return dict(r_drift=1.+.01*(dt-15.127), signed_error=np.ones_like(dt))
    monkeypatch.setattr(timing_module, 'drift_info', constant_radius)
    large = profile_t0(model, prepared, inputs['valid'])
    small = profile_t0(model, prepared, inputs['valid'], max_query_cells=7)
    for key in large:
        np.testing.assert_array_equal(large[key], small[key], err_msg=key)


def test_nonzero_doca_can_still_have_undefined_lorentz_angle():
    # A field-parallel wire and transverse separation give an undefined atan2(0,0).
    s, sensor = np.array([0., 0., 1.]), np.array([0., -1., 0.])
    actual = prepare_timing([[0., 0., 1., 0., .001]], [0.], [[0.]],
                            [[sensor+100*s]], [[sensor-100*s]], [[0]], [[100.]],
                            [[200.]], [[True]], bz=BZ, seed_valid=[True])
    assert actual['doca'][0, 0] == 1.
    assert actual['stationary'][0, 0] and actual['inside'][0, 0]
    assert actual['sign_valid'][0, 0] and not actual['parallel'][0, 0]
    assert not actual['angle_valid'][0, 0]
    assert actual['status'][0] == 'undefined_angle'
    assert not actual['row_valid'][0]
