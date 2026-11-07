"""Read-only original-state evidence and bounded root counterfactuals."""
import copy
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import least_squares

from trkfastsim.reco import driftfailure as failure, driftjacobian as analytic
from trkfastsim.reco.driftfitdiagnostic import _Evaluator, _empty_state
from trkfastsim.reco.helixfit import _at_z, _trajectory
from trkfastsim.reco.stabledriftfit import certify, _same_branch, _angle_branch
from test_driftjacobian import setup, nonconstant, independent_residual
from test_driftfitdiagnostic import affine_model
from test_helixdriftfit import model  # noqa: F401


def equal_dict(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def slice_inputs(inputs, start, stop):
    return {key: value[start:stop] if isinstance(value, np.ndarray) else value
            for key, value in inputs.items()}


@pytest.mark.parametrize('mode', ['valid', 'masked', 'invalid', 'candidate', 'nonconstant'])
def test_zero_round_every_state_field_exact_reconstruction(model, mode):
    inputs, _, _, _ = setup(nonconstant(model) if mode == 'nonconstant' else model)
    if mode == 'masked':
        inputs['valid'][:, -3:] = False
        inputs['wire_cal'][:, -3:] = np.nan
    if mode == 'invalid':
        inputs['seed_valid'][0] = False
    ev = _Evaluator(**inputs)
    p = ev.candidate.copy()
    if mode == 'candidate':
        p[0, 0] = np.nan
    original = ev(p)
    state, evidence = failure._PolishedEvaluator(ev, rounds=0).evaluate(p)
    equal_dict(state, original)
    assert not evidence['polish_changed'].any()
    assert not evidence['polish_rounds_accepted'].any()


def test_original_certificate_unchanged_and_public_state(model):
    _, ev, p, state = setup(nonconstant(model))
    out = failure.diagnose(ev, p)
    lin = analytic.linearize(ev, p, state)
    for name, value in certify(ev, p, state, lin).items():
        np.testing.assert_array_equal(out['original_'+name], value, err_msg=name)
    for name in ('gradient_norm', 'rank', 'condition', 'differentiable'):
        np.testing.assert_array_equal(out['original_'+name], lin[name])
    for name, value in state.items():
        key = ('original_' if value.ndim == 1 else 'hit_original_')+name
        np.testing.assert_array_equal(out[key], value, err_msg=key)
    assert 'success' not in out and 'polished_success' not in out
    assert all(isinstance(value, np.ndarray) and value.shape[0] == ev.batch for value in out.values())
    assert out['original_certificate_column_relative_error'].shape == (ev.batch, 4, 6)


@pytest.mark.parametrize('change', [False, True])
def test_polishing_independent_scalar_brent_reference_and_monotonic_roots(model, change):
    _, ev, p, original = setup(nonconstant(model), batch=1, change=change)
    wrapper = failure._PolishedEvaluator(ev)
    state, evidence = wrapper.evaluate(p)
    np.testing.assert_allclose(state['residuals'][0], independent_residual(ev, p), atol=2e-10, rtol=1e-8)
    hit = evidence['polished_root_valid']
    assert hit.all()
    assert np.all(np.abs(evidence['polished_correction'][hit]) <= np.abs(evidence['original_correction'][hit]))
    assert np.all(evidence['polish_rounds_accepted'] <= 3)
    assert evidence['polish_changed'].any()
    np.testing.assert_array_equal(evidence['poca_shift'], state['poca_z']-original['poca_z'])
    root = _at_z(p[:, :5], ev.zref, state['poca_z'], ev.mid, ev.direction)
    assert root['positive'].all() and (np.abs(root['u']) < ev.hlen).all()
    assert evidence['polish_arithmetic_floor'].any()
    assert np.any(evidence['polish_arithmetic_floor'] & (evidence['polished_correction'] != 0))
    out = failure.diagnose(ev, p)
    expected = certify(wrapper, p, state, analytic.linearize(wrapper, p, state))
    for name, value in expected.items():
        np.testing.assert_array_equal(out['polished_'+name], value, err_msg=name)
    same = _same_branch(original, state, ev.mask, ev.model)
    same &= np.all(_angle_branch(ev, p, original) == _angle_branch(ev, p, state), axis=(1, 2))
    np.testing.assert_array_equal(out['center_same_branch'], same)


def test_every_certificate_probe_starts_with_original_root(model):
    _, ev, p, _ = setup(model, batch=1)

    class Observed(failure._PolishedEvaluator):
        def __init__(self, evaluator):
            super().__init__(evaluator)
            self.calls = 0

        def __call__(self, candidate):
            self.calls += 1
            state, evidence = self.evaluate(candidate)
            original = self.original(candidate)
            root = _at_z(candidate[:, :5], self.zref, original['poca_z'], self.mid, self.direction)
            np.testing.assert_array_equal(evidence['original_correction'], root['correction'])
            np.testing.assert_array_equal(evidence['poca_shift'], state['poca_z']-original['poca_z'])
            return state

    wrapper = Observed(ev)
    state = wrapper(p)
    certify(wrapper, p, state, analytic.linearize(wrapper, p, state))
    assert wrapper.calls == 49


def test_nonlinear_noisy_scalar_reference_at_stationary_saved_parameters(model):
    inputs, _, _, _ = setup(nonconstant(model), batch=1)
    h = np.arange(17)
    inputs['early_time'] += .05*np.sin(.7*h)
    inputs['measured_u'] += .08*np.cos(.9*h)
    ev = _Evaluator(**inputs)
    initial = ev.candidate.copy()
    reference = least_squares(lambda x: independent_residual(ev, initial+x[None, :]*ev.scale).ravel(),
                              np.zeros(6), xtol=1e-11, ftol=1e-11, gtol=1e-9, max_nfev=100)
    assert reference.success
    p = initial+reference.x[None, :]*ev.scale
    out = failure.diagnose(ev, p)
    assert out['original_objective_valid'].all() and out['polished_objective_valid'].all()
    assert out['polished_objective'][0] > 0
    np.testing.assert_allclose(out['hit_polished_residuals'][0], independent_residual(ev, p), atol=2e-10, rtol=1e-7)
    np.testing.assert_allclose(out['polished_objective'][0], .5*np.sum(reference.fun**2), rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(out['original_parameters'], p)


@pytest.mark.parametrize('calibration', ['offset', 'signed'])
@pytest.mark.parametrize('kind', ['identical', 'jump', 'kink'])
def test_regression_boundary_classes_and_offset_jumps(model, calibration, kind):
    inputs, _, p, state = setup(affine_model(model), batch=1)
    m = inputs['model']
    edge = state['c_drift' if calibration == 'offset' else 'r_drift'][0, 3]
    intercept = np.zeros(2) if calibration == 'offset' else np.full(2, .11)
    slope = np.zeros(2)
    if kind == 'jump':
        intercept[1] += .01
    if kind == 'kink':
        slope[1] = .03
        intercept[1] -= .03*edge
    cal = replace(getattr(m, calibration), low=edge-1, width=1., intercept=intercept, slope=slope)
    inputs['model'] = replace(m, **{calibration: cal})
    ev = _Evaluator(**inputs)
    out = failure.diagnose(ev, p)
    name = 'offset_boundary' if calibration == 'offset' else 'signed_error_boundary'
    assert out['hit_'+name][0, 3] == (kind != 'identical')
    assert out['hit_differentiable'][0, 3] == (kind == 'identical')
    if calibration == 'offset':
        assert out['hit_offset_edge_distance'][0, 3] < 1e-14
        np.testing.assert_allclose(out['hit_offset_edge_jump'][0, 3], .01 if kind == 'jump' else 0., atol=1e-15)
        assert out['hit_offset_edge_nonidentical'][0, 3] == (kind != 'identical')
        assert out['hit_offset_edge_slope_jump'][0, 3] == (.03 if kind == 'kink' else 0.)


@pytest.mark.parametrize('kind', ['time', 'phi', 'angle', 'doca', 'hessian'])
def test_inventory_exact_reasons_and_arithmetic_slack(model, kind):
    inputs, ev, p, state = setup(model, batch=1)
    if kind == 'time':
        inputs['early_time'][0, 2] += 30*model.delta_t+1e-12-state['delta_t'][0, 2]
        ev = _Evaluator(**inputs)
        state = ev(p)
    elif kind == 'phi':
        state['phi'][0, 2] = 3*model.delta_phi
    elif kind == 'doca':
        state['doca'][0, 2] = 1e-10
    elif kind == 'angle':
        point, tangent, _ = _trajectory(p[:, :5], ev.zref, state['poca_z'])
        w = tangent[0, 2].copy()
        w[2] = 0
        w /= np.linalg.norm(w)
        q = np.cross(w, tangent[0, 2])
        q /= np.linalg.norm(q)
        ev.direction[0, 2] = w
        ev.mid[0, 2] = point[0, 2]-q
        ev.cal[0, 2], ev.hv[0, 2] = ev.mid[0, 2]+110*w, ev.mid[0, 2]-110*w
    else:
        point, _, second = _trajectory(p[:, :5], ev.zref, state['poca_z'])
        ev.mid[0, 2] = point[0, 2]+1e9*second[0, 2]
    lin = analytic.linearize(ev, p, state)
    flags = failure._inventory(ev, p, state, lin)
    np.testing.assert_array_equal(flags['differentiable'], lin['differentiable_hit'])
    name = dict(time='time_boundary', phi='phi_boundary', angle='angle_branch', doca='doca_branch', hessian='positive_hessian')[kind]
    assert flags[name][0, 2] == (kind != 'hessian')
    assert not flags['differentiable'][0, 2]


def test_no_offset_internal_edge_missing_policy(model):
    inputs, _, _, _ = setup(model, batch=1)
    inputs['model'] = replace(model, offset=replace(model.offset, slope=np.zeros(1), intercept=np.zeros(1)))
    ev = _Evaluator(**inputs)
    out = failure.diagnose(ev, ev.candidate)
    assert np.isnan(out['hit_offset_edge_distance']).all()
    assert np.isnan(out['hit_offset_edge_jump']).all()
    assert not out['hit_offset_boundary'].any()


@pytest.mark.parametrize('bad', ['seed', 'sigma', 'candidate', 'outside', 'empty'])
def test_invalid_singleton_batch_exact_no_rescue(model, bad):
    inputs, _, _, _ = setup(model, batch=3)
    if bad == 'seed':
        inputs['seed_valid'][1] = False
    elif bad == 'sigma':
        inputs['model'] = replace(model, signed=replace(model.signed, intercept=np.zeros_like(model.signed.intercept)))
    elif bad == 'candidate':
        inputs['local_parameters'][1, 0] = np.nan
    elif bad == 'outside':
        inputs['wire_cal'][1] = inputs['wire_hv'][1]+.01*(inputs['wire_cal'][1]-inputs['wire_hv'][1])
    else:
        inputs['valid'][1] = False
    inputs['valid'][2, -2:] = False
    ev = _Evaluator(**inputs)
    full = failure.diagnose(ev, ev.candidate)
    assert not full['original_objective_valid'][1] and not full['polished_objective_valid'][1]
    for prefix in ('original', 'polished'):
        assert not full[prefix+'_certified'][1]
        assert np.isnan(full[prefix+'_gradient_norm'][1])
        assert np.isnan(full['hit_'+prefix+'_residuals'][1]).all()
    assert not full['hit_polished_root_valid'][1].any()
    assert not full['hit_polish_changed'][1].any()
    parts = []
    for i in range(3):
        one = _Evaluator(**slice_inputs(inputs, i, i+1))
        parts.append(failure.diagnose(one, one.candidate))
    equal_dict(full, {name: np.concatenate([part[name] for part in parts]) for name in full})
    np.testing.assert_array_equal(full['hit_mask'], inputs['valid'])


@pytest.mark.parametrize('batch,hits', [(0, 17), (2, 0), (0, 0)])
def test_empty_shapes(model, batch, hits):
    inputs, _, _, _ = setup(model)
    args = {key: np.zeros((batch,)+value.shape[1:], dtype=value.dtype) if isinstance(value, np.ndarray) else value
            for key, value in inputs.items()}
    for key, value in args.items():
        if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[1] == 17:
            args[key] = value[:, :hits]
    ev = _Evaluator(**args)
    equal_dict(failure._PolishedEvaluator(ev, rounds=0)(ev.candidate), _empty_state(ev))
    out = failure.diagnose(ev, ev.candidate)
    assert out['hit_mask'].shape == (batch, hits)
    assert all(value.shape[0] == batch for value in out.values())
    assert not out['original_certified'].any() and not out['polished_certified'].any()


@pytest.mark.parametrize('bad', [np.nan, np.inf, 1e300, 0.])
def test_supplied_invalid_candidate_not_rescued_and_batch_exact(model, bad):
    inputs, ev, p, _ = setup(model)
    p[0, 4] = bad
    out = failure.diagnose(ev, p)
    assert not out['original_objective_valid'][0]
    assert not out['polished_objective_valid'][0]
    assert not out['hit_polish_changed'][0].any()
    for i in range(2):
        one = _Evaluator(**slice_inputs(inputs, i, i+1))
        part = failure.diagnose(one, p[i:i+1])
        equal_dict(part, {key: value[i:i+1] for key, value in out.items()})
    equal_dict(failure._PolishedEvaluator(ev, rounds=0)(p), ev(p))


def test_polished_jacobian_matches_independent_brent_probes(model):
    _, ev, p, _ = setup(nonconstant(model), batch=1)
    wrapper = failure._PolishedEvaluator(ev)
    state = wrapper(p)
    lin = analytic.linearize(wrapper, p, state)
    for j in range(6):
        step = np.zeros_like(p)
        step[:, j] = 2e-5*ev.scale[:, j]
        reference = (independent_residual(ev, p+step)-independent_residual(ev, p-step))/4e-5
        np.testing.assert_allclose(lin['jacobian'][0, ..., j], reference, atol=4e-7, rtol=2e-5)
        for candidate in (p+step, p-step):
            np.testing.assert_allclose(wrapper(candidate)['residuals'][0], independent_residual(ev, candidate),
                                       atol=2e-10, rtol=1e-8)


def test_local_neighborhood_rejects_displaced_supplied_root(model):
    _, ev, p, original = setup(model, batch=1)
    bad = {key: value.copy() for key, value in original.items()}
    bad['poca_z'][0, 0] += 600.
    state, evidence = failure._PolishedEvaluator(ev).evaluate(p, bad)
    assert not state['objective_valid'][0]
    assert not evidence['polished_root_valid'][0, 0]
    assert not evidence['polish_changed'][0, 0]
    assert evidence['poca_shift'][0, 0] == 0


def test_nonmutating_readonly_inputs_and_inherited_bounds(model):
    inputs, _, _, _ = setup(model)
    inputs['valid'][:, -2:] = False
    inputs['wire_cal'][:, -2:] = np.nan
    ev = _Evaluator(**inputs)
    snapshot = copy.deepcopy(ev.__dict__)
    for value in ev.__dict__.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    p = ev.candidate.copy()
    p.setflags(write=False)
    out = failure.diagnose(ev, p)
    for key, value in snapshot.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(ev, key), value)
    np.testing.assert_array_equal(p, snapshot['candidate'])
    np.testing.assert_array_equal(out['hit_original_residuals'][:, -2:], 0.)
    assert np.isnan(out['hit_poca_shift'][:, -2:]).all()
    for limits in ({'max_rows': 1}, {'max_cells': 33}):
        with pytest.raises(ValueError, match='bound exceeded'):
            _Evaluator(**inputs, **limits)
    with pytest.raises(ValueError, match='shape'):
        failure.diagnose(ev, p[:, :5])
    with pytest.raises(ValueError, match='rounds'):
        failure._PolishedEvaluator(ev, rounds=4)
