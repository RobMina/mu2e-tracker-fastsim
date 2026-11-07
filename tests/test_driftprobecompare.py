"""Counterfactual derivative decomposition, not acceptance tests on real data."""
import copy

import numpy as np
import pytest

from test_driftjacobian import setup
from test_helixdriftfit import model
from trkfastsim.reco import driftjacobian, driftprobecompare
from trkfastsim.reco.driftfitdiagnostic import _Evaluator, _empty_state


def test_identical_evaluator_zero_root_and_center_terms(model):
    _, ev, p, state = setup(model)
    before = copy.deepcopy(state)
    lin = driftjacobian.linearize(ev, p, state)
    out = driftprobecompare.compare(ev, ev, p, state, state, lin, lin)
    assert out['valid'].all()
    np.testing.assert_array_equal(out['probe_root_contribution'], 0.)
    np.testing.assert_array_equal(out['analytic_center_contribution'], 0.)
    np.testing.assert_array_equal(out['max_probe_poca_shift'], 0.)
    np.testing.assert_allclose(out['gradient_difference'], out['polished_fd_remainder'], atol=1e-20, rtol=3e-15)
    assert out['t0_original_poca_identical'].all() and out['t0_polished_poca_identical'].all()
    assert not out['unchanged_parameters'].any()
    steps = np.asarray(driftprobecompare.FD_STEPS)[None, :, None]*ev.scale[:, None, :]
    bound = 2*np.abs(np.spacing(p))[:, None, :]/steps
    assert np.all(np.abs(out['plus_increment_ratio']-1.) <= bound)
    assert np.all(np.abs(out['minus_increment_ratio']-1.) <= bound)
    for key in before:
        np.testing.assert_array_equal(state[key], before[key])


def test_known_residual_perturbation_exact_common_normalization(model):
    _, ev, p, state = setup(model)
    lin = driftjacobian.linearize(ev, p, state)
    slope = .125

    class Shift:
        def __getattr__(self, name):
            return getattr(ev, name)

        def __call__(self, q):
            result = ev(q)
            delta = slope*(q[:, 0]-p[:, 0])/ev.scale[:, 0]
            result['residuals'] = result['residuals'].copy()
            result['residuals'][..., 0] += ev.mask*delta[:, None]
            return result

    out = driftprobecompare.compare(ev, Shift(), p, state, state, lin, lin)
    expected = -slope*np.sum(state['residuals'][..., 0], axis=1)/lin['norm'][:, 0]
    np.testing.assert_allclose(out['probe_root_contribution'][:, :, 0], np.repeat(expected[:, None], 4, axis=1),
                               rtol=2e-8, atol=1e-8)
    np.testing.assert_array_equal(out['probe_root_contribution'][..., 1:], 0.)
    np.testing.assert_allclose(out['decomposition_roundoff'], 0., atol=3e-15)


def test_batch_invalid_and_empty(model):
    args, _, _, _ = setup(model, batch=3)
    args['seed_valid'][1] = False

    def run(a):
        ev = _Evaluator(**a)
        p = ev.candidate
        state = ev(p) if ev.batch and ev.hits else _empty_state(ev)
        lin = driftjacobian.linearize(ev, p, state)
        return driftprobecompare.compare(ev, ev, p, state, state, lin, lin)

    whole = run(args)
    for i in range(3):
        part = run({k: v[i:i+1] if isinstance(v, np.ndarray) else v for k, v in args.items()})
        for key in whole:
            np.testing.assert_array_equal(whole[key][i:i+1], part[key], err_msg=key)
    assert not whole['valid'][1].any()
    for value in whole.values():
        if value.dtype.kind == 'f':
            assert np.isnan(value[1]).all()
    zero = run({k: v[:0] if isinstance(v, np.ndarray) else v for k, v in args.items()})
    assert zero['gradient_difference'].shape == (0, 4, 6)


def test_branch_attribution_matches_endpoint_checks_and_ignores_identical_regressions(model):
    _, ev, p, state = setup(model)
    angle = driftprobecompare._angle_branch(ev, p, state)
    for i, key in enumerate(('time_bin', 'phi_bin_lower', 'offset_bin', 'signed_error_bin', 'signed_doca')):
        changed = copy.deepcopy(state)
        if key == 'signed_doca':
            changed[key][:, 0] *= -1
        else:
            changed[key][:, 0] += 1
        flags = driftprobecompare._branch_changes(ev, state, angle, changed, p)
        expected = not driftprobecompare._same_branch(state, changed, ev.mask, ev.model).all()
        assert flags[:, i].any() == expected
        assert not np.delete(flags, i, axis=1).any()
    lin = driftjacobian.linearize(ev, p, state)
    out = driftprobecompare.compare(ev, ev, p, state, state, lin, lin)
    np.testing.assert_array_equal(out['original_branch_changes'], out['polished_branch_changes'])
    np.testing.assert_array_equal(out['same_branches'], out['valid'] & ~np.any(out['original_branch_changes'], axis=-1))
