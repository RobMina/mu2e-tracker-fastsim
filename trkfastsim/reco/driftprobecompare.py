"""Controlled residual/gradient decomposition for read-only POCA sensitivity probes.

Every term uses the original center residual and original analytic column norms.
The polished evaluator is a diagnostic counterfactual, NOT an exact reference.
Differences isolate the changed root arithmetic at probes from changed analytic
center derivatives. They do not prove a sole cause or repair fit acceptance.
"""
import numpy as np

from .driftfitdiagnostic import FD_STEPS
from .stabledriftfit import _same_branch, _angle_branch

BRANCHES = ('time', 'phi', 'offset', 'signed_error', 'signed_doca', 'angle_fold')


def _branch_changes(ev, center, angle, state, p):
    mask = ev.mask
    changes = [center['time_bin'] != state['time_bin'],
               (center['phi_bin_lower'] != state['phi_bin_lower']) | (center['phi_bin_upper'] != state['phi_bin_upper'])]
    for key, cal in (('offset_bin', ev.model.offset), ('signed_error_bin', ev.model.signed)):
        a, b = np.maximum(center[key], 0), np.maximum(state[key], 0)
        changes.append((cal.intercept[a] != cal.intercept[b]) | (cal.slope[a] != cal.slope[b]))
    changes.append(np.sign(center['signed_doca']) != np.sign(state['signed_doca']))
    changes.append(np.any(_angle_branch(ev, p, state) != angle, axis=-1))
    return np.stack([np.any(mask & value, axis=1) for value in changes], axis=-1)


def compare(evaluator, polished, parameters, state, polished_state, analytic, polished_analytic):
    """Fixed four scales x six coordinates x two signs, O(B*H) live workspace.

    The exact algebraic split (up to reduction roundoff) in common normalization:
    (Jfd-Ja)R = (Jfd-Jpolished_fd)R + (Jpolished_fd-Jpolished_a)R
                + (Jpolished_a-Ja)R.
    All terms retain their signs and six coordinates. The middle term is not an
    error against truth: polishing is neither an independent root method nor an
    exact-arithmetic evaluator. Endpoint branch matching is not a path proof.
    t0 leaves spatial parameters unchanged and controls perturbation-dependent
    POCA effects. Realized floating-point increments and root shifts are saved.
    """
    b, h = evaluator.mask.shape
    shape = (b, len(FD_STEPS), 6)
    names = ('gradient_difference', 'probe_root_contribution', 'polished_fd_remainder',
             'analytic_center_contribution', 'decomposition_roundoff',
             'original_column_error', 'polished_column_error', 'max_probe_poca_shift',
             'plus_increment_ratio', 'minus_increment_ratio',
             'dominant_hit_probe_contribution', 'dominant_hit_gradient_difference')
    out = {name: np.full(shape, np.nan) for name in names}
    out['valid'] = np.zeros(shape, bool)
    out['same_branches'] = np.zeros(shape, bool)
    out['unchanged_parameters'] = np.zeros(shape+(2,), bool)
    out['original_branch_changes'] = np.zeros(shape+(len(BRANCHES),), bool)
    out['polished_branch_changes'] = np.zeros(shape+(len(BRANCHES),), bool)
    out['t0_original_poca_identical'] = np.zeros((b, len(FD_STEPS)), bool)
    out['t0_polished_poca_identical'] = np.zeros((b, len(FD_STEPS)), bool)
    base = state['objective_valid'] & polished_state['objective_valid'] & analytic['good'] & polished_analytic['good']
    if not b or not h or not np.any(base):
        return out
    r = state['residuals']
    center_angle = _angle_branch(evaluator, parameters, state)
    polished_angle = _angle_branch(polished, parameters, polished_state)
    for k, step in enumerate(FD_STEPS):
        for j in range(6):
            shift = np.zeros_like(parameters)
            shift[:, j] = step*evaluator.scale[:, j]
            plus_p, minus_p = parameters+shift, parameters-shift
            plus, minus = evaluator(plus_p), evaluator(minus_p)
            pp, pm = polished(plus_p), polished(minus_p)
            good = base & plus['objective_valid'] & minus['objective_valid'] & pp['objective_valid'] & pm['objective_valid']
            out['original_branch_changes'][:, k, j] = good[:, None] & (
                _branch_changes(evaluator, state, center_angle, plus, plus_p) |
                _branch_changes(evaluator, state, center_angle, minus, minus_p))
            out['polished_branch_changes'][:, k, j] = good[:, None] & (
                _branch_changes(polished, polished_state, polished_angle, pp, plus_p) |
                _branch_changes(polished, polished_state, polished_angle, pm, minus_p))
            same = good.copy()
            for ev, center, angle, trial, p in ((evaluator, state, center_angle, plus, plus_p),
                    (evaluator, state, center_angle, minus, minus_p),
                    (polished, polished_state, polished_angle, pp, plus_p),
                    (polished, polished_state, polished_angle, pm, minus_p)):
                same &= _same_branch(center, trial, evaluator.mask, evaluator.model)
                same &= np.all(~evaluator.mask[..., None] | (_angle_branch(ev, p, trial) == angle), axis=(1, 2))
            with np.errstate(all='ignore'):
                jp = (plus['residuals']-minus['residuals'])/(2*step)
                jq = (pp['residuals']-pm['residuals'])/(2*step)
                ja, jqa = analytic['jacobian'][..., j], polished_analytic['jacobian'][..., j]
                norm = analytic['norm'][:, j]
                hit_total = np.sum((jp-ja)*r, axis=2)/norm[:, None]
                hit_probe = np.sum((jp-jq)*r, axis=2)/norm[:, None]
                total = np.sum(hit_total, axis=1)
                probe = np.sum(hit_probe, axis=1)
                remainder = np.sum((jq-jqa)*r, axis=(1, 2))/norm
                center = np.sum((jqa-ja)*r, axis=(1, 2))/norm
                values = dict(gradient_difference=total, probe_root_contribution=probe,
                    polished_fd_remainder=remainder, analytic_center_contribution=center,
                    decomposition_roundoff=total-probe-remainder-center,
                    original_column_error=np.sqrt(np.sum((jp-ja)**2, axis=(1, 2)))/norm,
                    polished_column_error=np.sqrt(np.sum((jq-jqa)**2, axis=(1, 2)))/norm,
                    max_probe_poca_shift=np.max(np.where(evaluator.mask,
                        np.maximum(np.abs(pp['poca_z']-plus['poca_z']), np.abs(pm['poca_z']-minus['poca_z'])), 0.), axis=1),
                    plus_increment_ratio=(plus_p[:, j]-parameters[:, j])/shift[:, j],
                    minus_increment_ratio=(parameters[:, j]-minus_p[:, j])/shift[:, j],
                    dominant_hit_probe_contribution=np.take_along_axis(hit_probe, np.argmax(np.abs(hit_probe), axis=1)[:, None], axis=1)[:, 0],
                    dominant_hit_gradient_difference=np.take_along_axis(hit_total, np.argmax(np.abs(hit_total), axis=1)[:, None], axis=1)[:, 0])
            for name, value in values.items():
                out[name][:, k, j] = np.where(good, value, np.nan)
            out['valid'][:, k, j] = good
            out['same_branches'][:, k, j] = same
            out['unchanged_parameters'][:, k, j] = np.column_stack((np.all(minus_p == parameters, axis=1),
                                                                 np.all(plus_p == parameters, axis=1))) & good[:, None]
            if j == 5:
                out['t0_original_poca_identical'][:, k] = good & np.all(~evaluator.mask | (
                    (plus['poca_z'] == state['poca_z']) & (minus['poca_z'] == state['poca_z'])), axis=1)
                out['t0_polished_poca_identical'][:, k] = good & np.all(~evaluator.mask | (
                    (pp['poca_z'] == polished_state['poca_z']) & (pm['poca_z'] == polished_state['poca_z'])), axis=1)
    return out
