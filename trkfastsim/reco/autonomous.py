"""Opt-in detector-only F1 prototype with a fixed LineFinder candidate envelope.

No stock fit products are read. The existing fitter cannot resume a fitted seed
with new candidates: extension candidates are constructed on the measured seed,
not on its fitted owned trajectory. This is not native-exact reconstruction.
"""
import argparse
import hashlib
import functools
import json
import os
from pathlib import Path
import time

import awkward as ak
import numpy as np

from . import hitreco, linefind, linefit, tracktable


def poca(x, mid, direction):
    """Broadcast straight-line POCA: track length, wire length, unsigned DOCA."""
    p = np.stack([x[..., 0], np.zeros_like(x[..., 0]), x[..., 1]], -1)
    u = np.stack([x[..., 2], -np.ones_like(x[..., 2]), x[..., 3]], -1)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    c = np.sum(u * direction, -1)
    delta = p - mid
    du, dw = np.sum(delta * u, -1), np.sum(delta * direction, -1)
    ss = np.maximum(1.0 - c * c, 1e-12)
    st, sw = (dw * c - du) / ss, (dw - du * c) / ss
    doca = np.linalg.norm(mid + direction * sw[..., None] - p - u * st[..., None], axis=-1)
    return st, sw, doca


def seed_parameters(sd):
    """Native-style float-vector direction/momentum, not a native seed certificate."""
    x = linefit.seed_params(*(sd[k] for k in ('a0', 'b0', 'a1', 'b1', 't0')))
    direction = np.stack([x[:, 2], -np.ones(len(x)), x[:, 3]], 1).astype(np.float32)
    direction = (direction / np.linalg.norm(direction, axis=1, keepdims=True)).astype(np.float32)
    momentum = (direction * np.float32(linefit.MOM)).astype(np.float32).astype(np.float64)
    x[:, 2:4] = -momentum[:, (0, 2)] / momentum[:, 1:2]
    return np.column_stack([x, np.linalg.norm(momentum, axis=1)])


def geometry(cond, axes=None):
    st = cond['straw']
    mid = np.stack([st['mid' + c] for c in 'xyz'], 1)
    direction = np.stack([st['dir' + c] for c in 'xyz'], 1)
    if axes is not None:
        mid, direction = (np.asarray(v, np.float64) for v in axes)
    if mid.shape != direction.shape or mid.shape != (len(st['hlen']), 3):
        raise ValueError('straw geometry shapes disagree')
    valid = np.asarray(st['hlen']) > 0
    if not np.isfinite(mid[valid]).all() or not np.isfinite(direction[valid]).all():
        raise ValueError('nonfinite straw geometry')
    if not np.allclose(np.linalg.norm(direction[valid], axis=1), 1, atol=1e-6):
        raise ValueError('straw directions must be unit vectors')
    return mid, direction, valid


def _pad(rows, values, ntrack, fill):
    order = np.argsort(rows, kind='stable')
    return tracktable._pad(rows[order], np.asarray(values)[order], ntrack, fill)


def build_candidates(sh, sd, cond, axes=None, geometry_chunk=32):
    """Measured seeds plus same-event hits and finite-straw material candidates.

    The 10 mm / 50 ns hit envelope and 5 mm material envelope are evaluated on
    the initial line, not the fitted seed. Hitless material has no covariance
    consistency cut. Each measured hit keeps its own crossing association,
    including repeated hits on a straw. No-hit straws are unique per seed.
    """
    x = seed_parameters(sd)
    ntrack = len(x)
    if geometry_chunk < 1:
        raise ValueError('geometry_chunk must be positive')
    if not np.isfinite(x).all():
        raise ValueError('nonfinite LineFinder seed')
    smid, sdir, gvalid = geometry(cond, axes)
    hmid, hdir, _ = geometry(cond)
    trseed = np.repeat(np.arange(ntrack), sd['n'])
    member = np.asarray(sd['member'], np.int64)
    nhit = len(sh['ev'])
    if np.any(sh['ev'][member] != sd['ev'][trseed]):
        raise ValueError('seed membership crosses event identity')
    nevent = int(max(np.max(sh['ev'], initial=-1), np.max(sd['ev'], initial=-1))) + 1
    if np.any(np.diff(sh['ev']) < 0):
        raise ValueError('hits must be grouped by event')
    count = np.bincount(sh['ev'], minlength=nevent)
    grouped = ak.unflatten(np.arange(nhit, dtype=np.int64), count)
    candidate = grouped[sd['ev']]
    tr = np.repeat(np.arange(ntrack), ak.to_numpy(ak.num(candidate)))
    hi = ak.to_numpy(ak.flatten(candidate))
    us = hitreco.unique_straw(sh['sid'][hi].astype(np.int64))
    st, sw, doca = poca(x[tr], hmid[us], hdir[us])
    speed = x[tr, 5] / np.hypot(x[tr, 5], linefit.MMU) * linefit.C
    seedkey = trseed * max(nhit, 1) + member
    inseed = np.isin(tr * max(nhit, 1) + hi, seedkey)
    dt = sh['time'][hi] - (x[tr, 4] + st / speed)
    extra = ((np.abs(dt) < 50.0) & (doca < 10.0) & (np.abs(sw) < cond['straw']['hlen'][us])
             & ((sh['flag'][hi] & (1 << hitreco.FLAG['dead'])) == 0))
    keep = inseed | extra
    tr, hi, inseed = tr[keep], hi[keep], inseed[keep]
    H = linefit.build_hits(tr, hi, np.zeros(len(tr)), np.ones(len(tr), bool), sh, cond, ntrack)
    H['valid'] = H['sh'] >= 0
    H['inseed'] = _pad(tr, inseed, ntrack, False)
    hus = hitreco.unique_straw(sh['sid'][hi].astype(np.int64))
    pos = np.arange(len(tr)) - np.repeat(np.cumsum(np.bincount(tr, minlength=ntrack))
                                        - np.bincount(tr, minlength=ntrack), np.bincount(tr, minlength=ntrack))
    xr, xu, xh = [tr], [hus], [pos]
    nstraw = len(smid)
    existing = tr * nstraw + hus
    for start in range(0, ntrack, geometry_chunk):
        ids = np.arange(start, min(start + geometry_chunk, ntrack))
        _, along, distance = poca(x[ids, None], smid[None], sdir[None])
        admit = gvalid[None] & (distance < 5.0) & (np.abs(along) < cond['straw']['hlen'][None])
        rr, uu = np.nonzero(admit)
        rr = ids[rr]
        new = ~np.isin(rr * nstraw + uu, existing)
        xr.append(rr[new]); xu.append(uu[new]); xh.append(np.full(int(new.sum()), -1, np.int64))
    rows, unique, assoc = np.concatenate(xr), np.concatenate(xu), np.concatenate(xh)
    sid = ((unique // (6 * 96)) << 10) | (((unique // 96) % 6) << 7) | (unique % 96)
    X = {'mid': _pad(rows, smid[unique], ntrack, 0.), 'wdir': _pad(rows, sdir[unique], ntrack, 0.),
         'valid': _pad(rows, np.ones(len(rows), bool), ntrack, False),
         'hit': _pad(rows, assoc, ntrack, -1), 'sid': _pad(rows, sid, ntrack, 0)}
    X['wdir'][..., 0] = np.where(X['valid'], X['wdir'][..., 0], 1.)
    return x, H, X


def _frozen_tree(tree):
    import jax
    leaves, definition = jax.tree_util.tree_flatten(tree)
    return definition, tuple((np.asarray(v).shape, np.asarray(v).dtype.str, np.asarray(v).tobytes()) for v in leaves)


@functools.lru_cache(maxsize=4)
def _fit_kernel(fit_track, t0_piece, crossing_rows, frozen_drift, frozen_stages, frozen_ann, config=()):
    """Reuse executables only for identical immutable calibration/schedule/ANN contents."""
    import jax
    import jax.numpy as jnp
    def thaw(frozen):
        definition, leaves = frozen
        return jax.tree_util.tree_unflatten(definition, [jnp.asarray(np.frombuffer(data, dtype).reshape(shape))
                                                        for shape, dtype, data in leaves])
    drift, stages, ann = (thaw(v) for v in (frozen_drift, frozen_stages, frozen_ann))
    def one(xx, hh, cc):
        cf, _, good = fit_track(xx, hh, cc, drift, stages, ann, fixed=None, time_model='stock')
        xf, cov, _ = t0_piece(cf)
        material = crossing_rows(cf['xe'][hh['valid'].shape[0]:], cf['Ce'][hh['valid'].shape[0]:], cc, 1., 1.)
        return {'x': xf, 'cov': cov, 'chi2': cf['chi2'], 'ndof': cf['ndof'], 'good': good,
                'status': cf['status'], 'active': cf['st']['active'], 'lr': cf['st']['lr'],
                'effect_x': cf['xe'], 'effect_cov': cf['Ce'],
                'xactive': cf['xact'], 'material': {k: material[k] for k in ('wall', 'gas', 'doca', 'docavar', 'dirdot')}}
    return jax.jit(jax.vmap(one))


def fit_candidates(x, H, X, drift, chunk=32, buckets=False):
    """Unmodified fitter, optional device-width buckets, all seeds in original order."""
    import jax
    import jax.numpy as jnp
    from . import kalman_jax as K
    from .linefit_jax import HIT_KEYS, DRIFT_KEYS
    from .autonomous_batches import batch_plan, restore_width
    plan = batch_plan(H['valid'], X['valid'], chunk, buckets)
    if not plan:
        raise ValueError('fit_candidates requires at least one seed')
    dj = {k: jnp.asarray(np.asarray(drift[k], np.float64)) for k in DRIFT_KEYS}
    stages, ann = K.stock_stages(), K.load_anns()
    config = tuple((k, v) for k, v in sorted(vars(K).items())
                   if k.isupper() and isinstance(v, (str, int, float, bool)))
    config += (('GAS_SCATTERFRAC', K.kkmaterial.GAS_SCATTERFRAC),)
    fitted = _fit_kernel(K.fit_track, K.t0_piece, K.crossing_rows,
                         _frozen_tree(dj), _frozen_tree(stages), _frozen_tree(ann), config)
    outputs, records, validity, ordering, seen = [], [], [], [], set()
    import_start = time.perf_counter()
    for ids, nh, nx in plan:
        n = len(ids)
        pad = lambda a: np.concatenate([a[ids], np.repeat(a[ids[-1:]], chunk - n, axis=0)])
        args = (jnp.asarray(pad(x)), {k: jnp.asarray(pad(H[k][:, :nh])) for k in HIT_KEYS + ('valid', 'inseed')},
                {k: jnp.asarray(pad(X[k][:, :nx])) for k in ('mid', 'wdir', 'valid', 'hit')})
        t = time.perf_counter()
        result = fitted(*args)
        jax.block_until_ready(result)
        dt = time.perf_counter() - t
        first = (nh, nx) not in seen
        seen.add((nh, nx))
        records.append({'s': dt, 'real_tracks': n, 'lanes': chunk, 'hits': nh, 'crossings': nx,
                        'first_shape_call': first})
        host = jax.tree_util.tree_map(lambda a: np.asarray(a)[:n], result)
        from .fitvalidity import fit_checks
        checks, _ = fit_checks(host.pop('effect_x'), host.pop('effect_cov'),
                              np.concatenate([H['valid'][ids, :nh], X['valid'][ids, :nx]], axis=1),
                              host['x'], host['cov'], host['chi2'], host['ndof'], host['good'], np.zeros(n, bool))
        for k in ('active', 'lr'):
            host[k] = restore_width(host[k], H['valid'].shape[1])
        host['xactive'] = restore_width(host['xactive'], X['valid'].shape[1])
        host['material'] = {k: restore_width(v, X['valid'].shape[1]) for k, v in host['material'].items()}
        validity.append(checks)
        ordering.append(ids)
        outputs.append(host)
    inv = np.argsort(np.concatenate(ordering))
    result = jax.tree_util.tree_map(lambda *a: np.concatenate(a)[inv], *outputs)
    result['numerical_checks'] = {k: np.concatenate([v[k] for v in validity])[inv] for k in validity[0]}
    warm = [r for r in records if not r['first_shape_call']]
    return result, {'backend': jax.default_backend(), 'devices': [str(d) for d in jax.devices()],
                    'effective_fitter_config': dict(config),
                    'chunk': chunk, 'buckets': bool(buckets), 'batch_s': [r['s'] for r in records],
                    'batches': records, 'first_shape_calls_may_compile': True,
                    'batch_timer_scope': 'device call and synchronization; excludes packing/transfers/conversion',
                    'fit_loop_wall_s': time.perf_counter() - import_start,
                    'executed_lanes': len(records) * chunk, 'real_tracks': len(x),
                    'executed_effect_slots': sum(r['lanes'] * (r['hits'] + r['crossings']) for r in records),
                    'full_width_effect_slots': len(records) * chunk * (H['valid'].shape[1] + X['valid'].shape[1]),
                    'warm_batch_s': sum(r['s'] for r in warm), 'warm_tracks': sum(r['real_tracks'] for r in warm),
                    'warm_executed_lanes': len(warm) * chunk,
                    'host_candidate_bytes': sum(v.nbytes for v in list(H.values()) + list(X.values())),
                    'host_storage': 'full file-width padded arrays; device buckets do not bound host storage',
                    'hit_width': H['valid'].shape[1], 'crossing_width': X['valid'].shape[1]}


def reconstruct(path, cond, drift, settings='mc', axes=None, chunk=32, buckets=False):
    """One digi-only file to an all-event track table, with failed seeds retained."""
    from . import linefind_jax
    from ..kitpoint import validate_drift
    validate_drift(drift)
    t = time.perf_counter()
    d = hitreco.read_digis([path])
    keys = d['evkey']
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError('duplicate full event identities')
    sh = hitreco.straw_hits(d, cond, settings)
    tc = linefind.time_clusters(sh)
    sd = linefind_jax.line_seeds(sh, tc, cond)
    x0, H, X = build_candidates(sh, sd, cond, axes)
    prep = time.perf_counter() - t
    T = len(x0)
    if T:
        f, timing = fit_candidates(x0, H, X, drift, chunk, buckets=buckets)
    else:
        f = {'x': np.empty((0, 6)), 'cov': np.empty((0, 6, 6)), 'chi2': np.empty(0),
             'ndof': np.empty(0, np.int64), 'status': np.empty(0, np.int32), 'good': np.empty(0, bool),
             'active': H['valid'], 'lr': H['lr'], 'xactive': X['valid'],
             'material': {k: np.empty(X['valid'].shape) for k in ('wall', 'gas', 'doca', 'docavar', 'dirdot')}}
        timing = {'batch_s': [], 'fit_skipped_no_seeds': True}
    hits = {k: H[k] for k in tracktable.HIT_KEYS}
    hits.update(hvalid=H['valid'], active=f['active'], lr=f['lr'],
                state=np.where(H['valid'], tracktable._state(f['active'], f['lr']), -2).astype(np.int32))
    for k in tracktable.SH_EXTRA:
        values = np.append(np.asarray(sh[k]), 0)
        hits[k] = np.where(H['valid'], values[np.where(H['valid'], H['sh'], len(sh[k]))], 0)
    mat = f['material']
    xings = {'sid': X['sid'], 'hit': X['hit'], 'xvalid': X['valid'], 'active': f['xactive'] & X['valid'], **mat,
             'wire': np.full(X['valid'].shape, np.nan),
             'radlen': mat['wall'] / tracktable.X0_WALL + mat['gas'] / tracktable.X0_GAS}
    events = {k: keys[:, i].astype(np.int64) for i, k in enumerate(('run', 'subrun', 'event'))}
    events['pbt'] = d['pbt'].astype(np.float32)
    nseed = np.bincount(sd['ev'], minlength=len(keys))
    ordinal = np.arange(T) - np.repeat(np.cumsum(nseed) - nseed, nseed)
    meta = {'version': tracktable.VERSION, 'fit_type': 'F1', 'fit': 'autonomous fixed-envelope F1 prototype',
            'stock_conditioned': False, 'stock_exact': False, 'source': str(path), 'settings': settings,
            'candidate_model': 'LineFinder fixed envelope, not fitted-seed extension',
            'geometry': 'aligned straw axes' if axes is not None else 'wire-axis approximation',
            'time_model': 'stock-inspired owned pieces, not native-exact',
            'n_events': len(keys), 'n_hits': len(sh['ev']), 'n_clusters': len(tc['n']),
            'n_seeds': T, 'n_good': int(f['good'].sum()), 'events_without_seeds': int((nseed == 0).sum()),
            'events_with_multiple_seeds': int((nseed > 1).sum()), 'preparation_s': prep, 'timing': timing,
            'env_f1': {k: v for k, v in os.environ.items() if k.startswith('F1_')},
            'acceptance': 'unmodified fitter good flag; numerical checks do not redefine acceptance',
            'limits': ['no native seed covariance/range certificate', 'no fitted-seed candidate continuation',
                       'no hitless-crossing covariance-consistency cut', 'writer uses t0 line at surfaces']}
    checks = f.get('numerical_checks', {})
    meta['numerical_validity'] = {'diagnostic_only': True,
        'scope': 'final returned effect and t0 states, not intermediate histories',
        'evaluated_tracks': T if checks else 0,
        'failures_all': {k: int((~v).sum()) for k, v in checks.items()},
        'failures_accepted': {k: int((~v & f['good']).sum()) for k, v in checks.items()}}
    table = {'events': events, 'ev': sd['ev'].astype(np.int64), 'seed_index': ordinal,
             **{'numerical_' + k: v for k, v in checks.items()},
             'x': f['x'], 'cov': f['cov'], 'chi2': f['chi2'], 'ndof': f['ndof'],
             'status': np.where(f['status'] == 0, 1, np.where(f['status'] == 1, 2, -1)).astype(np.int32),
             'goodfit': f['good'].astype(np.int32), 'momfit': np.ones(T, np.int32),
             **hits, 'xings': xings, 'drift': drift, 'meta': meta}
    return table


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ann_identity():
    from . import driftann
    names = ('TrainBkgSeed.dat', 'TrainBkgFinal.dat', 'TrainSign_Stage0.dat', 'TrainCluster_Stage0.dat',
             'TrainSign_Stage1.dat', 'TrainCluster_Stage1.dat')
    paths = [driftann.find(name) for name in names]
    return {str(Path(p).resolve()): digest(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('digis')
    parser.add_argument('--conditions', required=True)
    parser.add_argument('--drift-source', required=True, help='reads conditions tree only')
    parser.add_argument('--straw-axis', help='TrkAlignStraw CSV; drift-source must include wire-end geometry')
    parser.add_argument('--settings', choices=('mc', 'data'), default='mc')
    parser.add_argument('--chunk', type=int, default=32)
    parser.add_argument('--buckets', action='store_true', help='opt-in device candidate-width buckets')
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    from . import ntuple
    from .cli import _drift_table
    from .vr_f1 import _straw_axes
    invocation_start = time.perf_counter()
    paths = [args.digis, args.conditions, args.drift_source] + ([args.straw_axis] if args.straw_axis else [])
    before = {p: digest(p) for p in paths}
    source = {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
    anns = ann_identity()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    cond = hitreco.load_conditions(args.conditions)
    axes = _straw_axes(args.drift_source, cond, args.straw_axis) if args.straw_axis else None
    start = time.perf_counter()
    table = reconstruct(args.digis, cond, _drift_table(args.drift_source), args.settings, axes, args.chunk, args.buckets)
    if before != {p: digest(p) for p in paths}:
        raise RuntimeError('input changed during reconstruction')
    if anns != ann_identity() or source != {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}:
        raise RuntimeError('ANN weights or reconstruction source changed during run')
    import jax
    import uproot
    table['meta'].update(input_sha256=before, source_sha256=source, ann_sha256=anns,
                         runtime_versions={'numpy': np.__version__, 'jax': jax.__version__, 'uproot': uproot.__version__})
    tracktable.save(table, out / 'tracks.npz')
    write_start = time.perf_counter()
    prov = ntuple.write(table, out / 'EventNtuple.root')
    from ..kitpoint import inspect_written
    persisted = inspect_written(out / 'EventNtuple.root',
        np.column_stack([table['events'][k] for k in ('run', 'subrun', 'event')]), int(table['goodfit'].sum()))
    report = {**table['meta'], 'writer': prov, 'persisted': persisted, 'writer_s': time.perf_counter() - write_start,
              'wall_s': time.perf_counter() - start,
              'wall_scope': 'reconstruct through product writing; excludes initial calibration/hashing',
              'invocation_work_s': time.perf_counter() - invocation_start}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'events': len(table['events']['event']), 'seeds': len(table['ev']),
                      'good': int(table['goodfit'].sum()), 'outdir': str(out)}))


if __name__ == '__main__':
    main()
