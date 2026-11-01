"""Opt-in donor-topology anchor mapping for secondary-placement diagnostics."""
import numpy as np
import pandas as pd

TOPOLOGY = ['source_anchor_plane', 'source_plane_min', 'source_plane_max',
            'source_exterior', 'source_plane_rank', 'source_plane_count']


def validate(st):
    missing = set(TOPOLOGY + ['dpl']) - set(st.columns)
    if missing:
        raise ValueError(f'topology anchor requires annotated library: missing {sorted(missing)}')
    a = st[TOPOLOGY + ['dpl']].to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError('noninteger or missing donor topology')
    lo, hi = st.source_plane_min, st.source_plane_max
    anchor = st.source_anchor_plane
    target = anchor + st.dpl
    exterior = np.where(target < lo, -1, np.where(target > hi, 1, 0))
    if ((lo < 0) | (hi >= 36) | (lo > anchor) | (anchor > hi) | (target < 0) | (target >= 36)).any():
        raise ValueError('invalid donor plane bounds')
    if not np.array_equal(exterior, st.source_exterior):
        raise ValueError('inconsistent donor exterior classification')
    if ((st.source_plane_count <= 0) | (st.source_plane_rank < 0) |
            (st.source_plane_rank >= st.source_plane_count)).any():
        raise ValueError('invalid donor within-plane rank')


def topology_anchor(mu, keys, first, st):
    """Choose recipient plane first, then within-plane rank; preserve source gaps on self-repasting."""
    validate(st)
    if not len(st):
        return np.zeros(0, dtype=np.int64)
    if 'i' not in st or not np.isfinite(st.i).all() or (st.i != np.floor(st.i)).any():
        raise ValueError('invalid recipient index')
    if ((st.i < 0) | (st.i >= len(first))).any():
        raise ValueError('invalid recipient index')
    recipient = mu[['plane', 'k']].to_numpy()
    if (not np.isfinite(recipient).all() or np.any(recipient != np.floor(recipient))
            or np.any(recipient < 0) or (mu.plane >= 36).any()):
        raise ValueError('invalid recipient plane or crossing rank')
    if first.duplicated(keys).any() or mu.duplicated(keys + ['k']).any():
        raise ValueError('duplicate recipient identity or crossing rank')
    bounds = mu.groupby(keys).plane.agg(['min', 'max']).reindex(pd.MultiIndex.from_frame(first[keys]))
    if bounds.isna().any().any():
        raise ValueError('missing recipient crossings')
    i = st.i.to_numpy(dtype=np.int64)
    lo, hi = bounds['min'].to_numpy(dtype=np.int64)[i], bounds['max'].to_numpy(dtype=np.int64)[i]
    denominator = np.maximum(st.source_plane_max - st.source_plane_min, 1).to_numpy(dtype=np.int64)
    offset = (st.source_anchor_plane - st.source_plane_min).to_numpy(dtype=np.int64)
    numerator = lo * denominator + offset * (hi - lo)
    numerator = np.where(st.source_exterior < 0, lo * denominator,
                         np.where(st.source_exterior > 0, hi * denominator, numerator))
    query = pd.DataFrame({**{k: first[k].to_numpy()[i] for k in keys},
                          'row': np.arange(len(st)), 'numerator': numerator, 'denominator': denominator})
    planes = mu[keys + ['plane']].drop_duplicates()
    candidate = query.merge(planes, on=keys, validate='many_to_many')
    candidate['distance'] = (candidate.plane * candidate.denominator - candidate.numerator).abs()
    chosen = candidate.sort_values(['row', 'distance', 'plane'], kind='stable').drop_duplicates('row').sort_values('row')
    crossings = mu[keys + ['plane', 'k']].sort_values(keys + ['plane', 'k']).copy()
    crossings['rank'] = crossings.groupby(keys + ['plane']).cumcount()
    crossings['count'] = crossings.groupby(keys + ['plane']).k.transform('size')
    candidates = chosen.merge(crossings, on=keys + ['plane'], validate='many_to_many')
    rows = candidates.row.to_numpy(dtype=np.int64)
    desired_rank = (st.source_plane_rank.to_numpy(dtype=np.int64)[rows] * candidates['count'].to_numpy() //
                    st.source_plane_count.to_numpy(dtype=np.int64)[rows])
    selected = candidates[candidates['rank'].to_numpy() == desired_rank].sort_values('row')
    if len(selected) != len(st) or selected.row.duplicated().any():
        raise ValueError('recipient topology mapping failed')
    return selected.k.to_numpy(dtype=np.int64)
