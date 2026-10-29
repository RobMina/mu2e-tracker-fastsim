"""Opt-in candidate-width batches for F1, without changing candidate membership."""
import numpy as np


def batch_plan(hvalid, xvalid, chunk, buckets=False):
    """Return original row indices and represented prefix widths for each batch.

    Prefix width uses the last valid slot, not a count: a masked interior hole
    must not truncate a later candidate or change crossing-to-hit identities.
    Host tables remain full-width; this bounds only device padding and work.
    """
    if isinstance(chunk, bool) or not isinstance(chunk, (int, np.integer)) or chunk < 1:
        raise ValueError('chunk must be a positive integer')
    hvalid, xvalid = np.asarray(hvalid, bool), np.asarray(xvalid, bool)
    if hvalid.ndim != 2 or xvalid.ndim != 2 or len(hvalid) != len(xvalid):
        raise ValueError('hit/crossing masks must be [track,slot] with equal track count')
    if min(hvalid.shape[1], xvalid.shape[1]) < 1:
        raise ValueError('at least one padding slot is required')
    n = len(hvalid)
    if not buckets:
        return [(np.arange(i, min(i + chunk, n)), hvalid.shape[1], xvalid.shape[1]) for i in range(0, n, chunk)]
    widths = []
    for valid in (hvalid, xvalid):
        extent = np.max(np.where(valid, np.arange(valid.shape[1])[None] + 1, 0), axis=1)
        rounded = np.maximum(16, 2 ** np.ceil(np.log2(np.maximum(extent, 1))).astype(np.int64))
        widths.append(np.minimum(valid.shape[1], rounded))
    if not n:
        return []
    order = np.lexsort((np.arange(n), widths[1], widths[0]))
    sizes = np.column_stack(widths)[order]
    begin = np.flatnonzero(np.r_[True, np.any(sizes[1:] != sizes[:-1], axis=1)])
    end = np.r_[begin[1:], n]
    return [(order[j:min(j+chunk, stop)], int(sizes[start, 0]), int(sizes[start, 1]))
            for start, stop in zip(begin, end) for j in range(start, stop, chunk)]


def restore_width(array, width, fill=0):
    """Restore a batched slot axis for the legacy full-width table interface."""
    array = np.asarray(array)
    if array.shape[1] > width:
        raise ValueError('cannot restore to a shorter width')
    out = np.full((len(array), width) + array.shape[2:], fill, dtype=array.dtype)
    out[:, :array.shape[1]] = array
    return out
