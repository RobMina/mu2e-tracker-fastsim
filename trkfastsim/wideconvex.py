"""Wider stored-plane intersection arithmetic; same binary64 planes/model and broad phase."""
import numpy as np

from . import convexpath

POLICY = "stored-plane-wide-intersections-v1"


def hits(pieces, origin, direction, lengths):
    """Promote stored planes/rays before dot products; export bounded binary64 parameters.

    This is a separate numerical stored-source arm, not a geometry enclosure.
    Existing explicit endpoint policy still decides any overlap; none is hidden.
    Geometry coefficients/boxes, candidate enumeration and physical labels remain
    unchanged. Near-parallel/face errors in the stored model are not corrected.
    """
    wide = np.longdouble
    if np.finfo(wide).nmant <= np.finfo(float).nmant:
        raise ValueError("wide stored planes require genuinely wider mantissa")
    for lane, piece in convexpath._pairs(pieces, origin, direction, lengths):
        chunk = max(1, convexpath._PAIR_CHUNK // pieces.P.shape[1])
        for start in range(0, len(lane), chunk):
            it, ip = lane[start:start+chunk], piece[start:start+chunk]
            planes = np.asarray(pieces.P[ip], wide)
            o, d = np.asarray(origin[it], wide), np.asarray(direction[it], wide)
            den = np.sum(d[:, None, :] * planes[:, :, :3], axis=2)
            num = planes[:, :, 3] - np.sum(o[:, None, :] * planes[:, :, :3], axis=2)
            with np.errstate(divide="ignore", invalid="ignore"):
                cross = num / den
            enter = np.maximum(wide(0), np.where(den < 0, cross, -np.inf).max(axis=1))
            leave = np.minimum(np.asarray(lengths[it], wide), np.where(den > 0, cross, np.inf).min(axis=1))
            keep = ~((den == 0) & (num < 0)).any(axis=1) & (leave > enter)
            a, b = enter[keep].astype(float), leave[keep].astype(float)
            positive = b > a
            if not positive.all():
                raise ValueError("positive wider stored interval collapses on binary64 lattice")
            yield it[keep], ip[keep], a, b


def local_hits(pieces, origin, direction, lengths, terms):
    """Intersect unflattened stored local planes with promoted stored R,t; separate numerical-model arm."""
    wide = np.longdouble
    if np.finfo(wide).nmant <= np.finfo(float).nmant:
        raise ValueError("local stored planes require genuinely wider mantissa")
    for lane, piece in convexpath._pairs(pieces, origin, direction, lengths):
        chunk = max(1, convexpath._PAIR_CHUNK // pieces.P.shape[1])
        for start in range(0, len(lane), chunk):
            it, ip = lane[start:start+chunk], piece[start:start+chunk]
            if not len(it):
                continue
            codes = np.asarray(pieces.mats, object)[pieces.mat[ip]].astype(int)
            planes = np.asarray([np.pad(terms[code][0], ((0, pieces.P.shape[1] - len(terms[code][0])), (0, 0)), mode="edge")
                                 for code in codes], wide)
            rotation = np.asarray([terms[code][1] for code in codes], wide)
            position = np.asarray([terms[code][2] for code in codes], wide)
            o = np.sum((np.asarray(origin[it], wide) - position)[:, :, None] * rotation, axis=1)
            d = np.sum(np.asarray(direction[it], wide)[:, :, None] * rotation, axis=1)
            den = np.sum(d[:, None, :] * planes[:, :, :3], axis=2)
            num = planes[:, :, 3] - np.sum(o[:, None, :] * planes[:, :, :3], axis=2)
            with np.errstate(divide="ignore", invalid="ignore"):
                cross = num / den
            enter = np.maximum(wide(0), np.where(den < 0, cross, -np.inf).max(axis=1))
            leave = np.minimum(np.asarray(lengths[it], wide), np.where(den > 0, cross, np.inf).min(axis=1))
            keep = ~((den == 0) & (num < 0)).any(axis=1) & (leave > enter)
            a, b = enter[keep].astype(float), leave[keep].astype(float)
            if (b <= a).any():
                raise ValueError("positive local-plane interval collapses on binary64 lattice")
            yield it[keep], ip[keep], a, b
