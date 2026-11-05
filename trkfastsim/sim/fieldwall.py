"""Finite outer-minus-inner straw-wall intervals for opt-in field transport.

This material-only ledger does not create gas steps. The homogeneous wall material
split and all gas/wire ownership remain the transport caller's responsibility.
"""
import dataclasses

import numpy as np


def _cylinder_interval(x, c, axis, half, radius):
    """Full-line finite cylinder intervals; invalid intersections have lo >= hi."""
    wa = np.einsum("ij,ij->i", x, axis)
    da = np.einsum("ij,ij->i", c, axis)
    v = x - wa[:, None] * axis
    u = c - da[:, None] * axis
    aa = np.einsum("ij,ij->i", u, u)
    bb = np.einsum("ij,ij->i", u, v)
    cc = np.einsum("ij,ij->i", v, v) - radius ** 2
    parallel = aa == 0
    cross = np.cross(u, v)
    disc = aa * radius ** 2 - np.einsum("ij,ij->i", cross, cross)
    valid = np.where(parallel, cc <= 0, disc > 0)
    root = np.sqrt(np.maximum(disc, 0.))
    den = np.where(parallel, 1., aa)
    q = -bb - np.copysign(root, bb)
    r0, r1 = q / den, cc / np.where(q == 0, 1., q)
    lo = np.where(parallel, -np.inf, np.minimum(r0, r1))
    hi = np.where(parallel, np.inf, np.maximum(r0, r1))
    stationary = da == 0
    cap_den = np.where(stationary, 1., da)
    z0, z1 = (-half - wa) / cap_den, (half - wa) / cap_den
    lo = np.maximum(lo, np.where(stationary, -np.inf, np.minimum(z0, z1)))
    hi = np.minimum(hi, np.where(stationary, np.inf, np.maximum(z0, z1)))
    valid &= ~stationary | (np.abs(wa) <= half)
    return np.where(valid, lo, np.inf), np.where(valid, hi, -np.inf)


def interval_columns(rows, length):
    """Per-lane integrals of 1, (L-s), (L-s)^2 over positive wall intervals, in mm powers."""
    out = tuple(np.zeros(len(length)) for _ in range(3))
    if len(rows):
        lane = rows[:, 0].astype(np.int64)
        a, b = rows[:, 2], rows[:, 3]
        ln = b - a
        h0, h1 = length[lane] - a, length[lane] - b
        np.add.at(out[0], lane, ln)
        np.add.at(out[1], lane, ln * (h0 + h1) * .5)
        np.add.at(out[2], lane, ln * (h0 * h0 + h0 * h1 + h1 * h1) / 3.)
    return out


class WallIndex:
    """Expanded outer-radius panel search followed by finite inner-cylinder subtraction."""

    def __init__(self, geom, r_out, index_factory):
        if not np.isfinite(r_out) or not 0 < geom.radius <= r_out:
            raise ValueError("invalid inner/outer wall radius")
        self.geom = geom
        self.outer = dataclasses.replace(geom, radius=float(r_out))
        self.index = index_factory(self.outer)

    def _candidates(self, x, c, length):
        """Box-only panel candidates, without the legacy gas search's near-parallel cylinder cut."""
        index = self.index
        z0 = np.minimum(x[:, 2], x[:, 2] + c[:, 2] * length)
        z1 = np.maximum(x[:, 2], x[:, 2] + c[:, 2] * length)
        a = np.searchsorted(index.lo_z, z0 - index.width, "left")
        b = np.searchsorted(index.lo_z, z1, "right")
        count = np.maximum(b - a, 0)
        lane = np.repeat(np.arange(len(x)), count)
        if not len(lane):
            return np.zeros((0, 2), dtype=np.int64)
        offset = np.arange(len(lane)) - np.repeat(np.cumsum(count) - count, count)
        panel = index.order[np.repeat(a, count) + offset]
        origin, direction = x[lane], c[lane]
        stationary = direction == 0
        den = np.where(stationary, 1., direction)
        s0, s1 = (index.lo[panel] - origin) / den, (index.hi[panel] - origin) / den
        lo = np.where(stationary, -np.inf, np.minimum(s0, s1)).max(1)
        hi = np.where(stationary, np.inf, np.maximum(s0, s1)).min(1)
        inside = np.all(~stationary | ((origin >= index.lo[panel]) & (origin <= index.hi[panel])), axis=1)
        keep = inside & (np.minimum(hi, length[lane]) >= np.maximum(lo, 0.))
        lane, panel = lane[keep], panel[keep]
        ns = index.straws_of.shape[1]
        return np.column_stack((np.repeat(lane, ns), index.straws_of[panel].ravel()))

    def intervals(self, x, c, length, batch=65536):
        """Positive rows (lane, straw, s_in, s_out) clipped to [0,L].

        batch bounds cylinder-subtraction work only; the outer candidate search
        expands all overlapping panel/straw candidates and is not memory-bounded here.
        """
        if isinstance(batch, bool) or not isinstance(batch, (int, np.integer)) or batch <= 0:
            raise ValueError("wall candidate batch must be a positive integer")
        if self.outer.radius == self.geom.radius:
            return np.zeros((0, 4))
        candidates = self._candidates(x, c, length)
        pieces = []
        for start in range(0, len(candidates), batch):
            rows = candidates[start:start + batch]
            lane, straw = rows[:, 0].astype(np.int64), rows[:, 1].astype(np.int64)
            args = (x[lane] - self.geom.mid[straw], c[lane], self.geom.dir[straw], self.geom.halflen[straw])
            o0, o1 = _cylinder_interval(*args, self.outer.radius)
            o0, o1 = np.maximum(o0, 0.), np.minimum(o1, length[lane])
            g0, g1 = _cylinder_interval(*args, self.geom.radius)
            g0, g1 = np.maximum(g0, o0), np.minimum(g1, o1)
            gas = g1 > g0
            g0, g1 = np.where(gas, g0, o1), np.where(gas, g1, o1)
            left = np.column_stack((lane, straw, o0, g0))
            right = np.column_stack((lane, straw, g1, o1))
            pieces.extend((left[left[:, 3] > left[:, 2]], right[right[:, 3] > right[:, 2]]))
        if not pieces:
            return np.zeros((0, 4))
        out = np.concatenate(pieces)
        return out[np.lexsort((out[:, 1], out[:, 2], out[:, 0]))]

    def columns(self, x, c, length):
        """Material length and exact lever moments independent of gas-step creation."""
        return interval_columns(self.intervals(x, c, length), length)
