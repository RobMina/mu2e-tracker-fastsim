"""Named edits of the stage C world (world.load pieces with names), for geometry sensitivity runs without Geant4.

Each modification takes (pieces, names) and returns new lists. Positions are relative to the tracker axis point
geometry.DETECTOR_ORIGIN_EXTRACTED; the hall walls run along z, so the edits are x slabs.

  issue2017_a1  Mu2e/Offline#2017 A1: the DS hatch ledges project 1'-2" past the upper wall face (drawing SC-4), not
                9": each ledge (dsAreaHatchN/SLedge, y 2717..3632 in the v04 world) gets 127 mm more concrete on its
                inner face (|x| 3454 -> 3327 mm).
  issue2017_b1  Mu2e/Offline#2017 B1, as the issue states it: 5" (127 mm) of the W2 walls' soil face is concrete in
                Offline where the drawing has backfill. In the v04 world the upper wall at the tracker (dsAreaUpper,
                |x| 3810..4420 mm) is 610 mm thick, which matches neither the issue's 889 mm nor the drawn 762 mm, so
                this applies the issue's direction to the outer 127 mm of that wall (concrete -> MBOverburden).
  issue2017     both.
"""
import numpy as np

from . import geometry

INCH = 25.4


def _slab(P, x_lo, x_hi):
    """Planes of a convex piece clipped to x_lo <= x <= x_hi (global x)."""
    return np.vstack([P, [[1.0, 0.0, 0.0, x_hi], [-1.0, 0.0, 0.0, -x_lo]]])


def _box(lo, hi):
    return np.array([[1.0, 0, 0, hi[0]], [-1.0, 0, 0, -lo[0]], [0, 1.0, 0, hi[1]], [0, -1.0, 0, -lo[1]],
                     [0, 0, 1.0, hi[2]], [0, 0, -1.0, -lo[2]]])


def issue2017_a1(pieces, names, ext=5 * INCH):
    """Extend each hatch ledge inwards by `ext` (default 127 mm) over its full height and length."""
    x0 = geometry.DETECTOR_ORIGIN_EXTRACTED[0]
    out_p, out_n = list(pieces), list(names)
    for (m, P, lo, hi), n in zip(pieces, names):
        if n not in ("dsAreaHatchNLedge", "dsAreaHatchSLedge"):
            continue
        north = lo[0] - x0 > 0
        if north:
            a, b = lo[0] - ext, lo[0]
        else:
            a, b = hi[0], hi[0] + ext
        nlo, nhi = np.array([a, lo[1], lo[2]]), np.array([b, hi[1], hi[2]])
        out_p.append((m, _box(nlo, nhi), nlo, nhi))
        out_n.append(n + "_ext2017")
    return out_p, out_n


def issue2017_b1(pieces, names, t=5 * INCH, band=(3700.0, 4500.0)):
    """Outer `t` of the dsAreaUpper walls (|x - x_axis| in `band`) from concrete to MBOverburden."""
    x0 = geometry.DETECTOR_ORIGIN_EXTRACTED[0]
    out_p, out_n = [], []
    for (m, P, lo, hi), n in zip(pieces, names):
        rl, rh = lo[0] - x0, hi[0] - x0
        if n == "dsAreaUpper" and m == "CONCRETE_MARS" and (band[0] <= rl and rh <= band[1] or -band[1] <= rl and rh <= -band[0]):
            if rl > 0:      # north wall: soil face at hi x
                cut = hi[0] - t
                keep, swap = _slab(P, -1e9, cut), _slab(P, cut, 1e9)
            else:           # south wall: soil face at lo x
                cut = lo[0] + t
                keep, swap = _slab(P, cut, 1e9), _slab(P, -1e9, cut)
            out_p += [(m, keep, lo, hi), ("MBOverburden", swap, lo, hi)]
            out_n += [n, n + "_soil2017"]
        else:
            out_p.append((m, P, lo, hi))
            out_n.append(n)
    return out_p, out_n


MODS = {"issue2017_a1": [issue2017_a1], "issue2017_b1": [issue2017_b1], "issue2017": [issue2017_a1, issue2017_b1]}


def apply(name, pieces, names):
    """Apply the named modification(s); returns (pieces, names)."""
    if name not in MODS:
        raise KeyError(f"unknown world modification {name!r}; known: {sorted(MODS)}")
    for fn in MODS[name]:
        pieces, names = fn(pieces, names)
    return pieces, names
