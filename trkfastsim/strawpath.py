"""Ordered straight-ray straw material intervals in the stored cylinder representation."""
from dataclasses import replace

import numpy as np
import pandas as pd

from mu2efastsim import materials
from . import crossings
from .materialpath import _partition, _rays
from .steps import STRAW_GAS


def straw_intervals(geom, origin, direction, length):
    """Partition bounded rays into concentric wall/metal/gas/wire/plate and vacuum.

    Uses the existing finite-cylinder cull with each stored radius. All layers use
    the stored gas half length; no end plugs, support or omitted GDML solids are
    inferred. wire_r denotes the outer plated radius, as in production transport.
    Positive overlaps between different straws are refused, even for equal media.
    This is an opt-in stored-geometry diagnostic, not a replacement for make_steps.
    """
    origin, direction, length = _rays(origin, direction, length)
    gas = float(geom.radius)
    outer = float(geom.wall_rmax if geom.wall_rmax is not None else gas + .015)
    metal = outer - float(geom.metal_t)
    wire = float(geom.wire_r)
    core = wire - float(geom.wire_plate_t)
    if (not np.isfinite([gas, outer, metal, wire, core]).all()
            or not (0 <= core <= wire < gas <= metal <= outer)):
        raise ValueError("ordered straw radii require 0 <= core <= wire < gas <= metal <= wall")
    radii = sorted(set([outer, metal, gas, wire, core]) - {0.})
    hits = {}
    for radius in radii:
        rows = crossings.straw_crossings(origin, direction, replace(geom, radius=radius))
        lane, straw = rows[:, :2].astype(np.int64).T
        end = np.minimum(rows[:, 3], length[lane])
        keep = end > rows[:, 2]
        hits[radius] = pd.DataFrame(dict(lane=lane[keep], straw=straw[keep],
                                        start=rows[keep, 2], end=end[keep]))
    solids = []
    layers = [(outer, metal, materials.ALUMINIUM_6061.name),
              (metal, gas, materials.MYLAR.name), (gas, wire, STRAW_GAS.name),
              (wire, core, materials.GOLD.name), (core, 0., materials.TUNGSTEN.name)]
    for ro, ri, material in layers:
        if ro == ri or ro == 0:
            continue
        outside = hits[ro]
        if ri == 0:
            spans = outside.rename(columns={"start": "s0_mm", "end": "s1_mm"})
        else:
            inner = hits[ri].rename(columns={"start": "inner0", "end": "inner1"})
            joined = outside.merge(inner, on=["lane", "straw"], how="left", validate="one_to_one")
            a = joined.assign(s0_mm=joined.start, s1_mm=joined.inner0.fillna(joined.end))
            b = joined.assign(s0_mm=joined.inner1.fillna(joined.end), s1_mm=joined.end)
            spans = pd.concat([a, b], ignore_index=True)
        spans = spans[spans.s1_mm > spans.s0_mm]
        solids.append(spans[["lane", "s0_mm", "s1_mm"]].assign(material=material))
    return _partition(solids, length)


def combined_intervals(shells, geom, origin, direction, length):
    """Combine stored support and straw partitions, refusing positive overlaps."""
    from .interiorpath import interior_intervals
    origin, direction, length = _rays(origin, direction, length)
    frames = [interior_intervals(shells, origin, direction, length),
              straw_intervals(geom, origin, direction, length)]
    return _partition([f[f.material != "vacuum"] for f in frames], length)
