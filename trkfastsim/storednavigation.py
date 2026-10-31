"""Explicit same-medium union of stored support, with strict straw ownership."""
import numpy as np
import pandas as pd

from . import materialpath, strawpath

POLICY = "stored-same-medium-union-v1"


def union_support(hits, length):
    """Take exact same-medium set unions; reject every cross-medium overlap.

    No endpoint moves and no epsilon is used. This treats stored support as a
    material field, not distinct volume identities. It cannot certify GDML leaf
    ownership or restore omitted solids. Duplicate same-medium columns are charged
    once under this explicit policy, including genuine same-medium overlaps.
    """
    if hits.empty:
        return materialpath._partition([], np.asarray(length))
    h = hits.sort_values(["lane", "material", "s0_mm", "s1_mm"], kind="stable").copy()
    group = h.groupby(["lane", "material"], sort=False)
    maximum = group.s1_mm.cummax()
    previous = maximum.groupby([h.lane, h.material], sort=False).shift()
    first = previous.isna() | (h.s0_mm > previous)
    h["block"] = first.cumsum().to_numpy()
    merged = h.groupby(["lane", "material", "block"], sort=False, as_index=False).agg(
        s0_mm=("s0_mm", "min"), s1_mm=("s1_mm", "max"))
    return materialpath._partition([merged[["lane", "s0_mm", "s1_mm", "material"]]], np.asarray(length))


def support_intervals(shells, origin, direction, length, *, endpoint_budget_mm=None, endpoint_evidence=None,
                      leading_resolution_mm=None):
    """Stored tube material field plus strict signed-convex calorimeter partition."""
    from .convexpath import convex_intervals
    coax = shells.coax if hasattr(shells, "coax") else shells
    hits = materialpath.tube_hits(coax, origin, direction, length)
    if endpoint_budget_mm is not None:
        from .boundarypolicy import coalesce
        hits, evidence = coalesce(hits.assign(sign=1), length, max_shift_mm=endpoint_budget_mm,
                                  leading_resolution_mm=leading_resolution_mm)
        if endpoint_evidence is not None:
            endpoint_evidence(evidence)
        hits = hits.drop(columns="sign")
    calo = getattr(shells, "calo", None)
    if calo is not None:
        convex = convex_intervals(calo, origin, direction, length, endpoint_budget_mm=endpoint_budget_mm,
                                  endpoint_evidence=endpoint_evidence, leading_resolution_mm=leading_resolution_mm)
        hits = pd.concat([hits, convex[convex.material != "vacuum"]], ignore_index=True)
    return union_support(hits, length)


def intervals(shells, geom, origin, direction, length, *, endpoint_budget_mm=None, endpoint_evidence=None,
              leading_resolution_mm=None):
    """Return a complete bounded partition with support/thin/vacuum provenance."""
    origin, direction, length = materialpath._rays(origin, direction, length)
    support = support_intervals(shells, origin, direction, length, endpoint_budget_mm=endpoint_budget_mm,
                                endpoint_evidence=endpoint_evidence, leading_resolution_mm=leading_resolution_mm)
    thin = strawpath.straw_intervals(geom, origin, direction, length)
    support = support[support.material != "vacuum"].assign(kind="support")
    thin = thin[thin.material != "vacuum"].assign(kind="thin")
    solids = pd.concat([support, thin], ignore_index=True)
    partition = materialpath._partition([solids[["lane", "s0_mm", "s1_mm", "material"]]], length)
    return partition.merge(solids, on=["lane", "s0_mm", "s1_mm", "material"], how="left",
                           validate="one_to_one").fillna({"kind": "vacuum"})
