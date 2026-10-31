"""Explicit straight stored-material candidate; never called by production make_steps."""
from . import geometry, historystate, historysurfaces, orderedtransport, strawpath

MODE = "ordered-stored-straw-mean-v1"


def transport(shells, geom, identity, origin_mm, direction, kinetic_MeV, length_mm, metadata, *,
              mode, max_step_mm=1., fractional_loss=.05, stop_ke_MeV=1., max_steps=10000):
    """Navigate supplied straight rays and evolve mean KE/time in actual interval order.

    Explicit opt-in only. The stored support model may omit GDML solids and fail
    strict overlap checks. No world, scattering or exit extension is manufactured.
    Positive material transfers have unmodeled deposited/escaping partitions; gas
    response and secondaries are not sampled. This is not stochastic production
    transport or the expectation of a stochastic range law. The cutoff is an
    absorbing model stop retaining residual KE, not physical muon decay/capture.
    Returned history is in detector coordinates and uses one linear-in-slice clock.
    """
    if mode != MODE:
        raise ValueError(f"explicit mode={MODE!r} is required")
    intervals = strawpath.combined_intervals(shells, geom, origin_mm, direction, length_mm)
    meta = dict(metadata)
    meta["settings"] = dict(meta["settings"], candidate=MODE,
                            geometry_scope="stored support and finite concentric straw cylinders; not GDML truth")
    if meta["frame"] != "detector":
        raise ValueError("ordered path history requires frame='detector'")
    result = orderedtransport.transport_intervals(None, identity, geometry.to_detector(origin_mm), direction,
        kinetic_MeV, intervals, meta, mode="ordered-support-deterministic-mean-v1", timing="linear-in-path",
        max_step_mm=max_step_mm, fractional_loss=fractional_loss, stop_ke_MeV=stop_ke_MeV, max_steps=max_steps)
    result["intervals"] = intervals
    return result


def at_entries(result, identity, path_mm):
    """Return the same history state used for physical surface projection; no extrapolation."""
    return historystate.at_path(result["history"], identity, path_mm, energy_evolution="linear-in-path")


def surface_steps(result, starts):
    """Project only traversed live history segments; no approach or downstream extension."""
    return historysurfaces.surface_steps(result["history"], starts, energy_evolution="linear-in-path")
