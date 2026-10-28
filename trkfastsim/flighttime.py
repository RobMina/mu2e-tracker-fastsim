"""Standalone timing for supplied linear-in-path kinetic states, not a transport or loss model."""
import numpy as np

from mu2efastsim.eloss import MMU
from trkfastsim.transportledger import SegmentHistory

C_MM_PER_NS = 299.792458


def linear_kinetic_dt_ns(path_mm, ke_in_MeV, ke_out_MeV):
    """Integrate ds/(beta*c) for explicitly supplied passive, linear-in-path muon KE.

    Inputs broadcast as NumPy arrays; the returned array has their broadcast shape.
    Length is the actual traversed arc length in mm, not a requested slab thickness.
    Endpoint KE is in MeV and must be finite, nonnegative and nonincreasing.
    The exact integral is L*(E0+E1)/(c*(p0+p1)), with E=K+MMU.
    This rationalized form also covers constant KE and a zero-energy exit without
    subtraction of nearby momenta. Zero path returns zero, including stopped lanes;
    positive path with both endpoints stopped is invalid. No momentum floor is used.
    Endpoint states alone do not establish that linear KE evolution is physical.
    """
    length, kin, kout = np.broadcast_arrays(
        np.asarray(path_mm, dtype=float), np.asarray(ke_in_MeV, dtype=float),
        np.asarray(ke_out_MeV, dtype=float))
    for name, value in (("path_mm", length), ("ke_in_MeV", kin), ("ke_out_MeV", kout)):
        if not np.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"{name} must be finite and nonnegative")
    if (kout > kin).any():
        raise ValueError("passive segments cannot gain kinetic energy")
    active = length > 0
    if (active & (kin == 0)).any():
        raise ValueError("stopped endpoints cannot traverse a positive path")
    result = np.zeros(length.shape, dtype=float)
    k0, k1 = kin[active], kout[active]
    # Separate roots and half sums avoid intermediate squares and doubled energies.
    p0 = np.sqrt(k0) * np.sqrt(k0 + 2 * MMU)
    p1 = np.sqrt(k1) * np.sqrt(k1 + 2 * MMU)
    inverse_beta = (0.5 * k0 + 0.5 * k1 + MMU) / (0.5 * p0 + 0.5 * p1)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        dt = (length[active] / C_MM_PER_NS) * inverse_beta
        exceptional = (dt == 0.) | ~np.isfinite(dt)
        ml, el = np.frexp(length[active][exceptional])
        mb, eb = np.frexp(inverse_beta[exceptional])
        dt[exceptional] = np.ldexp(ml * mb / C_MM_PER_NS, el + eb)
    if (dt <= 0.).any() or not np.isfinite(dt).all():
        raise ValueError("positive flight time is not representable as finite float64")
    result[active] = dt
    return result


def history_dt_ns(history, *, energy_evolution):
    """Return row-aligned increments only for caller-declared 'linear-in-path' KE.

    The declaration applies to every physical row and is an assumption supplied by
    the caller, never inferred from endpoint energies, media or existing dt values.
    Displacements and zero-path rows contribute zero. Existing history columns and
    metadata are unchanged; this neither fills dt_status nor groups/sums muon times.
    Stops, traversed paths and endpoint states must already have been established.
    """
    if energy_evolution != "linear-in-path":
        raise ValueError("explicit energy_evolution='linear-in-path' is required")
    if not isinstance(history, SegmentHistory):
        raise TypeError("history must be a SegmentHistory")
    return linear_kinetic_dt_ns(history["path_mm"], history["ke_in_MeV"], history["ke_out_MeV"])
