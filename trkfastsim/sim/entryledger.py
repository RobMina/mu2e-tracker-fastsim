"""Straw-entry writer diagnostics on fixed trajectories; not a material transport ledger."""
import numpy as np

from mu2efastsim import eloss, materials, transport

MODES = ("legacy", "energy", "vacuum")


def evaluate(p_start, support_loss, path, p_legacy, *, air_loss=True, mode="legacy"):
    """Return cumulative entry accounting in MeV, MeV/c and mm, before gas deposition.

    Energy mode debits the existing frozen-entry-momentum air estimate in energy space. Vacuum mode removes
    that estimate, retaining support loss. Neither mode changes crossings, support sampling or scattering.
    Positive energy_defect_MeV means the written kinetic energy exceeds the requested remaining energy.
    Gas/wall/wire transport loss and flight time are outside this ledger.
    """
    if mode not in MODES:
        raise ValueError(f"unknown entry ledger mode: {mode}")
    p0, support, distance, legacy = np.broadcast_arrays(
        np.asarray(p_start, float), np.asarray(support_loss, float),
        np.asarray(path, float), np.asarray(p_legacy, float))
    if any(not np.isfinite(x).all() or (x < 0).any() for x in (p0, support, distance, legacy)):
        raise ValueError("entry ledger inputs must be finite and nonnegative")
    kinetic = np.hypot(p0, eloss.MMU) - eloss.MMU
    air = eloss.dedx_restricted(materials.AIR, p0, 1e9) * distance if air_loss else np.zeros_like(p0)
    debit = np.zeros_like(air) if mode == "vacuum" else air
    requested = kinetic - support - debit
    momentum = legacy.copy() if mode == "legacy" else transport._p_after(p0, support + debit)
    remaining = np.hypot(momentum, eloss.MMU) - eloss.MMU
    return {"p_start_MeV_c": p0, "p_legacy_MeV_c": legacy, "p_written_MeV_c": momentum,
            "path_mm": distance, "ke_start_MeV": kinetic, "support_loss_MeV": support,
            "air_estimate_MeV": air, "air_debit_MeV": debit, "ke_requested_MeV": requested,
            "ke_written_MeV": remaining, "energy_defect_MeV": remaining - np.maximum(requested, 0.),
            "above_stop_threshold": requested > .5}


def keep_before_stop(track, path, above_threshold):
    """Keep no crossing at or beyond a track's first below-threshold entry, including unsorted rows."""
    track, path, above = np.asarray(track), np.asarray(path), np.asarray(above_threshold, bool)
    if not len(track):
        return np.zeros(0, bool)
    _, groups = np.unique(track, return_inverse=True)
    first_stop = np.full(groups.max() + 1, np.inf)
    np.minimum.at(first_stop, groups[~above], path[~above])
    return above & (path < first_stop[groups])
