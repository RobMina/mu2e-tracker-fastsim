"""Transport-ledger arms and helpers for the tracker fast simulation (audit item T3).

The stage C transport keeps the muon energy, the straw deposition and the step time in several places. This module
names the separately switchable conventions ("arms") and supplies the helpers that make them one ledger:

  writer  momentum   legacy: frozen air dE/dx x path subtracted from |p| in the step writer (the interior is vacuum)
          energy     the same air estimate debited from the kinetic energy (writer-only, entryledger "energy")
          none       no writer debit (the GDML tracker envelopes are DSVacuum): written |p| = transport |p|
  thin    False      legacy: straw walls, gas and sense wire scatter but take no energy
          True       their unrestricted mean loss (core eloss.dedx_restricted, no random numbers) is charged to the
                     transport kinetic energy, from the next straw on, before stopping and scattering decisions
  time    generated  legacy: t = t_gen + path / (beta(p_gen) c)
          integrated t = t_gen + sum over segments of the exact flight time for kinetic energy linear in path
                     (world height bands, interior segments between straw entries; condensed-scattering
                     displacement jumps take no time)

The world energy-to-momentum convention is a core property (mu2efastsim.transport.world_loss); core_world_convention
reports which one the imported core implements so every output records it.
"""
import dataclasses
import hashlib
import inspect

import numpy as np

from mu2efastsim import eloss
from mu2efastsim import transport as tr

C_MM_PER_NS = 299.792458
WRITERS = ("momentum", "energy", "none")
TIMES = ("generated", "integrated")


@dataclasses.dataclass(frozen=True)
class Arms:
    """One ledger configuration; the default is the legacy production behaviour."""
    writer: str = "momentum"
    thin: bool = False
    time: str = "generated"

    def __post_init__(self):
        if self.writer not in WRITERS:
            raise ValueError(f"unknown writer convention {self.writer!r} (one of {WRITERS})")
        if self.time not in TIMES:
            raise ValueError(f"unknown time convention {self.time!r} (one of {TIMES})")

    def as_dict(self):
        return dataclasses.asdict(self)


PRESETS = {
    "legacy": Arms(),
    "E": Arms(writer="energy"),
    "A": Arms(writer="none"),
    "M": Arms(thin=True),
    "T": Arms(time="integrated"),
    "AM": Arms(writer="none", thin=True),
    "AMT": Arms(writer="none", thin=True, time="integrated"),
}


def preset(name):
    if name not in PRESETS:
        raise ValueError(f"unknown ledger preset {name!r} (one of {sorted(PRESETS)})")
    return PRESETS[name]


def core_world_convention():
    """'energy' if the imported core world_loss converts the remaining kinetic energy to momentum before each loss
    call (core 2bcefde and later), 'momentum' if it passes p - loss (earlier cores); plus the source hash."""
    src = inspect.getsource(tr.world_loss)
    conv = "energy" if "_p_after(p[hit], loss[hit])" in src else "momentum" if "p[hit] - loss[hit]" in src else "unknown"
    return {"world_convention": conv, "world_loss_sha256": hashlib.sha256(src.encode()).hexdigest()}


def kinetic(p):
    p = np.asarray(p, float)
    return np.sqrt(p ** 2 + eloss.MMU ** 2) - eloss.MMU


def momentum(tk):
    """|p| [MeV/c] of kinetic energy tk [MeV], no floor (0 for tk <= 0)."""
    tk = np.maximum(np.asarray(tk, float), 0.0)
    return np.sqrt(tk * (tk + 2 * eloss.MMU))


def beta(p):
    p = np.asarray(p, float)
    return p / np.sqrt(p ** 2 + eloss.MMU ** 2)


def flight_time_ns(length, p_in, p_out):
    """Flight time [ns] over `length` [mm] with kinetic energy falling linearly in path from p_in to p_out [MeV/c].

    With T(s) linear, dt = ds (T + m) / (c sqrt(T (T + 2m))) integrates to dt = L (p_in - p_out) / (c (T_in - T_out))
    = L (E_in + E_out) / (c (p_in + p_out)) (no cancellation); for p_in = p_out it is L / (beta c). p_out = 0 is the
    stopping endpoint (finite time)."""
    L = np.asarray(length, float)
    pa, pb = np.broadcast_arrays(np.asarray(p_in, float), np.asarray(p_out, float))
    ea, eb = np.sqrt(pa ** 2 + eloss.MMU ** 2), np.sqrt(pb ** 2 + eloss.MMU ** 2)
    return L * (ea + eb) / (np.maximum(pa + pb, 1e-300) * C_MM_PER_NS)


def world_band_momenta(p0, band_p, band_len):
    """Entry and exit |p| of every world height band, from world_loss's band mean |p| (entry + exit) / 2.

    Bands a muon does not cross (zero length) keep its momentum. Returns (p_in, p_out) [N, nbands]."""
    p0 = np.asarray(p0, float)
    nb = band_p.shape[1]
    p_in, p_out = np.zeros_like(band_p), np.zeros_like(band_p)
    cur = p0.copy()
    for k in range(nb):
        crossed = band_len[:, k] > 0
        nxt = np.where(crossed, np.maximum(2.0 * band_p[:, k] - cur, 0.0), cur)
        p_in[:, k], p_out[:, k] = cur, nxt
        cur = nxt
    return p_in, p_out


def world_time_ns(o, d, y_plane, p0, band_p, band_len, L_plane, p_plane):
    """Integrated flight time [ns] from the generated vertex to the plane: the straight stretch above the first band
    at p0, each band with flight_time_ns, and the small scattering correction of the path (L_plane - straight
    length, transport._apply_msc) at the plane momentum."""
    L_straight = (o[:, 1] - y_plane) / -d[:, 1]
    p_in, p_out = world_band_momenta(p0, band_p, band_len)
    t = flight_time_ns(band_len, p_in, p_out).sum(1)
    t += flight_time_ns(np.clip(L_straight - band_len.sum(1), 0, None), p0, p0)
    t += (L_plane - L_straight) / (np.maximum(beta(p_plane), 1e-12) * C_MM_PER_NS)
    return t


def segment_times_ns(track, s, p_entry, p_start):
    """Cumulative interior flight time [ns] to each entry (rows sorted by track, then path s from the interior origin),
    piecewise linear kinetic energy between consecutive entries, from (0, p_start[track]) at the origin."""
    track = np.asarray(track)
    if not len(track):
        return np.zeros(0)
    first = np.r_[True, track[1:] != track[:-1]]
    s_prev = np.where(first, 0.0, np.r_[0.0, s[:-1]])
    p_prev = np.where(first, p_start[track], np.r_[0.0, p_entry[:-1]])
    dt = flight_time_ns(np.clip(s - s_prev, 0, None), p_prev, p_entry)
    grp = np.cumsum(first) - 1
    cum = np.cumsum(dt)
    return cum - np.r_[0.0, cum][np.flatnonzero(first)][grp]


def thin_mean_loss(mat, p, length):
    """Unrestricted mean loss [MeV] of `length` [mm] of `mat` at |p| (core restricted dE/dx with an infinite cut)."""
    length = np.asarray(length, float)
    out = np.zeros_like(length)
    h = np.flatnonzero(length > 0)
    if len(h):
        out[h] = eloss.dedx_restricted(mat, np.maximum(np.asarray(p, float)[h], 1.0), 1e9) * length[h]
    return out
