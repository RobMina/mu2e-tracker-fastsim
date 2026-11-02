"""Floating-point precision of the tracker step and digitisation JAX kernels: TRKFASTSIM_DEVICE_FLOAT=float64
(default) or float32 (opt-in).

The tracker does not follow the core device default (mu2efastsim.precision, MU2EFS_DEVICE_FLOAT) until its float32
validation is accepted: unset or float64 leaves every tracker kernel as before (jax.jit, float64). float32 runs the
step kernels (world column and loss, interior transport, exit scatter, gas eIon) and the JAX digitiser (clusters,
charge, electronics) in float32 through the core mechanism (dprec.jit, so it needs MU2EFS_DEVICE_FLOAT=float32, the
core default), with float32 cancellations removed by precision-aware forms that apply in float32 mode only:

- positions: the step kernels work in a frame centred on the straws (frame_of, whole mm), the digitiser in
  coordinates relative to each straw's middle;
- times: the digitiser electronics work relative to a per-straw reference time that is a multiple of COMMON_PERIOD
  (whole ADC periods, TDC and TOT counts), so integer TDC, ADC phase and TOT are exact offsets.

Hit reco, combo hits, line seeds and F2 keep their own precision in both settings (stock ports in stock's mixed
float/double; F2 float64): docs/device-precision.md. TRKFASTSIM_DRAW64=1 (float32 mode, diagnostic) draws the
digitiser's random numbers in float64 and rounds them, so float32 and float64 runs differ only by kernel arithmetic.
"""
import functools
import os

import numpy as np

from mu2efastsim import precision as dprec

DEFAULT = "float64"
NAME = os.environ.get("TRKFASTSIM_DEVICE_FLOAT", DEFAULT) or DEFAULT
if NAME not in ("float64", "float32"):
    raise ValueError(f"TRKFASTSIM_DEVICE_FLOAT must be float64 or float32, not {NAME!r}")
SINGLE = NAME == "float32"
if SINGLE and not dprec.SINGLE:
    raise ValueError("TRKFASTSIM_DEVICE_FLOAT=float32 needs the core device float in float32 mode "
                     "(MU2EFS_DEVICE_FLOAT=float32, the core default)")
FLOAT = np.dtype(NAME)
TINY = dprec.tiny_of(FLOAT)          # divide guard / smallest positive: 1e-300 in float64, as shipped
DRAW64 = SINGLE and os.environ.get("TRKFASTSIM_DRAW64", "") == "1"
COMMON_PERIOD = 20.0                 # ns: ADCPeriod; a multiple of TDCLSB and TOTLSB in the supported conditions


def jit(fun=None, **kw):
    """jax.jit in float64 mode; the core precision-aware jit (float64 arguments cast to float32) in float32 mode."""
    if SINGLE:
        return dprec.jit(fun, **kw)
    import jax
    return jax.jit(fun, **kw) if fun is not None else functools.partial(jax.jit, **kw)


def dev(tree):
    """Floating arrays of a pytree in the tracker device float (unchanged in float64 mode)."""
    return dprec.dev(tree) if SINGLE else tree


def host(x):
    """Device array as NumPy, float32 widened to float64 in float32 mode (host code is float64)."""
    a = np.asarray(x)
    return a.astype(np.float64) if SINGLE and a.dtype == np.float32 else a


def draw(sampler, key, shape=(), **kw):
    """jax.random sampler in the device float (float64 draws rounded with TRKFASTSIM_DRAW64=1)."""
    if DRAW64:
        return sampler(key, shape, dtype=np.float64, **kw).astype(FLOAT)
    return sampler(key, shape, dtype=FLOAT, **kw)


def frame_of(points):
    """Origin of the step kernels' frame: the straw centroid rounded to whole mm (float32 mode), else zero."""
    return np.round(np.mean(np.asarray(points, np.float64), 0)) if SINGLE else np.zeros(3)


def check_common_period(c):
    """The digitiser time reference needs COMMON_PERIOD to hold whole ADC periods, TDC and TOT counts and response bins."""
    for name, step in (("ADCPeriod", c.ADCPeriod), ("TDCLSB", c.TDCLSB), ("TOTLSB", c.TOTLSB),
                       ("1/sampleRate", 1.0 / c.sampleRate)):
        r = COMMON_PERIOD / float(step)
        if abs(r - round(r)) > 1e-9:
            raise NotImplementedError(f"float32 digitiser: {name}={step} does not divide {COMMON_PERIOD} ns")


def provenance():
    """Tracker device float for output metadata."""
    return {"tracker_device_float": NAME, "tracker_draw64": DRAW64}
