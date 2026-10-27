"""Throughput of trkfastsim.digi on synthetic events (no data files): N events of 22 minimum-ionising chords each.

Conditions: tests/data/digi_job_excerpt.fcl (the stock job's tables), uniform 12 mV thresholds, a synthetic straw
layout. For relative timing of code changes only; the GPVM numbers in the validation come from real step tables.
usage: PYTHONPATH=core:tracker python3 scripts/digi_bench.py [nevents] [repeats]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trkfastsim import digi as DG                                       # noqa: E402
from trkfastsim import digicond as DC                                   # noqa: E402
from trkfastsim import geometry as G                                    # noqa: E402


def conditions():
    n = DC.NUSTRAWS
    up = np.repeat(np.arange(216), 96)
    s = np.tile(np.arange(96), 216)
    mid = np.c_[np.zeros(n), 10.0 * s, 60.0 * up] + G.DETECTOR_ORIGIN_EXTRACTED
    g = G.StrawGeometry(plane=up // 6, panel=up % 6, straw=s, mid=mid, dir=np.tile([1.0, 0, 0], (n, 1)),
                        halflen=np.full(n, 500.0), radius=2.5085, panel_R=np.zeros((216, 3, 3)),
                        panel_t=np.zeros((216, 3)), plane_R=np.zeros((36, 3, 3)), plane_t=np.zeros((36, 3)),
                        wall_rmax=2.5235, wire_r=0.0125)
    pr = np.zeros((n, 6))
    pr[:, 0], pr[:, 3], pr[:, 4], pr[:, 5] = np.arange(n), 12.0, 12.0, 1.5e6
    d = DC.parse_fhicl_dump((Path(__file__).resolve().parent.parent / "tests/data/digi_job_excerpt.fcl").read_text())
    return DC.assemble(d, pr, g)


def events(c, nev, rng, nper=22):
    ev = np.repeat(np.arange(nev), nper)
    us = rng.integers(0, DC.NUSTRAWS, len(ev))
    r = c.strawInnerRadius
    b = rng.uniform(-r, r, len(ev)) * 0.98
    h = np.sqrt(r * r - b * b)
    zpos = rng.uniform(-450, 450, len(ev))
    base = c.straw_mid[us] + zpos[:, None] * c.straw_dir[us] + b[:, None] * np.array([0.0, 1.0, 0.0])
    p0, p1 = base + h[:, None] * [0, 0, 1.0], base - h[:, None] * [0, 0, 1.0]
    t = np.repeat(rng.uniform(500, 90000, nev), nper)
    return dict(ev=ev, sid=DG.unique_straw_to_sid(us), shape=np.zeros(len(ev), int), ion=np.zeros(len(ev), int),
                eion=np.clip(rng.lognormal(np.log(1.5e-3), 0.5, len(ev)), 1e-5, 0.05), len=2 * h,
                width=np.zeros(len(ev)), t=t, x0=p0[:, 0], y0=p0[:, 1], z0=p0[:, 2], x1=p1[:, 0], y1=p1[:, 1], z1=p1[:, 2])


if __name__ == "__main__":
    nev = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rep = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    c = conditions()
    rng = np.random.default_rng(1)
    st = events(c, nev, rng)
    for i in range(rep):
        t0 = time.perf_counter()
        d = DG.digitize(st, nev, c, np.random.default_rng(i))
        dt = time.perf_counter() - t0
        print(f"{nev} events, {len(d['sid'])} digis: {dt:.2f} s, {1e3 * dt / nev:.3f} ms/event")
