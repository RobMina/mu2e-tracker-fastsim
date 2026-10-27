"""Throughput of trkfastsim.digi on synthetic events (no data files): N events of 22 minimum-ionising chords each.

Conditions: tests/data/digi_job_excerpt.fcl (the stock job's tables), uniform 12 mV thresholds, a synthetic straw
layout. For relative timing of code changes only; the GPVM numbers in the validation come from real step tables.
usage: PYTHONPATH=core:tracker python3 scripts/digi_bench.py [nevents] [repeats] [--backend numpy|jax] [--json PATH]
"""
import argparse
import json
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("nevents", nargs="?", type=int, default=1000)
    p.add_argument("repeats", nargs="?", type=int, default=3)
    p.add_argument("--backend", choices=["numpy", "jax"], default="numpy")
    p.add_argument("--seed", type=int, default=1, help="digitizer seed, reused for every repeat")
    p.add_argument("--json", nargs="?", const="-", metavar="PATH", help="write timings as JSON (- for stdout)")
    args = p.parse_args(argv)
    nev, rep = args.nevents, args.repeats
    if nev <= 0 or rep <= 0:
        p.error("nevents and repeats must be positive")
    if args.backend == "jax":
        import jax
        from trkfastsim import digi_jax
        digitize = digi_jax.digitize
        device = ", ".join(str(d) for d in jax.devices())
    else:
        digitize = DG.digitize
        device = "cpu (numpy)"
    c = conditions()
    st = events(c, nev, np.random.default_rng(1))
    runs, first = [], None
    reproducible = True
    for i in range(rep):
        rng = np.random.default_rng(args.seed) if args.backend == "numpy" else jax.random.key(args.seed)
        t0 = time.perf_counter()
        d = digitize(st, nev, c, rng)
        dt = time.perf_counter() - t0  # Both backends return host NumPy arrays, synchronising device work.
        if first is None:
            first = d
        else:
            reproducible &= d.keys() == first.keys() and all(np.array_equal(d[k], first[k]) for k in d)
        phase = "cold" if i == 0 else "warm"
        runs.append({"phase": phase, "seconds": dt, "digis": len(d["sid"]), "ms_per_event": 1e3 * dt / nev})
        print(f"{args.backend} {phase}, {device}: {nev} events, {len(d['sid'])} digis: "
              f"{dt:.2f} s, {1e3 * dt / nev:.3f} ms/event", file=sys.stderr if args.json == "-" else sys.stdout)
    warm = [r["seconds"] for r in runs[1:]]
    report = {"backend": args.backend, "device": device, "events": nev, "steps": len(st["sid"]),
              "seed": args.seed, "event_seed": 1, "repeats": rep, "reproducible": bool(reproducible),
              "cold_s": runs[0]["seconds"], "warm_s": warm, "warm_mean_s": float(np.mean(warm)) if warm else None,
              "timing": "digitize only; first call cold, subsequent calls warm; host output includes device synchronisation",
              "runs": runs}
    if args.json:
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.json == "-":
            print(text, end="")
        else:
            Path(args.json).write_text(text)
    return report


if __name__ == "__main__":
    main()
