"""Profile and verify fixed-shape F2 on a bounded persisted digi event population.

Compare legacy and fixed-shape execution on the same selected events, then repeat
fixed execution at another event chunk size. No digitisation or stock reco is run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkfastsim.analyses import digicmp
from trkfastsim.reco import hitreco


def require_population(persisted, requested, legacy):
    if persisted < requested or requested <= 50 or legacy <= 0:
        raise ValueError("need the requested population (>50 events) and a nonempty legacy subset")


def require_exercised(legacy, fixed):
    if legacy["seeds"] <= 0 or legacy["good_fits"] <= 0 or fixed["seeds"] <= 0:
        raise ValueError("no successful fitter coverage; equality alone cannot certify batching")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--conditions", required=True)
    p.add_argument("--drift-source", required=True)
    p.add_argument("--events", type=int, default=100)
    p.add_argument("--legacy-events", type=int, default=32)
    p.add_argument("--outdir", required=True)
    a = p.parse_args(argv)
    if a.events <= 0 or not 0 < a.legacy_events <= a.events:
        p.error("events >= legacy-events > 0 required")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=False)
    cond = hitreco.load_conditions(a.conditions)
    d = hitreco.read_digis(a.inputs)
    persisted = len(d["evkey"])
    require_population(persisted, a.events, a.legacy_events)
    if len(np.unique(d["evkey"], axis=0)) != persisted:
        raise ValueError("duplicate persisted event identities")
    d = digicmp._restrict(d, np.arange(min(a.events, persisted)))
    small = digicmp._restrict(d, np.arange(min(a.legacy_events, len(d["evkey"]))))
    report = {"persisted_events": persisted, "requested_events": a.events,
              "legacy_events": len(small["evkey"]), "tested_events": len(d["evkey"]),
              "keys_sha256": hashlib.sha256(np.ascontiguousarray(d["evkey"]).tobytes()).hexdigest(), "calls": {}}

    def run(name, data, fixed, chunk):
        def progress(r):
            print(name + " " + json.dumps(r, sort_keys=True), flush=True)
        start = time.perf_counter()
        y = digicmp.f2_yield(data, cond, a.drift_source, fixed_shapes=fixed, chunk=chunk, progress=progress)
        np.savez_compressed(out / (name + ".npz"), evkey=data["evkey"], fits_per_event=y.pop("fits_per_event"))
        report["calls"][name] = {**y, "wall_s": time.perf_counter() - start,
                                 "maxrss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        print(name + " " + json.dumps(report["calls"][name], sort_keys=True), flush=True)
        return y

    legacy = run("legacy", small, False, 0)
    fixed_small = run("fixed_small", small, True, 0)
    for name in ("seeds", "good_fits", "events_with_fit"):
        if legacy[name] != fixed_small[name]:
            raise AssertionError((name, legacy[name], fixed_small[name]))
    np.testing.assert_array_equal(np.load(out / "legacy.npz")["fits_per_event"],
                                  np.load(out / "fixed_small.npz")["fits_per_event"])
    for name in ("nactive_q", "chi2ndf_q"):
        np.testing.assert_allclose(legacy[name], fixed_small[name], rtol=1e-8, atol=1e-8)
    first = run("fixed_50", d, True, 50)
    second = run("fixed_37", d, True, 37)
    for name in ("seeds", "good_fits", "events_with_fit"):
        if first[name] != second[name]:
            raise AssertionError((name, first[name], second[name]))
    np.testing.assert_array_equal(np.load(out / "fixed_50.npz")["fits_per_event"],
                                  np.load(out / "fixed_37.npz")["fits_per_event"])
    for name in ("nactive_q", "chi2ndf_q"):
        np.testing.assert_allclose(first[name], second[name], rtol=1e-8, atol=1e-8)
    require_exercised(legacy, first)
    permutation = np.arange(len(d["evkey"]))[::-1]
    reordered = digicmp._restrict(d, permutation)
    third = run("fixed_reordered", reordered, True, 37)
    counts = np.load(out / "fixed_reordered.npz")["fits_per_event"][np.argsort(permutation)]
    np.testing.assert_array_equal(counts, np.load(out / "fixed_50.npz")["fits_per_event"])
    for name in ("seeds", "good_fits", "events_with_fit"):
        if first[name] != third[name]:
            raise AssertionError((name, first[name], third[name]))
    for name in ("nactive_q", "chi2ndf_q"):
        np.testing.assert_allclose(first[name], third[name], rtol=1e-8, atol=1e-8)
    report["closure"] = True
    (out / "check.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("F2 batch equivalence PASS", flush=True)
    return report


if __name__ == "__main__":
    main()
