"""End-to-end tracker chain benchmark (trkfastsim.kit.devchain): cold and warm passes per generator table, per-stage
wall time, output digests (repeatability), peak host and device memory, optional cProfile and device trace of one
pass, and per-shard summaries for paired comparisons between backends (chain_pair).

The context (geometry, world, interior, conditions) is built once; pass 0 of the first table includes compilation.
Throughput is quoted from the warm passes as generated events per second of chain wall time.
"""
import argparse
import cProfile
import gzip
import json
import logging
import os
import re
from pathlib import Path
import platform
import pstats
import resource
import time

import numpy as np

from mu2efastsim import tables
from trkfastsim.kit import devchain
from trkfastsim.kit.kitpoint import file_identity, validate_drift
from trkfastsim.reco.cli import _drift_table

PRESETS = {"numpy": devchain.NUMPY, "device": devchain.DEVICE}
TRANSFER_FUNCS = ("asarray", "array", "device_put", "device_get", "__array__", "_value")
HISTS = {"steps_per_event": (0, 120, 60), "digis_per_event": (0, 120, 60), "hits_per_event": (0, 120, 60),
         "step_eion_keV": (0, 20, 80), "step_len_mm": (0, 6, 60), "digi_panel": (0, 216, 216),
         "hit_time_ns": (0, 100000, 100), "hit_edep_keV": (0, 10, 50), "seeds_per_event": (0, 8, 8),
         "fit_chi2ndf": (0, 10, 50), "fit_nactive": (0, 60, 60)}


def parse_backends(text):
    if text in PRESETS:
        return dict(PRESETS[text])
    b = dict(devchain.NUMPY)
    for item in text.split(","):
        k, v = item.split("=")
        b[k] = v
    return devchain.check_backends(b)


def histograms(r):
    """Per-pass histograms for distribution comparisons (fixed binning, overflow in the last bin)."""
    p, out = r["products"], {}
    s, d, sh, tb, sd = p["steps"], p["digis"], p["hits"], p["table"], p["seeds"]
    E = r["counts"]["generated_events"]
    ek = tables.KEY

    def per_event(n):
        n = np.asarray(n)
        return n[n > 0]
    sev = s.groupby(ek, sort=False).size().to_numpy() if len(s) else np.zeros(0)
    vals = {"steps_per_event": sev, "digis_per_event": per_event(np.bincount(d["ev"], minlength=E)),
            "hits_per_event": per_event(np.bincount(sh["ev"], minlength=E)),
            "step_eion_keV": 1e3 * s["eion"].to_numpy(), "step_len_mm": s["len"].to_numpy(),
            "digi_panel": sh["upanel"], "hit_time_ns": sh["time"], "hit_edep_keV": 1e3 * sh["edep"],
            "seeds_per_event": per_event(np.bincount(sd["ev"], minlength=E)) if len(sd["n"]) else np.zeros(0)}
    good = tb["goodfit"].astype(bool)
    vals["fit_chi2ndf"] = (tb["chi2"] / np.maximum(tb["ndof"], 1))[good]
    vals["fit_nactive"] = (tb["state"] != -2).sum(1)[good] if len(good) else np.zeros(0)
    for k, (lo, hi, n) in HISTS.items():
        v = np.asarray(vals[k], np.float64)
        edges = np.linspace(lo, hi, n + 1)
        out[k] = np.histogram(np.clip(v, lo, np.nextafter(hi, lo)), edges)[0].tolist()
    return out


def device_memory():
    try:
        import jax
        ms = jax.devices()[0].memory_stats()
    except Exception:
        return None
    return {k: int(ms[k]) for k in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit") if ms and k in ms} or None


def transfer_calls(stats):
    """cProfile call counts of the Python-level host/device conversion entry points (jax.numpy, jax, jax.Array)."""
    out = {}
    for (fname, _, func), v in stats.stats.items():
        if func in TRANSFER_FUNCS and ("jax" in fname or "jaxlib" in fname):
            k = f"{func}@{Path(fname).parent.name}/{Path(fname).name}"
            out[k] = out.get(k, 0) + v[1]
    return out


def trace_transfers(trace_dir):
    """Host/device copies in a JAX profiler perfetto trace: event count and bytes per copy kind."""
    files = sorted(Path(trace_dir).rglob("perfetto_trace.json.gz"))
    if not files:
        return None
    with gzip.open(files[-1], "rt") as f:
        ev = json.load(f)
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    out = {}
    for e in ev:
        name = str(e.get("name", ""))
        if not name.startswith("Memcpy"):
            continue
        kind = ("H2D" if "h2d" in name.lower() or "htod" in name.lower() else
                "D2H" if "d2h" in name.lower() or "dtoh" in name.lower() else
                "D2D" if "d2d" in name.lower() or "dtod" in name.lower() else name[:40])
        a = e.get("args", {}) or {}
        nbytes = next((int(v) for k, v in a.items() if "bytes" in k.lower() and str(v).isdigit()), 0)
        m = re.search(r"size:(\d+)", str(a.get("memcpy_details", "")))
        nbytes = int(m.group(1)) if m else nbytes
        o = out.setdefault(kind, {"events": 0, "bytes": 0, "dur_us": 0.0, "events_le_64B": 0})
        o["events"] += 1
        o["events_le_64B"] += nbytes <= 64
        o["bytes"] += nbytes
        o["dur_us"] += float(e.get("dur", 0.0))
    return out


class CompileCounter(logging.Handler):
    """Counts JAX compilations (jax_log_compiles messages) per function: backend compiles and their seconds,
    persistent-cache hits, and the distinct argument shapes compiled per function."""
    PAT = re.compile(r"Finished XLA compilation of (?:jit\()?([^\s)]+)\)? in ([0-9.eE+-]+) ?s")
    SHAPES = re.compile(r"Compiling (?:jit\()?([^\s)]+)\)? with global shapes and types (.*?)\. Argument mapping")

    def __init__(self):
        super().__init__(logging.DEBUG)
        import jax
        jax.config.update("jax_log_compiles", True)
        for name in ("jax", "jax._src"):
            lg = logging.getLogger(name)
            lg.addHandler(self)
            lg.setLevel(logging.DEBUG)
            lg.propagate = False
        self.reset()

    def reset(self):
        self.n, self.secs, self.hits, self.shapes = {}, {}, 0, {}

    def emit(self, record):
        msg = record.getMessage()
        m = self.PAT.search(msg)
        if m:
            f = m.group(1)
            self.n[f] = self.n.get(f, 0) + 1
            self.secs[f] = self.secs.get(f, 0.0) + float(m.group(2))
        elif "cache hit" in msg.lower():
            self.hits += 1
        m = self.SHAPES.search(msg)
        if m:
            self.shapes.setdefault(m.group(1), []).append(m.group(2)[:300])

    def summary(self):
        return {"compiles": sum(self.n.values()), "compile_s": sum(self.secs.values()), "cache_hits": self.hits,
                "by_function": {f: [self.n[f], round(self.secs[f], 3)] for f in sorted(self.n, key=lambda f: -self.secs[f])},
                "shapes": {f: v for f, v in self.shapes.items()}}


def summarise_profile(prof, path, top=40):
    stats = pstats.Stats(prof)
    rows = sorted(({"function": f"{Path(k[0]).name}:{k[1]}:{k[2]}", "calls": v[1], "self_s": v[2], "cum_s": v[3]}
                   for k, v in stats.stats.items()), key=lambda r: -r["self_s"])[:top]
    prof.dump_stats(str(path) + ".pstats")
    return {"top_self": rows, "transfer_calls": transfer_calls(stats)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen", nargs="+", required=True, help="generator tables, one pass set per table")
    p.add_argument("--gdml", required=True)
    p.add_argument("--digi-cond", required=True)
    p.add_argument("--reco-cond", required=True)
    p.add_argument("--drift-source", required=True)
    p.add_argument("--backends", default="numpy", help="numpy, device, or stage=backend[,stage=backend]")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1, help="passes per table (identical inputs and seed)")
    p.add_argument("--profile-pass", type=int, help="global pass index to run under cProfile")
    p.add_argument("--trace-pass", type=int, help="global pass index to run under the JAX profiler")
    p.add_argument("--digi-compact", choices=("auto", "on", "off"), default="auto",
                   help="digitise in chunks of events with steps (auto: on for the JAX digitiser)")
    p.add_argument("--shape-growth", type=int, help="JAX step-kernel shape growth (default devchain.DEVICE_SHAPE_GROWTH)")
    p.add_argument("--step-settings", help="JAX step settings, module.NAME=value[,...]; 'default' for the steps "
                   "--backend jax defaults (default: devchain.DEVICE_STEP_SETTINGS)")
    p.add_argument("--log-compiles", action="store_true", help="count JAX compilations per pass and function")
    p.add_argument("--save-steps", action="store_true", help="write each table's first-pass steps and digis (.npz)")
    p.add_argument("--outdir", type=Path, required=True)
    a = p.parse_args(argv)
    b = parse_backends(a.backends)
    compact = {"auto": None, "on": True, "off": False}[a.digi_compact]
    step_settings = None
    if a.step_settings == "default":
        step_settings = {}
    elif a.step_settings:
        step_settings = {k: (v.lower() == "true" if v.lower() in ("true", "false") else int(v))
                         for k, v in (x.split("=") for x in a.step_settings.split(","))}
    a.outdir.mkdir(parents=True, exist_ok=False)
    counter = CompileCounter() if a.log_compiles else None
    t0 = time.perf_counter()
    drift = validate_drift(_drift_table(a.drift_source))
    ctx = devchain.Context(a.gdml, a.digi_cond, a.reco_cond, drift)
    prep_s = time.perf_counter() - t0
    try:
        import jax
        runtime = {"jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
                   "device_kind": jax.devices()[0].device_kind, "x64": bool(jax.config.jax_enable_x64)}
        from mu2efastsim import precision
        runtime["precision"] = precision.provenance() if hasattr(precision, "provenance") else precision.NAME
    except Exception as e:                     # NumPy-only environments
        runtime = {"jax": None, "error": str(e)}
    report = {"backends": b, "seed": a.seed, "repeat": a.repeat, "prepare_s": prep_s, "runtime": runtime,
              "host": platform.node(), "cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "env": {k: v for k, v in sorted(os.environ.items()) if k in ("XLA_FLAGS", "JAX_PLATFORMS", "OMP_NUM_THREADS")
                      or k.startswith(("MU2EFS_", "TRKFASTSIM_", "TRKFS_"))},
              "inputs": {"gen": [file_identity(g) for g in a.gen], "digi_cond": file_identity(a.digi_cond),
                         "reco_cond": file_identity(a.reco_cond), "drift_source": file_identity(a.drift_source)},
              "shards": []}
    gi = 0
    for path in a.gen:
        gen = tables.read(path, "gen")
        shard = {"gen": Path(path).name, "passes": []}
        for rep in range(a.repeat):
            prof = cProfile.Profile() if a.profile_pass == gi else None
            tracing = a.trace_pass == gi
            if tracing:
                import jax
                jax.profiler.start_trace(str(a.outdir / f"trace_{gi:03d}"), create_perfetto_trace=True)
            if prof:
                prof.enable()
            t = time.perf_counter()
            if counter:
                counter.reset()
            r = devchain.run(gen, ctx, a.seed, b, digi_compact=compact, shape_growth=a.shape_growth,
                             step_settings=step_settings)
            wall = time.perf_counter() - t
            if prof:
                prof.disable()
            if tracing:
                jax.profiler.stop_trace()
            rec = {"pass": gi, "repeat": rep, "wall_s": wall, "timing": r["timing"], "counts": r["counts"],
                   "hashes": r["hashes"], "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "device_memory": device_memory(), "profiled": bool(prof), "traced": tracing,
                   "digi_compact": r["digi_compact"], "step_settings": r["step_settings"]}
            if counter:
                rec["compiles"] = counter.summary()
            if prof:
                rec["profile"] = summarise_profile(prof, a.outdir / f"profile_{gi:03d}")
            if tracing:
                rec["transfers"] = trace_transfers(a.outdir / f"trace_{gi:03d}")
            if rep == 0 and a.save_steps:
                np.savez_compressed(a.outdir / f"products_{Path(path).stem}.npz",
                                    **{"steps_" + c: r["products"]["steps"][c].to_numpy() for c in r["products"]["steps"]},
                                    **{"digi_" + k: v for k, v in r["products"]["digis"].items()})
            if rep == 0:
                shard["histograms"] = histograms(r)
                shard["counts"] = r["counts"]
                shard["hashes"] = r["hashes"]
            shard["passes"].append(rec)
            print(json.dumps({"gen": shard["gen"], **{k: rec[k] for k in ("pass", "wall_s", "counts")}}), flush=True)
            gi += 1
        shard["repeatable"] = all(q["hashes"] == shard["passes"][0]["hashes"] for q in shard["passes"])
        report["shards"].append(shard)
        (a.outdir / "report.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    plain = [q for s in report["shards"] for q in s["passes"] if not q["profiled"] and not q["traced"]]
    # warm: repeats of a table already run in this process; steady: every pass after the first (new tables may
    # still compile new shapes, as in production streaming)
    for name, sel in (("warm", [q for q in plain if q["repeat"] > 0]), ("steady", [q for q in plain if q["pass"] > 0])):
        if sel:
            ev = sum(q["counts"]["generated_events"] for q in sel)
            wall = sum(q["wall_s"] for q in sel)
            report[name] = {"passes": len(sel), "generated_events": ev, "wall_s": wall, "events_per_s": ev / wall,
                            "stage_s": {k: sum(q["timing"].get(k, 0.0) for q in sel) for k in sel[0]["timing"]}}
    report["all_repeatable"] = all(s["repeatable"] for s in report["shards"])
    (a.outdir / "report.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("warm", "steady", "all_repeatable", "prepare_s")}), flush=True)


if __name__ == "__main__":
    main()
