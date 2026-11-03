"""End-to-end tracker chain in memory, one backend choice per stage, for identity and throughput measurements.

steps (make_steps stage C, sliced world MSC, every primary muon) -> digis (digitize_steps) -> straw hits -> panel combo
hits -> time clusters -> line seeds -> F2 (tracktable.from_f2_digis). With every backend "numpy" this is the
tracker_point chain (kitpoint.run) without its file handling and writer: same keys, seeds and outputs. The
combo hits are made and counted but not consumed: the extracted reconstruction is combo-free (as stock).

Backends: steps and digi numpy or jax (independent random streams: compare as distributions); hits and combos
numpy or jax (twins that are bitwise equal to NumPy on the same digis). Time clusters are NumPy; line seeds and F2
are the JAX kernels in every configuration (as in tracker_point).
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from trkfastsim import geometry
from trkfastsim.digi import digicond, fastdigi
from trkfastsim.reco import combo, hitreco, tracktable
from trkfastsim.sim import steps as steps_mod

BACKENDS = {"steps": ("numpy", "jax"), "digi": ("numpy", "jax"), "hits": ("numpy", "jax"), "combos": ("numpy", "jax")}
NUMPY = {k: "numpy" for k in BACKENDS}
DEVICE = {k: "jax" for k in BACKENDS}


class Context:
    """Geometry, world, interior and conditions, built once and reused by every pass."""

    def __init__(self, gdml, digi_cond, reco_cond, drift):
        self.geom = geometry.from_gdml(str(gdml))
        self.world = steps_mod.load_world(str(gdml), self.geom)
        self.interior = steps_mod.TrackerInterior(str(gdml), self.geom)
        self.dc = digicond.Conditions.load(digi_cond) if isinstance(digi_cond, str) else digi_cond
        self.rc = hitreco.load_conditions(reco_cond) if isinstance(reco_cond, str) else reco_cond
        self.drift = drift


def check_backends(backends):
    b = dict(NUMPY, **(backends or {}))
    bad = {k: v for k, v in b.items() if k not in BACKENDS or v not in BACKENDS[k]}
    if bad:
        raise ValueError(f"unsupported devchain backends {bad}; choices {BACKENDS}")
    return b


def digi_rng(seed, backend):
    """The tracker_point digitiser stream: NumPy [seed, 0, 3], or JAX key(seed) folded with file 0, domain 3."""
    if backend == "numpy":
        return np.random.default_rng([seed, 0, 3])
    from jax import random as jr
    return jr.fold_in(jr.fold_in(jr.key(seed), 0), 3)


def digest(arrays):
    """SHA256 over the named arrays (sorted names, dtype, shape and bytes)."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(np.asarray(arrays[k]))
        h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _frame_arrays(df):
    return {c: df[c].to_numpy() for c in df.columns}


def straw_hits(d, rc, backend, settings="mc"):
    if backend == "numpy":
        return hitreco.straw_hits(d, rc, settings)
    from trkfastsim.reco import hitreco_jax
    return hitreco_jax.straw_hits(d, rc, settings)


def panel_hits(sh, backend, settings="mc"):
    if backend == "numpy":
        return combo.panel_hits(sh, settings)
    from trkfastsim.reco import combo_jax
    return combo_jax.panel_hits(sh, settings)


# JAX settings of this chain (module attributes, set for the stage and restored): fewer compiled shapes (SHAPE_GROWTH;
# step floats differ from the steps --backend jax defaults at rounding level), the loss sampler's fixed-trip loops
# unrolled inside the slab loop (SAMPLER_UNROLL; same outputs, fewer device launches), the device-resident slab loss
# (SLAB_KERNEL="device": mu2efastsim.slab_jax, one call per interior slab and per world chunk, the Triton kernel on
# CUDA and the XLA while loop elsewhere; same keys and draws, values equal to rounding, counts statistically equal to
# NumPy) and, for the JAX digitiser, the minion electron fill with 16 loop bodies per device iteration and
# comparison-pass searches (same outputs). SLICE_ROUNDS, SLAB_UNROLL and world_jax.SYNC_FREE_PAIRS stay opt-in.
DEVICE_STEP_SETTINGS = {"steps_jax.SHAPE_GROWTH": 8, "steps_jax.SAMPLER_UNROLL": True, "steps_jax.SLAB_KERNEL": "device",
                        "digi_clusters_jax.MINION_UNROLL": 16, "digi_clusters_jax.SEARCH_METHOD": "compare_all"}
DEVICE_SHAPE_GROWTH = DEVICE_STEP_SETTINGS["steps_jax.SHAPE_GROWTH"]


class _module_settings:
    """Set module attributes ("module.NAME" in trkfastsim.sim, or trkfastsim.digi for digi* modules) for a block,
    then restore them."""

    def __init__(self, settings):
        import importlib
        pkg = lambda m: "trkfastsim.digi." if m.startswith("digi") else "trkfastsim.sim."
        self.items = [(importlib.import_module(pkg(k.split(".")[0]) + k.split(".")[0]), k.split(".")[1], v)
                      for k, v in settings.items()]

    def __enter__(self):
        self.old = [(m, n, getattr(m, n)) for m, n, _ in self.items]
        for m, n, v in self.items:
            if not hasattr(m, n):
                raise ValueError(f"unknown step setting {m.__name__}.{n}")
            setattr(m, n, type(getattr(m, n))(v))
        return {f"{m.__name__.rsplit('.', 1)[1]}.{n}": getattr(m, n) for m, n, _ in self.items}

    def __exit__(self, *exc):
        for m, n, v in self.old:
            setattr(m, n, v)


def run(gen, ctx, seed, backends=None, sync=None, digi_compact=None, shape_growth=None, step_settings=None):
    """One pass over a generator table. Returns counts, per-stage wall seconds, per-stage output digests and the
    products (steps, digis, hits, combos, clusters, seeds, table). `sync`: called after each stage (e.g. to wait for
    the device) so stage times do not leak into the next stage. digi_compact (default: on for the JAX digitiser, off
    for NumPy, which then equals tracker_point): digitise in chunks of events with steps (fastdigi.digitize_steps).
    step_settings (JAX steps only; default DEVICE_STEP_SETTINGS, {} for the steps --backend jax defaults) and
    shape_growth (overrides its SHAPE_GROWTH entry)."""
    from trkfastsim.kit.kitpoint import generated_keys
    b = check_backends(backends)
    compact = (b["digi"] == "jax") if digi_compact is None else bool(digi_compact)
    sync = sync or (lambda: None)
    t_all = t = time.perf_counter()
    timing = {}
    keys = generated_keys(gen)
    settings = {}
    if b["steps"] == "jax":
        settings = dict(DEVICE_STEP_SETTINGS if step_settings is None else step_settings)
        if shape_growth:
            settings["steps_jax.SHAPE_GROWTH"] = int(shape_growth)
        g = settings.get("steps_jax.SHAPE_GROWTH", 2)
        if g < 2 or g & (g - 1):
            raise ValueError("shape_growth must be a power of two >= 2")
    digi_settings = {k: v for k, v in settings.items() if k.startswith("digi")} if b["digi"] == "jax" else {}
    settings = {k: v for k, v in settings.items() if not k.startswith("digi")}
    with _module_settings(settings) as used:
        sgs, parts = steps_mod.make_steps(gen, ctx.geom, np.random.default_rng([seed, 0]), stage="C", world=ctx.world,
                                          interior=ctx.interior, sliced_msc=True, all_muons=True, backend=b["steps"])
        if b["steps"] == "jax":
            from trkfastsim.sim import steps_jax
            used["slab_impl"] = steps_jax.slab_impl() if steps_jax.SLAB_KERNEL != "loop" else "loop"
    growth = used or None
    sync()
    timing["steps_s"] = time.perf_counter() - t
    sub = {f"steps.{k}_s": float(v) for k, v in getattr(steps_mod.make_steps, "timing", {}).items() if k != "total"}
    t = time.perf_counter()
    with _module_settings(digi_settings) as used_digi:
        _, digis, _ = fastdigi.digitize_steps(sgs, ctx.dc, digi_rng(seed, b["digi"]), keys=keys, chunk=1000,
                                              backend=b["digi"], compact=compact)
    if used_digi:
        growth = dict(growth or {}, **used_digi)
    d = fastdigi.digis_as_read(keys, digis, ctx.dc.nADCSamples)
    sync()
    timing["digi_s"] = time.perf_counter() - t
    t = time.perf_counter()
    sh = straw_hits(d, ctx.rc, b["hits"])
    sync()
    timing["hits_s"] = time.perf_counter() - t
    t = time.perf_counter()
    ph = panel_hits(sh, b["combos"])
    sync()
    timing["combos_s"] = time.perf_counter() - t
    stages = {}
    tb = tracktable.from_f2_digis(d, fastdigi.events_as_read(keys), ctx.rc, settings="mc", niter=4, all_fits=True,
                                  drift=ctx.drift, sh=sh, stages=stages)
    sync()
    for k in ("clusters_s", "seeds_s", "fit_s", "table_s"):
        timing[k] = stages["timing"][k]
    timing["total_s"] = time.perf_counter() - t_all
    timing.update(sub)
    good = tb["goodfit"].astype(bool)
    counts = {"generated_events": int(len(keys)), "primary_muons": int(len(parts)),
              "events_with_steps": int(len(sgs[["run", "subrun", "event"]].drop_duplicates())), "steps": int(len(sgs)),
              "digis": int(len(d["sid"])), "events_with_digis": int(len(np.unique(d["ev"]))),
              "hits": int(len(sh["ev"])), "combos": int(len(ph["ev"])), "clusters": int(len(stages["clusters"]["n"])),
              "seeds": int(tb["meta"]["n_seeds"]), "good_fits": int(good.sum()),
              "events_with_fit": int(len(np.unique(tb["ev"][good])))}
    seeds = stages["seeds"]
    table = {k: tb[k] for k in ("ev", "x", "cov", "chi2", "ndof", "status", "goodfit", "state", "hvalid")}
    hashes = {"steps": digest(_frame_arrays(sgs)), "digis": digest(digis), "hits": digest(sh), "combos": digest(ph),
              "clusters": digest(stages["clusters"]), "seeds": digest({k: v for k, v in seeds.items()
                                                                       if isinstance(v, np.ndarray)}),
              "table": digest(table)}
    return {"backends": b, "digi_compact": compact, "step_settings": growth, "seed": seed, "counts": counts, "timing": timing, "hashes": hashes,
            "products": {"steps": sgs, "digis": d, "hits": sh, "combos": ph, "clusters": stages["clusters"],
                         "seeds": seeds, "table": tb}}
