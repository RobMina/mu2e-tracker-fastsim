"""jaxcheck -- the jax backend of `steps` against numpy on the same primaries: statistics, reproducibility, timing.

Runs make_steps on each gen table with the numpy backend (the reference) and with the jax backend `--repeat` times on
the same seed (the first jax pass includes compilation; later passes are warm), then compares:
  * per-seed reproducibility: the jax passes (and --numpy-repeat numpy passes) bit for bit;
  * straws per muon (muons with steps), steps per (event, straw), per-straw and per-panel occupancy (chi2 / ndf);
  * eIon: Moyal MPV with its fit error, binned spectrum ratio (stats.ratio_band), mean;
  * step time after generation and path length; entry positions x0, y0, z0 (means, binned ratio);
  * the PrimaryFilter tracker term (>= 12 good steps of one particle), paired per event: both, numpy-only, jax-only,
    difference / sqrt(numpy-only + jax-only);
  * wall time per stage (make_steps.timing) for every pass, and the device.
The two arms share the primaries. Legacy step-level "sig" and occupancy statistics are descriptive only: repeated
steps of one event are correlated, particularly after independent world random streams. The supplemental
`event_clustered` block uses paired event-cluster influence covariance for step-weighted means and normalized panel
occupancy. The PrimaryFilter term and straws per muon are also compared paired;
  * --cross DIR: the jax pass against the jax table of the same seed made on another device (--save of that run): steps
    matched on (event, particle, straw), largest differences, the PrimaryFilter decisions;
  * --kernel-times: one more jax pass with every steps_jax kernel synchronised and timed (device + dispatch per kernel).
--concat runs all inputs as one make_steps call (throughput at a large batch), --lanes the muons per device chunk.
Writes jaxcheck.json and jaxcheck.txt to the outdir; --save also writes both step tables (fasttrk<stage>{np,jx}.*).

    jaxcheck gen.001400_00001001.root --gdml G.gdml --stage C --outdir DIR [--seed 1] [--repeat 2] [--no-exit-scatter]
"""
import collections
import hashlib
import json
import os
import platform
import time
import warnings

import numpy as np
import pandas as pd

from mu2efastsim import stats
from mu2efastsim import tables as core_tables
from .. import geometry, steps as st, surfaces, tables
from .steps import default_gdml

KEY = tables.KEY


def _err_mean(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def _cmp_mean(a, b):
    (ma, ea), (mb, eb) = _err_mean(a), _err_mean(b)
    e = float(np.hypot(ea, eb))
    return {"numpy": ma, "numpy_err": ea, "jax": mb, "jax_err": eb, "diff_sigma": (mb - ma) / e if e > 0 else float("nan")}


def _band(a, b, **kw):
    r = stats.ratio_band(b, a, **kw)
    return {k: r[k] for k in ("max_abs_dev", "max_abs_dev_minus_2sigma", "nbins_used")}


def _mpv(x):
    m, e = stats.mpv_fit(np.asarray(x) * 1e3, lo=0.0, hi=8.0)      # keV
    return m, e


def compare(a, b, gen):
    """Comparison block of two step tables (a = numpy, b = jax) made from the same gen table."""
    out = {"steps": [len(a), len(b)]}
    mu_a, mu_b = a.groupby(KEY + ["sp"]).size(), b.groupby(KEY + ["sp"]).size()
    out["muons_with_steps"] = [int(len(mu_a)), int(len(mu_b)),
                               float((len(mu_b) - len(mu_a)) / np.sqrt(len(mu_a) + len(mu_b)))]
    out["straws_per_muon"] = _cmp_mean(mu_a, mu_b)
    out["straws_per_muon_band"] = _band(mu_a.to_numpy(), mu_b.to_numpy(), min_ref=100)
    # paired per muon (both arms share the primaries; a muon without steps in one arm counts 0 there)
    pr = pd.concat([mu_a.rename("a"), mu_b.rename("b")], axis=1).fillna(0)
    dd = (pr.b - pr.a).to_numpy()
    out["straws_per_muon_paired_diff"] = {"mean": float(dd.mean()), "err": float(dd.std(ddof=1) / np.sqrt(len(dd))),
                                          "muons": int(len(dd))}
    sa, sb = a.groupby(KEY + ["sid"]).size(), b.groupby(KEY + ["sid"]).size()
    out["steps_per_event_straw"] = _cmp_mean(sa, sb)
    for unit, shift in (("straw", 0), ("panel", 7)):
        cnt = lambda s, n: pd.Series(s.sid.to_numpy().astype(np.int64) >> shift).value_counts().rename(n)
        occ = pd.concat([cnt(a, "a"), cnt(b, "b")], axis=1).fillna(0)
        na, nb = occ.a.sum(), occ.b.sum()
        m = (occ.a + occ.b) >= 20
        chi2 = float((((occ.a[m] / na - occ.b[m] / nb) ** 2) / (occ.a[m] / na ** 2 + occ.b[m] / nb ** 2)).sum())
        out[f"{unit}_occupancy_chi2_ndf"] = [chi2, int(m.sum() - 1)]
    for c in ("eion", "len"):
        out[c] = _cmp_mean(a[c], b[c])
        out[c + "_band"] = _band(a[c].to_numpy(), b[c].to_numpy())
    (ma, ea_), (mb, eb_) = _mpv(a.eion), _mpv(b.eion)
    out["eion_mpv_keV"] = {"numpy": ma, "numpy_err": ea_, "jax": mb, "jax_err": eb_,
                           "rel": mb / ma - 1, "diff_sigma": (mb - ma) / np.hypot(ea_, eb_)}
    t0 = gen.drop_duplicates(KEY).set_index(KEY).t
    dta = a.t.to_numpy() - t0.reindex(pd.MultiIndex.from_frame(a[KEY])).to_numpy()
    dtb = b.t.to_numpy() - t0.reindex(pd.MultiIndex.from_frame(b[KEY])).to_numpy()
    out["time_after_gen_ns"] = _cmp_mean(dta, dtb)
    out["time_after_gen_band"] = _band(dta, dtb)
    for c in ("x0", "y0", "z0"):
        out[c] = _cmp_mean(a[c], b[c])
        out[c + "_band"] = _band(a[c].to_numpy(), b[c].to_numpy())
    ev = gen[KEY].drop_duplicates()
    fa, fb = tables.trk_filter(a), tables.trk_filter(b)
    j = ev.set_index(KEY).join(fa.pass_trk.rename("a")).join(fb.pass_trk.rename("b")).fillna(False).astype(bool)
    both, ao, bo = int((j.a & j.b).sum()), int((j.a & ~j.b).sum()), int((~j.a & j.b).sum())
    out["primaryfilter_trk"] = {"events": int(len(j)), "numpy": both + ao, "jax": both + bo, "both": both,
                                "numpy_only": ao, "jax_only": bo,
                                "rel": (bo - ao) / max(both + ao, 1),
                                "rel_err": float(np.sqrt(ao + bo) / max(both + ao, 1)),
                                "diff_sigma": float((bo - ao) / np.sqrt(ao + bo)) if ao + bo else 0.0}
    out["event_clustered"] = event_clustered(a, b, gen)
    return out


def event_clustered(a, b, gen):
    """Paired event-cluster errors for step-weighted means and normalized panel occupancy.

    All steps and primaries of an event stay together. The influence-function covariance includes the random
    step denominators; zero-step generator events contribute zero influence, not extra independent steps.
    """
    ev = gen[KEY].drop_duplicates()
    n = len(ev)
    if n < 2 or not len(a) or not len(b):
        raise ValueError("event-cluster covariance requires at least two generated events and nonempty arms")
    if gen[KEY + ["t"]].isna().any().any() or not np.isfinite(gen.t.to_numpy()).all():
        raise ValueError("invalid generator event keys or times")
    index = pd.MultiIndex.from_frame(ev)
    for df in (a, b):
        if df[KEY + ["sid"]].isna().any().any() or not pd.MultiIndex.from_frame(df[KEY]).isin(index).all():
            raise ValueError("step event keys must belong to the generator table")
        if not np.isfinite(df[["t", "x0", "y0", "z0", "len", "eion"]].to_numpy()).all():
            raise ValueError("nonfinite step quantity in event-cluster covariance")
    correction = n / (n - 1)
    joined = pd.concat([a.groupby(KEY).size().rename("na"), b.groupby(KEY).size().rename("nb")], axis=1).fillna(0)
    na, nb = joined.na.to_numpy(), joined.nb.to_numpy()
    out = {"generated_events": n, "events_with_steps_union": len(joined), "means": {}}
    count_delta = nb - na
    count_mean = float(count_delta.sum() / n)
    count_se = float(np.sqrt(max(float(count_delta @ count_delta) - n * count_mean ** 2, 0.0) / (n * (n - 1))))
    out["steps_per_generated_event"] = {"numpy": float(na.sum() / n), "jax": float(nb.sum() / n),
                                       "difference": count_mean, "paired_event_se": count_se,
                                       "diff_sigma": count_mean / count_se if count_se else 0.0}
    t0 = gen.drop_duplicates(KEY).set_index(KEY).t
    out["time_origin"] = "first generator row per event (legacy jaxcheck convention)"
    for col in ("eion", "len", "t", "x0", "y0", "z0"):
        sums = []
        means = []
        for df, count in ((a, na), (b, nb)):
            val = df[col].to_numpy()
            if col == "t":
                val = val - t0.reindex(pd.MultiIndex.from_frame(df[KEY])).to_numpy()
            grouped = df[KEY].assign(value=val).groupby(KEY).value.sum().reindex(joined.index, fill_value=0).to_numpy()
            mean = grouped.sum() / count.sum()
            sums.append((grouped - mean * count) / count.sum())
            means.append(float(mean))
        se = float(np.sqrt(correction * np.sum((sums[1] - sums[0]) ** 2)))
        delta = means[1] - means[0]
        out["means"]["time_after_gen_ns" if col == "t" else col] = {
            "numpy": means[0], "jax": means[1], "difference": delta, "paired_event_se": se,
            "diff_sigma": delta / se if se else (0.0 if delta == 0 else float(np.copysign(np.inf, delta)))}
    counts = []
    for df in (a, b):
        counts.append(df[KEY].assign(panel=df.sid.to_numpy().astype(np.int64) >> 7).groupby(KEY + ["panel"]).size().unstack("panel", fill_value=0))
    panels = counts[0].columns.union(counts[1].columns)
    ca, cb = (q.reindex(index=joined.index, columns=panels, fill_value=0).to_numpy(dtype=float) for q in counts)
    fa, fb = ca.sum(0) / na.sum(), cb.sum(0) / nb.sum()
    influence = (cb - nb[:, None] * fb) / nb.sum() - (ca - na[:, None] * fa) / na.sum()
    cov = correction * influence.T @ influence
    delta = fb - fa
    vals, vec = np.linalg.eigh(cov)
    rank_tol = float(vals.max(initial=0) * 1e-10)
    negative_tol = float(vals.max(initial=0) * 1e-12)
    if (vals < -negative_tol).any():
        raise ValueError("event-cluster covariance is not positive semidefinite")
    keep = vals > rank_tol
    null_delta = float(np.linalg.norm(vec[:, ~keep].T @ delta))
    unresolved = null_delta >= 1e-12
    chi2 = float(np.sum((vec[:, keep].T @ delta) ** 2 / vals[keep]))
    from scipy.stats import chi2 as chi2_dist
    pvalue = None if unresolved else (float(chi2_dist.sf(chi2, keep.sum())) if keep.any() else 1.0)
    out["panel_occupancy"] = {"chi2": chi2, "rank": int(keep.sum()), "discarded_mode_difference": null_delta,
                              "unresolved_discarded_modes": unresolved, "eigenvalues": vals.tolist(),
                              "rank_tolerance": rank_tol, "negative_tolerance": negative_tol,
                              "pvalue": pvalue,
                              "max_abs_pull": float(np.max(np.abs(delta) / np.sqrt(np.maximum(np.diag(cov), 1e-300)), initial=0))}
    return out


def cross_backend(a, b, gen):
    """Same seed, two JAX devices (a = this run, b = the --cross table): per-step match on (event, particle, straw),
    largest differences, and the statistical comparison."""
    on = KEY + ["sp", "sid"]
    m = a.merge(b, on=on, how="outer", suffixes=("_a", "_b"), indicator=True)
    both = m[m._merge == "both"]
    out = {"steps": [len(a), len(b)], "matched": int(len(both)), "only_this": int((m._merge == "left_only").sum()),
           "only_cross": int((m._merge == "right_only").sum()),
           "identical_frames": len(a) == len(b) and all(np.array_equal(a[c].to_numpy(), b[c].to_numpy())
                                                        for c in b.columns if c in a)}
    unmatched = m[m._merge != "both"]
    out["unmatched_events"] = len(unmatched[KEY].drop_duplicates())
    out["unmatched_keys_first100"] = unmatched[on + ["_merge"]].head(100).astype({"_merge": str}).to_dict("records")
    for c in ("x0", "y0", "z0", "t", "eion", "len"):
        dlt = np.abs(both[c + "_a"].to_numpy() - both[c + "_b"].to_numpy())
        out[f"max_abs_diff_{c}"] = float(dlt.max()) if len(dlt) else 0.0
        out[f"frac_exact_{c}"] = float(np.mean(dlt == 0)) if len(dlt) else 1.0
        out[f"abs_diff_quantiles_{c}"] = np.quantile(dlt, [.5, .9, .99, .999, 1]).tolist() if len(dlt) else [0.0] * 5
        if len(dlt):
            worst = both.iloc[int(np.argmax(dlt))]
            out[f"worst_key_{c}"] = {k: int(worst[k]) for k in on}
    ev = lambda s: set(map(tuple, tables.trk_filter(s).query("pass_trk").reset_index()[KEY].to_numpy()))
    ea, eb = ev(a), ev(b)
    out["primaryfilter_trk"] = {"this": len(ea), "cross": len(eb), "both": len(ea & eb)}
    out["stats"] = compare(b, a, gen)
    return out


def _kernel_times(sj, run_once):
    """Run `run_once` with world/interior kernels synchronized; times include dispatch and can nest on retracing."""
    import jax
    tsum, ncall = collections.defaultdict(float), collections.Counter()
    from .. import world_jax
    orig = {(mod, n): f for mod in (sj, world_jax) for n, f in vars(mod).items()
            if n.startswith("_") and not n.startswith("__") and hasattr(f, "lower")}

    def wrap(n, f):
        def g(*a, **k):
            t = time.perf_counter()
            r = jax.block_until_ready(f(*a, **k))
            tsum[n] += time.perf_counter() - t
            ncall[n] += 1
            return r
        return g

    for (mod, n), f in orig.items():
        setattr(mod, n, wrap(mod.__name__.rsplit(".", 1)[-1] + "." + n, f))
    try:
        t = time.perf_counter()
        run_once()
        wall = time.perf_counter() - t
    finally:
        for (mod, n), f in orig.items():
            setattr(mod, n, f)
    ks = sorted(tsum, key=lambda n: -tsum[n])
    return {"wall": wall, "kernels_s": float(sum(tsum.values())), "calls": int(sum(ncall.values())),
            "per_kernel": {n: [tsum[n], ncall[n]] for n in ks}}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same(x, y):
    try:
        pd.testing.assert_frame_equal(x, y, check_exact=True)
        return True
    except AssertionError:
        return False


def run(inputs, outdir, seed=1, gdml=None, stage="C", repeat=2, numpy_repeat=1, no_exit_scatter=False, save=False,
        sliced_msc=True, concat=False, lanes=None, cross=None, kernel_times=False, **kw):
    warnings.filterwarnings("ignore", message=".*donated.*")
    import jax
    from .. import steps_jax
    if lanes:
        steps_jax.LANES = int(lanes)
    os.makedirs(outdir, exist_ok=True)
    gdml = gdml or default_gdml()
    t = time.perf_counter()
    geom = geometry.from_gdml(gdml)
    world = st.load_world(gdml, geom) if stage == "C" else None
    interior = st.TrackerInterior(gdml, geom) if stage == "C" else None
    setup_s = time.perf_counter() - t
    kws = dict(stage=stage, world=world, interior=interior, sliced_msc=sliced_msc, exit_scatter=not no_exit_scatter)
    report = {"device": str(jax.devices()[0]), "backend": jax.default_backend(), "host": platform.node(),
              "cpu": platform.processor() or platform.machine(), "setup_s": setup_s, "lanes": steps_jax.LANES,
              "files": []}
    lines = [f"# jaxcheck on {report['device']} ({report['host']}), stage {stage}, seed {seed}, setup {setup_s:.1f} s, "
             f"lanes {steps_jax.LANES}"]
    files = tables.expand(inputs)
    if not files:
        raise ValueError("jaxcheck needs input generator tables")
    from .. import world_jax
    from mu2efastsim import eloss_jax, transport
    report["versions"] = {"jax": jax.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    report["settings"] = {"seed": int(seed), "stage": stage, "sliced_msc": sliced_msc,
                          "exit_scatter": not no_exit_scatter, "concat": concat,
                          "world_lanes_max": world_jax.WORLD_LANES}
    report["sha256"] = {p: _sha256(p) for p in [gdml, *files, st.__file__, steps_jax.__file__,
                                                 world_jax.__file__, eloss_jax.__file__, transport.__file__]}
    items = [(os.path.basename(f), lambda f=f: tables.read(f, "gen")) for f in files]
    if concat:   # one make_steps call on all inputs (throughput at a large batch; KEY keeps the events apart)
        items = [(f"concat{len(files)}." + os.path.basename(files[0]),
                  lambda: pd.concat([tables.read(f, "gen") for f in files], ignore_index=True))]
    for i, (fname, load) in enumerate(items):
        gen = load()
        rec = {"file": fname, "generated": int(len(gen[KEY].drop_duplicates())),
               "generator_rows": int(len(gen)), "numpy": [], "jax": []}
        outs = {"numpy": [], "jax": []}
        for backend, n in (("numpy", numpy_repeat), ("jax", repeat)):
            for r in range(n):
                t = time.perf_counter()
                sgs, parts = st.make_steps(gen, geom, np.random.default_rng([int(seed), i]), backend=backend, **kws)
                wall = time.perf_counter() - t
                pv = getattr(st.make_steps, "path_vertices", None)
                surf = surfaces.surface_steps(sgs, pv) if pv is not None else None
                outs[backend].append((sgs, parts, surf))
                tm = dict(st.make_steps.timing, wall=wall)
                rec[backend].append({"timing": tm, "stats": {k: int(v) for k, v in
                                                             getattr(st.make_steps, "interior_stats", {}).items()}})
                lines.append(f"{rec['file']} {backend} pass {r}: " + ", ".join(f"{k} {v:.3f}" for k, v in tm.items())
                             + f" s; {len(sgs)} steps")
                print(lines[-1], flush=True)
                checkpoint = os.path.join(outdir, "jaxcheck.partial.json")
                with open(checkpoint + ".tmp", "w") as stream:
                    json.dump(dict(report, files=report["files"] + [rec], incomplete=True), stream, indent=1, default=float)
                os.replace(checkpoint + ".tmp", checkpoint)
        if kernel_times:
            kt = rec["kernel_times"] = _kernel_times(steps_jax, lambda: st.make_steps(
                gen, geom, np.random.default_rng([int(seed), i]), backend="jax", **kws))
            lines.append(f"  kernel times (synchronised pass): wall {kt['wall']:.3f} s, in kernels {kt['kernels_s']:.3f} s, "
                         f"{kt['calls']} calls; " + ", ".join(f"{n} {v[0]:.3f} s/{v[1]}" for n, v in
                                                            list(kt["per_kernel"].items())[:10]))
        rec["reproducible"] = {b: all(_same(outs[b][0][k], o[k]) for o in outs[b][1:] for k in (0, 1))
                               for b in outs if len(outs[b]) > 1}
        if all(len(outs[b]) and outs[b][0][2] is not None for b in outs):
            rec["surfsteps"] = [len(outs["numpy"][0][2]), len(outs["jax"][0][2])]
        if outs["numpy"] and outs["jax"]:
            rec["compare"] = compare(outs["numpy"][0][0], outs["jax"][0][0], gen)
        xf = os.path.join(cross, f"fasttrk{stage}jx." + fname.replace("gen_tables.", "").replace("gen.", "")) if cross else None
        if xf and outs["jax"] and os.path.exists(xf):
            rec["cross"] = cross_backend(outs["jax"][0][0], tables.read(xf, "sgsteps"), gen)
            x = rec["cross"]
            lines.append(f"  cross-device (this jax vs {xf}): steps {x['steps']}, matched {x['matched']}, only "
                         f"{x['only_this']}/{x['only_cross']}, identical {x['identical_frames']}, max |d| x0 "
                         f"{x['max_abs_diff_x0']:.2g} mm, t {x['max_abs_diff_t']:.2g} ns, eion {x['max_abs_diff_eion']:.2g} MeV; "
                         f"exact eion {x['frac_exact_eion']:.4f}; PrimaryFilter trk {x['primaryfilter_trk']}")
        report["files"].append(rec)
        if save:
            for b, tag in (("numpy", "np"), ("jax", "jx")):
                if not outs[b]:
                    continue
                sgs, parts, surf = outs[b][0]
                name = f"fasttrk{stage}{tag}." + fname.replace("gen_tables.", "").replace("gen.", "")
                core_tables.write(os.path.join(outdir, name), {"sgsteps": sgs, "particles": parts, "gen": gen,
                                                               "surfsteps": surf}, as_ttree=True)
        lines += _summary(rec) if "compare" in rec else [f"  reproducible per seed: {rec['reproducible']}"]
        print("\n".join(lines[-40:]), flush=True)
    with open(os.path.join(outdir, "jaxcheck.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    with open(os.path.join(outdir, "jaxcheck.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    partial_path = os.path.join(outdir, "jaxcheck.partial.json")
    if os.path.exists(partial_path):
        os.remove(partial_path)


def _summary(rec):
    c = rec["compare"]
    f = lambda d, u="": f"{d['numpy']:.5g}{u} +- {d['numpy_err']:.2g} vs {d['jax']:.5g} +- {d['jax_err']:.2g} ({d['diff_sigma']:+.1f} sig)"
    pf = c["primaryfilter_trk"]
    mp = c["eion_mpv_keV"]
    out = [f"  reproducible per seed: {rec['reproducible']}",
           f"  steps {c['steps'][0]} vs {c['steps'][1]}; muons with steps {c['muons_with_steps'][0]} vs "
           f"{c['muons_with_steps'][1]} ({c['muons_with_steps'][2]:+.1f} sig)",
           f"  straws per muon {f(c['straws_per_muon'])}; band {c['straws_per_muon_band']['max_abs_dev_minus_2sigma']}"
           + (f"; paired jax - numpy {c['straws_per_muon_paired_diff']['mean']:+.4f} +- "
              f"{c['straws_per_muon_paired_diff']['err']:.4f} over {c['straws_per_muon_paired_diff']['muons']} muons"
              if "straws_per_muon_paired_diff" in c else ""),
           f"  steps per (event, straw) {f(c['steps_per_event_straw'])}; occupancy chi2/ndf straws (>= 20 steps) "
           f"{c['straw_occupancy_chi2_ndf'][0]:.1f}/{c['straw_occupancy_chi2_ndf'][1]}, panels "
           f"{c['panel_occupancy_chi2_ndf'][0]:.1f}/{c['panel_occupancy_chi2_ndf'][1]}",
           f"  eIon MPV {mp['numpy']:.4f} +- {mp['numpy_err']:.4f} vs {mp['jax']:.4f} +- {mp['jax_err']:.4f} keV "
           f"({mp['rel'] * 100:+.2f} %, {mp['diff_sigma']:+.1f} sig); mean {f(c['eion'])}; band {c['eion_band']['max_abs_dev_minus_2sigma']}",
           f"  path {f(c['len'])}; band {c['len_band']['max_abs_dev_minus_2sigma']}",
           f"  time after gen {f(c['time_after_gen_ns'])}; band {c['time_after_gen_band']['max_abs_dev_minus_2sigma']}"]
    out += [f"  {x} {f(c[x])}; band {c[x + '_band']['max_abs_dev_minus_2sigma']}" for x in ("x0", "y0", "z0")]
    out.append(f"  PrimaryFilter trk (paired, {pf['events']} events): numpy {pf['numpy']} jax {pf['jax']} both {pf['both']} "
               f"numpy-only {pf['numpy_only']} jax-only {pf['jax_only']}: {pf['rel'] * 100:+.2f} % +- {pf['rel_err'] * 100:.2f} % "
               f"({pf['diff_sigma']:+.1f} sig)")
    if "event_clustered" in c:
        ec = c["event_clustered"]
        out.append("  paired event-cluster mean pulls: " + ", ".join(f"{k} {v['diff_sigma']:+.2f}" for k, v in ec["means"].items()))
        pc = ec["panel_occupancy"]
        pv = f"{pc['pvalue']:.4g}" if pc['pvalue'] is not None else "undefined (discarded-mode difference)"
        out.append(f"  paired event-cluster panel occupancy chi2/rank {pc['chi2']:.2f}/{pc['rank']}, p={pv}")
        if "steps_per_generated_event" in ec:
            sc = ec["steps_per_generated_event"]
            out.append(f"  steps/generated event paired difference {sc['difference']:+.6g} +- {sc['paired_event_se']:.3g} ({sc['diff_sigma']:+.2f} sig)")
    if "surfsteps" in rec:
        out.append(f"  surfsteps {rec['surfsteps'][0]} vs {rec['surfsteps'][1]}")
    return out
