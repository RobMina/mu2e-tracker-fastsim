"""stepcheck -- diagnose the T-V1 path-length and eIon checks: what drives a fast-vs-G4 difference, and how big is noise.

Same selection as v1compare (primary-muon StrawGasSteps of events passing the PrimaryFilter tracker term). Three parts:

  1. path length: per bin of the crossing angle (|cos| between the step direction and the straw axis, from the step
     momentum and the GDML straw), the share of steps and the mean path, fast against G4; then the fast mean path
     reweighted to the G4 angle population. If the reweighted mean matches, the path difference comes from the
     muon direction population (world model, selection), not from the chords. Also per bin of impact parameter
     (distance of the step chord's midpoint from the straw axis) and of |p|.
  2. noise floor: every T-V1 statistic (path mean, eIon mean, eIon MPV, the binned eIon and path ratio bands) between
     the two halves of the G4 sample (shards in alternating order), so a fast-vs-G4 value can be compared with what
     G4 gives against itself; and the MPV estimators (stats.mpv argmax, stats.mpv_fit Moyal fit) at several bin widths.
  3. eIon band: the per-bin fast/G4 ratios and 2-sigma allowances behind the band, so the failing bins are visible.

    stepcheck FAST.root [...] --compare G4.root [...] --gdml mu2e_extracted_v04.gdml --outdir DIR
"""
import numpy as np

from mu2efastsim import plots, stats
from .. import geometry, tables as T
from .v1compare import _primary_steps, _select

COS_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
B_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0])         # mm, impact parameter (straw radius 2.5085)
P_EDGES = np.array([0.0, 500.0, 1000.0, 3000.0, 10000.0, 1e9])
MPV_BINS = (100, 200, 400)                                             # bins on 0-4 keV (v1compare uses 200)


def _features(s, geom):
    """Per step: |cos| to the straw axis, impact parameter of the chord midpoint, |p|."""
    js = (s.upanel.to_numpy() * 96 + s.straw.to_numpy()).astype(np.int64)
    u = s[["px", "py", "pz"]].to_numpy(float)
    p = np.linalg.norm(u, axis=1)
    u = u / np.maximum(p, 1e-12)[:, None]
    c = np.abs(np.einsum("ij,ij->i", u, geom.dir[js]))
    a = geometry.to_mu2e(s[["x0", "y0", "z0"]].to_numpy(float))
    b = geometry.to_mu2e(s[["x1", "y1", "z1"]].to_numpy(float))
    w = 0.5 * (a + b) - geom.mid[js]
    imp = np.linalg.norm(w - np.einsum("ij,ij->i", w, geom.dir[js])[:, None] * geom.dir[js], axis=1)
    return c, imp, p


def _binned(xf, xg, lf, lg, edges):
    rows = []
    bf, bg = np.digitize(xf, edges) - 1, np.digitize(xg, edges) - 1
    for k in range(len(edges) - 1):
        mf, mg = bf == k, bg == k
        if mg.sum() < 50:
            continue
        rows.append({"bin": [float(edges[k]), float(min(edges[k + 1], 1e6))],
                     "share": [float(mf.mean()), float(mg.mean())],
                     "len_mean": [float(lf[mf].mean()) if mf.any() else None, float(lg[mg].mean())],
                     "n": [int(mf.sum()), int(mg.sum())]})
    return rows


def _reweighted_mean(xf, xg, lf, edges):
    """Fast mean path with each angle bin weighted to the G4 share."""
    bf, bg = np.digitize(xf, edges) - 1, np.digitize(xg, edges) - 1
    tot = 0.0
    for k in range(len(edges) - 1):
        mf = bf == k
        if mf.any():
            tot += np.mean(bg == k) * lf[mf].mean()
    return float(tot)


def _v1stats(a, b):
    """The T-V1 statistics of sample a against sample b (each a selected step frame)."""
    ea, eb = 1e3 * a.eion.to_numpy(), 1e3 * b.eion.to_numpy()
    out = {"len_mean_ratio_minus_1": float(a.len.mean() / b.len.mean() - 1),
           "len_band_minus_2s": stats.ratio_band(a.len, b.len)["max_abs_dev_minus_2sigma"],
           "eion_mean_ratio_minus_1": float(ea.mean() / eb.mean() - 1),
           "eion_band_minus_2s": stats.ratio_band(ea, eb)["max_abs_dev_minus_2sigma"]}
    for nb in MPV_BINS:
        out[f"mpv{nb}_ratio_minus_1"] = float(stats.mpv(ea, 0, 4, nb) / stats.mpv(eb, 0, 4, nb) - 1)
        (fa, sa), (fb, sb) = stats.mpv_fit(ea, 0, 4, nb), stats.mpv_fit(eb, 0, 4, nb)
        out[f"mpvfit{nb}_ratio_minus_1"] = float(fa / fb - 1)
        out[f"mpvfit{nb}_ratio_err"] = float(fa / fb * np.hypot(sa / fa, sb / fb))
    return out


def _band_bins(fast, ref):
    r = stats.ratio_band(fast, ref)
    hf, hr, e = r["h_fast"], r["h_ref"], r["edges"]
    ok = hr >= 400
    ratio = (hf / len(fast)) / np.maximum(hr / len(ref), 1e-30)
    err = ratio * np.sqrt(1 / np.maximum(hf, 1) + 1 / np.maximum(hr, 1))
    return [{"lo": float(e[k]), "hi": float(e[k + 1]), "ratio": float(ratio[k]), "err": float(err[k]),
             "dev_minus_2s": float(max(abs(ratio[k] - 1) - 2 * err[k], 0)), "n_ref": int(hr[k])}
            for k in range(len(hr)) if ok[k]]


def run(inputs, outdir, compare=None, gdml=None, **kw):
    if not compare:
        raise SystemExit("stepcheck FAST --compare G4REF")
    from .steps import default_gdml
    geom = geometry.from_gdml(gdml or default_gdml())
    gpaths = sorted(T.expand(compare))
    f, g = _select(_primary_steps(T.expand(inputs))), _select(_primary_steps(gpaths))
    cf, bf, pf = _features(f, geom)
    cg, bg, pg = _features(g, geom)
    lf, lg = f.len.to_numpy(), g.len.to_numpy()
    out = {"steps": [int(len(f)), int(len(g))],
           "len_mean": [float(lf.mean()), float(lg.mean())],
           "len_mean_ratio_minus_1": float(lf.mean() / lg.mean() - 1),
           "len_mean_fast_reweighted_to_g4_angle": _reweighted_mean(cf, cg, lf, COS_EDGES),
           "by_cos_to_straw": _binned(cf, cg, lf, lg, COS_EDGES),
           "by_impact_mm": _binned(bf, bg, lf, lg, B_EDGES),
           "by_p": _binned(pf, pg, lf, lg, P_EDGES)}
    out["len_mean_reweighted_ratio_minus_1"] = out["len_mean_fast_reweighted_to_g4_angle"] / float(lg.mean()) - 1
    out["fast_vs_g4"] = _v1stats(f, g)
    h1, h2 = gpaths[0::2], gpaths[1::2]
    g1, g2 = _select(_primary_steps(h1)), _select(_primary_steps(h2))
    out["g4_half_vs_half"] = _v1stats(g1, g2)
    out["eion_band_bins"] = _band_bins(1e3 * f.eion.to_numpy(), 1e3 * g.eion.to_numpy())
    out["len_band_bins"] = _band_bins(lf, lg)
    _print(out)
    plots.write_json(outdir, "stepcheck.json", out)


def _print(o):
    print(f"steps fast/g4 {o['steps']}  len mean {o['len_mean'][0]:.4f}/{o['len_mean'][1]:.4f} ({o['len_mean_ratio_minus_1']:+.4f})"
          f"  fast reweighted to the G4 angle population: {o['len_mean_reweighted_ratio_minus_1']:+.4f}")
    for key in ("by_cos_to_straw", "by_impact_mm", "by_p"):
        print(f"  {key}:")
        for r in o[key]:
            lm = r["len_mean"]
            print(f"    [{r['bin'][0]:g}, {r['bin'][1]:g})  share {r['share'][0]:.4f}/{r['share'][1]:.4f}"
                  f"  len {lm[0] if lm[0] is None else round(lm[0], 3)}/{lm[1]:.3f}  n {r['n']}")
    print("statistic                     fast-vs-g4   g4-half-vs-half")
    for k in o["fast_vs_g4"]:
        a, b = o["fast_vs_g4"][k], o["g4_half_vs_half"][k]
        print(f"  {k:28s} {a if a is None else round(a, 4):>10}   {b if b is None else round(b, 4):>10}")
    for key in ("eion_band_bins", "len_band_bins"):
        print(f"  {key} (lo-hi ratio+-err  dev-2s):")
        print("   " + "  ".join(f"{r['lo']:.2f}-{r['hi']:.2f} {r['ratio']:.3f}+-{r['err']:.3f}"
                                + (f" !{r['dev_minus_2s']:.3f}" if r["dev_minus_2s"] > 0 else "") for r in o[key]))
