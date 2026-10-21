"""slowcheck -- where the fast sim loses slow muons in the tracker: per-muon straw counts, entry state, in-tracker loss.

Unpaired, per second of live time (the reference may be the MDC2025as production). The reference dts keep only
PrimaryFilter-accepted events (tracker OR CRV term), so its muons with fewer than 12 good tracker steps are
under-counted: only the >= 12 class (and its step-count distribution) is a like-for-like comparison. For the MC primary
muon of every
event with at least one good StrawGasStep (PrimaryFilter good-step definition), binned by |p| at its first straw:
  1. rate of muons with any good step, with 1-11 and with >= 12 good steps (the tracker term), fast/G4;
  2. the good-step count distribution (quantiles), the fraction of muons whose last good step is far above the
     bottom of the straw stack (stopped or left the stack sideways), and |p| at the last straw;
  3. entry state at the first straw: |cos| to vertical (uy), x and z (Mu2e frame) quantiles;
  4. in-tracker loss: |p| first minus |p| last, per straw crossed;
  5. for G4 tables with particles end points: the share of slow primary muons that stop inside the tracker region
     (end kinetic energy < 1 MeV) against the fast muons that run out of good steps.
A deficit concentrated in muons with few steps and a normal entry distribution points at the transport inside the
tracker; a deficit already in the any-step rate with a shifted entry distribution points above the tracker.

    slowcheck FAST.root [...] --compare G4.root [...] --outdir DIR
"""
import numpy as np
import pandas as pd

from mu2efastsim import eloss, plots
from mu2efastsim.tables import KEY
from .. import geometry, tables as T

P_EDGES = np.array([0.0, 300.0, 500.0, 750.0, 1000.0, 1500.0, 3000.0, 1e9])
QS = (0.1, 0.25, 0.5, 0.75, 0.9)


N_EDGES = np.array([12, 14, 16, 18, 20, 24, 30, 1000])


def _muons(paths):
    s = T.read_many(paths, "sgsteps")
    p = T.read_many(paths, "particles")
    pm = p[(p.ismcprim == 1) & (p.pdg.abs() == 13)][KEY + ["key"] + [c for c in ("ee", "ey") if c in p.columns]]
    s = s.merge(pm[KEY + ["key"]].rename(columns={"key": "sp"}), on=KEY + ["sp"])
    s["p"] = np.sqrt(s.px ** 2 + s.py ** 2 + s.pz ** 2)
    s["good"] = T.good_steps(s)
    ga = s.groupby(KEY).agg(n_all=("good", "size"), n_low_eion=("eion", lambda e: int((e <= T.FILTER_MIN_STEP_EION).sum())),
                            n_low_p=("p", lambda x: int((x <= T.FILTER_MIN_MOM).sum())))
    s = s[s.good].sort_values(KEY + ["t"])
    s["p"] = np.sqrt(s.px ** 2 + s.py ** 2 + s.pz ** 2)
    pos = geometry.to_mu2e(s[["x0", "y0", "z0"]].to_numpy())
    s["X"], s["Y"], s["Z"] = pos[:, 0], pos[:, 1], pos[:, 2]
    g = s.groupby(KEY, sort=False)
    m = pd.DataFrame({"n": g.size(), "p_first": g.p.first(), "p_last": g.p.last(), "uy": (g.py.first() / g.p.first()).abs(),
                      "x": g.X.first(), "z": g.Z.first(), "y_first": g.Y.first(), "y_last": g.Y.last()}).reset_index()
    m = m.merge(ga.reset_index(), on=KEY, how="left")
    if "ee" in pm.columns:
        end = pm.groupby(KEY).first().reset_index()
        end["stopped"] = (end.ee - eloss.MMU) < 1.0
        m = m.merge(end[KEY + ["stopped", "ey"]], on=KEY, how="left")
    return m


def _q(x):
    return np.quantile(x, QS).round(3).tolist() if len(x) else []


def run(inputs, outdir, compare=None, **kw):
    if not compare:
        raise SystemExit("slowcheck FAST --compare G4REF")
    fp, rp = T.expand(inputs), T.expand(compare)
    lt_f, lt_r = T.livetime(fp), T.livetime(rp)
    f, r = _muons(fp), _muons(rp)
    y_bottom = min(f.y_last.min(), r.y_last.min())
    out = {"livetime_s": [lt_f, lt_r], "muons_any_step": [int(len(f)), int(len(r))], "by_p_first": []}
    bf = np.clip(np.digitize(f.p_first, P_EDGES) - 1, 0, len(P_EDGES) - 2)
    br = np.clip(np.digitize(r.p_first, P_EDGES) - 1, 0, len(P_EDGES) - 2)
    for k in range(len(P_EDGES) - 1):
        a, b = f[bf == k], r[br == k]
        if len(b) < 20:
            continue
        row = {"p": [float(P_EDGES[k]), float(min(P_EDGES[k + 1], 1e6))]}
        for lab, sel in (("any", lambda x: x.n >= 1), ("few_1_11", lambda x: (x.n >= 1) & (x.n < 12)),
                         ("pass_ge12", lambda x: x.n >= 12)):
            na, nb = int(sel(a).sum()), int(sel(b).sum())
            rr = (na / lt_f) / max(nb / lt_r, 1e-30)
            row[f"rate_{lab}"] = {"n": [na, nb], "ratio_minus_1": float(rr - 1),
                                  "err": float(rr * np.sqrt(1 / max(na, 1) + 1 / max(nb, 1)))}
        row["n_good_q"] = [_q(a.n), _q(b.n)]
        row["uy_q"] = [_q(a.uy), _q(b.uy)]
        row["x_q"] = [_q(a.x), _q(b.x)]
        row["z_q"] = [_q(a.z), _q(b.z)]
        row["dp_per_straw_q"] = [_q((a.p_first - a.p_last) / np.maximum(a.n - 1, 1)),
                                 _q((b.p_first - b.p_last) / np.maximum(b.n - 1, 1))]
        row["share_ending_above_stack_bottom_300mm"] = [float(np.mean(a.y_last > y_bottom + 300)),
                                                        float(np.mean(b.y_last > y_bottom + 300))]
        if "stopped" in b.columns:
            row["g4_share_stopped"] = float(b.stopped.fillna(False).astype(bool).mean())
        # selected muons only (>= 12 good steps): the class both tables select the same way
        sa, sb = a[a.n >= 12], b[b.n >= 12]
        ha, _ = np.histogram(sa.n, N_EDGES)
        hb, _ = np.histogram(sb.n, N_EDGES)
        row["pass_rate_by_ngood"] = [{"n": [int(N_EDGES[j]), int(N_EDGES[j + 1]) - 1], "count": [int(ha[j]), int(hb[j])],
                                      "ratio_minus_1": float((ha[j] / lt_f) / max(hb[j] / lt_r, 1e-30) - 1),
                                      "err": float(np.sqrt(1 / max(ha[j], 1) + 1 / max(hb[j], 1)))}
                                     for j in range(len(N_EDGES) - 1)]
        row["pass_n_all_q"] = [_q(sa.n_all), _q(sb.n_all)]
        row["pass_p_last_q"] = [_q(sa.p_last), _q(sb.p_last)]
        row["pass_dp_track_q"] = [_q(sa.p_first - sa.p_last), _q(sb.p_first - sb.p_last)]
        row["pass_share_p_last_below_100"] = [float(np.mean(sa.p_last < 100)), float(np.mean(sb.p_last < 100))]
        row["pass_dy_track_q"] = [_q(sa.y_first - sa.y_last), _q(sb.y_first - sb.y_last)]
        row["pass_nongood_share"] = [float((sa.n_all - sa.n).sum() / max(sa.n_all.sum(), 1)),
                                     float((sb.n_all - sb.n).sum() / max(sb.n_all.sum(), 1))]
        row["pass_low_eion_share"] = [float(sa.n_low_eion.sum() / max(sa.n_all.sum(), 1)),
                                      float(sb.n_low_eion.sum() / max(sb.n_all.sum(), 1))]
        row["pass_low_p_share"] = [float(sa.n_low_p.sum() / max(sa.n_all.sum(), 1)),
                                   float(sb.n_low_p.sum() / max(sb.n_all.sum(), 1))]
        out["by_p_first"].append(row)
    print(f"live time fast {lt_f:.2f} s, G4 {lt_r:.2f} s; muons with a good step fast {len(f):,}, G4 {len(r):,}")
    print(f"{'|p| first':>12s} {'any step':>17s} {'1-11 steps':>17s} {'>=12 steps':>17s}  n_good q50 f/g  "
          f"uy q50 f/g      dp/straw q50 f/g  end>bottom+300 f/g  G4 stopped")
    for x in out["by_p_first"]:
        cell = lambda k: f"{x[k]['ratio_minus_1']:+.3f}+-{x[k]['err']:.3f}"
        print(f"{x['p'][0]:5.0f}-{x['p'][1]:6.0f} {cell('rate_any'):>17s} {cell('rate_few_1_11'):>17s} {cell('rate_pass_ge12'):>17s}"
              f"  {x['n_good_q'][0][2]:4.0f}/{x['n_good_q'][1][2]:4.0f}   {x['uy_q'][0][2]:.3f}/{x['uy_q'][1][2]:.3f}"
              f"   {x['dp_per_straw_q'][0][2]:6.2f}/{x['dp_per_straw_q'][1][2]:6.2f}"
              f"   {x['share_ending_above_stack_bottom_300mm'][0]:.3f}/{x['share_ending_above_stack_bottom_300mm'][1]:.3f}"
              f"   {x.get('g4_share_stopped', float('nan')):.3f}")
    print("NOTE: the any-step and 1-11 columns are only fair against a reference that keeps every event; the production"
          " and g4ref dts keep only PrimaryFilter-accepted events (tracker OR CRV), so G4 muons with < 12 tracker steps are"
          " under-counted there. The >= 12 class is selected identically in both.")
    print("selected muons (>= 12 good steps): rate fast/G4 - 1 by good-step count, and the share of muon steps that are"
          " not good (eIon <= 1 eV / |p| <= 50 MeV/c):")
    for x in out["by_p_first"]:
        print(f"  {x['p'][0]:5.0f}-{x['p'][1]:6.0f}  " + "  ".join(
            f"{y['n'][0]}-{y['n'][1]}: {y['ratio_minus_1']:+.2f}+-{y['err']:.2f}" for y in x["pass_rate_by_ngood"])
              + f"   non-good {x['pass_nongood_share'][0]:.4f}/{x['pass_nongood_share'][1]:.4f}"
              f" (eIon {x['pass_low_eion_share'][0]:.4f}/{x['pass_low_eion_share'][1]:.4f},"
              f" |p| {x['pass_low_p_share'][0]:.4f}/{x['pass_low_p_share'][1]:.4f})"
              f"  all-steps q50 {x['pass_n_all_q'][0][2]:.0f}/{x['pass_n_all_q'][1][2]:.0f}")
    print("selected muons: |p| at the last good straw, loss first->last straw, vertical extent of the track [mm]:")
    for x in out["by_p_first"]:
        pl, dp, dy = x["pass_p_last_q"], x["pass_dp_track_q"], x["pass_dy_track_q"]
        print(f"  {x['p'][0]:5.0f}-{x['p'][1]:6.0f}  p_last q10/50 f {pl[0][0]:.0f}/{pl[0][2]:.0f} g {pl[1][0]:.0f}/{pl[1][2]:.0f}"
              f"  dp q50/q90 f {dp[0][2]:.1f}/{dp[0][4]:.1f} g {dp[1][2]:.1f}/{dp[1][4]:.1f} MeV"
              f"  p_last<100 f {x['pass_share_p_last_below_100'][0]:.3f} g {x['pass_share_p_last_below_100'][1]:.3f}"
              f"  dy q50/q90 f {dy[0][2]:.0f}/{dy[0][4]:.0f} g {dy[1][2]:.0f}/{dy[1][4]:.0f}")
    print("counts n fast/g4 (any, few, pass) per bin:", [(int(x['p'][0]), x['rate_any']['n'], x['rate_few_1_11']['n'],
                                                          x['rate_pass_ge12']['n']) for x in out["by_p_first"]])
    plots.write_json(outdir, "slowcheck.json", out)
