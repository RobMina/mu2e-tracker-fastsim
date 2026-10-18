"""elosscheck -- the primary muon's own eIon per straw: fast Urban sampler against Geant4, per path-length bin.

Reference: StrawGasStepTableDumper tables; steps of the MC primary muon (production MakeStrawGasSteps keeps
KeepDeltasModule = g4run, so no delta is folded into the muon step: its eIon is the muon's restricted loss with the
990 eV tracker cut, the quantity steps.make_steps samples).

Two tests, so population and model are separated:
  1. model on G4 kinematics: for every G4 muon step, resample eIon with mu2efastsim.eloss.sample_loss at the step's
     own |p| and path length; variants split the path into n equal sub-steps and sum (G4 may split a gas crossing);
     compared per path-length bin by quantiles and the share of steps in the 2.2-3.3 keV window.
  2. population (with --fast): the fast run's muon steps against G4 in path length and |p|.
"""
import numpy as np

from mu2efastsim import eloss, plots, stats
from .. import steps as st, tables as T

LEN_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 8.0, 1e9])
QS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
WINDOW = (2.2, 3.3)   # keV


def _muon_steps(paths):
    s = T.read_many(paths, "sgsteps")
    p = T.read_many(paths, "particles", T.KEY + ["key", "pdg", "ismcprim"]).rename(columns={"key": "sp"})
    s = s.merge(p, on=T.KEY + ["sp"], how="inner")
    s = s[(s.ismcprim == 1) & (s.pdg.abs() == 13)].reset_index(drop=True)
    s["p"] = np.sqrt(s.px ** 2 + s.py ** 2 + s.pz ** 2)
    return s


def _summary(e_kev):
    e = np.asarray(e_kev)
    if not len(e):
        return {"n": 0}
    return {"n": int(len(e)), "mean": float(e.mean()), "q": np.quantile(e, QS).round(4).tolist(),
            "window_share": float(np.mean((e >= WINDOW[0]) & (e < WINDOW[1]))), "zero_share": float(np.mean(e <= 0))}


def _sample(rng, p, length, nsplit, tcut):
    if nsplit == 1:
        return eloss.sample_loss(rng, st.STRAW_GAS, p, length, tcut)
    tot = np.zeros(len(p))
    for _ in range(nsplit):
        tot += eloss.sample_loss(rng, st.STRAW_GAS, p, length / nsplit, tcut)
    return tot


def run(inputs, outdir, fast=None, seed=1, splits="1,2,4", **kw):
    g = _muon_steps(T.expand(inputs))
    rng = np.random.default_rng(int(seed))
    tcut = st.delta_cut_mev()
    eg = 1e3 * g.eion.to_numpy()
    lb = np.clip(np.digitize(g.len, LEN_EDGES) - 1, 0, len(LEN_EDGES) - 2)
    nsp = [int(x) for x in str(splits).split(",")]
    model = {n: 1e3 * _sample(rng, g.p.to_numpy(), g.len.to_numpy(), n, tcut) for n in nsp}
    out = {"g4_muon_steps": int(len(g)), "tcut_MeV": tcut, "all": {"g4": _summary(eg)}, "by_len": []}
    for n, e in model.items():
        b = stats.ratio_band(e, eg)
        out["all"][f"model_split{n}"] = {**_summary(e), "band": b["max_abs_dev"], "band_minus_2sigma": b["max_abs_dev_minus_2sigma"],
                                          "mean_ratio_minus_1": float(e.mean() / eg.mean() - 1)}
    for k in range(len(LEN_EDGES) - 1):
        m = lb == k
        if m.sum() < 200:
            continue
        row = {"len_mm": [float(LEN_EDGES[k]), float(LEN_EDGES[k + 1])], "g4": _summary(eg[m])}
        for n, e in model.items():
            row[f"model_split{n}"] = _summary(e[m])
        out["by_len"].append(row)
    if fast:
        f = _muon_steps(T.expand(fast))
        ef = 1e3 * f.eion.to_numpy()
        out["population"] = {
            "fast_steps": int(len(f)), "len_band": stats.ratio_band(f.len, g.len)["max_abs_dev_minus_2sigma"],
            "p_band": stats.ratio_band(f.p, g.p)["max_abs_dev_minus_2sigma"],
            "len_q": {"fast": np.quantile(f.len, QS).round(3).tolist(), "g4": np.quantile(g.len, QS).round(3).tolist()},
            "p_q": {"fast": np.quantile(f.p, QS).round(0).tolist(), "g4": np.quantile(g.p, QS).round(0).tolist()},
            "fast_eion": _summary(ef),
            "fast_eion_band_minus_2sigma": stats.ratio_band(ef, eg)["max_abs_dev_minus_2sigma"]}
    _print(out, nsp)
    plots.write_json(outdir, "elosscheck.json", out)


def _print(out, nsp):
    a = out["all"]
    print(f"G4 muon steps: {out['g4_muon_steps']:,}   tcut {1e6 * out['tcut_MeV']:.0f} eV")
    print(f"{'':18s} {'mean':>7s} {'q10':>7s} {'q25':>7s} {'q50':>7s} {'q75':>7s} {'q90':>7s} {'q99':>7s} {'2.2-3.3':>8s} {'band-2s':>8s}")
    for k in ["g4"] + [f"model_split{n}" for n in nsp]:
        r = a[k]
        print(f"{k:18s} {r['mean']:7.3f} " + " ".join(f"{q:7.3f}" for q in r["q"]) + f" {r['window_share']:8.4f} "
              + (f"{r['band_minus_2sigma']:8.3f}" if "band_minus_2sigma" in r else ""))
    print("by path length (window share 2.2-3.3 keV; q50 / q90):")
    for row in out["by_len"]:
        s = f"  len {row['len_mm'][0]:4.1f}-{min(row['len_mm'][1], 99):4.1f} mm n {row['g4']['n']:6d}  G4 {row['g4']['window_share']:.4f} {row['g4']['q'][2]:.3f}/{row['g4']['q'][4]:.3f}"
        for n in nsp:
            r = row[f"model_split{n}"]
            s += f"  split{n} {r['window_share']:.4f} {r['q'][2]:.3f}/{r['q'][4]:.3f}"
        print(s)
    if "population" in out:
        p = out["population"]
        print(f"population: fast muon steps {p['fast_steps']:,}  len band-2s {p['len_band']:.3f}  |p| band-2s {p['p_band']:.3f}"
              f"  fast eIon band-2s {p['fast_eion_band_minus_2sigma']:.3f}  window {p['fast_eion']['window_share']:.4f}")
        print(f"  len q fast {p['len_q']['fast']}  g4 {p['len_q']['g4']}")
        print(f"  |p| q fast {p['p_q']['fast']}  g4 {p['p_q']['g4']}")
