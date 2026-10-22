"""Tracker fast reco commands: `python3 -m trkfastsim reco-<cmd> REF.root [...]`

  conditions  StrawHitTableDumper conditions tree -> text (JSON) conditions file for the hit reco
  vr3         T-VR3 (statistical): full fast chain from digis (hit reco, combo-free SimpleTimeCluster + LineFinder,
              F2 JAX fit with its own hit states) vs stock KKLine on the same events: yield, hits, chi2, parameters
  vr2         T-VR2 (F2): batched Gauss-Newton line fit on the stock seed, hit list and drift states vs stock KKLine
  vr1b        T-VR1b: SimpleTimeCluster + LineFinder ports vs the stock clusters and seeds (dump_tracks_* tables)
  vr1         T-VR1: dumped StrawDigis -> numpy makeSH + makePH vs the stock hits in the same tables
              (--jax: also the JAX straw hits vs numpy, with warm throughput)
"""
import argparse
import json
import os

import awkward as ak
import numpy as np
import uproot

from . import combo, hitreco

THRESH = {"time_tot_identical": 1.0, "energy_0p5pct": 0.99, "flags": 0.999, "combo_lists_identical": 1.0}
TIMECOLS = ("time", "et0", "et1", "tot0", "tot1")


def _ref(paths, pre):
    cols = ["sid", "flag", "edep", "ncombo", "ind"] + list(TIMECOLS)
    out = {k: [] for k in cols + ["ev"]}
    off = 0
    for p in paths:
        a = uproot.open(p)["trkdump/events"].arrays([pre + c for c in cols])
        out["ev"].append(np.repeat(np.arange(len(a)) + off, ak.to_numpy(ak.num(a[pre + "sid"]))))
        for c in cols:
            out[c].append(ak.to_numpy(ak.flatten(a[pre + c])))
        off += len(a)
    return {k: np.concatenate(v) for k, v in out.items()}


def vr1(inputs, outdir, conditions, settings, use_jax=False):
    os.makedirs(outdir, exist_ok=True)
    cond = hitreco.load_conditions(conditions)
    d = hitreco.read_digis(inputs)
    sh = hitreco.straw_hits(d, cond, settings)
    res = {"n_digis": int(len(d["ev"]))}
    if use_jax:
        res["jax_vs_numpy"] = _jax_twin(d, cond, settings, sh)
        res["jax_vs_numpy_combo"] = _jax_combo_twin(sh, settings)
    rs = _ref(inputs, "s_")
    res["n_stock_sh"] = int(len(rs["ev"]))
    same = len(rs["ev"]) == len(sh["ev"]) and np.array_equal(rs["sid"], sh["sid"])
    res["sh_same_list"] = bool(same)
    if same:
        for c in TIMECOLS:
            res[f"sh_{c}_identical"] = float(np.mean(sh[c] == rs[c]))
            dd = sh[c].astype(np.float64) - rs[c].astype(np.float64)
            res[f"sh_{c}_diff_ns"] = {"median": float(np.median(dd)), "max_abs": float(np.abs(dd).max()),
                                      "max_abs_ulp": float((np.abs(dd) / np.spacing(np.abs(rs[c]).astype(np.float32))).max())}
        rel = np.abs(sh["edep"] / np.where(rs["edep"] == 0, 1, rs["edep"]) - 1)
        res["sh_energy_within_0p5pct"] = float(np.mean(rel <= 0.005))
        res["sh_flags_agree"] = float(np.mean(sh["flag"] == rs["flag"]))
        bad = sh["flag"] != rs["flag"]
        res["sh_flag_xor_bits"] = np.bincount(np.log2((sh["flag"] ^ rs["flag"])[bad] & -(sh["flag"] ^ rs["flag"])[bad]).astype(int), minlength=27).tolist() if bad.any() else []
    ph = combo.panel_hits(sh, settings)
    rp = _ref(inputs, "p_")
    res["n_stock_ph"], res["n_fast_ph"] = int(len(rp["ev"])), int(len(ph["ev"]))
    lists = len(rp["ev"]) == len(ph["ev"]) and np.array_equal(rp["ncombo"], ph["ncombo"])
    lists = lists and np.array_equal(rp["ind"], ph["ind"].astype(rp["ind"].dtype))  # ind is per-event in stock
    res["ph_lists_identical_global"] = bool(lists)
    if len(rp["ev"]) == len(ph["ev"]):
        res["ph_ncombo_identical"] = float(np.mean(rp["ncombo"] == ph["ncombo"]))
        res["ph_sid_identical"] = float(np.mean(rp["sid"] == ph["sid"]))
        if len(rp["ind"]) == len(ph["ind"]):
            bad = rp["ind"] != ph["ind"].astype(rp["ind"].dtype)
            res["ph_ind_identical"] = float(1 - bad.mean())
            cev = np.repeat(rp["ev"], rp["ncombo"])                            # event of each constituent
            res["ph_events_with_ind_mismatch"] = int(len(np.unique(cev[bad])))
            res["n_events"] = int(len(np.unique(rp["ev"])))
    if len(rp["ev"]) == len(ph["ev"]):
        res["ph_time_identical"] = float(np.mean(ph["time"] == rp["time"]))
        dd = ph["time"].astype(np.float64) - rp["time"].astype(np.float64)
        res["ph_time_diff_ns"] = {"median": float(np.median(dd)), "max_abs": float(np.abs(dd).max())}
        res["ph_flags_agree"] = float(np.mean(ph["flag"] == rp["flag"]))
    with open(os.path.join(outdir, "vr1.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


def _jax_twin(d, cond, settings, sh):
    """JAX straw hits vs the numpy reference on the same digis: identical fraction, max ulp, warm throughput."""
    import time
    import jax
    from . import hitreco_jax
    dc = hitreco_jax.device_conditions(cond)
    packed = hitreco_jax.pack_digis(d, dc["npre"])
    t0 = time.perf_counter()
    sj = hitreco_jax.straw_hits(d, cond, settings, dcond=dc, packed=packed)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(3):
        sj = hitreco_jax.straw_hits(d, cond, settings, dcond=dc, packed=packed)
    warm = (time.perf_counter() - t0) / 3
    out = {"backend": jax.default_backend(), "first_call_s": first, "warm_s": warm,
           "digis_per_s_warm": len(d["ev"]) / warm, "n_pad": int(len(packed["us"]))}
    for k in sh:
        a, b = np.asarray(sh[k]), np.asarray(sj[k])
        out[k + "_identical"] = float(np.mean(a == b))
        if a.dtype.kind == "f" and a.dtype.itemsize == 4:
            ulp = np.abs(a.astype(np.float64) - b) / np.spacing(np.abs(a))
            out[k + "_max_ulp"] = float(ulp.max()) if len(ulp) else 0.0
    return out


def _jax_combo_twin(sh, settings):
    """JAX panel combo hits vs the numpy port on the same straw hits: identical lists, columns, warm throughput."""
    import time
    from . import combo_jax
    ref = combo.panel_hits(sh, settings)
    t0 = time.perf_counter()
    out = combo_jax.panel_hits(sh, settings)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(3):
        out = combo_jax.panel_hits(sh, settings)
    warm = (time.perf_counter() - t0) / 3
    res = {"first_call_s": first, "warm_s": warm, "straw_hits_per_s_warm": len(sh["ev"]) / warm,
           "max_run_len": int(combo_jax.run_index(sh, settings)[2].max()),
           "n_combos": int(len(ref["ncombo"]))}
    same = len(ref["ncombo"]) == len(out["ncombo"])
    res["same_count"] = bool(same)
    if same:
        for k in ("ncombo", "sid", "flag", "ind", "edep", "et0", "et1", "time", "wdist", "x", "y", "z", "uvar"):
            res[k + "_identical"] = float(np.mean(ref[k] == out[k])) if len(ref[k]) == len(out[k]) else 0.0
    return res


def _ref_col(paths, name):
    return np.concatenate([ak.to_numpy(ak.flatten(uproot.open(p)["trkdump/events"][name].array())) for p in paths])


def _ref_tracks(paths):
    """Stock time clusters and line seeds from DumpTracks tables, flattened with a global event index."""
    cols = ["tc_n", "tc_ind", "tc_t0", "tc_t0err", "sd_tc", "sd_n", "sd_ind", "sd_t0", "sd_a0", "sd_b0", "sd_a1", "sd_b1"]
    out = {k: [] for k in cols + ["tc_ev", "sd_ev"]}
    off = 0
    for p in paths:
        a = uproot.open(p)["trkdump/events"].arrays(cols)
        out["tc_ev"].append(np.repeat(np.arange(len(a)) + off, ak.to_numpy(ak.num(a.tc_n))))
        out["sd_ev"].append(np.repeat(np.arange(len(a)) + off, ak.to_numpy(ak.num(a.sd_n))))
        for c in cols:
            out[c].append(ak.to_numpy(ak.flatten(a[c])))
        off += len(a)
    return {k: np.concatenate(v) for k, v in out.items()}


def vr1b(inputs, outdir, conditions, settings, stock_hits=False):
    from . import linefind
    os.makedirs(outdir, exist_ok=True)
    cond = hitreco.load_conditions(conditions)
    d = hitreco.read_digis(inputs)
    sh = hitreco.straw_hits(d, cond, settings)
    if stock_hits:   # isolate the clustering/seeding logic from the 1-ulp float32 hit-time differences (T-VR1)
        rs = _ref(inputs, "s_")
        for c in ("time", "et0", "et1", "edep", "flag"):
            sh[c] = rs[c].astype(sh[c].dtype)
        for c in ("dtime", "ptime", "wdist", "uvar", "x", "y", "z"):
            sh[c] = _ref_col(inputs, "s_" + c).astype(sh[c].dtype)
    r = _ref_tracks(inputs)
    tc = linefind.time_clusters(sh)
    res = {"hits": "stock" if stock_hits else "fast", "n_events": int(d["evkey"].shape[0]), "n_stock_tc": int(len(r["tc_n"])), "n_fast_tc": int(len(tc["n"]))}
    same = len(r["tc_n"]) == len(tc["n"]) and np.array_equal(r["tc_ev"], tc["ev"]) and np.array_equal(r["tc_n"], tc["n"])
    res["tc_same_counts"] = bool(same)
    if same:
        res["tc_members_identical"] = bool(np.array_equal(r["tc_ind"], tc["ind"]))
        res["tc_t0_max_abs_ns"] = float(np.abs(r["tc_t0"] - tc["t0"]).max()) if len(tc["t0"]) else 0.0
        res["tc_t0err_max_abs_ns"] = float(np.abs(r["tc_t0err"] - tc["t0err"]).max()) if len(tc["t0"]) else 0.0
    if len(r["tc_n"]) == len(tc["n"]) and np.array_equal(r["tc_ev"], tc["ev"]):
        res["tc_nhits_identical"] = float(np.mean(r["tc_n"] == tc["n"]))
        res["tc_clusters_differing"] = int((r["tc_n"] != tc["n"]).sum())
    if not same:
        ne = int(d["evkey"].shape[0])
        cs, cf = np.bincount(r["tc_ev"], minlength=ne), np.bincount(tc["ev"], minlength=ne)
        res["tc_events_count_mismatch"] = int((cs != cf).sum())
    sd = linefind.line_seeds(sh, tc, cond)
    res["n_stock_seeds"], res["n_fast_seeds"] = int(len(r["sd_n"])), int(len(sd["n"]))
    ssame = len(r["sd_n"]) == len(sd["n"]) and np.array_equal(r["sd_ev"], sd["ev"])
    res["sd_same_list"] = bool(ssame)
    if ssame and len(sd["n"]):
        res["sd_nhits_identical"] = float(np.mean(r["sd_n"] == sd["n"]))
        for k in ("t0", "a0", "b0", "a1", "b1"):
            dd = np.abs(r["sd_" + k] - sd[k])
            res[f"sd_{k}_abs_diff"] = {"q50": float(np.median(dd)), "q99": float(np.quantile(dd, 0.99)), "max": float(dd.max())}
        res["sd_members_identical"] = bool(len(r["sd_ind"]) == len(sd["ind"]) and np.array_equal(r["sd_ind"], sd["ind"]))
        same_n = r["sd_n"] == sd["n"]
        rs, fs = np.repeat(np.arange(len(r["sd_n"])), r["sd_n"]), np.repeat(np.arange(len(sd["n"])), sd["n"])
        rk, fk = same_n[rs], same_n[fs]
        diff = np.zeros(len(r["sd_n"]), bool)
        diff[rs[rk][r["sd_ind"][rk] != sd["ind"][fk]]] = True     # same counts: compare member by member
        res["sd_seeds_with_member_mismatch"] = int((~same_n).sum() + diff.sum())
    with open(os.path.join(outdir, "vr1b.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


def _drift_table(path):
    c = uproot.open(path)["trkdump/conditions"].arrays(["dtgrid", "rdrift", "dvel", "sderr", "uderr"], library="np")
    return {"dt": c["dtgrid"][0], "rdrift": c["rdrift"][0], "dvel": c["dvel"][0], "sderr": c["sderr"][0], "uderr": c["uderr"][0]}


def vr2(inputs, outdir, conditions, settings, niter=8, max_tracks=None, use_tot=True, use_long=True, drift_dt=False,
        use_jax=False):
    """F2 on stock inputs: stock LineFinder seed, the stock fit's hit list, active flags and left/right states; compares
    the fitted KinematicLine parameters with stock KKLine per track, in units of the stock errors."""
    import time
    from . import linefit
    os.makedirs(outdir, exist_ok=True)
    cond = hitreco.load_conditions(conditions)
    d = hitreco.read_digis(inputs)
    sh = hitreco.straw_hits(d, cond, settings)
    rs = _ref(inputs, "s_")                                   # stock makeSH (the fit's input hits)
    for c in ("time", "et0", "et1", "edep", "flag"):
        sh[c] = rs[c].astype(sh[c].dtype)
    drift = _drift_table(inputs[0])
    cols = ["sd_a0", "sd_b0", "sd_a1", "sd_b1", "sd_t0", "kl_d0", "kl_phi0", "kl_z0", "kl_theta", "kl_t0",
            "kl_d0err", "kl_phi0err", "kl_z0err", "kl_thetaerr", "kl_t0err", "kl_chi2", "kl_ndof", "kl_status",
            "kh_trk", "kh_ind", "kh_ambig", "kh_active", "kh_rdoca", "kh_rdt", "kh_dt"]
    acc = {k: [] for k in cols}
    evh, off_ev, ntrk_off, evoff_sh = [], 0, 0, []
    tr_ev, hit_trk, hit_sh, hit_lr, hit_act = [], [], [], [], []
    ev0 = np.searchsorted(sh["ev"], np.arange(d["evkey"].shape[0] + 1), side="left")
    for p in inputs:
        a = uproot.open(p)["trkdump/events"].arrays(cols)
        one = (ak.num(a.kl_d0) == 1) & (ak.num(a.sd_a0) == 1)        # one seed and one fit: unambiguous pairing
        ie = np.flatnonzero(ak.to_numpy(one))
        b = a[one]
        for k in cols:
            acc[k].append(ak.to_numpy(ak.flatten(b[k])) if k.startswith(("kh_",)) else ak.to_numpy(ak.flatten(b[k])))
        nh = ak.to_numpy(ak.num(b.kh_ind))
        tidx = np.repeat(np.arange(len(b)) + ntrk_off, nh)
        hit_trk.append(tidx)
        hit_sh.append(ak.to_numpy(ak.flatten(b.kh_ind)) + np.repeat(ev0[ie + off_ev], nh))
        tr_ev.append(ie + off_ev)
        ntrk_off += len(b)
        off_ev += len(a)
    A = {k: np.concatenate(v) for k, v in acc.items()}
    trk, shi = np.concatenate(hit_trk), np.concatenate(hit_sh)
    amb, act = A["kh_ambig"], A["kh_active"].astype(bool)
    lr = np.where(np.abs(amb) == 1, amb, 0).astype(np.float64)
    ntrk = ntrk_off
    if max_tracks:
        keep = trk < max_tracks
        trk, shi, lr, act = trk[keep], shi[keep], lr[keep], act[keep]
        ntrk = min(ntrk, max_tracks)
    H = linefit.build_hits(trk, shi, lr, act, sh, cond, ntrk)
    # model check: our per-hit DOCA and deltaT at the STOCK final parameters vs the stock reference values
    xs = linefit.from_kinematic_line(*(A["kl_" + k][:ntrk] for k in ("d0", "phi0", "z0", "theta", "t0")))
    _, doca_s, dT_s, _ = linefit._model(xs, H, drift)
    sel = H["sh"] >= 0
    hidx = np.full(H["sh"].shape, -1)
    o = np.lexsort((np.arange(len(trk)), trk))
    cnt = np.bincount(trk, minlength=ntrk)
    posn = np.arange(len(trk)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    hidx[trk[o], posn] = o
    flat = hidx[sel]
    keepm = np.ones(len(A["kh_trk"]), bool) if not max_tracks else np.concatenate(hit_trk) < max_tracks
    rdoca = A["kh_rdoca"][keepm][flat] * H["esign"][sel] * -1.0    # TrkStrawHitSeed stores rdoca * endSign (hv = +1)
    rdt = A["kh_rdt"][keepm][flat]
    actf = H["active"][sel]
    dd, dt_ = (doca_s[sel] - rdoca)[actf], (dT_s[sel] - rdt)[actf]
    mc = {"doca_absdiff_q50_q99_mm": [float(np.median(np.abs(dd))), float(np.quantile(np.abs(dd), .99))],
          "doca_sum_q50_q99_mm": [float(np.median(np.abs(doca_s[sel] + rdoca)[actf])), float(np.quantile(np.abs(doca_s[sel] + rdoca)[actf], .99))],
          "dT_diff_q50_mean_q99_ns": [float(np.median(dt_)), float(np.mean(dt_)), float(np.quantile(np.abs(dt_), .99))]}
    x0 = linefit.seed_params(*(A[k][:ntrk] for k in ("sd_a0", "sd_b0", "sd_a1", "sd_b1", "sd_t0")))
    t0c = time.perf_counter()
    if use_jax:
        from . import linefit_jax
        import jax
        fj = linefit_jax.fit(x0, H, drift, niter=niter, drift_dt=drift_dt, use_tot=use_tot, use_long=use_long)
        first = time.perf_counter() - t0c
        t0c = time.perf_counter()
        f = linefit_jax.fit(x0, H, drift, niter=niter, drift_dt=drift_dt, use_tot=use_tot, use_long=use_long)
        wall = time.perf_counter() - t0c
        jax_info = {"backend": jax.default_backend(), "first_call_s": first, "warm_s": wall}
    else:
        f = linefit.fit(x0, H, drift, niter=niter, use_tot=use_tot, use_long=use_long, drift_dt=drift_dt)
        wall = time.perf_counter() - t0c
        jax_info = None
    kl = linefit.to_kinematic_line(f["x"])
    res = {"jax": jax_info, "model_check_at_stock_params": mc, "n_tracks": int(ntrk), "niter": niter, "use_tot": use_tot, "use_long": use_long, "drift_dt": drift_dt, "fit_wall_s": wall, "tracks_per_s": ntrk / wall,
           "active_hits_q50": float(np.median(H["active"].sum(1)))}
    ok = np.isfinite(f["chi2"])
    res["fit_finite_frac"] = float(ok.mean())
    for k in ("d0", "phi0", "z0", "theta", "t0"):
        ref, err = A["kl_" + k][:ntrk], A["kl_" + k + "err"][:ntrk]
        dv = kl[k] - ref
        if k == "phi0":
            dv = (dv + np.pi) % (2 * np.pi) - np.pi
        pull = dv / np.where(err > 0, err, np.nan)
        m = ok & np.isfinite(pull)
        res[f"{k}_dsigma"] = {"q50": float(np.median(pull[m])), "q16_84": [float(np.quantile(pull[m], .16)), float(np.quantile(pull[m], .84))],
                              "frac_within_0p1": float(np.mean(np.abs(pull[m]) < 0.1)), "frac_within_1": float(np.mean(np.abs(pull[m]) < 1.0))}
    J = np.zeros((ntrk, 5, 5))
    eps = np.array([1e-3, 1e-3, 1e-6, 1e-6, 1e-3])
    klv = lambda xx: np.stack([linefit.to_kinematic_line(xx)[k] for k in ("d0", "phi0", "z0", "theta", "t0")], 1)
    for k in range(5):
        dx = np.zeros(5); dx[k] = eps[k]
        J[:, :, k] = (klv(f["x"] + dx) - klv(f["x"] - dx)) / (2 * eps[k])
    C = np.einsum("tij,tjk,tlk->til", J, f["cov"], J)
    for i, k in enumerate(("d0", "phi0", "z0", "theta", "t0")):
        res[f"{k}_err_ratio_q50"] = float(np.median(np.sqrt(C[:, i, i]) / A["kl_" + k + "err"][:ntrk]))
    rc = A["kl_chi2"][:ntrk] / np.maximum(A["kl_ndof"][:ntrk], 1)
    fc = f["chi2"] / np.maximum(f["ndof"], 1)
    res["chi2ndf_stock_q50"], res["chi2ndf_fast_q50"] = float(np.median(rc)), float(np.median(fc[ok]))
    with open(os.path.join(outdir, "vr2.json"), "w") as fo:
        json.dump(res, fo, indent=1)
    print(json.dumps(res, indent=1))


def vr3(inputs, outdir, conditions, settings, niter=4):
    """Fast chain on the same digis as the stock DumpTracks reference; distributions compared, not track by track."""
    import time
    from . import linefind, linefit, linefit_jax
    os.makedirs(outdir, exist_ok=True)
    cond = hitreco.load_conditions(conditions)
    d = hitreco.read_digis(inputs)
    tt = {}
    t0 = time.perf_counter(); sh = hitreco.straw_hits(d, cond, settings); tt["hits_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); tc = linefind.time_clusters(sh); tt["clusters_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); sd = linefind.line_seeds(sh, tc, cond); tt["seeds_s"] = time.perf_counter() - t0
    drift = _drift_table(inputs[0])
    # fit every seed with all its hits as candidates; states decided by the fit
    ns = sd["n"]
    trk = np.repeat(np.arange(len(ns)), ns)
    H = linefit.build_hits(trk, sd["member"], np.zeros(len(trk)), np.ones(len(trk), bool), sh, cond, len(ns))
    x0 = linefit.seed_params(sd["a0"], sd["b0"], sd["a1"], sd["b1"], sd["t0"])
    t0 = time.perf_counter()
    f = linefit_jax.fit_own_states(x0, H, drift, niter=niter)
    tt["fit_first_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    f = linefit_jax.fit_own_states(x0, H, drift, niter=niter)
    tt["fit_warm_s"] = time.perf_counter() - t0
    kl = linefit.to_kinematic_line(f["x"])
    nact = f["active"].sum(1)
    good = (nact >= 5) & np.isfinite(f["chi2"]) & (f["ndof"] >= 1)          # KKFIT MinNStrawHits 5, MinNDOF 1
    # stock reference
    cols = ["kl_d0", "kl_phi0", "kl_z0", "kl_theta", "kl_t0", "kl_chi2", "kl_ndof", "kl_nactive", "kl_status"]
    r = {k: [] for k in cols + ["ev"]}
    off = 0
    for p in inputs:
        a = uproot.open(p)["trkdump/events"].arrays(cols)
        r["ev"].append(np.repeat(np.arange(len(a)) + off, ak.to_numpy(ak.num(a.kl_d0))))
        for c in cols:
            r[c].append(ak.to_numpy(ak.flatten(a[c])))
        off += len(a)
    r = {k: np.concatenate(v) for k, v in r.items()}
    ne = int(d["evkey"].shape[0])
    res = {"n_events": ne, "timing_cpu_or_device": tt, "jax_backend": __import__("jax").default_backend(),
           "stock_fits": int(len(r["kl_d0"])), "fast_fits": int(good.sum()),
           "stock_events_with_fit": int(len(np.unique(r["ev"]))), "fast_events_with_fit": int(len(np.unique(sd["ev"][good])))}
    res["yield_ratio_fast_over_stock"] = res["fast_fits"] / max(res["stock_fits"], 1)
    def qs(a): return [float(x) for x in np.quantile(a, [.05, .16, .5, .84, .95])]
    comp = {}
    stock_chi = r["kl_chi2"] / np.maximum(r["kl_ndof"], 1)
    fast_chi = f["chi2"][good] / np.maximum(f["ndof"][good], 1)
    comp["nactive"] = {"stock": qs(r["kl_nactive"]), "fast": qs(nact[good])}
    comp["chi2ndf"] = {"stock": qs(stock_chi), "fast": qs(fast_chi)}
    for k in ("d0", "phi0", "z0", "theta", "t0"):
        comp[k] = {"stock": qs(r["kl_" + k]), "fast": qs(kl[k][good])}
    res["quantiles_5_16_50_84_95"] = comp
    # matched events (one stock fit and one fast fit): per-track differences, for orientation only
    evs, evf = r["ev"], sd["ev"][good]
    us, cs = np.unique(evs, return_counts=True)
    uf, cf = np.unique(evf, return_counts=True)
    both = np.intersect1d(us[cs == 1], uf[cf == 1])
    si = np.searchsorted(evs, both)
    fi = np.flatnonzero(good)[np.searchsorted(evf, both)]
    diff = {}
    for k in ("d0", "phi0", "z0", "theta", "t0"):
        dv = kl[k][fi] - r["kl_" + k][si]
        if k == "phi0":
            dv = (dv + np.pi) % (2 * np.pi) - np.pi
        diff[k] = qs(dv)
    res["matched_events"] = int(len(both))
    res["matched_diff_quantiles"] = diff
    with open(os.path.join(outdir, "vr3.json"), "w") as fo:
        json.dump(res, fo, indent=1)
    print(json.dumps(res, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="trkfastsim reco", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("conditions")
    c.add_argument("ref")
    c.add_argument("--out", required=True)
    v = sub.add_parser("vr1")
    v.add_argument("inputs", nargs="+")
    v.add_argument("--conditions", required=True)
    v.add_argument("--settings", default="mc", choices=["mc", "data"])
    v.add_argument("--outdir", default="plots/reco-vr1")
    v.add_argument("--jax", action="store_true", help="also run the JAX twin and compare it with numpy")
    b = sub.add_parser("vr1b")
    b.add_argument("inputs", nargs="+")
    b.add_argument("--conditions", required=True)
    b.add_argument("--settings", default="mc", choices=["mc", "data"])
    b.add_argument("--outdir", default="plots/reco-vr1b")
    b.add_argument("--stock-hits", action="store_true", help="cluster and seed the stock makeSH hits (logic-only test)")
    f2 = sub.add_parser("vr2")
    f2.add_argument("inputs", nargs="+")
    f2.add_argument("--conditions", required=True)
    f2.add_argument("--settings", default="mc", choices=["mc", "data"])
    f2.add_argument("--niter", type=int, default=8)
    f2.add_argument("--max-tracks", type=int, default=None)
    f2.add_argument("--outdir", default="plots/reco-vr2")
    f2.add_argument("--no-tot", action="store_true", help="drop the TOT drift-time residual")
    f2.add_argument("--no-long", action="store_true", help="drop the longitudinal (time-division) residual")
    f2.add_argument("--drift-dt", action="store_true", help="let drift radii constrain t0 (KKSHFlag DriftDt; stock: off)")
    f2.add_argument("--jax", action="store_true", help="fit with the JAX twin (exact Jacobian, vmapped); timing is warm")
    v3 = sub.add_parser("vr3")
    v3.add_argument("inputs", nargs="+")
    v3.add_argument("--conditions", required=True)
    v3.add_argument("--settings", default="mc", choices=["mc", "data"])
    v3.add_argument("--niter", type=int, default=4)
    v3.add_argument("--outdir", default="plots/reco-vr3")
    a = p.parse_args(argv)
    if a.cmd == "vr3":
        return vr3(a.inputs, a.outdir, a.conditions, a.settings, a.niter)
    if a.cmd == "vr2":
        return vr2(a.inputs, a.outdir, a.conditions, a.settings, a.niter, a.max_tracks, not a.no_tot, not a.no_long, a.drift_dt, a.jax)
    if a.cmd == "conditions":
        hitreco.extract_conditions(a.ref, a.out)
    elif a.cmd == "vr1b":
        vr1b(a.inputs, a.outdir, a.conditions, a.settings, a.stock_hits)
    else:
        vr1(a.inputs, a.outdir, a.conditions, a.settings, a.jax)
