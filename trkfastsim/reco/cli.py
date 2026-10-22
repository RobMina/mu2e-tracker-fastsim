"""Tracker fast reco commands: `python3 -m trkfastsim reco-<cmd> REF.root [...]`

  conditions  StrawHitTableDumper conditions tree -> text (JSON) conditions file for the hit reco
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
    a = p.parse_args(argv)
    if a.cmd == "conditions":
        hitreco.extract_conditions(a.ref, a.out)
    elif a.cmd == "vr1b":
        vr1b(a.inputs, a.outdir, a.conditions, a.settings, a.stock_hits)
    else:
        vr1(a.inputs, a.outdir, a.conditions, a.settings, a.jax)
