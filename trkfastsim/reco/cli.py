"""Tracker fast reco commands: `python3 -m trkfastsim reco-<cmd> REF.root [...]`

  conditions  StrawHitTableDumper conditions tree -> text (JSON) conditions file for the hit reco
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
    a = p.parse_args(argv)
    if a.cmd == "conditions":
        hitreco.extract_conditions(a.ref, a.out)
    else:
        vr1(a.inputs, a.outdir, a.conditions, a.settings, a.jax)
