"""Tracker fast reco commands: `python3 -m trkfastsim reco-<cmd> REF.root [...]`

  conditions  StrawHitTableDumper conditions tree -> text (JSON) conditions file for the hit reco
  vr1         T-VR1: dumped StrawDigis -> numpy makeSH + makePH vs the stock hits in the same tables
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


def vr1(inputs, outdir, conditions, settings):
    os.makedirs(outdir, exist_ok=True)
    cond = hitreco.load_conditions(conditions)
    d = hitreco.read_digis(inputs)
    sh = hitreco.straw_hits(d, cond, settings)
    rs = _ref(inputs, "s_")
    res = {"n_digis": int(len(d["ev"])), "n_stock_sh": int(len(rs["ev"]))}
    same = len(rs["ev"]) == len(sh["ev"]) and np.array_equal(rs["sid"], sh["sid"])
    res["sh_same_list"] = bool(same)
    if same:
        for c in TIMECOLS:
            res[f"sh_{c}_identical"] = float(np.mean(sh[c] == rs[c]))
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
        res["ph_time_identical"] = float(np.mean(ph["time"] == rp["time"]))
        res["ph_flags_agree"] = float(np.mean(ph["flag"] == rp["flag"]))
    with open(os.path.join(outdir, "vr1.json"), "w") as f:
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
    a = p.parse_args(argv)
    if a.cmd == "conditions":
        hitreco.extract_conditions(a.ref, a.out)
    else:
        vr1(a.inputs, a.outdir, a.conditions, a.settings)
