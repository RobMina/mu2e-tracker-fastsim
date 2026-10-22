"""Vectorised numpy port of Offline StrawHitReco (FitType 1, peakminuspedavg) on dumped StrawDigis.

Reference: TrkHitReco/src/StrawHitReco_module.cc + StrawHitRecoUtils::createComboHit, StrawResponse
(calibrateTimes, wireDistance/halfPropV/wpRes, TOTdriftTime). Inputs are the StrawHitTableDumper tables;
conditions come from the text file written by `conditions` (extracted from the dumper's conditions tree).
float32 casts follow the stock types (TDCTimes/TOTTimes/ComboHit members are float).
"""
import json

import awkward as ak
import numpy as np
import uproot

F32 = np.float32
FLAG = {"energysel": 1, "radsel": 2, "timesel": 3, "tdiv": 10, "panelcombo": 21, "dead": 23, "noisy": 24, "nhitsel": 25}
NSTRAWS, NPANELS = 96, 6
CENTRAL_WIRE_POS = 65.0   # TrackerConditions/fcl/prolog.fcl centralWirePos
WBUF, SLFAC, ERRFAC = 2.0, 0.9, 1.0   # wireLengthBuffer, strawLengthFactor, errorFactor (same prolog)
EBIN = 0.1                # eBinWidth (keV grid of eHalfPVScale/tdCentralRes/tdResSlope)

# makeSH settings: TrkHitReco prolog + NoPileup_epilog + OffSpillTiming_epilog (MC); data adds reco_data_vst.fcl
SH_SETTINGS = {
    "mc":   dict(minT=0.0, maxT=100.0e3, minE=0.0, maxE=1.0, minR=300.0, maxR=800.0),
    "data": dict(minT=0.0, maxT=256000.0, minE=0.0, maxE=1.0, minR=300.0, maxR=800.0),
}


def unique_panel(sid):
    return (sid >> 10) * NPANELS + ((sid & 0x380) >> 7)


def unique_straw(sid):
    return unique_panel(sid) * NSTRAWS + (sid & 0x7F)


def extract_conditions(ref_path, out_txt):
    """Conditions tree of one dumper file -> JSON text file of every constant the hit reco needs."""
    t = uproot.open(ref_path)["trkdump/conditions"].arrays(library="np")
    g = {k: np.asarray(v[0]) for k, v in t.items()}
    k = g["kgrid"]
    idx = np.rint(np.arange(59) * EBIN / 0.005).astype(int)        # probe points at the 0.1 keV knots
    hpv_scale0 = g["hpv_s0"][idx]                                     # halfvpscale_i * meanv(straw 0)
    centres = g["wpres0"][idx]
    resslope = (g["wpres200"][idx] - centres) / (200.0 - CENTRAL_WIRE_POS)
    it, ie = np.meshgrid(np.arange(100), np.arange(10), indexing="ij")
    tb, eb = np.minimum(15, (2 * it + 1) // 5), np.minimum(4, ie)
    totdt, totde = np.zeros(80), np.zeros(80)
    totdt[(tb * 5 + eb).ravel()] = g["totdt"]
    totde[(tb * 5 + eb).ravel()] = g["totde"]
    us = unique_straw(g["sid"].astype(np.int64))
    n = int(us.max()) + 1
    per = {}
    for name in ("midx", "midy", "midz", "dirx", "diry", "dirz", "hlen", "pmpscale", "toff_hv", "toff_cal", "dead", "noisy"):
        a = np.zeros(n)
        a[us] = g[name]
        per[name] = a.tolist()
    hv = np.zeros(n)
    hv[us] = g["hpv2"] / _pieceline(np.arange(59) * EBIN, hpv_scale0 / hpv_scale0[0], np.array([2.0]))[0]
    per["halfvp"] = hv.tolist()
    cond = {"source": str(ref_path), "kgrid_check": [float(k[0]), float(k[-1])],
            "scalar": {s: float(g[s]) for s in ("tdcLSB", "adcLSB", "totLSB", "strawGain", "eionPerQ", "pmpScaleAvg",
                                                 "strawOuterRadius")} | {"nADCPreSamples": int(g["nADCPreSamples"])},
            "halfvpscale": (hpv_scale0 / hpv_scale0[0]).tolist(), "centres": centres.tolist(),
            "resslope": resslope.tolist(), "totdtime": totdt.tolist(), "totderror": totde.tolist(),
            "tot_bins": [16, 5.0, 5, 0.0005], "straw": per}
    with open(out_txt, "w") as f:
        json.dump(cond, f, indent=1)
    return cond


def load_conditions(path):
    with open(path) as f:
        c = json.load(f)
    c["straw"] = {k: np.asarray(v) for k, v in c["straw"].items()}
    for k in ("halfvpscale", "centres", "resslope", "totdtime", "totderror"):
        c[k] = np.asarray(c[k])
    return c


def _pieceline(xv, yv, x):
    """StrawResponse::PieceLine: uniform-grid linear interpolation, linear extrapolation beyond the ends."""
    xbin = (xv[-1] - xv[0]) / (len(xv) - 1)
    ib = np.clip(np.floor((x - xv[0]) / xbin).astype(np.int64), 0, len(xv) - 2)
    return yv[ib] + (x - xv[ib]) * (yv[ib + 1] - yv[ib]) / xbin


def read_digis(paths):
    """Dumper events tree -> flat per-digi numpy arrays (+ event index and per-event pbt/spill)."""
    out = {k: [] for k in ("ev", "sid", "tdc0", "tdc1", "tot0", "tot1", "nadc", "adc", "pbt", "evkey")}
    off = 0
    for p in paths:
        a = uproot.open(p)["trkdump/events"].arrays(["run", "subrun", "event", "pbt", "d_sid", "d_tdc0", "d_tdc1",
                                                      "d_tot0", "d_tot1", "d_nadc", "d_adc"])
        nd = ak.to_numpy(ak.num(a.d_sid))
        out["ev"].append(np.repeat(np.arange(len(a)) + off, nd))
        out["pbt"].append(ak.to_numpy(a.pbt))
        out["evkey"].append(np.stack([ak.to_numpy(a.run), ak.to_numpy(a.subrun), ak.to_numpy(a.event)], 1))
        for k in ("sid", "tdc0", "tdc1", "tot0", "tot1", "nadc", "adc"):
            out[k].append(ak.to_numpy(ak.flatten(a["d_" + k])))
        off += len(a)
    return {k: np.concatenate(v) for k, v in out.items()}


def peak_minus_ped(adc_flat, nadc, npre):
    """StrawHitRecoUtils::peakMinusPedWF: max over samples >= npre minus the float mean of the first npre."""
    start = np.concatenate([[0], np.cumsum(nadc)[:-1]])
    pos = np.arange(len(adc_flat)) - np.repeat(start, nadc)
    ped = (np.add.reduceat(np.where(pos < npre, adc_flat, 0), start) / F32(npre)).astype(F32)
    peak = np.maximum.reduceat(np.where(pos >= npre, adc_flat, np.iinfo(np.int32).min), start)
    return peak.astype(np.float64) - ped.astype(np.float64)


def straw_hits(d, cond, settings="mc"):
    """All digis -> single-straw ComboHit columns (FilterHits false: one hit per digi, flags carry the cuts)."""
    s, sc, st = SH_SETTINGS[settings], cond["scalar"], cond["straw"]
    us = unique_straw(d["sid"].astype(np.int64))
    pmp = peak_minus_ped(d["adc"].astype(np.int64), d["nadc"], sc["nADCPreSamples"])
    charge = pmp * (sc["adcLSB"] / sc["strawGain"]) * sc["pmpScaleAvg"]
    energy = F32(sc["eionPerQ"] * charge)
    pbt = F32(d["pbt"][d["ev"]])
    t_cal = (F32(d["tdc0"] * sc["tdcLSB"] + st["toff_cal"][us]) - pbt).astype(F32)
    t_hv = (F32(d["tdc1"] * sc["tdcLSB"] + st["toff_hv"][us]) - pbt).astype(F32)
    eend = (t_hv < t_cal).astype(np.int64)                 # 1 = hv, 0 = cal
    tot_cal, tot_hv = F32(d["tot0"] * sc["totLSB"]), F32(d["tot1"] * sc["totLSB"])
    tot_sel = np.where(eend == 1, tot_hv, tot_cal)
    dt = t_hv.astype(np.float64) - t_cal.astype(np.float64)
    kedep = 1000.0 * energy.astype(np.float64)
    egrid = np.arange(len(cond["halfvpscale"])) * EBIN
    halfpv = _pieceline(egrid, cond["halfvpscale"], kedep) * st["halfvp"][us]
    wdist = halfpv * dt
    wres = _pieceline(egrid, cond["centres"], kedep)
    wres = np.maximum(30.0, wres + np.where(np.abs(wdist) > CENTRAL_WIRE_POS,
                                            (np.abs(wdist) - CENTRAL_WIRE_POS) * _pieceline(egrid, cond["resslope"], kedep), 0.0))
    hlen = st["hlen"][us]
    trunc = np.abs(wdist) > hlen + WBUF * wres
    wdist = np.where(trunc, np.copysign(hlen * SLFAC, wdist), wdist)
    wres = np.where(trunc, wres * ERRFAC, wres)
    esign = 2.0 * (eend - 0.5)
    propd = F32(hlen + esign * wdist)
    ptime = F32(propd / (2 * halfpv))
    x = F32(st["midx"][us] + wdist * st["dirx"][us])
    y = F32(st["midy"][us] + wdist * st["diry"][us])
    z = F32(st["midz"][us] + wdist * st["dirz"][us])
    rho = np.hypot(x, y)
    ntb, wtb, neb, web = cond["tot_bins"]
    tbin = np.minimum(int(ntb) - 1, (tot_sel / wtb).astype(np.int64))
    ebin = np.minimum(int(neb) - 1, (energy / web).astype(np.int64))
    dtime = cond["totdtime"][tbin * int(neb) + ebin]                     # probed TOTdriftTime already includes its -0.1 ns
    tres = cond["totderror"][tbin * int(neb) + ebin]
    t_early = np.where(eend == 1, t_hv, t_cal).astype(np.float64)
    time = F32(t_early - ptime - dtime)
    flag = np.zeros(len(us), np.uint32)
    for bit, ok in ((FLAG["dead"], st["dead"][us] > 0), (FLAG["noisy"], st["noisy"][us] > 0),
                    (FLAG["energysel"], (energy <= s["maxE"]) & (energy >= s["minE"])),
                    (FLAG["radsel"], (rho >= s["minR"]) & (rho <= s["maxR"])),
                    (FLAG["timesel"], (time >= s["minT"]) & (time <= s["maxT"])), (FLAG["tdiv"], ~trunc)):
        flag |= np.where(ok, np.uint32(1 << bit), np.uint32(0))
    rout = sc["strawOuterRadius"]
    return {"ev": d["ev"], "sid": d["sid"], "upanel": unique_panel(d["sid"].astype(np.int64)), "straw": d["sid"] & 0x7F,
            "time": time, "et0": t_cal, "et1": t_hv, "tot0": tot_cal, "tot1": tot_hv, "edep": energy, "eend": eend,
            "dtime": F32(dtime), "ptime": ptime, "wdist": F32(wdist), "uvar": F32(wres * wres),
            "vvar": np.full(len(us), F32(rout * rout / 3.0)), "timevar": F32(tres * tres), "x": x, "y": y, "z": z,
            "udx": F32(st["dirx"][us]), "udy": F32(st["diry"][us]), "flag": flag, "halfpv": halfpv}
