"""Conditions of the stock tracker digitiser (Offline StrawDigisFromStrawGasSteps) as arrays, from the stock job's sources.

Sources, the same ones the stock job reads (Offline v13_39_00):
- ``fhicl-dump`` of the digitisation job: ProditionsService ``strawPhysics``, ``strawDrift``, ``strawElectronics``,
  ``eventTiming``; ``physics.producers.makeSD`` (module defaults where unset); ``EWMProducer.SpillType``;
  ``GeometryService.bFieldFile`` (only the field-free case is supported);
- DbService tables of the run (``dbTool print-run --purpose P --version V --run R --table T --content`` CSVs):
  ``TrkPreampStraw`` (thresholds, ADC gain), ``TrkAlignStrawSim`` (wire-end offsets of ``alignedTrackerSim``, which runs
  with onlyWireOffsets), ``TrkDelayPanel``/``TrkDelayRStraw`` (read only when overrideDbTimeOffsets is false);
- the GDML export for the nominal straws (``trkfastsim.geometry``), moved to the detector frame;
- GlobalConstantsService physicsParams (DR period, DAQ clock) from its text file, or the defaults below.

Derived tables follow StrawPhysicsMaker, StrawDriftMaker, StrawElectronicsMaker::fromDb, AlignedTrackerMaker and
StrawElectronics::linearResponse/maxResponseTime/maxLinearResponse. Units: mm, ns, MeV, pC, mV.
"""
import json
import math
import re

import numpy as np

from . import geometry

NSTRAWS, NPANELS, NPLANES = 96, 6, 36
NUSTRAWS = NPLANES * NPANELS * NSTRAWS
NADC_MIN, NADC_PERPACKET = 3, 12            # TrkTypes.hh
# StrawDigisFromStrawGasSteps::Config defaults (used where the job's makeSD table leaves them unset)
MAKESD_DEFAULTS = {"MaxNClusters": 20, "addCrossTalk": False, "DriftSingleElectrons": False, "RandomizeRadius": True,
                   "addNoise": False, "preAmplificationCrossTalk": 0.0, "postAmplificationCrossTalk": 0.02,
                   "minstepE": 2.0e-6, "StrawGasStepTimeBuffer": 100.0, "FlashTimeBuffer": 10.0, "TDCTimeBuffer": 2.0,
                   "AllHitsStraw": 90, "AllHitsPlanes": [], "UseStatus": False}
# GlobalConstantsService/data/globalConstants_01.txt physicsParams
GLOBAL_DEFAULTS = {"nominalDRPeriod": 1695.0, "nominalDAQFrequency": 40.0}


# ---------------------------------------------------------------- fhicl-dump reader
_TOK = re.compile(r'"(?:[^"\\]|\\.)*"|[{}\[\],:]|[^\s{}\[\],:"]+')


def parse_fhicl_dump(text):
    """The nested table printed by ``fhicl-dump`` -> dict/list/str/float/bool (comments dropped, no @-references)."""
    toks = _TOK.findall("\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#")))
    pos = 0

    def atom(t):
        if t.startswith('"'):
            return t[1:-1]
        if t in ("true", "false"):
            return t == "true"
        if t == "nil":
            return None
        try:
            return int(t) if re.fullmatch(r"[+-]?\d+", t) else float(t)
        except ValueError:
            return t

    def value():
        nonlocal pos
        t = toks[pos]
        pos += 1
        if t == "{":
            return table("}")
        if t == "[":
            out = []
            while toks[pos] != "]":
                out.append(value())
                if toks[pos] == ",":
                    pos += 1
            pos += 1
            return out
        return atom(t)

    def table(end):
        nonlocal pos
        out = {}
        while pos < len(toks) and toks[pos] != end:
            key = toks[pos]
            if toks[pos + 1] != ":":
                raise ValueError(f"fhicl-dump: expected ':' after {key!r}")
            pos += 2
            out[key] = value()
        pos += 1
        return out

    return table(None)


def fcl_get(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def read_db_csv(path):
    """dbTool --content CSV -> 2D float array (index column first; StrawId text columns like 0_0_1 dropped)."""
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or not ln[0].isdigit():
                continue
            cols = [c.strip() for c in ln.split(",")]
            rows.append([float(c) for c in cols if not re.fullmatch(r"\d+_\d+_\d+", c)])
    return np.array(rows)


def read_global_constants(path=None):
    out = dict(GLOBAL_DEFAULTS)
    if path:
        with open(path) as f:
            for ln in f:
                m = re.match(r"\s*(?:double|int)\s+physicsParams\.(\w+)\s*=\s*([-+.\deE]+)", ln)
                if m and m.group(1) in out:
                    out[m.group(1)] = float(m.group(2))
    return out


# ---------------------------------------------------------------- StrawPhysicsMaker
def straw_physics(cfg):
    p = np.asarray(cfg["probPerCharge"], float)
    eion = np.asarray(cfg["ionizationEnergyTable"], float)
    ke = float(cfg["ionizedElectronKE"])
    EIonize = np.cumsum(1.0e-6 * eion + ke)                     # MeV to free n electrons, n = 1..len
    norm = p / p.sum()
    intNProb = np.cumsum(norm)
    nele = np.arange(1, len(p) + 1)
    e_n = EIonize[np.minimum(nele, len(EIonize)) - 1]
    return {"EIonize": EIonize, "intNProb": intNProb, "EAverage": float((norm * e_n / nele).sum()),
            "NAverage": float((norm * nele).sum()), "meanFreePath": float(cfg["meanFreePath"]),
            "Qe": float(cfg["electronCharge"]), "gasGain": float(cfg["gasGain"]), "polyaA": float(cfg["polyaA"]),
            "gainRMSSlope": float(cfg["gainRMSSlope"]), "nGainGauss": int(cfg["nGainGauss"]),
            "vprop": float(cfg["propagationVelocity"]), "dtvar": np.asarray(cfg["driftTimeVariance"], float),
            "nonlindrift": bool(cfg["useNonLinearDrift"])}


# ---------------------------------------------------------------- StrawDriftMaker::fromFcl
def straw_drift(cfg, wire_radius, straw_outer_radius, bz=0.0):
    """Distance -> time table on the deltaD grid for phiBins Lorentz angles (times_dbins[d, phi])."""
    V = float(cfg["wireVoltage"])
    E = np.asarray(cfg["kVcm"], float) * 100.0
    vinst = np.asarray(cfg["cmus"], float) * 0.01
    logR = math.log(straw_outer_radius / wire_radius)
    dd = [V / (e * logR) for e in E]
    nphi = int(cfg["phiBins"])
    CC = bz * logR / V
    dD = float(cfg["deltaDistance"])
    dphi = math.pi / 2.0 / (nphi - 1)
    dist, times = [0.0], [[0.0] * nphi]
    index = len(dd) - 2
    tempD, tempT = dD, 0.0
    while tempD < straw_outer_radius:
        while dd[index] < tempD:
            if index == 0:
                break
            index -= 1
        d0, d1, s0, s1 = dd[index + 1], dd[index], vinst[index + 1], vinst[index]
        speed = s0 + (s1 - s0) * (tempD - d0) / (d1 - d0)
        dist.append(tempD)
        tempT += dD / speed
        C = CC * (tempD / tempT) * 1.0e6
        row = []
        for ip in range(nphi):
            zeta = C * tempD * 0.001
            gam = (1 + zeta ** 2 / 3.0) / (1 + (zeta * math.cos(dphi * ip)) ** 2 / 3.0)
            row.append(tempT * gam)
        times.append(row)
        tempD += dD
    return {"drift_dist": np.array(dist), "drift_times": np.array(times), "drift_dD": dD, "drift_dphi": dphi}


def d2t(dist, phi, dd, tt, dD, dphi):
    """StrawDrift::D2T, vectorised: linear in distance on the grid, linear in folded phi between phi bins."""
    dist = np.asarray(dist, float)
    nphi = tt.shape[1]
    fphi = np.abs(np.fmod(np.asarray(phi, float), math.pi))
    fphi = np.where(fphi > math.pi / 2, math.pi - fphi, fphi)
    p0 = np.maximum(0, np.floor(fphi / dphi)).astype(np.int64)
    p1 = np.minimum(nphi - 1, np.ceil(fphi / dphi)).astype(np.int64)
    di = np.clip(np.floor(dist / dD), 0, len(dd) - 2).astype(np.int64)
    fr = (dist - dd[di]) / dD
    lo = tt[di, p0] + fr * (tt[di + 1, p0] - tt[di, p0])
    hi = tt[di, p1] + fr * (tt[di + 1, p1] - tt[di, p1])
    return np.where(np.asarray(phi) == 0, lo, lo + (fphi - p0 * dphi) / dphi * (hi - lo))


# ---------------------------------------------------------------- StrawElectronics responses
def _comb(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def _poly(roots):
    """DigitalFiltering::zpk2tf: monic polynomial coefficients (highest power first) of the given roots."""
    c = [1.0]
    for r in roots:
        c = [a - r * b for a, b in zip(c + [0.0], [0.0] + c)]
    return c


def _bilinear(num, den, fs):
    """DigitalFiltering::bilinear (num = zeros polynomial, den = poles polynomial), normalised to aprime[0] = 1."""
    N, D = len(num) - 1, len(den) - 1
    M = max(N, D)

    def tr(c, n):
        out = []
        for j in range(M + 1):
            v = 0.0
            for i in range(n + 1):
                for k in range(i + 1):
                    for l in range(M - i + 1):
                        if k + l == j:
                            v += _comb(i, k) * _comb(M - i, l) * c[n - i] * (2 * fs) ** i * (-1) ** k
            out.append(v)
        return out
    bp, ap = tr(num, N), tr(den, D)
    return [b / ap[0] for b in bp], [a / ap[0] for a in ap]


def calculate_response(poles, zeros, inp, sample_rate):
    """StrawElectronics::calculateResponse: bilinear IIR of the pole/zero filter (MHz), gain-normalised at 160 MHz."""
    pa = [-p * 2 * math.pi for p in poles if p != 0]
    za = [-z * 2 * math.pi for z in zeros if z != 0]
    num, den = _poly(za), _poly(pa)
    bp, ap = _bilinear(num, den, sample_rate * 1000.0)
    w = complex(0, 160 * 2 * math.pi)
    g160 = abs(sum(c * w ** (len(num) - i - 1) for i, c in enumerate(num)) /
               sum(c * w ** (len(den) - i - 1) for i, c in enumerate(den)))
    x = list(map(float, inp))
    y = [0.0] * len(x)
    for i in range(len(x)):
        r = 0.0
        for j in range(len(bp)):
            if i >= j:
                r += x[i - j] * bp[j]
        for j in range(1, len(ap)):
            if i >= j:
                r += -1 * y[i - j] * ap[j]
        y[i] = r
    return np.array(y) / g160


def straw_electronics(cfg, event_timing, preamp_rows, nominal_halflen, delays=None):
    """StrawElectronicsMaker::fromDb arrays: response tables per wire-distance point, thresholds, pedestals, dVdI."""
    rb, sr = int(cfg["responseBins"]), float(cfg["sampleRate"])
    wd = np.asarray(cfg["wireDistances"], float)
    means, norms = np.asarray(cfg["currentMeans"], float), np.asarray(cfg["currentNormalizations"], float)
    sigmas, t0s = np.asarray(cfg["currentSigmas"], float), np.asarray(cfg["currentT0s"], float)
    integral_norm = math.log((rb // 2) / sr + t0s[0]) - math.log(t0s[0])
    tg = (np.arange(rb) - rb // 2) / sr
    tail_t = np.arange(rb) / sr
    pulses, pre, adc, p2a1 = [], [], [], []
    for ai in range(len(wd)):
        g = 1 / np.sqrt(2 * math.pi * sigmas[ai] ** 2) * np.exp(-((tg - means[ai]) ** 2) / (2 * sigmas[ai] ** 2))
        pulse = np.convolve(g, 1.0 / (tail_t + t0s[ai]))[:rb]
        pulse = pulse / (sr * 1000.0) / integral_norm * norms[ai]
        r_pre = calculate_response(cfg["preampPoles"], cfg["preampZeros"], pulse, sr)
        r_adc = calculate_response(cfg["ADCPoles"], cfg["ADCZeros"], pulse, sr)
        r_p1 = calculate_response(cfg["preampToAdc1Poles"], cfg["preampToAdc1Zeros"], pulse, sr)
        r_p1 = r_p1 * (max(0.0, r_pre.max()) / max(0.0, r_p1.max()))
        pulses.append(pulse), pre.append(r_pre), adc.append(r_adc), p2a1.append(r_p1)
    imp = np.zeros(rb)
    imp[rb // 2] = 1.0
    p2a2 = calculate_response(cfg["preampToAdc2Poles"], cfg["preampToAdc2Zeros"], imp, sr)
    test = calculate_response(cfg["preampToAdc2Poles"], cfg["preampToAdc2Zeros"], p2a1[0], sr)
    p2a2 = p2a2 * (max(0.0, adc[0].max()) / max(0.0, test.max()))
    resp = {"resp_thresh": np.array(pre), "resp_adc": np.array(adc), "resp_sat": np.array(p2a1), "resp_imp": p2a2,
            "current_pulse": np.array(pulses)}
    # thresholds, ADC gains and pedestals from TrkPreampStraw (index, delay_hv, delay_cal, thr_hv, thr_cal, gain)
    pr = preamp_rows
    if len(pr) != NUSTRAWS:
        raise ValueError(f"TrkPreampStraw has {len(pr)} rows, expected {NUSTRAWS}")
    vthresh = np.stack([pr[:, 4], pr[:, 3]], 1)                  # [ustraw, end], end 0 = cal, 1 = hv
    maxADC, adclsb = int(cfg["maxADC"]), float(cfg["ADCLSB"])
    ped = np.floor((maxADC + 1) / 2.0 - vthresh.mean(1) / adclsb * 20).astype(np.int64)
    dvdi_thresh = np.full(NUSTRAWS, float(cfg["defaultThresholddVdI"])) if not cfg["thresholddVdI"] \
        else np.asarray(cfg["thresholddVdI"], float)
    dvdi_adc = pr[:, 5].astype(float)
    off = np.zeros((NUSTRAWS, 2))
    if not cfg["overrideDbTimeOffsets"]:
        if delays is None:
            raise ValueError("overrideDbTimeOffsets is false: pass the TrkDelayPanel/TrkDelayRStraw tables")
        panel, rstraw = delays
        us = np.arange(NUSTRAWS)
        off[:, 0] = panel[us // NSTRAWS, 1] + rstraw[us % NSTRAWS, 2] + pr[:, 2]   # cal
        off[:, 1] = panel[us // NSTRAWS, 1] + rstraw[us % NSTRAWS, 1] + pr[:, 1]   # hv
    lsb, etd = float(cfg["TDCLSB"]), float(cfg["electronicsTimeDelay"])
    tp2dr = float(event_timing["TimeFromProtonsToDRMarker"])

    def tdc_response(t):
        return min(max(int(math.floor((t + etd) / lsb)), 0), (1 << int(cfg["numTDCbits"])) - 1)
    out = {"wire_points": wd, "vthresh": vthresh, "adc_ped": ped, "dvdi_thresh": dvdi_thresh, "dvdi_adc": dvdi_adc,
           "time_offsets": off,
           "digitizationStartTDC": tdc_response(float(cfg["digitizationStart"]) - etd - tp2dr),
           "digitizationEndTDC": tdc_response(float(cfg["digitizationEnd"]) - etd - tp2dr),
           "timeFromProtonsToDRMarker": tp2dr, "maxTDC": (1 << int(cfg["numTDCbits"])) - 1}
    out.update(resp)
    # maxResponseTime / maxLinearResponse tables per wire point, path (0 thresh, 1 adc), straw number (nominal lengths)
    k = np.arange(int(-5 * sr), int(20 * sr))
    times = k / sr
    tmax = np.zeros((len(wd), 2, NSTRAWS))
    linmax = np.zeros((len(wd), 2, NSTRAWS))
    for ip in range(len(wd)):
        for path, tab in ((0, resp["resp_thresh"]), (1, resp["resp_adc"])):
            for s in range(NSTRAWS):
                r = linear_response_table(tab, times, np.full_like(times, wd[ip]), np.full_like(times, nominal_halflen[s]),
                                          cfg, wd) * 1e-9
                i = int(np.argmax(r))
                tmax[ip, path, s], linmax[ip, path, s] = times[i], r[i] * 1e9
    out["tmax"], out["linmax"] = tmax, linmax
    return out


def dist_interp(wdist, wd):
    """StrawElectronics wire-distance interpolation: (lower point index, weight of the lower point) per distance."""
    wdist = np.asarray(wdist, float)
    k = np.zeros(wdist.shape, np.int64)
    for i in range(1, len(wd) - 1):
        k = np.where(wdist >= wd[i], i, k)
    frac = 1 - (wdist - wd[k]) / (wd[k + 1] - wd[k])
    return k, frac


def reflection(wdist, halflen, cfg):
    """Reflection delay [ns] and amplitude of the pulse reflected at the far end (StrawElectronics::linearResponse)."""
    L = 2 * np.asarray(halflen, float)
    path = 2 * L - 2 * np.asarray(wdist, float)
    return (float(cfg["reflectionTimeShift"]) + path / float(cfg["reflectionVelocity"]),
            float(cfg["reflectionFrac"]) * np.exp(-path / float(cfg["reflectionALength"])))


def resp_index(t, sr, rb):
    """int(time*sampleRate + responseBins/2.), clamped to the table (C++ truncation toward zero)."""
    return np.clip(np.trunc(np.asarray(t) * sr + rb / 2.0), 0, rb - 1).astype(np.int64)


def linear_response_table(tab, t, wdist, halflen, cfg, wd):
    """StrawElectronics::linearResponse per unit charge and unit dVdI, for table tab[point, bin]."""
    rb, sr = tab.shape[1], float(cfg["sampleRate"])
    k, f = dist_interp(wdist, wd)
    tr, sc = reflection(wdist, halflen, cfg)
    i, ir = resp_index(t, sr, rb), resp_index(np.asarray(t) - tr, sr, rb)
    p0 = tab[k, i] + tab[k, ir] * sc
    p1 = tab[k + 1, i] + tab[k + 1, ir] * sc
    return p0 * f + p1 * (1 - f)


# ---------------------------------------------------------------- straws (alignedTrackerSim, onlyWireOffsets)
def sim_straws(g, align_rows=None):
    """Straw axes (nominal) and wires (nominal + TrkAlignStrawSim wire-end offsets in the panel UVW frame), detector frame.

    Offline Panel UVW: u = wire direction of straw 0, v = unit(origin(straw 94) - origin(straw 0)), w = u x v.
    Straw direction points from the HV end to the cal end (cal end = mid + halflength * dir).
    """
    mid = geometry.to_detector(g.mid)
    sdir = g.dir / np.linalg.norm(g.dir, axis=1)[:, None]
    hl = np.asarray(g.halflen, float)
    up = np.asarray(g.upanel)
    first = up * NSTRAWS
    u = sdir[first]
    v = mid[first + 94] - mid[first]
    v /= np.linalg.norm(v, axis=1)[:, None]
    w = np.cross(u, v)
    cal, hv = mid + hl[:, None] * sdir, mid - hl[:, None] * sdir
    if align_rows is not None:
        a = align_rows
        if len(a) != len(mid):
            raise ValueError(f"TrkAlignStrawSim has {len(a)} rows, expected {len(mid)}")
        if np.any(a[:, 5:9] != 0):
            raise ValueError("TrkAlignStrawSim straw-end offsets are not zero (onlyWireOffsets expects none)")
        cal = cal + a[:, 1:2] * v + a[:, 2:3] * w
        hv = hv + a[:, 3:4] * v + a[:, 4:5] * w
    wmid = 0.5 * (cal + hv)
    wdir = cal - hv
    whl = 0.5 * np.linalg.norm(wdir, axis=1)
    wdir /= (2 * whl)[:, None]
    return {"straw_mid": mid, "straw_dir": sdir, "straw_halflen": hl, "wire_mid": wmid, "wire_dir": wdir,
            "wire_halflen": whl}


# ---------------------------------------------------------------- build / save / load
def build(fcl_dump, preamp_csv, gdml, align_csv=None, delay_csvs=None, global_constants=None):
    """Conditions from the files the stock job reads (see the module docstring)."""
    with open(fcl_dump) as f:
        d = parse_fhicl_dump(f.read())
    sources = {"fcl_dump": str(fcl_dump), "preamp": str(preamp_csv), "align": str(align_csv), "gdml": str(gdml),
               "global_constants": str(global_constants)}
    return assemble(d, read_db_csv(preamp_csv), geometry.from_gdml(gdml),
                    read_db_csv(align_csv) if align_csv else None,
                    [read_db_csv(p) for p in delay_csvs] if delay_csvs else None,
                    read_global_constants(global_constants), sources)


def assemble(d, preamp_rows, g, align_rows=None, delays=None, gc=None, sources=None):
    """Conditions from a parsed fhicl-dump table, DB rows and a StrawGeometry (Mu2e frame)."""
    gc = gc or dict(GLOBAL_DEFAULTS)
    pro = fcl_get(d, "services.ProditionsService")
    bfile = fcl_get(d, "services.GeometryService.bFieldFile")
    if "no_field" not in bfile:
        raise NotImplementedError(f"bFieldFile {bfile}: only the field-free drift (Bz = 0) is implemented")
    sd = dict(MAKESD_DEFAULTS)
    sd.update(fcl_get(d, "physics.producers.makeSD"))
    if sd["addCrossTalk"] or sd["addNoise"] or sd["DriftSingleElectrons"] or sd["UseStatus"]:
        raise NotImplementedError("makeSD with cross talk, noise, DriftSingleElectrons or UseStatus is not implemented")
    if sd.get("MixedDigisTag"):
        raise NotImplementedError("digi mixing is not implemented")
    ewm = fcl_get(d, "physics.producers.EWMProducer")
    et = pro["eventTiming"]
    ele = pro["strawElectronics"]
    phys = straw_physics(pro["strawPhysics"])
    drift = straw_drift(pro["strawDrift"], g.wire_r, g.wall_rmax, bz=0.0)
    el = straw_electronics(ele, et, preamp_rows, g.halflen[:NSTRAWS], delays)
    straws = sim_straws(g, align_rows)
    clock = 1000.0 / gc["nominalDAQFrequency"]
    onspill = int(ewm["SpillType"]) == 1
    if onspill:
        raise NotImplementedError("on-spill digitisation (microbunch folding, ghosts, EWM phase) is not implemented")
    scal = {
        "onspill": onspill, "eventWindowLength": float(et["OffSpillEventLength"]) * clock,
        "mbtime": gc["nominalDRPeriod"], "strawInnerRadius": float(g.radius), "strawOuterRadius": float(g.wall_rmax),
        "wireRadius": float(g.wire_r), "minstepE": float(sd["minstepE"]), "steptimebuf": float(sd["StrawGasStepTimeBuffer"]),
        "flashBuffer": float(sd["FlashTimeBuffer"]), "tdcbuf": float(sd["TDCTimeBuffer"]),
        "maxnclu": int(sd["MaxNClusters"]), "randrad": bool(sd["RandomizeRadius"]), "allStraw": int(sd["AllHitsStraw"]),
        "allPlanes": [int(x) for x in sd["AllHitsPlanes"]],
        **{k: float(ele[k]) for k in ("deadTimeAnalog", "deadTimeDigital", "saturationVoltage", "strawNoise",
                                      "thresholdAnalogNoise", "adcAnalogNoise", "ADCLSB", "ADCPeriod", "ADCOffset",
                                      "TDCLSB", "TOTLSB", "TDCResolution", "electronicsTimeDelay",
                                      "eventWindowMarkerROCJitter", "digitizationStart", "digitizationEnd", "sampleRate",
                                      "triggerHysteresis", "clusterLookbackTime", "reflectionTimeShift",
                                      "reflectionVelocity", "reflectionALength", "reflectionFrac")},
        **{k: int(ele[k]) for k in ("maxADC", "nADCPackets", "nADCPresamples", "maxThreshTimeSeparation", "maxTOT",
                                    "responseBins", "saturationSampleFactor")},
        "overrideDbTimeOffsets": bool(ele["overrideDbTimeOffsets"]),
    }
    scal["nADCSamples"] = NADC_MIN + scal["nADCPackets"] * NADC_PERPACKET
    scal["ttrunc"] = (scal["responseBins"] // 2) / scal["sampleRate"]
    for k in ("digitizationStartTDC", "digitizationEndTDC", "timeFromProtonsToDRMarker", "maxTDC"):
        scal[k] = el.pop(k)
    for k in ("EAverage", "NAverage", "meanFreePath", "Qe", "gasGain", "polyaA", "gainRMSSlope", "nGainGauss", "vprop",
              "nonlindrift"):
        scal[k] = phys.pop(k)
    if not scal["nonlindrift"]:
        raise NotImplementedError("useNonLinearDrift false (constant drift speed) is not implemented")
    scal["drift_dD"], scal["drift_dphi"] = drift.pop("drift_dD"), drift.pop("drift_dphi")
    scal["sources"] = dict(sources or {}, bFieldFile=bfile)
    return Conditions(scal, {**phys, **drift, **el, **straws})


class Conditions:
    """Scalars (attribute access) and arrays of the digitiser conditions; save/load as one .npz."""

    def __init__(self, scalars, arrays):
        self.s = dict(scalars)
        self.a = {k: np.asarray(v) for k, v in arrays.items()}

    def __getattr__(self, k):
        if k in ("s", "a"):
            raise AttributeError(k)
        if k in self.s:
            return self.s[k]
        if k in self.a:
            return self.a[k]
        raise AttributeError(k)

    def save(self, path):
        np.savez_compressed(path, __scalars__=np.array(json.dumps(self.s)), **self.a)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(json.loads(str(z["__scalars__"])), {k: z[k] for k in z.files if k != "__scalars__"})

    def tdc_response(self, t):
        x = np.floor((np.asarray(t, float) + self.electronicsTimeDelay) / self.TDCLSB)
        return np.clip(x, 0, self.maxTDC).astype(np.int64)
