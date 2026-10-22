"""JAX twin of hitreco.straw_hits (StrawHitReco port): same float32/float64 casts, padded shape-stable batches.

The numpy module is the reference. On CPU the outputs are bitwise equal to it (tests/test_reco_jax.py); on GPU
XLA may contract multiply-adds, so agreement there is checked in float32 ulp (`reco-vr1 --jax`).
Settings are runtime data (no recompile per Mu2eBO point); only the padded sizes set the compiled shape.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .hitreco import CENTRAL_WIRE_POS, EBIN, ERRFAC, FLAG, SH_SETTINGS, SLFAC, WBUF, unique_panel, unique_straw

jax.config.update("jax_enable_x64", True)

F32, F64 = jnp.float32, jnp.float64
SH_KEYS = ("minT", "maxT", "minE", "maxE", "minR", "maxR")


def bucket(n, lo=1024):
    """Padded size: next power of two (>= lo), so a run of batches compiles a handful of shapes."""
    return max(lo, 1 << int(np.ceil(np.log2(max(n, 1)))))


def pack_digis(d, npre, n_pad=None, ns_pad=None):
    """Flat digi arrays (hitreco.read_digis) -> padded device inputs; host glue, O(n)."""
    n = len(d["ev"])
    n_pad = n_pad or bucket(n)
    ns = int(d["nadc"].max()) if n else npre + 1
    ns_pad = ns_pad or max(ns, npre + 1)
    start = np.concatenate([[0], np.cumsum(d["nadc"])[:-1]]).astype(np.int64)
    pos = np.arange(len(d["adc"])) - np.repeat(start, d["nadc"])
    adc = np.zeros((n_pad, ns_pad), np.int64)
    adc[np.repeat(np.arange(n), d["nadc"]), pos] = d["adc"]
    nadc = np.zeros(n_pad, np.int64)
    nadc[:n] = d["nadc"]

    def pad(a, dt):
        out = np.zeros(n_pad, dt)
        out[:n] = a
        return out

    sid = d["sid"].astype(np.int64)
    return {"n": n, "us": pad(unique_straw(sid), np.int64), "tdc0": pad(d["tdc0"], np.int64), "tdc1": pad(d["tdc1"], np.int64),
            "tot0": pad(d["tot0"], np.int64), "tot1": pad(d["tot1"], np.int64), "pbt": pad(d["pbt"][d["ev"]], np.float64),
            "adc": adc, "nadc": nadc, "valid": pad(np.ones(n, bool), bool)}


def device_conditions(cond):
    """Conditions dict (hitreco.load_conditions) -> arrays and scalars for the jitted kernel."""
    sc, st = cond["scalar"], cond["straw"]
    c = {k: jnp.asarray(np.asarray(st[k], np.float64)) for k in
         ("midx", "midy", "midz", "dirx", "diry", "dirz", "hlen", "toff_hv", "toff_cal", "dead", "noisy", "halfvp")}
    for k in ("halfvpscale", "centres", "resslope", "totdtime", "totderror"):
        c[k] = jnp.asarray(np.asarray(cond[k], np.float64))
    for k in ("tdcLSB", "adcLSB", "totLSB", "strawGain", "eionPerQ", "pmpScaleAvg", "strawOuterRadius"):
        c[k] = jnp.float64(sc[k])
    c["tot_bins"] = tuple(cond["tot_bins"])
    c["npre"] = int(sc["nADCPreSamples"])
    return c


def _pieceline(xv, yv, x):
    xbin = (xv[-1] - xv[0]) / (xv.shape[0] - 1)
    ib = jnp.clip(jnp.floor((x - xv[0]) / xbin).astype(jnp.int64), 0, xv.shape[0] - 2)
    return yv[ib] + (x - xv[ib]) * (yv[ib + 1] - yv[ib]) / xbin


def _straw_hits(p, c, s, npre, tot_bins):
    us, valid = p["us"], p["valid"]
    ns = p["adc"].shape[1]
    pos = jnp.arange(ns)[None, :]
    inwf = pos < p["nadc"][:, None]
    ped = (jnp.sum(jnp.where(pos < npre, p["adc"], 0), axis=1) / F64(npre)).astype(F32)
    peak = jnp.max(jnp.where((pos >= npre) & inwf, p["adc"], jnp.iinfo(jnp.int32).min), axis=1)
    pmp = peak.astype(F64) - ped.astype(F64)
    charge = pmp * (c["adcLSB"] / c["strawGain"]) * c["pmpScaleAvg"]
    energy = (c["eionPerQ"] * charge).astype(F32)
    pbt = p["pbt"].astype(F32)
    t_cal = ((p["tdc0"] * c["tdcLSB"] + c["toff_cal"][us]).astype(F32) - pbt).astype(F32)
    t_hv = ((p["tdc1"] * c["tdcLSB"] + c["toff_hv"][us]).astype(F32) - pbt).astype(F32)
    eend = (t_hv < t_cal).astype(jnp.int64)
    tot_cal, tot_hv = (p["tot0"] * c["totLSB"]).astype(F32), (p["tot1"] * c["totLSB"]).astype(F32)
    tot_sel = jnp.where(eend == 1, tot_hv, tot_cal)
    dt = t_hv.astype(F64) - t_cal.astype(F64)
    kedep = 1000.0 * energy.astype(F64)
    egrid = jnp.arange(c["halfvpscale"].shape[0]) * EBIN
    halfpv = _pieceline(egrid, c["halfvpscale"], kedep) * c["halfvp"][us]
    wdist = halfpv * dt
    wres = _pieceline(egrid, c["centres"], kedep)
    wres = jnp.maximum(30.0, wres + jnp.where(jnp.abs(wdist) > CENTRAL_WIRE_POS,
                                              (jnp.abs(wdist) - CENTRAL_WIRE_POS) * _pieceline(egrid, c["resslope"], kedep), 0.0))
    hlen = c["hlen"][us]
    trunc = jnp.abs(wdist) > hlen + WBUF * wres
    wdist = jnp.where(trunc, jnp.copysign(hlen * SLFAC, wdist), wdist)
    wres = jnp.where(trunc, wres * ERRFAC, wres)
    esign = 2.0 * (eend - 0.5)
    propd = (hlen + esign * wdist).astype(F32)
    ptime = (propd / (2 * halfpv)).astype(F32)
    x = (c["midx"][us] + wdist * c["dirx"][us]).astype(F32)
    y = (c["midy"][us] + wdist * c["diry"][us]).astype(F32)
    z = (c["midz"][us] + wdist * c["dirz"][us]).astype(F32)
    rho = jnp.hypot(x, y)
    ntb, wtb, neb, web = tot_bins
    tbin = jnp.minimum(int(ntb) - 1, (tot_sel / F32(wtb)).astype(jnp.int64))
    ebin = jnp.minimum(int(neb) - 1, (energy / F32(web)).astype(jnp.int64))
    dtime = c["totdtime"][tbin * int(neb) + ebin]
    tres = c["totderror"][tbin * int(neb) + ebin]
    t_early = jnp.where(eend == 1, t_hv, t_cal).astype(F64)
    time = (t_early - ptime - dtime).astype(F32)
    flag = jnp.zeros(us.shape, jnp.uint32)
    for bit, ok in ((FLAG["dead"], c["dead"][us] > 0), (FLAG["noisy"], c["noisy"][us] > 0),
                    (FLAG["energysel"], (energy <= s["maxE"]) & (energy >= s["minE"])),
                    (FLAG["radsel"], (rho >= s["minR"]) & (rho <= s["maxR"])),
                    (FLAG["timesel"], (time >= s["minT"]) & (time <= s["maxT"])), (FLAG["tdiv"], ~trunc)):
        flag = flag | jnp.where(ok, jnp.uint32(1 << bit), jnp.uint32(0))
    flag = jnp.where(valid, flag, jnp.uint32(0))
    rout = c["strawOuterRadius"]
    return {"time": time, "et0": t_cal, "et1": t_hv, "tot0": tot_cal, "tot1": tot_hv, "edep": energy, "eend": eend,
            "dtime": dtime.astype(F32), "ptime": ptime, "wdist": wdist.astype(F32), "uvar": (wres * wres).astype(F32),
            "vvar": jnp.full(us.shape, (rout * rout / 3.0)).astype(F32), "timevar": (tres * tres).astype(F32),
            "x": x, "y": y, "z": z, "udx": c["dirx"][us].astype(F32), "udy": c["diry"][us].astype(F32), "flag": flag,
            "halfpv": halfpv}


_straw_hits_jit = jax.jit(_straw_hits, static_argnames=("npre", "tot_bins"))


def straw_hits(d, cond, settings="mc", dcond=None, packed=None):
    """numpy-in, numpy-out wrapper with the hitreco.straw_hits columns (padding trimmed)."""
    dcond = dcond or device_conditions(cond)
    p = packed or pack_digis(d, dcond["npre"])
    s = {k: jnp.float64(SH_SETTINGS[settings][k]) if isinstance(settings, str) else jnp.float64(settings[k]) for k in SH_KEYS}
    arrs = {k: jnp.asarray(v) for k, v in p.items() if k != "n"}
    tab = {k: v for k, v in dcond.items() if k not in ("tot_bins", "npre")}
    out = _straw_hits_jit(arrs, tab, s, npre=dcond["npre"], tot_bins=dcond["tot_bins"])
    n = p["n"]
    res = {k: np.asarray(v)[:n] for k, v in out.items()}
    sid = d["sid"]
    res.update(ev=d["ev"], sid=sid, upanel=unique_panel(sid.astype(np.int64)), straw=sid & 0x7F)
    return res
