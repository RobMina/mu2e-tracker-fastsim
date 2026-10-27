"""Opt-in fixed-shape JAX electronics with explicit random keys.

Host code packs complete straw instances into power-of-two cluster buckets. The
compiled kernel runs threshold searches and dead time in masked lock step. Pair
capacity is checked and retried, never silently truncated. NumPy digi.py remains
the independent default/reference. Same keys, inputs and bucket configuration
reproduce a run; random streams need not match NumPy or another JAX version.
"""
from functools import lru_cache

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, random

from . import digi as reference

jax.config.update("jax_enable_x64", True)

BATCH = 32
MAX_ROW_CLUSTERS = 4096


def _empty(c):
    return {"ev": np.zeros(0, np.int64), "sid": np.zeros(0, np.int64),
            "tdc": np.zeros((0, 2), np.int64), "tot": np.zeros((0, 2), np.int64),
            "pmp": np.zeros(0, np.int64), "adc": np.zeros((0, c.nADCSamples), np.int64),
            "step": np.zeros((0, 2), np.int64), "t_analog": np.zeros((0, 2))}


@lru_cache(maxsize=8)
def _charge_kernel(c):
    a = {k: jnp.asarray(v) for k, v in c.a.items()}

    @jax.jit
    def run(key, us, pos, ne, time):
        k1, k2, k3 = random.split(key, 3)
        aval = jnp.where(ne == 1, c.polyaA, ne * (c.polyaA + 1) - 1).astype(jnp.float64)
        gain = jnp.where(ne < c.nGainGauss, random.gamma(k1, aval) / aval,
                         1 + c.gainRMSSlope / jnp.sqrt(ne) * random.normal(k2, ne.shape))
        charge = (c.Qe * ne * c.gasGain * gain).astype(jnp.float32)
        d = pos - a["straw_mid"][us]
        dw = jnp.sum(d * a["straw_dir"][us], 1)
        cp = d - dw[:, None] * a["straw_dir"][us]
        rho = jnp.linalg.norm(cp, axis=1)
        pos = pos - cp / jnp.maximum(rho, 1e-30)[:, None] * jnp.maximum(rho - c.strawInnerRadius, 0)[:, None]
        d = pos - a["wire_mid"][us]
        zw = jnp.sum(d * a["wire_dir"][us], 1)
        rho = jnp.linalg.norm(d - zw[:, None] * a["wire_dir"][us], axis=1)
        di = jnp.clip(jnp.floor(rho / c.drift_dD).astype(jnp.int32), 0, len(c.drift_dist) - 2)
        fr = (rho - a["drift_dist"][di]) / c.drift_dD
        td = a["drift_times"][di, 0] + fr * (a["drift_times"][di + 1, 0] - a["drift_times"][di, 0])
        td = (td + (c.dtvar[0] + c.dtvar[1] * rho) * random.normal(k3, ne.shape)).astype(jnp.float32)
        hl = a["wire_halflen"][us]
        wd = jnp.stack([hl - zw, hl + zw], 1).astype(jnp.float32)
        prop = (wd / c.vprop).astype(jnp.float32)
        gt = (time[:, None] + td[:, None].astype(jnp.float64) + prop.astype(jnp.float64)).astype(jnp.float32)
        return gt, (jnp.float32(0.5) * charge), wd
    return run


@lru_cache(maxsize=32)
def _electronics_kernel(c, cap, pair_cap, batch):
    rb, sr, LB = c.responseBins, c.sampleRate, c.clusterLookbackTime
    R = batch * 2
    rows = jnp.arange(R)
    ci = jnp.arange(cap)[None, :]
    cond = {k: jnp.asarray(v) for k, v in c.a.items()}
    tables = {k: cond["resp_" + k].astype(jnp.float32) for k in ("thresh", "adc", "sat")}
    dts = c.saturationSampleFactor / sr
    sat_steps = int(np.ceil((c.ttrunc + c.nADCSamples * c.ADCPeriod + 2 * LB) / dts)) + 2
    ticks = jnp.arange(1, c.maxTOT)
    adc_k = jnp.arange(c.nADCSamples)

    @jax.jit
    def run(key, t, q, wd, step, n, straw_us, straw_ra, straw_key, live_straw, jitter):
        n = n.astype(jnp.int32)
        us = jnp.repeat(straw_us, 2)
        live = jnp.repeat(live_straw, 2)
        valid = (ci < n[:, None]) & live[:, None]
        wp = cond["wire_points"]
        dk = jnp.clip(jnp.searchsorted(wp, wd, side="right") - 1, 0, len(c.wire_points) - 2)
        frac = 1 - (wd - wp[dk]) / (wp[dk + 1] - wp[dk])
        path = 4 * cond["wire_halflen"][us, None] - 2 * wd
        tr = float(c.s["reflectionTimeShift"]) + path / float(c.s["reflectionVelocity"])
        sc = (float(c.s["reflectionFrac"]) * jnp.exp(-path / float(c.s["reflectionALength"]))).astype(jnp.float32)
        aa, bb = (q * frac).astype(jnp.float32), (q * (1 - frac)).astype(jnp.float32)
        dv_t, dv_a = cond["dvdi_thresh"][us], cond["dvdi_adc"][us]
        sn = us % 96
        tmax = cond["tmax"][dk, 0, sn[:, None]] * frac + cond["tmax"][dk + 1, 0, sn[:, None]] * (1 - frac)
        maxlin = (cond["linmax"][dk, 0, sn[:, None]] * frac + cond["linmax"][dk + 1, 0, sn[:, None]] * (1 - frac)) * q
        maxlin = jnp.where(valid, maxlin * dv_t[:, None], 0)
        sums = jnp.cumsum(maxlin, axis=1)
        maxposs = jnp.sum(maxlin, axis=1)
        keys = jax.vmap(lambda x: random.fold_in(random.fold_in(key, (x >> 32).astype(jnp.uint32)), x.astype(jnp.uint32)))(straw_key)

        def normal(stage, iteration, shape):
            kk = jax.vmap(lambda k: random.fold_in(random.fold_in(k, stage), iteration))(keys)
            return jax.vmap(lambda k: random.normal(k, shape, dtype=jnp.float64))(kk)

        def at(x, i):
            return x[rows, jnp.clip(i, 0, cap - 1)]

        def count(tlim, inclusive=False):
            cmp = t <= tlim[:, None] if inclusive else t < tlim[:, None]
            return jnp.sum(valid & cmp, axis=1).astype(jnp.int32)

        def volt(tq, path="thresh", cmin=None, lattice=None, kmax=None):
            tq = tq.reshape(R, -1)
            T = tables[path]
            if lattice is None:
                u = ((tq[:, None, :] - t[:, :, None]) * sr + rb / 2.).astype(jnp.float32)
                ix = jnp.clip(u, 0, rb - 1).astype(jnp.int32)
                ir = jnp.clip(u - (tr * sr).astype(jnp.float32)[:, :, None], 0, rb - 1).astype(jnp.int32)
            else:
                u = (tq[:, 0, None] - t) * sr + rb / 2.
                stride = jnp.arange(tq.shape[1]) * lattice
                ix = jnp.clip(jnp.floor(u)[:, :, None] + stride, 0, rb - 1).astype(jnp.int32)
                ir = jnp.clip(jnp.floor(u - tr * sr)[:, :, None] + stride, 0, rb - 1).astype(jnp.int32)
            k = dk[:, :, None]
            v = aa[:, :, None] * (T[k, ix] + sc[:, :, None] * T[k, ir]) + bb[:, :, None] * (T[k + 1, ix] + sc[:, :, None] * T[k + 1, ir])
            ok = valid[:, :, None] & (t[:, :, None] - LB < tq[:, None, :])
            if cmin is not None:
                ok &= (ci >= cmin[:, None])[:, :, None]
            v = jnp.sum(jnp.where(ok, v, 0).astype(jnp.float64), axis=1)
            v *= (dv_a if path == "adc" else dv_t)[:, None]
            if kmax is not None:
                v = jnp.where(jnp.arange(tq.shape[1])[None, :] < kmax[:, None], v, 0)
            return v

        def crosses(ic, time, thr, vcross, active):
            ic = jnp.where(active, jnp.maximum(ic, count(time + LB)), ic)
            active &= ic < n
            vstart = volt(at(t, ic) - LB)[:, 0]
            retact = active & (vstart > thr)

            def retbody(state):
                ic, time, vs, act, it = state
                nxt = at(t, ic) + LB + 2 * at(tmax, ic)
                newic = jnp.maximum(ic, count(nxt + LB))
                ic = jnp.where(act, newic, ic)
                act &= ic < n
                time = jnp.where(act, at(t, ic) - LB, time)
                vs = jnp.where(act, volt(time)[:, 0], vs)
                return ic, time, vs, act & (vs > thr), it + 1
            ic, time, vstart, stuck, _ = lax.while_loop(
                lambda x: jnp.any(x[3]) & (x[4] <= cap), retbody,
                (ic, time, vstart, retact, jnp.int32(0)))
            active &= ic < n
            before = jnp.where(ic > 0, at(sums, ic - 1), 0.)
            j = jnp.maximum(ic, jnp.sum(valid & (sums <= (thr - vstart + before)[:, None]), 1).astype(jnp.int32))
            ic = jnp.where(active & (j >= n), n, ic)
            active &= j < n
            time = jnp.where(active, at(t, j) - LB, time)
            star = jnp.full(R, -1, jnp.int32)
            response = jnp.zeros(R)

            def peakbody(state):
                j, star, response, act = state
                cand = j[:, None] + jnp.arange(8)[None, :]
                gc = jnp.clip(cand, 0, cap - 1)
                mt = t[rows[:, None], gc] + tmax[rows[:, None], gc]
                v = volt(mt)
                hit = (cand < n[:, None]) & (v > thr[:, None]) & act[:, None]
                anyhit = jnp.any(hit, 1)
                first = jnp.argmax(hit, 1).astype(jnp.int32)
                star = jnp.where(anyhit, j + first, star)
                response = jnp.where(anyhit, v[rows, first], response)
                j = jnp.where(act & ~anyhit, j + 8, j)
                return j, star, response, act & ~anyhit & (j < n)
            _, star, response, _ = lax.while_loop(lambda x: jnp.any(x[3]), peakbody, (j, star, response, active))
            ic = jnp.where(active & (star < 0), n, ic)
            found = active & (star >= 0)
            pre, post = at(t, star) - LB, at(t, star) + at(tmax, star)
            pres, posts = volt(pre)[:, 0], response
            tt = pre + (post - pre) / (posts - pres) * (thr - pres)
            dt = post - pre

            def finebody(_, state):
                pre, post, pres, posts, tt, dt = state
                act = found & (jnp.abs(dt) > .020)
                smp = volt(tt)[:, 0]
                up = smp > thr
                post = jnp.where(act & up, tt, post)
                posts = jnp.where(act & up, smp, posts)
                pre = jnp.where(act & ~up, tt, pre)
                pres = jnp.where(act & ~up, smp, pres)
                newt = pre + (post - pre) / (posts - pres) * (thr - pres)
                return pre, post, pres, posts, jnp.where(act, newt, tt), jnp.where(act, newt - tt, dt)
            _, _, _, _, tt, _ = lax.fori_loop(0, 10, finebody, (pre, post, pres, posts, tt, dt))
            time = jnp.where(found, tt, time)
            vcross = jnp.where(found, thr, vcross)
            advance = jnp.maximum(star, count(tt + LB))
            ic = jnp.where(found, jnp.maximum(advance - 1, 0), ic)
            return ic, time, thr, vcross, found, jnp.any(stuck)

        noise = normal(0, 0, ()) * c.strawNoise
        threshold = (cond["vthresh"][straw_us] + noise[:, None] + c.thresholdAnalogNoise * normal(1, 0, (2,))).reshape(R)
        ts = jnp.where(straw_ra, -c.deadTimeAnalog, c.digitizationStart - c.timeFromProtonsToDRMarker - c.flashBuffer)
        ic, tt, th, vv, cr, failed = crosses(jnp.zeros(R, jnp.int32), jnp.repeat(ts, 2), threshold, jnp.zeros(R), live)
        end = max(c.eventWindowLength, c.mbtime) - c.electronicsTimeDelay + c.tdcbuf

        def active(cr, tt):
            t0, t1 = tt[::2], tt[1::2]
            return live_straw & cr[::2] & cr[1::2] & (jnp.where(t0 < t1, t1, t0) < end)

        output_t = jnp.zeros((batch, pair_cap, 2))
        output_v = jnp.zeros_like(output_t)
        output_i = jnp.zeros((batch, pair_cap, 2), jnp.int32)
        number = jnp.zeros(batch, jnp.int32)

        def pairbody(state):
            ic, tt, th, vv, cr, ot, ov, oi, number, it, fail = state
            act = active(cr, tt)
            t0, t1 = tt[::2], tt[1::2]
            finite = jnp.isfinite(t0) & jnp.isfinite(t1)
            comb = act & finite & (jnp.abs(jnp.floor(t0 / c.ADCPeriod) - jnp.floor(t1 / c.ADCPeriod)) < c.maxThreshTimeSeparation)
            rec = comb & (jnp.minimum(t0, t1) > 0.)
            loc = jnp.minimum(number, pair_cap - 1)
            bi = jnp.arange(batch)
            ot = ot.at[bi, loc].set(jnp.where(rec[:, None], tt.reshape(batch, 2), ot[bi, loc]))
            ov = ov.at[bi, loc].set(jnp.where(rec[:, None], vv.reshape(batch, 2), ov[bi, loc]))
            oi = oi.at[bi, loc].set(jnp.where(rec[:, None], ic.reshape(batch, 2), oi[bi, loc]))
            fail |= jnp.any(rec & (number >= pair_cap))
            number += rec.astype(jnp.int32)
            early = jnp.where(t0 < t1, 0, 1)
            adv = jnp.repeat(comb, 2) | (jnp.repeat(act & ~comb, 2) & (rows % 2 == jnp.repeat(early, 2)))
            tt = jnp.where(adv, tt + c.deadTimeAnalog, tt)
            ic = ic + adv.astype(jnp.int32)
            fresh = (cond["vthresh"][straw_us] + c.thresholdAnalogNoise * normal(2, it, (2,))).reshape(R)
            th = jnp.where(jnp.repeat(comb, 2), fresh, th)
            ic, tt, th, vv, newcr, stuck = crosses(ic, tt, th, vv, adv)
            cr = jnp.where(adv, newcr, cr)
            return ic, tt, th, vv, cr, ot, ov, oi, number, it + 1, fail | stuck
        state = lax.while_loop(lambda x: jnp.any(active(x[4], x[1])) & (x[9] <= 2 * cap), pairbody,
                              (ic, tt, th, vv, cr, output_t, output_v, output_i, number, jnp.int32(0), failed))
        ic, tt, th, vv, cr, pt, pv, pi, number, _, failed = state
        failed |= jnp.any(active(cr, tt))
        ready = jnp.full(batch, -9e9)
        sat_overflow = jnp.bool_(False)
        adc_out = jnp.zeros((batch, pair_cap, c.nADCSamples), jnp.int64)
        tdc_out = jnp.zeros((batch, pair_cap, 2), jnp.int64)
        tot_out = jnp.zeros_like(tdc_out)
        accepted = jnp.zeros((batch, pair_cap), bool)

        def digitbody(k, state):
            ready, adc_out, tdc_out, tot_out, accepted, overflow = state
            present = live_straw & (k < number)
            T, V = pt[:, k], pv[:, k]
            xt = T + jitter[:, None] + c.TDCResolution * normal(3, k, (2,))
            tot_thr = V + c.thresholdAnalogNoise * normal(4, k, (2,))
            tq = (T + jitter[:, None]).reshape(R, 1) + ticks[None, :] * c.TOTLSB
            volts_tot = volt(tq, lattice=int(round(c.TOTLSB * sr)))
            below = volts_tot < (tot_thr.reshape(R, 1) - c.triggerHysteresis)
            tot = jnp.where(jnp.any(below, 1), jnp.argmax(below, 1) + 1, c.maxTOT).reshape(batch, 2)
            phase = jnp.maximum(0, jnp.ceil(T[:, 0] / c.ADCPeriod).astype(jnp.int64) - c.nADCPresamples)
            times = jnp.repeat((phase[:, None] + adc_k) * c.ADCPeriod + c.ADCOffset, 2, axis=0)
            volts = volt(times, "adc", lattice=int(round(c.ADCPeriod * sr)))
            sat = (maxposs > c.saturationVoltage) & jnp.repeat(present, 2)

            def saturation(v):
                i0 = jnp.minimum(count(times[:, 0] + LB - c.ttrunc, inclusive=True), jnp.maximum(n - 1, 0))
                t0 = at(t, i0) - LB
                nstep = jnp.maximum(0, jnp.ceil((times[:, -1] - t0) / dts).astype(jnp.int32))
                tq = t0[:, None] + dts * jnp.arange(sat_steps)[None, :]
                resp = volt(tq, "sat", cmin=i0, lattice=c.saturationSampleFactor, kmax=nstep)
                resp = jnp.minimum(resp, c.saturationVoltage)
                scale = c.saturationSampleFactor * dv_a / dv_t
                def sample(tm):
                    ii = jnp.clip(jnp.trunc((tm[:, None] - tq) * sr + rb / 2.), 0, rb - 1).astype(jnp.int32)
                    return jnp.sum(resp * cond["resp_imp"][ii], 1) * scale
                result = jax.vmap(sample, in_axes=1, out_axes=1)(times)
                return jnp.where(sat[:, None], result, v), jnp.any(sat & (nstep > sat_steps))
            volts, of = lax.cond(jnp.any(sat), saturation, lambda v: (v, jnp.bool_(False)), volts)
            overflow |= of
            volts = volts.reshape(batch, 2, c.nADCSamples).sum(1)
            dtime = jnp.max(xt, 1)
            if not c.overrideDbTimeOffsets:
                xt -= cond["time_offsets"][straw_us]
            wsum = volts + c.adcAnalogNoise * normal(5, k, (c.nADCSamples,))
            tdc = jnp.clip(jnp.floor((xt + c.electronicsTimeDelay) / c.TDCLSB), 0, c.maxTDC).astype(jnp.int64)
            ewend = int(c.tdc_response(c.eventWindowLength - c.electronicsTimeDelay))
            ok_all = jnp.all((xt + c.electronicsTimeDelay > 0) & (tdc < ewend), 1)
            upper = c.digitizationEndTDC if c.onspill else ewend
            ok_norm = jnp.all((tdc > c.digitizationStartTDC) & (tdc < upper), 1)
            keep = present & jnp.where(straw_ra, ok_all, ok_norm) & (dtime >= ready)
            ready = jnp.where(keep, dtime + c.deadTimeDigital, ready)
            adc = jnp.clip(jnp.floor(wsum / c.ADCLSB).astype(jnp.int64) + cond["adc_ped"][straw_us, None], 0, c.maxADC)
            return ready, adc_out.at[:, k].set(adc), tdc_out.at[:, k].set(tdc), tot_out.at[:, k].set(tot), accepted.at[:, k].set(keep), overflow
        _, adc_out, tdc_out, tot_out, accepted, sat_overflow = lax.fori_loop(
            0, jnp.minimum(pair_cap, jnp.max(number)), digitbody,
            (ready, adc_out, tdc_out, tot_out, accepted, sat_overflow))
        pmp = (adc_out.max(2) - adc_out[:, :, :c.nADCPresamples].sum(2) // c.nADCPresamples) & 0xffff
        step_out = jnp.stack([step[2 * jnp.arange(batch)[:, None], jnp.clip(pi[:, :, 0], 0, cap - 1)],
                              step[2 * jnp.arange(batch)[:, None] + 1, jnp.clip(pi[:, :, 1], 0, cap - 1)]], 2)
        return {"adc": adc_out, "tdc": tdc_out, "tot": tot_out, "pmp": pmp, "step": step_out,
                "t_analog": pt, "keep": accepted, "failed": failed, "sat_overflow": sat_overflow,
                "pairs": number.sum(), "nan_ends": jnp.sum(jnp.isnan(tt) & live),
                "saturated_ends": jnp.sum((maxposs > c.saturationVoltage) & live)}
    return run


def digitize(st, nev, c, key, stats=None):
    """Digitise flat steps with an explicit JAX key, returning host arrays like digi.digitize.

    Host boundaries select and pack variable-sized arrays; ionisation and electronics
    are JAX kernels. Conditions must be immutable for the life of the kernel cache.
    """
    from .digi_clusters_jax import make_clusters
    random.key_data(key)
    if not np.all(c.drift_times == c.drift_times[:, :1]) or c.onspill:
        raise NotImplementedError("JAX digitiser requires off-spill field-free conditions")
    st = {k: np.asarray(v) for k, v in st.items()}
    for field in ("t", "eion", "len", "width", "x0", "y0", "z0", "x1", "y1", "z1"):
        if not np.isfinite(st[field]).all():
            raise ValueError(f"nonfinite step {field}")
    sid, ev = st["sid"].astype(np.int64), st["ev"].astype(np.int64)
    if np.any(ev < 0) or np.any(ev >= nev):
        raise ValueError("step event index outside event population")
    us = reference.unique_straw(sid)
    if np.any(us < 0) or np.any(us >= len(c.wire_halflen)):
        raise ValueError("invalid StrawId")
    st["_us"] = us
    ra = ((sid & 0x7f) >= c.allStraw) & np.isin(sid >> 10, c.allPlanes)
    time = st["t"].astype(np.float64)
    start = c.digitizationStart - c.timeFromProtonsToDRMarker
    inwin = (time > start - c.electronicsTimeDelay - c.steptimebuf) & (time < max(c.mbtime, c.eventWindowLength) - c.electronicsTimeDelay + c.steptimebuf)
    idx = np.flatnonzero((st["eion"] > c.minstepE) & (inwin | ra))
    if not len(idx):
        return _empty(c)
    cs, pos, ne = make_clusters(random.fold_in(key, 0), c, st, idx)
    if not len(cs):
        return _empty(c)
    charge_blocks = []
    for first in range(0, len(cs), 4096):
        last = min(first + 4096, len(cs))
        size = last - first
        pad = 4096 - size
        args = (np.pad(us[cs[first:last]], (0, pad)), np.pad(pos[first:last], ((0, pad), (0, 0))),
                np.pad(ne[first:last], (0, pad), constant_values=1), np.pad(time[cs[first:last]], (0, pad)))
        result = _charge_kernel(c)(random.fold_in(random.fold_in(key, 1), first), *map(jnp.asarray, args))
        charge_blocks.append(tuple(np.asarray(x)[:size] for x in result))
    gt, charge, wd = (np.concatenate([b[i] for b in charge_blocks]) for i in range(3))
    if not all(np.isfinite(x).all() for x in (gt, charge, wd)):
        raise ValueError("nonfinite drift/charge cluster before waveform packing")
    ukey, inst = np.unique(ev[cs] * 65536 + sid[cs], return_inverse=True)
    ns = len(ukey)
    straw_ev, straw_sid = ukey // 65536, ukey % 65536
    straw_us = reference.unique_straw(straw_sid)
    straw_ra = ((straw_sid & 0x7f) >= c.allStraw) & np.isin(straw_sid >> 10, c.allPlanes)
    panel = reference.unique_panel(straw_sid)
    jp = random.normal(random.fold_in(key, 2), (nev, reference.NUPANELS), dtype=jnp.float64) * c.eventWindowMarkerROCJitter
    jitter = np.asarray(jp)[straw_ev, panel]
    counts = np.bincount(inst, minlength=ns)
    if counts.max() > MAX_ROW_CLUSTERS:
        raise OverflowError(f"waveform cluster count exceeds MAX_ROW_CLUSTERS={MAX_ROW_CLUSTERS}; no output returned")
    caps = np.left_shift(np.int64(1), np.ceil(np.log2(counts)).astype(np.int64))
    order = np.argsort(inst, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    outs = []
    for cap in np.unique(caps):
        bucket = np.flatnonzero(caps == cap)
        for b0 in range(0, len(bucket), BATCH):
            which = bucket[b0:b0 + BATCH]
            nb = len(which)
            gather = starts[which, None] + np.arange(cap)[None, :]
            mask = np.arange(cap)[None, :] < counts[which, None]
            ci_ = order[np.minimum(gather, len(order) - 1)]
            t = np.zeros((BATCH, 2, cap)); q = np.zeros_like(t); w = np.zeros_like(t)
            steps = np.zeros((BATCH, 2, cap), np.int64)
            n = np.zeros((BATCH, 2), np.int32)
            for end in (0, 1):
                oo = np.argsort(np.where(mask, gt[ci_, end], np.inf), axis=1, kind="stable")
                ix = np.take_along_axis(ci_, oo, 1)
                t[:nb, end] = gt[ix, end]
                q[:nb, end] = charge[ix]
                w[:nb, end] = wd[ix, end]
                steps[:nb, end] = cs[ix]
                n[:nb, end] = counts[which]
            su = np.zeros(BATCH, np.int64); sr = np.zeros(BATCH, bool); sk = np.zeros(BATCH, np.int64)
            live = np.arange(BATCH) < nb; ji = np.zeros(BATCH)
            su[:nb], sr[:nb], sk[:nb], ji[:nb] = straw_us[which], straw_ra[which], ukey[which], jitter[which]
            pair_cap = 4
            while True:
                result = _electronics_kernel(c, int(cap), pair_cap, BATCH)(random.fold_in(key, 3),
                    *[jnp.asarray(x.reshape(2 * BATCH, cap)) for x in (t, q, w, steps)], jnp.asarray(n.ravel()),
                    jnp.asarray(su), jnp.asarray(sr), jnp.asarray(sk), jnp.asarray(live), jnp.asarray(ji))
                result = {k: np.asarray(v) for k, v in result.items()}
                if result["sat_overflow"]:
                    raise RuntimeError("JAX ADC saturation time bucket overflow; no output returned")
                if not result["failed"]:
                    break
                if pair_cap >= 2 * cap:
                    raise RuntimeError("JAX threshold search failed to progress; no output returned")
                pair_cap *= 2
            bi, pi = np.nonzero(result["keep"])
            out = {k: result[k][bi, pi] for k in ("adc", "tdc", "tot", "pmp", "step", "t_analog")}
            out.update(ev=straw_ev[which[bi]], sid=straw_sid[which[bi]])
            outs.append(out)
            if stats is not None:
                for field, val in (("pairs", result["pairs"]), ("nan_crossing_ends", result["nan_ends"]),
                                   ("saturated_ends", result["saturated_ends"]), ("jax_buckets", 1)):
                    stats[field] = stats.get(field, 0) + int(val)
    out = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    order = np.lexsort((out["t_analog"][:, 0], out["sid"], out["ev"]))
    if stats is not None:
        for field, val in (("clusters", len(cs)), ("straws", ns), ("ends", 2 * ns), ("digis", len(order))):
            stats[field] = stats.get(field, 0) + int(val)
    return {k: v[order] for k, v in out.items()}
