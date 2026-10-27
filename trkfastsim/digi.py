"""Fast tracker digitisation: a vectorised numpy port of Offline StrawDigisFromStrawGasSteps (TrackerMC, v13_39_00).

StrawGasSteps -> ionisation clusters -> drift to the wire -> propagation to both ends -> shaped waveform per end ->
threshold crossings (both ends, combined) -> TDC, TOT, 15-sample ADC waveform, PMP -> StrawDigis, in the layout of
the StrawHitTableDumper `trkdump/events` tree that trkfastsim.reco.hitreco reads. Conditions: trkfastsim.digicond.

The per-straw sequential logic of the stock module (StrawWaveform::crossesThreshold with its rough/fine crossing and
return-below-threshold steps, the pair loop of findThresholdCrossings, the digital dead time between digis) runs in
lock step over all straw ends of a batch: every iteration evaluates the waveform at one or a few query times per
active row (a segment sum over the row's clusters), and finished rows drop out. The waveform evaluation is the
stock one (0.1 ns response tables, wire-distance interpolation, end reflection, 5 ns cluster look-back).

Random numbers come from one numpy Generator; the draws are statistically equivalent to the stock CLHEP draws, not
the same numbers. Rejection loops of the stock module (cluster sizes of minimum-ionising steps) are sampled as the
equivalent truncated distributions. Only the off-spill, field-free, no-cross-talk, no-noise configuration of the
extracted (KPP) digitisation is implemented; digicond.build refuses others.
"""
import numpy as np

from .digicond import NSTRAWS, NPANELS, d2t, dist_interp, reflection

F32 = np.float32
NUPANELS = 216


def unique_straw(sid):
    sid = np.asarray(sid, np.int64)
    return ((sid >> 10) * NPANELS + ((sid >> 7) & 0x7)) * NSTRAWS + (sid & 0x7F)


def unique_straw_to_sid(us):
    us = np.asarray(us, np.int64)
    up = us // NSTRAWS
    return ((up // NPANELS) << 10) | ((up % NPANELS) << 7) | (us % NSTRAWS)


def unique_panel(sid):
    sid = np.asarray(sid, np.int64)
    return (sid >> 10) * NPANELS + ((sid >> 7) & 0x7)


def _ranges(starts, n):
    """Flat indices of the concatenated ranges [starts[i], starts[i] + n[i])."""
    tot = int(n.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    off = np.cumsum(n) - n
    return np.repeat(starts - off, n) + np.arange(tot)


# ---------------------------------------------------------------- ionisation clusters
def _trunc_ne(rng, intNProb, EIonize, emax):
    """nePerIon(u) drawn until EIonize[ne-1] < emax (the stock rejection loops), as one truncated draw."""
    kmax = np.searchsorted(EIonize, emax, side="left")                  # number of sizes with energy < emax
    u = rng.random(len(emax)) * intNProb[np.maximum(kmax, 1) - 1]
    return np.searchsorted(intNProb, u, side="right") + 1


def _minion_clusters(rng, c, etot, nc):
    """StrawDigisFromStrawGasSteps::fillClusterMinion for many steps: electrons per cluster, [sum(nc)]."""
    EI, P = c.EIonize, c.intNProb
    E1 = EI[0]
    ns = len(etot)
    off = np.cumsum(nc) - nc
    ne = np.zeros(int(nc.sum()), np.int64)
    esum = np.zeros(ns)
    for m in range(int(nc.max()) if ns else 0):
        a = np.nonzero(nc > m)[0]
        emax = etot[a] - esum[a] - (nc[a] - m - 1) * E1
        ie = _trunc_ne(rng, P, EI, emax)
        ne[off[a] + m] = ie
        esum[a] += EI[np.minimum(ie, len(EI)) - 1]
    # residual energy: further ionisations added to random clusters until less than one electron's energy is left
    res = etot - esum
    emaxion = EI[-1]
    a = np.nonzero(res >= E1)[0]
    block = 64
    while a.size:
        big = a[res[a] > 2 * emaxion]
        if big.size:                                     # every draw fits: accept the prefix that keeps res > max
            u = rng.random((big.size, block))
            ie = np.searchsorted(P, u, side="right") + 1
            e = EI[np.minimum(ie, len(EI)) - 1]
            acc = np.cumsum(e, 1) <= (res[big] - emaxion)[:, None]
            cl = np.minimum(nc[big][:, None] - 1, np.floor(rng.random((big.size, block)) * nc[big][:, None]).astype(np.int64))
            r, k = np.nonzero(acc)
            np.add.at(ne, off[big[r]] + cl[r, k], ie[r, k])
            res[big] -= (e * acc).sum(1)
        small = a[res[a] <= 2 * emaxion]
        if small.size:
            ie = _trunc_ne(rng, P, EI, res[small])
            cl = np.minimum(nc[small] - 1, np.floor(rng.random(small.size) * nc[small]).astype(np.int64))
            np.add.at(ne, off[small] + cl, ie)
            res[small] -= EI[np.minimum(ie, len(EI)) - 1]
        a = a[res[a] >= E1]
    return ne


def make_clusters(rng, c, st, idx):
    """divideStep for the steps idx -> per-cluster (step index, position [n,3], electron count)."""
    eion = st["eion"][idx].astype(np.float64)
    ln = st["len"][idx].astype(np.float64)
    single = (st["shape"][idx] == 3) | (ln < c.meanFreePath)
    minion = st["ion"][idx] == 0
    p0 = np.stack([st["x0"][idx], st["y0"][idx], st["z0"][idx]], 1).astype(np.float64)
    p1 = np.stack([st["x1"][idx], st["y1"][idx], st["z1"][idx]], 1).astype(np.float64)
    out_s, out_p, out_n = [], [], []
    # single cluster at the start point
    s1 = np.nonzero(single)[0]
    if s1.size:
        out_s.append(idx[s1])
        out_p.append(p0[s1])
        out_n.append(np.maximum(rng.poisson(eion[s1] / c.EAverage), 1))
    m = np.nonzero(~single)[0]
    if m.size:
        nc = np.maximum(rng.poisson(ln[m] / c.meanFreePath), 1)
        nc = np.where(minion[m], nc, np.minimum(nc, c.maxnclu))
        nc = np.minimum(nc, np.floor(eion[m] / c.EIonize[0]).astype(np.int64))
        keep = nc > 0
        m, nc = m[keep], nc[keep]
        stp = np.repeat(m, nc)
        path = p1[stp] - p0[stp]
        pos = p0[stp] + rng.random(len(stp))[:, None] * path
        if c.randrad:
            sdir = c.straw_dir[st["_us"][idx[stp]]]
            q1 = np.cross(path, sdir)
            q1 /= np.where((n1 := np.linalg.norm(q1, axis=1)) > 0, n1, 1.0)[:, None]
            q2 = np.cross(path, q1)
            q2 /= np.where((n2 := np.linalg.norm(q2, axis=1)) > 0, n2, 1.0)[:, None]
            w = st["width"][idx[stp]].astype(np.float64)[:, None]
            pos = pos + q1 * rng.standard_normal(len(stp))[:, None] * w + q2 * rng.standard_normal(len(stp))[:, None] * w
        mi = minion[m]
        ne = np.zeros(len(stp), np.int64)
        isminion = np.repeat(mi, nc)
        nm = np.nonzero(~mi)[0]
        if nm.size:
            fne = np.repeat(eion[m[nm]] / (nc[nm] * c.EAverage), nc[nm])
            ne[~isminion] = np.maximum(rng.poisson(fne), 1)
        if mi.any():
            ne[isminion] = _minion_clusters(rng, c, eion[m[mi]], nc[mi])
            # minimum-ionising steps: every electron drifts on its own
            e1 = np.nonzero(isminion)[0]
            rep = np.ones(len(stp), np.int64)
            rep[e1] = ne[e1]
            stp, pos = np.repeat(stp, rep), np.repeat(pos, rep, axis=0)
            ne = np.where(np.repeat(isminion, rep), 1, np.repeat(ne, rep))
        out_s.append(idx[stp])
        out_p.append(pos)
        out_n.append(ne)
    if not out_s:
        return np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0, np.int64)
    return np.concatenate(out_s), np.concatenate(out_p), np.concatenate(out_n)


def wire_coordinates(c, us, pos):
    """strawCoordinates: (rho, along-wire z, phi) of positions about the (sim-aligned) wire, after truncating to the straw."""
    smid, sdir = c.straw_mid[us], c.straw_dir[us]
    d = pos - smid
    dw = (d * sdir).sum(1)
    cp = d - dw[:, None] * sdir
    rho = np.linalg.norm(cp, axis=1)
    r = c.strawInnerRadius
    over = rho > r
    pos = np.where(over[:, None], pos - cp / np.where(rho > 0, rho, 1)[:, None] * (rho - r)[:, None], pos)
    wmid, wdir = c.wire_mid[us], c.wire_dir[us]
    d = pos - wmid
    dw = (d * wdir).sum(1)
    cp = d - dw[:, None] * wdir
    zdir = np.array([0.0, 0.0, 1.0])
    pdir = np.cross(wdir, zdir)
    pdir *= np.where((pdir * wmid).sum(1) < 0, -1.0, 1.0)[:, None]
    phi = np.arctan2((cp * pdir).sum(1), cp[:, 2])
    return np.linalg.norm(cp, axis=1), dw, phi


def cluster_gain(rng, c, ne):
    """StrawPhysics::clusterGain: Polya (gamma) below nGainGauss electrons, Gaussian above."""
    A = c.polyaA
    small = ne < c.nGainGauss
    aval = np.where(ne == 1, A, ne * (A + 1) - 1).astype(np.float64)
    g = np.empty(len(ne))
    g[small] = rng.standard_gamma(aval[small]) / aval[small]
    big = ~small
    g[big] = rng.normal(1.0, c.gainRMSSlope / np.sqrt(ne[big]))
    return c.gasGain * g


# ---------------------------------------------------------------- waveforms
class Waveforms:
    """Clusters of every straw end of a batch, sorted by time per row (row = 2 * straw instance + end, 0 cal, 1 hv)."""

    def __init__(self, c, row, t, q, wdist, us, ifstep):
        o = np.lexsort((t, row))
        self.row, self.t, self.q, self.step = row[o], t[o], q[o], ifstep[o]
        nrow = int(row.max()) + 1 if len(row) else 0
        self.n = np.bincount(self.row, minlength=nrow)
        self.s = np.cumsum(self.n) - self.n
        self.c = c
        rb = c.responseBins
        self.rb, self.sr, self.LB = rb, c.sampleRate, c.clusterLookbackTime
        wd = c.wire_points
        sus = us[o]
        k, f = dist_interp(wdist[o], wd)
        tr, sc = reflection(wdist[o], c.wire_halflen[sus], c.s)
        self.base0, self.base1 = (k * rb).astype(np.int32), ((k + 1) * rb).astype(np.int32)
        self.a, self.b = (self.q * f).astype(F32), (self.q * (1 - f)).astype(F32)
        self.tr, self.sc, self.trs = tr, sc.astype(F32), (tr * self.sr).astype(F32)
        sn = sus % NSTRAWS
        self.tmax = c.tmax[k, 0, sn] * f + c.tmax[k + 1, 0, sn] * (1 - f)
        maxlin = (c.linmax[k, 0, sn] * f + c.linmax[k + 1, 0, sn] * (1 - f)) * self.q
        self.us_row = np.zeros(nrow, np.int64)
        self.us_row[self.row] = sus
        self.dvdi_t = c.dvdi_thresh[self.us_row]
        self.dvdi_a = c.dvdi_adc[self.us_row]
        maxlin = maxlin * self.dvdi_t[self.row]
        self.S = np.cumsum(maxlin)                                  # within-row inclusive cumsum (rows contiguous)
        self.S -= np.repeat(self.S[self.s] - maxlin[self.s], self.n)
        self.maxposs = np.add.reduceat(maxlin, self.s) if len(maxlin) else np.zeros(0)
        self.tab = {k: np.ascontiguousarray(getattr(c, "resp_" + k).ravel(), F32) for k in ("thresh", "adc", "sat")}

    def _gather(self, rows, cmin=None):
        n = self.n[rows]
        st = self.s[rows]
        if cmin is not None:
            st, n = st + cmin, n - cmin
        idx = _ranges(st, n)
        return idx, n

    def volt(self, rows, tq, path="thresh", cmin=None):
        """Waveform [mV] of the rows at query times tq ([R] or [R, K]); clusters before row index cmin excluded."""
        tq = np.asarray(tq, np.float64)
        out_shape = tq.shape
        tq2 = tq.reshape(len(rows), -1)
        # only clusters that start before the latest query time contribute (sampleWaveform's look-back cut)
        nlast = self.count_before(rows, tq2.max(1) + self.LB)
        st = self.s[rows] + (0 if cmin is None else cmin)
        n = np.maximum(nlast - (0 if cmin is None else cmin), 0)
        idx = _ranges(st, n)
        if idx.size == 0:
            return np.zeros(out_shape)
        rr = np.repeat(np.arange(len(rows)), n)
        tc = self.t[idx][:, None]
        tqr = tq2[rr]
        u = ((tqr - tc) * self.sr + self.rb / 2.0).astype(F32)
        i = np.clip(u, 0, self.rb - 1).astype(np.int32)            # truncation toward zero (u >= 0 after the clip)
        ir = np.clip(u - self.trs[idx][:, None], 0, self.rb - 1).astype(np.int32)
        T = self.tab[path]
        b0, b1, sc = self.base0[idx][:, None], self.base1[idx][:, None], self.sc[idx][:, None]
        v = self.a[idx][:, None] * (T[b0 + i] + sc * T[b0 + ir]) + self.b[idx][:, None] * (T[b1 + i] + sc * T[b1 + ir])
        v = np.where(tc - self.LB < tqr, v, F32(0))
        off = np.cumsum(n) - n
        nz = n > 0
        out = np.zeros(tq2.shape)
        out[nz] = np.add.reduceat(v, off[nz], axis=0, dtype=np.float64)
        dv = (self.dvdi_a if path == "adc" else self.dvdi_t)[rows]
        return (out * dv[:, None]).reshape(out_shape)

    def volt_lattice(self, rows, t0, step_bins, K, path="thresh", cmin=None, kmax=None):
        """volt at the regular times t0 + k * step_bins / sampleRate, k < K (kmax per row: later k left at 0).

        With an integer step in response bins the table index of cluster c at k is floor(u0_c) + k * step_bins, so
        only the first index is computed per cluster (same indices as volt).
        """
        t0 = np.asarray(t0, np.float64)
        dt = step_bins / self.sr
        kmax = np.full(len(rows), K) if kmax is None else np.minimum(kmax, K)
        nlast = self.count_before(rows, t0 + (kmax - 1) * dt + self.LB)
        st = self.s[rows] + (0 if cmin is None else cmin)
        n = np.maximum(nlast - (0 if cmin is None else cmin), 0)
        idx = _ranges(st, n)
        out = np.zeros((len(rows), K))
        if idx.size == 0:
            return out
        rr = np.repeat(np.arange(len(rows)), n)
        tc = self.t[idx]
        u0 = (t0[rr] - tc) * self.sr + self.rb / 2.0
        i0 = np.floor(u0).astype(np.int64)
        ir0 = np.floor(u0 - self.tr[idx] * self.sr).astype(np.int64)
        ks = (np.arange(K) * step_bins)[None, :]
        i = np.clip(i0[:, None] + ks, 0, self.rb - 1).astype(np.int32)
        ir = np.clip(ir0[:, None] + ks, 0, self.rb - 1).astype(np.int32)
        T = self.tab[path]
        b0, b1, sc = self.base0[idx][:, None], self.base1[idx][:, None], self.sc[idx][:, None]
        v = self.a[idx][:, None] * (T[b0 + i] + sc * T[b0 + ir]) + self.b[idx][:, None] * (T[b1 + i] + sc * T[b1 + ir])
        # included while tc - LB < t0 + k dt, i.e. from the first k at which the cluster has started
        kfirst = np.floor((tc - self.LB - t0[rr]) / dt).astype(np.int64) + 1
        v = np.where(np.arange(K)[None, :] >= kfirst[:, None], v, F32(0))
        off = np.cumsum(n) - n
        nz = n > 0
        out[nz] = np.add.reduceat(v, off[nz], axis=0, dtype=np.float64)
        dv = (self.dvdi_a if path == "adc" else self.dvdi_t)[rows]
        out *= dv[:, None]
        out[np.arange(K)[None, :] >= kmax[:, None]] = 0.0
        return out

    def count_before(self, rows, tlim, inclusive=False):
        """Number of row clusters with time < tlim (<= with inclusive): the first index at or after tlim."""
        idx, n = self._gather(rows)
        if idx.size == 0:
            return np.zeros(len(rows), np.int64)
        rr = np.repeat(np.arange(len(rows)), n)
        m = (self.t[idx] <= tlim[rr]) if inclusive else (self.t[idx] < tlim[rr])
        return np.bincount(rr, weights=m, minlength=len(rows)).astype(np.int64)

    def count_le_S(self, rows, target):
        idx, n = self._gather(rows)
        rr = np.repeat(np.arange(len(rows)), n)
        return np.bincount(rr, weights=self.S[idx] <= target[rr], minlength=len(rows)).astype(np.int64)


def _crosses(wf, st, rows, K=8):
    """StrawWaveform::crossesThreshold for the rows (state arrays in st, modified in place) -> bool per row."""
    LB = wf.LB
    ret = np.zeros(len(rows), bool)
    if rows.size == 0:
        return ret
    ic, T, thr = st["ic"], st["T"], st["thr"]
    ic[rows] = np.maximum(ic[rows], wf.count_before(rows, T[rows] + LB))
    r = rows[ic[rows] < wf.n[rows]]
    if r.size == 0:
        return ret
    vstart = wf.volt(r, wf.t[wf.s[r] + ic[r]] - LB)
    # returnCrossing: the waveform starts above threshold; skip past the response of that cluster
    sub = np.nonzero(vstart > thr[r])[0]
    while sub.size:
        rs = r[sub]
        g = wf.s[rs] + ic[rs]
        time = wf.t[g] + LB + 2 * wf.tmax[g]
        ic[rs] = np.maximum(ic[rs], wf.count_before(rs, time + LB))
        ok = ic[rs] < wf.n[rs]
        sub, rs = sub[ok], rs[ok]
        if not sub.size:
            break
        T[rs] = wf.t[wf.s[rs] + ic[rs]] - LB
        vstart[sub] = wf.volt(rs, T[rs])
        sub = sub[vstart[sub] > thr[rs]]
    live = ic[r] < wf.n[r]
    r, vstart = r[live], vstart[live]
    if r.size == 0:
        return ret
    # roughCrossing: first cluster at which the sum of maximal responses passes the threshold
    g0 = wf.s[r] + ic[r]
    s_before = np.where(ic[r] > 0, wf.S[g0 - 1], 0.0)
    j = np.maximum(ic[r], wf.count_le_S(r, thr[r] - vstart + s_before))
    found = j < wf.n[r]
    ic[r[~found]] = wf.n[r[~found]]
    r, j = r[found], j[found]
    if r.size == 0:
        return ret
    T[r] = wf.t[wf.s[r] + j] - LB
    # first cluster from j whose response at its maximum passes the threshold (K candidates per pass)
    jstar = np.full(len(r), -1)
    mresp = np.zeros(len(r))
    act = np.arange(len(r))
    while act.size:
        ra = r[act]
        cand = j[act][:, None] + np.arange(K)[None, :]
        valid = cand < wf.n[ra][:, None]
        g = wf.s[ra][:, None] + np.minimum(cand, wf.n[ra][:, None] - 1)
        mt = wf.t[g] + wf.tmax[g]
        v = wf.volt(ra, mt)
        hit = valid & (v > thr[ra][:, None])
        anyh = hit.any(1)
        first = np.argmax(hit, 1)
        h = act[anyh]
        jstar[h] = cand[anyh, first[anyh]]
        mresp[h] = v[anyh, first[anyh]]
        rest = ~anyh
        j[act[rest]] += K
        act = act[rest]
        act = act[j[act] < wf.n[r[act]]]
    ok = jstar >= 0
    ic[r[~ok]] = wf.n[r[~ok]]
    r, jstar, mresp = r[ok], jstar[ok], mresp[ok]
    if r.size == 0:
        return ret
    g = wf.s[r] + jstar
    ic[r] = jstar
    pre = wf.t[g] - LB
    post = wf.t[g] + wf.tmax[g]
    presample = wf.volt(r, pre)
    postsample = mresp
    th = thr[r]
    # fineCrossing: regula falsi between the cluster start and its maximum, to 0.02 ns or 10 steps
    with np.errstate(divide="ignore", invalid="ignore"):         # flat waveform between the samples: NaN, as stock
        time = pre + (post - pre) / (postsample - presample) * (th - presample)
    dt = post - pre
    act = np.nonzero(np.abs(dt) > 0.020)[0]
    for _ in range(10):
        if not act.size:
            break
        smp = wf.volt(r[act], time[act])
        up = smp > th[act]
        post[act] = np.where(up, time[act], post[act])
        postsample[act] = np.where(up, smp, postsample[act])
        pre[act] = np.where(up, pre[act], time[act])
        presample[act] = np.where(up, presample[act], smp)
        old = time[act]
        with np.errstate(divide="ignore", invalid="ignore"):
            time[act] = pre[act] + (post[act] - pre[act]) / (postsample[act] - presample[act]) * (th[act] - presample[act])
        dt[act] = time[act] - old
        act = act[np.abs(dt[act]) > 0.020]
    T[r] = time
    st["vcross"][r] = th
    adv = np.maximum(ic[r], wf.count_before(r, time + LB))
    ic[r] = np.where(adv > 0, adv - 1, 0)
    ret[np.searchsorted(rows, r)] = True
    return ret


def find_crossings(wf, rng, straw_us, straw_readall, stats=None):
    """findThresholdCrossings for every straw instance -> pairs (straw, T[2], vcross[2], ic[2]) in time order per straw."""
    c = wf.c
    ns = len(straw_us)
    nrow = 2 * ns
    st = {"ic": np.zeros(nrow, np.int64), "T": np.zeros(nrow), "thr": np.zeros(nrow), "vcross": np.zeros(nrow)}
    noise = rng.normal(0.0, c.strawNoise, ns)
    vth = c.vthresh[straw_us]                                             # [ns, 2]
    st["thr"][:] = rng.normal((vth + noise[:, None]).ravel(), c.thresholdAnalogNoise)
    tstart = np.where(straw_readall, -c.deadTimeAnalog, c.digitizationStart - c.timeFromProtonsToDRMarker - c.flashBuffer)
    st["T"][:] = np.repeat(tstart, 2)
    tend = max(c.eventWindowLength, c.mbtime) - c.electronicsTimeDelay + c.tdcbuf
    allrows = np.arange(nrow)
    cr = _crosses(wf, st, allrows)
    pairs = []

    def cxx_max(a, b):                                   # std::max(a, b): a NaN cal time stops, a NaN hv time does not
        return np.where(a < b, b, a)
    act = np.nonzero(cr[0::2] & cr[1::2] & (cxx_max(st["T"][0::2], st["T"][1::2]) < tend))[0]
    P = c.ADCPeriod
    while act.size:
        r0, r1 = 2 * act, 2 * act + 1
        t0, t1 = st["T"][r0], st["T"][r1]
        fin = np.isfinite(t0) & np.isfinite(t1)
        k0 = np.floor(np.where(fin, t0, 0.0) / P).astype(np.int64)
        k1 = np.floor(np.where(fin, t1, 0.0) / P).astype(np.int64)
        comb = fin & (np.abs(k0 - k1) < c.maxThreshTimeSeparation)
        a = act[comb]
        if a.size:
            rec = a[np.minimum(st["T"][2 * a], st["T"][2 * a + 1]) > 0.0]
            if rec.size:
                pairs.append((rec, st["T"][2 * rec].copy(), st["T"][2 * rec + 1].copy(), st["vcross"][2 * rec].copy(),
                              st["vcross"][2 * rec + 1].copy(), st["ic"][2 * rec].copy(), st["ic"][2 * rec + 1].copy()))
            rows = np.sort(np.concatenate([2 * a, 2 * a + 1]))
            st["T"][rows] += c.deadTimeAnalog
            st["ic"][rows] += 1
            st["thr"][rows] = rng.normal(c.vthresh[straw_us[rows // 2], rows % 2], c.thresholdAnalogNoise)
            crr = _crosses(wf, st, rows)
            cr[rows] = crr
        b = act[~comb]
        if b.size:
            early = np.where(st["T"][2 * b] < st["T"][2 * b + 1], 2 * b, 2 * b + 1)
            early = np.sort(early)
            st["ic"][early] += 1
            st["T"][early] += c.deadTimeAnalog
            cr[early] = _crosses(wf, st, early)
        act = act[cr[2 * act] & cr[2 * act + 1] & (cxx_max(st["T"][2 * act], st["T"][2 * act + 1]) < tend)]
    if stats is not None:
        stats["straws"] = stats.get("straws", 0) + ns
        stats["nan_crossing_ends"] = stats.get("nan_crossing_ends", 0) + int(np.isnan(st["T"]).sum())
    if not pairs:
        return np.zeros(0, np.int64), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2), np.int64)
    s = np.concatenate([p[0] for p in pairs])
    T = np.stack([np.concatenate([p[1] for p in pairs]), np.concatenate([p[2] for p in pairs])], 1)
    V = np.stack([np.concatenate([p[3] for p in pairs]), np.concatenate([p[4] for p in pairs])], 1)
    I = np.stack([np.concatenate([p[5] for p in pairs]), np.concatenate([p[6] for p in pairs])], 1)
    o = np.lexsort((T[:, 0], s))
    return s[o], T[o], V[o], I[o]


def _tot(wf, rows, thr, t0):
    """StrawWaveform::digitizeTOT: first TOT tick at which the waveform is below threshold - hysteresis."""
    c = wf.c
    ticks = np.arange(1, c.maxTOT)
    step = int(round(c.TOTLSB * c.sampleRate))
    v = wf.volt_lattice(rows, t0 + c.TOTLSB, step, len(ticks))
    below = v < (thr - c.triggerHysteresis)[:, None]
    return np.where(below.any(1), ticks[np.argmax(below, 1)], c.maxTOT)


def _adc_volts(wf, rows, times, chunk=256):
    """StrawWaveform::sampleADCWaveform for the rows at the sample times [R, nsamp] (saturation model where needed)."""
    c = wf.c
    out = np.zeros(times.shape)
    sat = wf.maxposs[rows] > c.saturationVoltage
    lin = np.nonzero(~sat)[0]
    if lin.size:
        out[lin] = wf.volt_lattice(rows[lin], times[lin, 0], int(round(c.ADCPeriod * c.sampleRate)), times.shape[1], "adc")
    srow = np.nonzero(sat)[0]
    LB, dts = wf.LB, c.saturationSampleFactor / c.sampleRate
    imp = c.resp_imp
    for k in range(0, srow.size, chunk):
        sr = srow[k:k + chunk]
        rr = rows[sr]
        i0 = wf.count_before(rr, times[sr, 0] + LB - c.ttrunc, inclusive=True)
        i0 = np.minimum(i0, wf.n[rr] - 1)
        t0 = wf.t[wf.s[rr] + i0] - LB
        nstep = np.ceil((times[sr, -1] - t0) / dts).astype(np.int64)
        smax = int(nstep.max())
        tq = t0[:, None] + dts * np.arange(smax)[None, :]
        resp = wf.volt_lattice(rr, t0, c.saturationSampleFactor, smax, "sat", cmin=i0, kmax=nstep)
        resp = np.minimum(resp, c.saturationVoltage)
        x = times[sr][:, :, None] - tq[:, None, :]
        ii = np.clip(np.trunc(x * c.sampleRate + c.responseBins / 2.0), 0, c.responseBins - 1).astype(np.int64)
        scale = c.saturationSampleFactor * wf.dvdi_a[rr] / wf.dvdi_t[rr]
        out[sr] = (resp[:, None, :] * imp[ii]).sum(2) * scale[:, None]
    return out


def digitize(st, nev, c, rng, stats=None):
    """StrawGasSteps of nev events (flat arrays; positions detector frame, mm; t ns; eion MeV) -> digi arrays.

    st keys: ev (event index), sid, shape, ion, eion, len, width, t, x0, y0, z0, x1, y1, z1. Returns a dict of per-digi
    arrays (ev, sid, tdc [n,2], tot [n,2], pmp, adc [n, nsamp], step [n,2] = index of the step whose cluster is
    referenced at the crossing of each end, t_analog [n,2]) in the stock order (event, StrawId, time). stats (dict,
    optional) accumulates counters: straws, clusters, saturated straw ends, NaN regula-falsi crossings, pairs, digis.
    """
    st = dict(st)
    nst = len(st["sid"])
    sid = np.asarray(st["sid"], np.int64)
    us = unique_straw(sid)
    st["_us"] = us
    readall = ((sid & 0x7F) >= c.allStraw) & np.isin(sid >> 10, c.allPlanes)
    jitter = rng.normal(0.0, c.eventWindowMarkerROCJitter, (nev, NUPANELS))
    ctime = np.asarray(st["t"], np.float64)                             # off spill: no PBT offset, no folding
    dstart = c.digitizationStart - c.timeFromProtonsToDRMarker
    dend = c.eventWindowLength
    inwin = (ctime > dstart - c.electronicsTimeDelay - c.steptimebuf) & \
            (ctime < max(c.mbtime, dend) - c.electronicsTimeDelay + c.steptimebuf)
    sel = np.nonzero((np.asarray(st["eion"]) > c.minstepE) & (inwin | readall))[0]
    empty = {"ev": np.zeros(0, np.int64), "sid": np.zeros(0, np.int64), "tdc": np.zeros((0, 2), np.int64),
             "tot": np.zeros((0, 2), np.int64), "pmp": np.zeros(0, np.int64),
             "adc": np.zeros((0, c.nADCSamples), np.int64), "step": np.zeros((0, 2), np.int64),
             "t_analog": np.zeros((0, 2))}
    if nst == 0 or sel.size == 0:
        return empty
    cs, cpos, cne = make_clusters(rng, c, st, sel)
    if cs.size == 0:
        return empty
    cus = us[cs]
    gain = cluster_gain(rng, c, cne)
    charge = F32(c.Qe * cne * gain)
    rho, zw, phi = wire_coordinates(c, cus, cpos)
    tdrift = d2t(rho, phi, c.drift_dist, c.drift_times, c.drift_dD, c.drift_dphi)
    tdrift = rng.normal(tdrift, c.dtvar[0] + c.dtvar[1] * rho).astype(F32)       # WireCharge::_time is float
    hl = c.wire_halflen[cus]
    wd = np.stack([hl - zw, hl + zw], 1).astype(F32)                   # cal, hv
    tprop = (wd / c.vprop).astype(F32)
    gt = (ctime[cs][:, None] + tdrift[:, None].astype(np.float64) + tprop.astype(np.float64)).astype(F32)
    qe = (F32(0.5) * charge)
    # straw instances (event, straw) in StrawId order, rows 2*i + end
    key = np.asarray(st["ev"], np.int64)[cs] * 65536 + sid[cs]
    ukey, inst = np.unique(key, return_inverse=True)
    straw_ev, straw_sid = ukey // 65536, ukey % 65536
    straw_us = unique_straw(straw_sid)
    straw_ra = ((straw_sid & 0x7F) >= c.allStraw) & np.isin(straw_sid >> 10, c.allPlanes)
    row = np.concatenate([2 * inst, 2 * inst + 1])
    wf = Waveforms(c, row, np.concatenate([gt[:, 0], gt[:, 1]]).astype(np.float64),
                   np.concatenate([qe, qe]).astype(np.float64), np.concatenate([wd[:, 0], wd[:, 1]]).astype(np.float64),
                   np.concatenate([cus, cus]), np.concatenate([cs, cs]))
    if stats is not None:
        stats["clusters"] = stats.get("clusters", 0) + len(cs)
        stats["saturated_ends"] = stats.get("saturated_ends", 0) + int((wf.maxposs > c.saturationVoltage).sum())
        stats["ends"] = stats.get("ends", 0) + len(wf.maxposs)
    ps, pT, pV, pI = find_crossings(wf, rng, straw_us, straw_ra, stats)
    if ps.size == 0:
        return empty
    npair = len(ps)
    r0, r1 = 2 * ps, 2 * ps + 1
    dt = jitter[straw_ev[ps], unique_panel(straw_sid[ps])]
    xt = pT + dt[:, None] + rng.normal(0.0, c.TDCResolution, (npair, 2))
    thr_tot = rng.normal(pV, c.thresholdAnalogNoise)
    tot = np.stack([_tot(wf, r0, thr_tot[:, 0], pT[:, 0] + dt), _tot(wf, r1, thr_tot[:, 1], pT[:, 1] + dt)], 1)
    nsamp = c.nADCSamples
    phase = np.maximum(0, np.ceil(pT[:, 0] / c.ADCPeriod).astype(np.int64) - c.nADCPresamples)
    times = (phase[:, None] + np.arange(nsamp)[None, :]) * c.ADCPeriod + c.ADCOffset
    volts = _adc_volts(wf, r0, times) + _adc_volts(wf, r1, times)
    dtime = xt.max(1)
    if not c.overrideDbTimeOffsets:
        xt = xt - c.time_offsets[straw_us[ps]]
    wsum = volts + rng.normal(0.0, c.adcAnalogNoise, volts.shape)
    tdc = c.tdc_response(xt)
    ewend = int(c.tdc_response(c.eventWindowLength - c.electronicsTimeDelay))
    ra = straw_ra[ps]
    ok_all = np.all((xt + c.electronicsTimeDelay > 0) & (tdc < ewend), 1)
    upper = c.digitizationEndTDC if c.onspill else ewend
    ok_norm = np.all((tdc > c.digitizationStartTDC) & (tdc < upper), 1)
    ok = np.where(ra, ok_all, ok_norm)
    # digital dead time: a later pair of the same straw is dropped if it digitises before the previous digi + tdead
    keep = np.zeros(npair, bool)
    first = np.r_[True, ps[1:] != ps[:-1]]
    rank = np.arange(npair) - np.maximum.accumulate(np.where(first, np.arange(npair), 0))
    ready = np.full(len(straw_us), -9e9)
    for k in range(int(rank.max()) + 1):
        m = np.nonzero(rank == k)[0]
        acc = ok[m] & (dtime[m] >= ready[ps[m]])
        keep[m[acc]] = True
        ready[ps[m[acc]]] = dtime[m[acc]] + c.deadTimeDigital
    kk = np.nonzero(keep)[0]
    ped = c.adc_ped[straw_us[ps[kk]]]
    adc = np.clip(np.floor(wsum[kk] / c.ADCLSB).astype(np.int64) + ped[:, None], 0, c.maxADC)
    pmp = (adc.max(1) - adc[:, :c.nADCPresamples].sum(1) // c.nADCPresamples) & 0xFFFF
    step = np.stack([wf.step[wf.s[r0[kk]] + pI[kk, 0]], wf.step[wf.s[r1[kk]] + pI[kk, 1]]], 1)
    out = {"ev": straw_ev[ps[kk]], "sid": straw_sid[ps[kk]], "tdc": tdc[kk], "tot": tot[kk], "pmp": pmp, "adc": adc,
           "step": step, "t_analog": pT[kk]}
    if stats is not None:
        stats["pairs"] = stats.get("pairs", 0) + npair
        stats["digis"] = stats.get("digis", 0) + len(kk)
    o = np.lexsort((out["t_analog"][:, 0], out["sid"], out["ev"]))
    return {k: v[o] for k, v in out.items()}
