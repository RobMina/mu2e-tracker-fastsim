"""Straw axes (KKStrawXing's Mu2eKinKal::strawLine, from Straw::strawEnd) rebuilt from the dumped aligned wire ends and
the TrkAlignStraw table: AlignedTrackerMaker moves each wire end and each straw end by its own (dV, dW) in panel UVW,
so straw end = wire end + (straw delta - wire delta) in the aligned panel frame. The panel frame (U along the wire
cal<-hv, V from straw 0 to straw 94 in the same layer, W = U x V) is taken from the dumped wire geometry; it differs
from the exact aligned frame by the panel rotations (<= 1e-4 rad), i.e. < 0.1 um on these sub-mm deltas.
Table text from `dbTool print-run --purpose P --version V --run R --table TrkAlignStraw --content`."""
import numpy as np

NSTRAWS = 96


def read_trkalignstraw(path):
    """-> array [20736, 8]: wire_cal_dV, wire_cal_dW, wire_hv_dV, wire_hv_dW, straw_cal_dV, straw_cal_dW, straw_hv_dV,
    straw_hv_dW, indexed by unique straw (the table index)."""
    rows = [l.split(",") for l in open(path) if l[:1].isdigit()]
    idx = np.array([int(r[0]) for r in rows])
    v = np.array([[float(x) for x in r[2:10]] for r in rows])
    out = np.zeros((idx.max() + 1, 8))
    out[idx] = v
    return out


def straw_axes(cond_straw, wire_cal, wire_hv, align):
    """Straw midpoints and unit directions [S, 3] (direction hv -> cal, as Straw::_sdir).
    cond_straw: the conditions 'straw' dict (midx.., dirx..) ordered by unique straw; wire_cal/wire_hv [S, 3]."""
    S = wire_cal.shape[0]
    u = wire_cal - wire_hv
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    mid = 0.5 * (wire_cal + wire_hv)
    up = np.arange(S) // NSTRAWS
    m0 = mid[up * NSTRAWS]
    m94 = mid[up * NSTRAWS + NSTRAWS - 2]
    v = m94 - m0
    v -= np.sum(v * u, 1, keepdims=True) * u
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    w = np.cross(u, v)
    a = align[:S]
    dcal = (a[:, 4] - a[:, 0])[:, None] * v + (a[:, 5] - a[:, 1])[:, None] * w
    dhv = (a[:, 6] - a[:, 2])[:, None] * v + (a[:, 7] - a[:, 3])[:, None] * w
    scal, shv = wire_cal + dcal, wire_hv + dhv
    sd = scal - shv
    return 0.5 * (scal + shv), sd / np.linalg.norm(sd, axis=1, keepdims=True)
