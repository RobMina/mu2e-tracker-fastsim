"""EventNtuple-compatible output of the tracker fast reco: `evtinfo`, `trk`, `trksegs`, `trkhits`, `trksegpars_kl`
(and `trkmats` when the fit has a material model) in `EventNtuple/ntuple`, written with uproot.

Layout. uproot cannot write ROOT split-class TTree branches or `std::vector<std::vector<T>>` (its TTree writer takes
one level of jaggedness and flat leaves only), so the tree is written as an RNTuple with the stock field names, field
order, primitive types and nesting, taken from a stored stock schema (`data/eventntuple_*.json`, made by
`reco-ntschema` on a stock file). Readers that use `tree["evtinfo"].array()`, `tree["evtinfo/run"]`,
`tree["trk"].array()` (a record of `trk.<leaf>` lists), `tree.arrays([...])` or `tree["trksegs"|"trkhits"|...].array()`
get the same awkward types from both files.

Split-branch paths. Analyses read single `trk` leaves as `tree["trk/trk.status"]`. uproot resolves that path on an
RNTuple as the field chain trk -> trk -> status, so the `trk` record carries one extra trailing field `trk` (SPLIT_ALIAS):
a record of the same per-track lists under the bare leaf names. `tree["trk/trk.<leaf>"].array()` then returns the
stock jagged array; `tree["trk"].array()` has the stock fields plus that one (`without_alias` drops it).
`--no-split-alias` writes the bare stock layout.

Not reproducible with uproot 5.7: `tree.arrays(["trk", "trksegs", ...])` (pyutils Importer) returns empty records for
every record field of an RNTuple, and `"trk/trk.status" in tree.keys()` is False.

Units and frames are the stock ones: mm, ns, MeV/c, detector (tracker) frame.

Unfilled fields. Every field of the schema is written. Fields the fast reco does not compute hold the sentinel
UNSET_F = -9999.0 (floating point), UNSET_I = -9999 (integers), False (booleans; no sentinel is possible). The
realised per-field coverage (filled / approx / constant / sentinel) is in the provenance record.

Provenance: `EventNtuple/fastreco_provenance` (TObjString, JSON): fit type (stock / F1 / F2), stock conditioning,
package commit, track-table metadata, schema source, sentinel values, coverage, counts and a content hash over every
written buffer (the replay check: the same input gives the same hash; the ROOT file bytes carry a random UUID).
"""
import hashlib
import json
import os
import subprocess

import awkward as ak
import numpy as np
import uproot

from . import ntschema, tracktable

TREE = "EventNtuple/ntuple"
PROV = "EventNtuple/fastreco_provenance"
UNSET_F, UNSET_I = -9999.0, -9999
DEFAULT_SCHEMA = os.path.join(os.path.dirname(__file__), "data", "eventntuple_v06_13_02_ssmc.json")
BRANCHES = ("evtinfo", "trk", "trksegs", "trkhits", "trksegpars_kl", "trkmats")
SPLIT_ALIAS = "trk"                 # extra trk sub-record that makes tree["trk/trk.<leaf>"] resolve on the RNTuple
C_LIGHT, M_MU = 299.792458, 105.6583755
PDG_MU = 13                         # KKLine fit hypothesis (stock trk.pdg is 13 for every track of the sample)
FITALG_KKLINE = 3                   # InfoStructHelper: KKLine fit flag -> fitalg 3
# KinKalGeom tracker surfaces in the detector frame (GeometryService KinKalGeomMaker::makeTracker, extracted v04):
# TT_Front / TT_Back disks at the TT VD centres out to the DS inner radius, TT_Outer cylinder of the tracker mother
# (rOut 850.1 mm) centred on zMid 10.1 mm with half length (zBack - zFront) / 2
SURF = {"TT_Front": 0, "TT_Back": 2, "TT_Outer": 4}
ZFRONT, ZBACK, ZMID, ROUT, RDISK = -1631.11, 1639.11, 10.1, 850.1, 950.0      # RDISK: DS rIn1 (ds.rIn.in)
HALFLEN = 0.5 * (ZBACK - ZFRONT)
# StrawHitFlag bits (RecoDataProducts/inc/StrawHitFlag.hh)
F_ESEL, F_RSEL, F_TSEL, F_BKG, F_DEAD, F_NOISY = 1, 2, 3, 6, 23, 24

# classification of every field the writer can compute; anything else in the schema is a sentinel
COVERAGE = {
    "evtinfo": {"event": "filled", "run": "filled", "subrun": "filled", "pbtime": "filled: dump pbt (PBTFSD)"},
    "trk": {
        "status": "filled: 1 converged, 2 OK not converged, -1 failed (F2: 1 when goodfit, no convergence test)",
        "goodfit": "filled: stock FitOK / F1 good / F2 KKFit gates (>= 5 active, ndof >= 1)",
        "seedalg": "constant -1 (stock: no helix seed flag for LineFinder seeds)", "fitalg": "constant 3 (KKLine)",
        "pdg": "constant 13", "nhits": "filled", "ndof": "filled", "nactive": "filled",
        "ndouble": "filled (stock rule: consecutive hits in list order on the same panel; list order differs from stock for F2)",
        "ndactive": "filled (same rule)", "nplanes": "filled", "npanels": "filled",
        "planespan": "filled (stock rule: max plane of active hits, minplane starts at 0)", "nnullambig": "filled",
        "nmat": "filled where the fit has straw crossings", "nmatactive": "filled where the fit has straw crossings",
        "radlen": "filled (stock) / approx (F1, nominal X0) where the fit has straw crossings",
        "nesel": "filled (hit flags)", "nrsel": "filled", "ntsel": "filled", "nbkg": "filled", "nsel": "filled",
        "nipaup": "constant 0 (no IPA in the extracted geometry)", "nipadown": "constant 0", "nstup": "constant 0",
        "nstdown": "constant 0", "tsdainter": "constant False", "opainter": "constant False",
        "firststinter": "constant -1", "chisq": "filled", "fitcon": "filled: TMath::Prob(chisq, ndof)",
        "firsthit": "filled: min ptoca of active hits", "lasthit": "filled: max ptoca of active hits",
        "maxgap": "constant 0 (no trajectory gaps)", "avggap": "constant 0 (no trajectory gaps)",
        "avgedep": "filled: mean edep of active hits"},
    "trksegs": {
        "mom": "approx: fit momentum x direction at every crossing, no energy loss along the line (F2: fixed 2000 MeV/c)",
        "pos": "filled", "time": "filled",
        "dmom": "constant 0 at the tracker surfaces (stock 0 there)", "momerr": "filled when mom is fitted (stock, F1)",
        "inbounds": "constant True (only in-bounds crossings are written, as in stock)", "gap": "constant False",
        "sid": "filled: every intersection of the line with TT_Front 0, TT_Back 2 (r < 950 mm) and TT_Outer 4",
        "sindex": "constant 0"},
    "trksegpars_kl": {**{k: "filled: the line parameters at each crossing" for k in
                         ("d0", "phi0", "z0", "theta", "t0", "d0err", "phi0err", "z0err", "thetaerr", "momerr", "t0err")},
                      "mom": "approx: fit momentum, no energy loss along the line"},
    "trkhits": {
        "plane": "filled", "panel": "filled", "layer": "filled", "straw": "filled",
        "state": "filled: -2 inactive, -1/+1 drift, 0 null (stock tables keep -3 unusable)",
        "earlyend": "filled", "edep": "filled", "wdist": "filled", "werr": "approx: sqrt(ComboHit uvar)",
        "tottdrift": "filled: TOT drift time", "etime": "filled", "tot": "filled",
        "ptoca": "filled", "stoca": "filled: wire signal time at the POCA", "rdoca": "filled: -endSign x signed DOCA",
        "rdocavar": "approx: DOCA gradient x fit covariance (stock closure: diagonal KinematicLine errors)",
        "rdt": "filled: stoca - ptoca", "rupos": "filled: POCA along the wire from the straw middle",
        "rdrift": "filled: drift table at rdt", "sderr": "filled: drift table at rdt", "uderr": "filled: drift table at rdt",
        "dvel": "filled: drift table at rdt", "wdot": "filled: track . wire direction",
        "poca": "filled: track point at the POCA", "dhit": "filled", "dactive": "filled"},
    "trkmats": {
        "active": "filled", "hashit": "filled", "activehit": "filled", "drifthit": "filled", "plane": "filled",
        "panel": "filled", "layer": "filled", "straw": "filled", "doca": "filled (F1)", "docavar": "filled (F1)",
        "radlen": "filled (stock) / approx (F1)", "dirdot": "filled (F1)", "gaspath": "filled", "wallpath": "filled",
        "wirepath": "filled (stock)"},
}


def load_schema(path=DEFAULT_SCHEMA):
    with open(path) as f:
        s = json.load(f)
    return {b: ntschema.signature(s["branches"][b]["form"]) for b in BRANCHES if s["branches"].get(b)}, s


def _sentinel(prim):
    dt = np.dtype(prim)
    return False if dt.kind == "b" else (UNSET_I if dt.kind in "iu" else UNSET_F)


def _provided(v):
    """A column counts as computed unless it is absent or entirely non-finite (e.g. a fixed-momentum error)."""
    if v is None:
        return False
    a = np.asarray(v)
    return not (a.dtype.kind == "f" and a.size and not np.isfinite(a).any())


def _leaf(sig, val, n):
    prim, shape = (sig, ()) if isinstance(sig, str) else (sig[2], tuple(np.atleast_1d(sig[1]).tolist()))
    dt = np.dtype(prim)
    if not _provided(val):
        arr = np.full((n,) + shape, _sentinel(prim), dt)
    else:
        arr = np.asarray(val)
        if arr.dtype.kind == "f" and dt.kind != "b":
            arr = np.where(np.isfinite(arr), arr, _sentinel(prim))
        arr = arr.astype(dt).reshape((n,) + shape)
    return ak.from_numpy(arr, regulararray=True) if shape else ak.from_numpy(arr)


def _record(sig, flat, n, used, prefix=""):
    fields = {}
    for f, s in sig["record"]:
        path = f"{prefix}.{f}" if prefix else f
        if isinstance(s, dict):
            fields[f] = _record(s, flat, n, used, path)
        else:
            v = flat.get(path)
            if _provided(v):
                used.add(path)
            fields[f] = _leaf(s, v, n)
    return ak.zip(fields, depth_limit=1)


def build_branch(sig, flat, counts, used):
    """Awkward array of one branch from flat leaf columns. counts: [] (one row per event), [ntrk] (per-track lists)
    or [ntrk, nper] (per-track lists of lists)."""
    if isinstance(sig, dict) and not counts:
        return _record(sig, flat, len(next(iter(flat.values()))) if flat else 0, used)
    if isinstance(sig, dict):                                        # record of per-track lists (split `trk`)
        fields = {}
        for f, s in sig["record"]:
            inner = s[1]
            v = flat.get(f)
            if _provided(v):
                used.add(f)
            fields[f] = ak.unflatten(_leaf(inner, v, int(np.sum(counts[0]))), counts[0])
        return ak.zip(fields, depth_limit=1)
    assert sig[0] == "var" and sig[1][0] == "var", sig
    rec = _record(sig[1][1], flat, int(np.sum(counts[1])), used)
    return ak.unflatten(ak.unflatten(rec, counts[1]), counts[0])


# ---- geometry and derived quantities (vectorised over tracks, padded hits / crossings) -------------------------

def line(x):
    """Point (a0, 0, b0), unit direction of motion, t0 at the point, momentum, speed."""
    p = np.stack([x[:, 0], np.zeros(len(x)), x[:, 1]], 1)
    d = np.stack([x[:, 2], -np.ones(len(x)), x[:, 3]], 1)
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    mom = x[:, 5]
    return p, u, x[:, 4], mom, mom / np.hypot(mom, M_MU) * C_LIGHT


def kl_params(x):
    """(a0, b0, a1, b1, t0, mom) -> KinematicLine (d0, phi0, z0, theta, t0, mom) at the POCA to the z axis."""
    p, u, t0, mom, v = line(x)
    uz = u[:, 2]
    s1 = (p[:, 2] * uz - np.sum(p * u, 1)) / np.maximum(1.0 - uz * uz, 1e-12)
    pca = p + u * s1[:, None]
    sign = np.copysign(1.0, -(pca[:, 0] * u[:, 1] - pca[:, 1] * u[:, 0]))
    return np.stack([sign * np.hypot(pca[:, 0], pca[:, 1]), np.arctan2(u[:, 1], u[:, 0]), pca[:, 2], np.arccos(uz),
                     t0 + s1 / v, mom], 1)


def kl_cov(x, cov):
    """KinematicLine covariance through the numerical Jacobian of kl_params (NaN rows stay NaN)."""
    eps = np.array([1e-3, 1e-3, 1e-6, 1e-6, 1e-3, 1e-3])
    J = np.zeros((len(x), 6, 6))
    for i in range(6):
        dx = np.zeros(6)
        dx[i] = eps[i]
        hi, lo = kl_params(x + dx), kl_params(x - dx)
        dd = hi - lo
        dd[:, 1] = (dd[:, 1] + np.pi) % (2 * np.pi) - np.pi
        J[:, :, i] = dd / (2 * eps[i])
    Cz = np.where(np.isfinite(cov), cov, 0.0)
    out = np.einsum("tij,tjk,tlk->til", J, Cz, J)
    bad = ~np.isfinite(cov)
    # a term with an undefined input variance (fixed momentum) leaves that output undefined only for mom itself
    out[:, 5, 5] = np.where(bad[:, 5, 5], np.nan, out[:, 5, 5])
    return out


def hit_geometry(x, tb):
    """Per hit: track length st and wire position sw of the POCA, signed DOCA (KinKal convention), particle and
    sensor times, track point, wire-track cosine."""
    p, u, t0, mom, v = line(x)
    m, w = tb["mid"], tb["wdir"]
    c = np.einsum("ti,tni->tn", u, w)
    s2 = np.maximum(1.0 - c * c, 1e-12)
    dl = p[:, None] - m
    du, dw = np.einsum("tni,ti->tn", dl, u), np.einsum("tni,tni->tn", dl, w)
    st = (dw * c - du) / s2
    sw = (dw - du * c) / s2
    ptrk = p[:, None] + u[:, None] * st[..., None]
    delta = (m + w * sw[..., None]) - ptrk
    sdir = w * tb["esign"][..., None]
    lsign = np.sign(np.einsum("tni,tni->tn", np.cross(sdir, u[:, None]), delta))
    doca = np.linalg.norm(delta, axis=-1) * np.where(lsign == 0, 1.0, lsign)
    ptoca = t0[:, None] + st / v[:, None]
    stoca = tb["tmeas"] - (tb["hlen"] - tb["esign"] * sw) / tb["vsig"]
    return {"st": st, "sw": sw, "doca": doca, "ptoca": ptoca, "stoca": stoca, "poca": ptrk, "wdot": c}


def doca_var(x, cov, tb):
    """Variance of the hit DOCA from the fit covariance of the five line parameters (numerical gradient)."""
    eps = np.array([1e-3, 1e-3, 1e-6, 1e-6, 1e-3])
    g = np.zeros(tb["mid"].shape[:2] + (5,))
    for i in range(5):
        dx = np.zeros(6)
        dx[i] = eps[i]
        g[..., i] = (hit_geometry(x + dx, tb)["doca"] - hit_geometry(x - dx, tb)["doca"]) / (2 * eps[i])
    C5 = np.where(np.isfinite(cov[:, :5, :5]), cov[:, :5, :5], 0.0)
    return np.einsum("tni,tij,tnj->tn", g, C5, g)


def tracker_crossings(x):
    """Intersections of each (infinite) line with the bounded KinKalGeom tracker surfaces: the TT_Front and TT_Back
    disks (r < RDISK, the DS inner radius) and the TT_Outer cylinder (|z - ZMID| <= HALFLEN). Returns track length
    s [T, 4] (NaN where absent) and SurfaceId [T, 4], in the fixed slot order front, back, cylinder (x2)."""
    p, u, *_ = line(x)
    T = len(x)
    S = np.full((T, 4), np.nan)
    sid = np.array([SURF["TT_Front"], SURF["TT_Back"], SURF["TT_Outer"], SURF["TT_Outer"]])
    uz = u[:, 2]
    for j, z in enumerate((ZFRONT, ZBACK)):
        ok = np.abs(uz) > 1e-12
        s = (z - p[:, 2]) / np.where(ok, uz, 1.0)
        q = p + u * s[:, None]
        S[:, j] = np.where(ok & (np.hypot(q[:, 0], q[:, 1]) < RDISK), s, np.nan)
    a = u[:, 0] ** 2 + u[:, 1] ** 2
    b = 2 * (p[:, 0] * u[:, 0] + p[:, 1] * u[:, 1])
    c = p[:, 0] ** 2 + p[:, 1] ** 2 - ROUT ** 2
    disc = b * b - 4 * a * c
    ok = (disc > 0) & (a > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    for j, sgn in ((2, -1.0), (3, 1.0)):
        s = (-b + sgn * sq) / (2 * np.where(ok, a, 1.0))
        z = p[:, 2] + uz * s
        S[:, j] = np.where(ok & (np.abs(z - ZMID) <= HALFLEN), s, np.nan)
    return S, np.broadcast_to(sid, (T, 4))


def _strawid(sid):
    sid = np.asarray(sid, np.int64)
    return sid >> 10, (sid >> 7) & 7, (sid & 0x7F) & 1, sid & 0x7F


def _fitcon(chi2, ndof):
    from scipy.special import gammaincc
    return np.where(ndof > 0, gammaincc(np.maximum(ndof, 1) / 2.0, np.maximum(chi2, 0) / 2.0), 0.0)


def _drift(tb, dT, key):
    d = tb["drift"]
    return np.interp(dT, np.asarray(d["dt"]), np.asarray(d[key]))


def compute(tb, only_good=True):
    """Track table -> flat columns per branch, the event/track/element counts and bookkeeping."""
    E = len(tb["events"]["run"])
    keep = (tb["goodfit"] == 1) if only_good else np.ones(len(tb["ev"]), bool)
    # tracks grouped by event, stable in table order (deterministic)
    order = np.flatnonzero(keep)[np.argsort(tb["ev"][keep], kind="stable")]
    sel = lambda a: np.asarray(a)[order]
    x, cov, ev = sel(tb["x"]).astype(np.float64), sel(tb["cov"]), sel(tb["ev"])
    T = len(order)
    hv = sel(tb["hvalid"])
    hit = {k: sel(tb[k]) for k in ("mid", "wdir", "hlen", "esign", "vsig", "tmeas", "totdt", "uvar", "wdist", "edep",
                                    "sid", "flag", "et0", "et1", "tot0", "tot1", "eend", "state")}
    hit["drift"] = tb["drift"]
    ntrk = np.bincount(ev, minlength=E).astype(np.int64)
    out = {"counts": {"events": E, "tracks_in_table": int(len(tb["ev"])), "tracks_written": T,
                      "not_written_failed_fits": int((~keep).sum()),
                      "events_with_track": int((ntrk > 0).sum())}, "ntrk": ntrk}
    # ---- evtinfo
    evs = tb["events"]
    out["evtinfo"] = {"event": evs["event"], "run": evs["run"], "subrun": evs["subrun"], "pbtime": evs["pbt"]}
    # ---- hits: order along the track (ptoca) for the written lists
    g = hit_geometry(x, hit) if T else None
    if T:
        key = np.where(hv, g["st"], np.inf)
        o = np.argsort(key, axis=1, kind="stable")
        tk = lambda a: np.take_along_axis(a, o.reshape(o.shape + (1,) * (a.ndim - 2)), 1)
        hv = tk(hv)
        hit = {k: (tk(v) if k != "drift" else v) for k, v in hit.items()}
        g = {k: tk(v) for k, v in g.items()}
        dvar = doca_var(x, cov, hit)
    nh = hv.sum(1).astype(np.int64) if T else np.zeros(0, np.int64)
    state = hit["state"] if T else np.zeros((0, 1), np.int32)
    act = hv & (state > -2)                                            # strawHitState() > inactive
    plane, panel, layer, straw = _strawid(hit["sid"]) if T else (np.zeros((0, 1), np.int64),) * 4
    upanel = plane * 6 + panel
    flag = hit["flag"].astype(np.int64) if T else np.zeros((0, 1), np.int64)
    bit = lambda b: (flag >> b) & 1 == 1
    tr = {}
    nact = act.sum(1)
    tr["nhits"], tr["nactive"] = nh, nact
    tr["nnullambig"] = (act & (state == 0)).sum(1)
    if T:
        tr["nplanes"] = _nunique(np.where(act, plane, -1))
        tr["npanels"] = _nunique(np.where(act, upanel, -1))
        tr["planespan"] = np.max(np.where(act, plane, 0), 1)
        nxt_same = np.zeros_like(act)
        nxt_same[:, :-1] = (upanel[:, 1:] == upanel[:, :-1]) & hv[:, 1:]
        nxt_act = np.zeros_like(act)
        nxt_act[:, :-1] = act[:, 1:]
        tr["ndouble"] = (act & nxt_same).sum(1)
        tr["ndactive"] = (act & nxt_same & nxt_act).sum(1)
        tr["avgedep"] = np.where(nact > 0, np.where(act, hit["edep"], 0).sum(1) / np.maximum(nact, 1), np.nan)
        for name, b in (("nesel", F_ESEL), ("nrsel", F_RSEL), ("ntsel", F_TSEL), ("nbkg", F_BKG)):
            tr[name] = (act & bit(b)).sum(1)
        tr["nsel"] = (act & bit(F_ESEL) & bit(F_RSEL) & bit(F_TSEL) & ~(bit(F_BKG) | bit(F_DEAD) | bit(F_NOISY))).sum(1)
        pt = np.where(act, g["ptoca"], np.nan)
        tr["firsthit"] = np.where(nact > 0, np.nanmin(np.where(act, pt, np.inf), 1), np.nan)
        tr["lasthit"] = np.where(nact > 0, np.nanmax(np.where(act, pt, -np.inf), 1), np.nan)
    chi2, ndof = sel(tb["chi2"]), sel(tb["ndof"])
    tr.update(status=sel(tb["status"]), goodfit=sel(tb["goodfit"]), seedalg=np.full(T, -1), fitalg=np.full(T, FITALG_KKLINE),
              pdg=np.full(T, PDG_MU), chisq=chi2, ndof=ndof, fitcon=_fitcon(chi2, ndof), nipaup=np.zeros(T),
              nipadown=np.zeros(T), nstup=np.zeros(T), nstdown=np.zeros(T), tsdainter=np.zeros(T, bool),
              opainter=np.zeros(T, bool), firststinter=np.full(T, -1), maxgap=np.zeros(T), avggap=np.zeros(T))
    X = tb.get("xings")
    if X is not None:
        X = {k: sel(v) for k, v in X.items()}
        xv = X["xvalid"]
        xa = xv & X["active"].astype(bool)
        tr["nmat"], tr["nmatactive"] = xv.sum(1), xa.sum(1)
        tr["radlen"] = np.where(xa, np.nan_to_num(X["radlen"]), 0.0).sum(1)
    out["trk"] = {"trk." + k: v for k, v in tr.items()}
    # ---- trkhits
    hflat = {}
    if T:
        m = hv
        f = lambda a: np.asarray(a)[m]
        dT = g["stoca"] - g["ptoca"]
        dhit, dactive = _same_panel(upanel, m, act)
        hflat = {"plane": f(plane), "panel": f(panel), "layer": f(layer), "straw": f(straw), "state": f(state),
                 "earlyend": f(hit["eend"]), "edep": f(hit["edep"]), "wdist": f(hit["wdist"]),
                 "werr": np.sqrt(f(hit["uvar"])), "tottdrift": f(hit["totdt"]),
                 "etime": np.stack([f(hit["et0"]), f(hit["et1"])], 1), "tot": np.stack([f(hit["tot0"]), f(hit["tot1"])], 1),
                 "ptoca": f(g["ptoca"]), "stoca": f(g["stoca"]), "rdoca": f(-hit["esign"] * g["doca"]),
                 "rdocavar": f(dvar), "rdt": f(dT), "rupos": f(g["sw"]), "rdrift": f(_drift(hit, dT, "rdrift")),
                 "sderr": f(_drift(hit, dT, "sderr")), "uderr": f(_drift(hit, dT, "uderr")), "dvel": f(_drift(hit, dT, "dvel")),
                 "wdot": f(g["wdot"]), "poca.fCoordinates.fX": f(g["poca"][..., 0]),
                 "poca.fCoordinates.fY": f(g["poca"][..., 1]), "poca.fCoordinates.fZ": f(g["poca"][..., 2]),
                 "dhit": f(dhit), "dactive": f(dactive)}
    out["trkhits"] = hflat
    out["nhit"] = nh
    # ---- trksegs and trksegpars_kl at the envelope crossings
    p, u, t0, mom, v = line(x) if T else (None,) * 5
    sflat, pflat = {}, {}
    if T:
        S, I = tracker_crossings(x)
        o = np.argsort(np.where(np.isfinite(S), S, np.inf), axis=1, kind="stable")       # time order
        S, I = np.take_along_axis(S, o, 1), np.take_along_axis(I, o, 1)
        ok = np.isfinite(S)
        nseg = ok.sum(1).astype(np.int64)
        s, sid = S[ok], I[ok]
        ti = np.repeat(np.arange(T), nseg)
        pos = p[ti] + u[ti] * s[:, None]
        momv = mom[ti][:, None] * u[ti]
        momfit = sel(tb["momfit"]).astype(bool)
        momerr = np.where(momfit[ti], np.sqrt(np.where(np.isfinite(cov[ti, 5, 5]), cov[ti, 5, 5], np.nan)), np.nan)
        sflat = {"mom.fCoordinates.fX": momv[:, 0], "mom.fCoordinates.fY": momv[:, 1], "mom.fCoordinates.fZ": momv[:, 2],
                 "pos.fCoordinates.fX": pos[:, 0], "pos.fCoordinates.fY": pos[:, 1], "pos.fCoordinates.fZ": pos[:, 2],
                 "time": t0[ti] + s / v[ti], "dmom": np.zeros(len(s)), "momerr": momerr,
                 "inbounds": np.ones(len(s), bool), "gap": np.zeros(len(s), bool), "sid": sid, "sindex": np.zeros(len(s))}
        kp = kl_params(x)
        kc = kl_cov(x, cov)
        names = ("d0", "phi0", "z0", "theta", "t0", "mom")
        pflat = {k: kp[ti, i] for i, k in enumerate(names)}
        pflat.update({k + "err": np.sqrt(np.where(kc[ti, i, i] >= 0, kc[ti, i, i], np.nan)) for i, k in enumerate(names)})
        if not momfit.all():
            pflat["momerr"] = np.where(momfit[ti], pflat["momerr"], np.nan)
    else:
        nseg = np.zeros(0, np.int64)
    out["trksegs"], out["trksegpars_kl"], out["nseg"] = sflat, pflat, nseg
    # ---- trkmats
    if X is not None:
        xv = X["xvalid"]
        nx = xv.sum(1).astype(np.int64)
        f = lambda a: np.asarray(a)[xv]
        xpl, xpa, xla, xst = _strawid(X["sid"])
        if "hit" in X:
            hidx = np.asarray(X["hit"])
            if hidx.shape != xv.shape or hidx.dtype.kind not in "iu":
                raise ValueError("crossing hit associations must be integer slots matching crossing shape")
            if np.any(xv & ((hidx < -1) | (hidx >= tb["hvalid"].shape[1]))):
                raise ValueError("crossing hit association outside original padded hit block")
            safe = np.where(xv & (hidx >= 0), hidx, 0)
            original_valid, original_state, original_sid = sel(tb["hvalid"]), sel(tb["state"]), sel(tb["sid"])
            hashit = xv & (hidx >= 0)
            if np.any(hashit & (~np.take_along_axis(original_valid, safe, 1)
                               | (np.take_along_axis(original_sid, safe, 1) != X["sid"]))):
                raise ValueError("crossing association must reference a valid hit on the same straw")
            associated_state = np.take_along_axis(original_state, safe, 1)
            hact = hashit & (associated_state > -2)
            hdrift = hact & (np.abs(associated_state) == 1)
        else:
            hashit, hidx = _straw_lookup(xpl * 6 * 96 + xpa * 96 + xst, xv, plane * 6 * 96 + panel * 96 + straw, hv)
            hact = np.take_along_axis(act, hidx, 1) & hashit
            hdrift = hact & (np.abs(np.take_along_axis(state, hidx, 1)) == 1)
        out["trkmats"] = {"active": f(X["active"].astype(bool)), "hashit": f(hashit), "activehit": f(hact),
                          "drifthit": f(hdrift), "plane": f(xpl), "panel": f(xpa), "layer": f(xla), "straw": f(xst),
                          "doca": f(X["doca"]), "docavar": f(X["docavar"]), "radlen": f(X["radlen"]),
                          "dirdot": f(X["dirdot"]), "gaspath": f(X["gas"]), "wallpath": f(X["wall"]), "wirepath": f(X["wire"])}
        out["nmat"] = nx
    else:
        out["trkmats"], out["nmat"] = None, None
    return out


def _same_panel(upanel, valid, act):
    """Per hit: another hit of the track on the same panel (dhit), and an active one (dactive)."""
    T, N = upanel.shape
    key = np.where(valid, np.arange(T)[:, None] * 1000 + upanel, -1)
    flat, fa = key.ravel(), act.ravel().astype(np.int64)
    u, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    nact = np.bincount(inv, weights=fa, minlength=len(u))
    inv = inv.reshape(T, N)
    dhit = valid & (cnt[inv] >= 2)
    dactive = valid & ((nact[inv] - act) >= 1)
    return dhit, dactive


def _straw_lookup(xkey, xvalid, hkey, hvalid):
    """Per crossing: does the track have a hit on that straw, and the hit slot (0 where none)."""
    T, M = xkey.shape
    N = hkey.shape[1]
    hk = np.where(hvalid, np.arange(T)[:, None] * 10 ** 6 + hkey, -1).ravel()
    o = np.argsort(hk, kind="stable")
    hs = hk[o]
    xk = np.where(xvalid, np.arange(T)[:, None] * 10 ** 6 + xkey, -2).ravel()
    i = np.clip(np.searchsorted(hs, xk), 0, max(len(hs) - 1, 0))
    found = (hs[i] == xk) if len(hs) else np.zeros(len(xk), bool)
    slot = np.where(found, o[i] % max(N, 1), 0).reshape(T, M)
    return found.reshape(T, M) & xvalid, slot


def _nunique(a):
    """Distinct non-negative values per row."""
    s = np.sort(a, 1)
    first = np.ones_like(s, bool)
    first[:, 1:] = s[:, 1:] != s[:, :-1]
    return (first & (s >= 0)).sum(1)


def _commit():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        c = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        d = subprocess.run(["git", "-C", here, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True, timeout=10)
        if c.returncode == 0:
            return {"commit": c.stdout.strip(), "dirty": bool(d.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    return {"commit": os.environ.get("TRKFS_COMMIT", "unknown"), "dirty": None}


def content_hash(arrays):
    h = hashlib.sha256()
    for b in sorted(arrays):
        form, length, bufs = ak.to_buffers(ak.to_packed(arrays[b]))
        h.update(b.encode())
        h.update(json.dumps(form.to_dict(), sort_keys=True).encode())
        h.update(str(length).encode())
        for k in sorted(bufs):
            h.update(k.encode())
            h.update(np.ascontiguousarray(bufs[k]).tobytes())
    return h.hexdigest()


# leaves whose class depends on the fit: (fit_type, branch, leaf name) -> class
FIT_CLASS = {("F1", "trk", "radlen"): "approx", ("F1", "trkmats", "radlen"): "approx"}


def coverage(branch, sig, used, fit_type=None):
    """{leaf path: [class, note]}: class is the first word of the COVERAGE note (filled / approx / constant) for a
    computed leaf, `sentinel` otherwise."""
    rows = {}
    for path, _, _ in ntschema.leaves(sig):
        name = path[4:] if branch == "trk" else path.split(".")[0]
        note = COVERAGE.get(branch, {}).get(name, "filled")
        cls = FIT_CLASS.get((fit_type, branch, name), note.split(":")[0].split(" ")[0])
        rows[path] = [cls, note] if path in used else ["sentinel", ""]
    return rows


def add_split_alias(trk):
    """Append the SPLIT_ALIAS sub-record (bare leaf names, same lists) to a `trk` record of per-track lists."""
    leaves = [f for f in trk.fields if f.startswith("trk.")]
    fields = {f: trk[f] for f in trk.fields}
    fields[SPLIT_ALIAS] = ak.zip({f[4:]: trk[f] for f in leaves}, depth_limit=1)
    return ak.zip(fields, depth_limit=1)


def without_alias(trk):
    """The stock `trk` record: drops the SPLIT_ALIAS sub-record when present (no-op on a stock file)."""
    if trk is None or SPLIT_ALIAS not in trk.fields:
        return trk
    return trk[[f for f in trk.fields if f != SPLIT_ALIAS]]


def build(tb, schema=DEFAULT_SCHEMA, only_good=True, split_alias=True):
    """Track table -> {branch: awkward array} in the schema layout, plus the provenance record."""
    sigs, sraw = load_schema(schema)
    c = compute(tb, only_good)
    arrays, used = {}, {}
    for b in BRANCHES:
        if b not in sigs or (b == "trkmats" and c["trkmats"] is None):
            continue
        u = set()
        if b == "evtinfo":
            arrays[b] = build_branch(sigs[b], c[b], [], u)
        elif b == "trk":
            arrays[b] = build_branch(sigs[b], c[b], [c["ntrk"]], u)
            if split_alias:
                arrays[b] = add_split_alias(arrays[b])
        else:
            inner = {"trksegs": c["nseg"], "trksegpars_kl": c["nseg"], "trkhits": c["nhit"], "trkmats": c["nmat"]}[b]
            arrays[b] = build_branch(sigs[b], c[b], [c["ntrk"], inner], u)
        used[b] = u
    cov = {b: coverage(b, sigs[b], u, tb["meta"].get("fit_type")) for b, u in used.items()}
    prov = {"writer": "trkfastsim.reco.ntuple", "layout": "RNTuple (uproot); stock names, types and nesting",
            "fit_type": tb["meta"].get("fit_type"), "stock_conditioned": tb["meta"].get("stock_conditioned"),
            "table_meta": tb["meta"], "package": _commit(), "schema": {"path": os.path.basename(schema),
                                                                        "source_file": sraw.get("source", "")},
            "sentinel": {"float": UNSET_F, "int": UNSET_I, "bool": False}, "only_good": only_good,
            "counts": c["counts"], "coverage": cov, "branches": sorted(arrays),
            "split_alias": (f"trk.{SPLIT_ALIAS}: bare-name copy of the trk leaves for tree['trk/trk.<leaf>']"
                            if split_alias else None),
            "units": "mm, ns, MeV/c, detector frame (as stock)",
            "surfaces": {"TT_Front_z": ZFRONT, "TT_Back_z": ZBACK, "disk_rmax": RDISK, "TT_Outer_r": ROUT,
                         "TT_Outer_zmid": ZMID, "TT_Outer_halflen": HALFLEN}}
    prov["content_sha256"] = content_hash(arrays)
    return arrays, prov


def write(tb, out, schema=DEFAULT_SCHEMA, only_good=True, split_alias=True):
    arrays, prov = build(tb, schema, only_good, split_alias)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with uproot.recreate(out) as f:
        f[TREE] = arrays
        f[PROV] = json.dumps(prov, sort_keys=True)
    return prov


def read_provenance(path):
    return json.loads(str(uproot.open(path)[PROV]))


def main(dump, out, fit, conditions, settings="mc", f1_tracks=None, niter=4, all_fits=False, schema=DEFAULT_SCHEMA,
         table_out=None, split_alias=True):
    from . import hitreco
    cond = hitreco.load_conditions(conditions)
    if fit == "stock":
        tb = tracktable.from_stock(dump, cond, settings)
    elif fit == "f2":
        tb = tracktable.from_f2(dump, cond, settings, niter=niter, all_fits=all_fits)
    elif fit == "f1":
        if not f1_tracks:
            raise SystemExit("--fit f1 needs --f1-tracks (reco-vr2f1 --save-tracks output)")
        tb = tracktable.from_f1(dump, cond, f1_tracks, settings)
    else:
        raise SystemExit(f"unknown fit {fit}")
    if table_out:
        tracktable.save(tb, table_out)
    prov = write(tb, out, schema, only_good=not all_fits, split_alias=split_alias)
    print(json.dumps({k: prov[k] for k in ("fit_type", "stock_conditioned", "counts", "content_sha256", "package")}, indent=1))
    return prov
