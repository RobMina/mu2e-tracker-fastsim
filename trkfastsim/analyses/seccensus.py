"""seccensus -- T-V1b secondary census and step-level panel occupancy, fast sim against Geant4.

A secondary step is every StrawGasStep not of a primary muon (fast: the pasted library steps; G4: delta rays, their
progeny, showers from the material above, and the rare non-muon CRY particles). Primary muons are particles with
isprim = 1 and |pdg| = 13 in both arms (every CRY muon: the fast sim transports all of them, and G4 flags only one
of them ismcprim). Events: the PrimaryFilter tracker term passed by any one primary muon; with
--paired (both arms on the same CRY primaries, files matched by (run, subrun)) only the events selected in both arms.

Rows, each as fast/G4 - 1 of a rate per primary-muon step with a jackknife error over file groups (file index mod 20):
  agreed T-V1b row: secondary steps per primary-muon step; secondary share of the eIon (threshold 10 % relative);
  breakdowns: species; eIon and path-length bins; position relative to the muon (in a muon straw, same panel within
  3 straws, rest of the same plane, 1-2 planes away, further); the step midpoint's distance from the wire and
  along-wire position over the half length; the event's earliest primary-muon straw-step |p| (per muon step of that event bin); in-straw
  secondaries per muon crossing by the muon's path in the straw; planes from the nearest muon plane (steps outside the
  muon straws: the off-axis plane profile); along-wire position by location (location * 10 + bin); secondary steps per
  straw holding any (muon straws 0x, others 1x; counted per straw); G4 only: creation code and start kinetic energy.
Also the combined eIon per straw (muon + secondaries, what the digitiser sees; v1compare's 'eIon per straw' band) with
a G4-against-G4 floor from two disjoint halves of the event groups, and the step-level panel occupancy (straws with a
step per uniquePanel, all particles and muon straws only) with Poisson and event-clustered errors (occupancy.py).

    seccensus FAST_TABLES [...] --compare G4_TABLES [...] [--paired] [--gdml G] --outdir DIR
"""
import json
import os

import numpy as np
import pandas as pd

from mu2efastsim import stats
from .. import geometry, tables as T
from . import occupancy as O

NG = 20
P_EDGES = np.array([0.0, 300.0, 600.0, 1000.0, 2000.0, 4000.0, 10000.0, 1e9])     # MeV/c, as library.P_BINS
EION_EDGES = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 1e9])                   # keV
LEN_EDGES = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 1e9])                   # mm
MULEN_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 1e9])             # mm, muon path in the straw
R_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.6])                                # mm from the wire
U_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.01])                             # |along-wire| / half length
KE_EDGES = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1e9])                  # MeV, G4 start kinetic energy
DPL_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 1e9])                   # planes to the nearest muon plane
MULT_EDGES = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 1e9])                             # secondary steps in one straw
LOC = ["muon_straw", "same_panel_le3", "same_plane", "dplane_1_2", "dplane_ge3"]
SPECIES = {11: "e-", -11: "e+", 13: "mu", -13: "mu", 2212: "p", 22: "gamma"}
ME = 0.51099895
PROCESS = {12: "compt", 13: "conv", 14: "Decay", 16: "eBrem", 17: "eIoni", 21: "hIoni", 29: "msc", 30: "muBrems", 31: "muIoni",
           34: "muPairProd", 40: "phot", 56: "mu2ePrimary", 114: "DIO", 116: "muonNuclear", -1: "none"}   # Mu2e ProcessCode


def _species(pdg):
    return pd.Series(pdg).map(SPECIES).fillna("other").to_numpy()


def _bin(x, edges):
    return np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)


def _load(path, g4):
    s = T.read(path, "sgsteps")
    cols = T.KEY + ["key", "pdg", "isprim", "ismcprim"] + (["ccode", "spx", "spy", "spz"] if g4 else [])
    p = T.read(path, "particles", cols).rename(columns={"key": "sp"})
    s = s.merge(p, on=T.KEY + ["sp"], how="left")
    s["pdg"] = s.pdg.fillna(0).astype(np.int64)
    s["isprim"] = s.isprim.fillna(0).astype(np.int64)
    s["ismcprim"] = s.ismcprim.fillna(0).astype(np.int64)
    s["mu"] = (s.isprim == 1) & (s.pdg.abs() == 13)
    return s


def _selected_keys(s):
    m = s[(s.isprim == 1) & (s.pdg.abs() == 13)]
    f = T.trk_filter(m)
    return f[f.pass_trk].reset_index()[T.KEY]


def _classify(s, geom):
    """Add per-step columns: loc (index into LOC, -1 for muon steps), r and |u|/half of the step midpoint, the event's
    earliest primary-muon straw-step |p| bin (all event steps share it; input order breaks time ties),
    and for in-straw secondaries the muon's path length in that straw."""
    s = s.reset_index(drop=True)
    s["row"] = np.arange(len(s))
    mu = s[s.mu]
    sec = s[~s.mu]
    loc = np.full(len(s), -1, np.int64)
    dmin = np.full(len(s), -1.0)
    mst = mu[T.KEY + ["sid", "len"]].groupby(T.KEY + ["sid"], as_index=False).len.sum().rename(columns={"len": "mulen"})
    x = sec[T.KEY + ["sid", "row"]].merge(mst, on=T.KEY + ["sid"], how="left")
    inmu = x.mulen.notna().to_numpy()
    s["mulen"] = np.nan
    s.loc[x.row.to_numpy()[inmu], "mulen"] = x.mulen.to_numpy()[inmu]
    loc[x.row.to_numpy()[inmu]] = 0
    rest = sec[~np.isin(sec.row.to_numpy(), x.row.to_numpy()[inmu])]
    if len(rest):
        mpl = mu[T.KEY + ["plane"]].drop_duplicates().rename(columns={"plane": "mpl"})
        y = rest[T.KEY + ["plane", "row"]].merge(mpl, on=T.KEY, how="left")
        y["dpl"] = (y.plane - y.mpl).abs()
        dpl = y.groupby("row").dpl.min()
        mps = mu[T.KEY + ["upanel", "straw"]].rename(columns={"straw": "mst"})
        z = rest[T.KEY + ["upanel", "straw", "row"]].merge(mps, on=T.KEY + ["upanel"], how="inner")
        dst = (z.straw - z.mst).abs().groupby(z.row).min()
        rows = rest.row.to_numpy()
        d = dpl.reindex(rows).to_numpy()
        ds = dst.reindex(rows).to_numpy()
        lr = np.where(np.isnan(d), 4, np.where(d == 0, np.where(ds <= 3, 1, 2), np.where(d <= 2, 3, 4)))
        loc[rows] = lr
        dmin[rows] = np.where(np.isnan(d), 99.0, d)
    s["loc"] = loc
    s["dplmin"] = dmin
    js = (s.upanel.to_numpy() * geometry.NSTRAWS + s.straw.to_numpy()).astype(np.int64)
    xm = geometry.to_mu2e(0.5 * (s[["x0", "y0", "z0"]].to_numpy(float) + s[["x1", "y1", "z1"]].to_numpy(float)))
    v = xm - geom.mid[js]
    u = np.einsum("ij,ij->i", v, geom.dir[js])
    s["r"] = np.linalg.norm(v - u[:, None] * geom.dir[js], axis=1)
    s["uf"] = np.abs(u) / geom.halflen[js]
    f = mu.sort_values("t", kind="stable").drop_duplicates(T.KEY).set_index(T.KEY)[["px", "py", "pz"]]
    f["pb"] = _bin(np.sqrt(f.px ** 2 + f.py ** 2 + f.pz ** 2).to_numpy(), P_EDGES)
    s = s.merge(f[["pb"]].reset_index(), on=T.KEY, how="left")
    s["pb"] = s.pb.fillna(0).astype(np.int64)
    return s


def _counts(s, group, g4):
    """Aggregated counts of one file's selected steps for every breakdown: {name: DataFrame(group, bin, n, e)} plus the
    denominators (muon steps per |p| bin and per path bin) and totals."""
    mu, sec = s[s.mu], s[~s.mu]
    e = 1e3 * sec.eion.to_numpy()
    out = {}

    def add(name, bins, sub=None, weights_e=None):
        d = sec if sub is None else sub
        ee = 1e3 * d.eion.to_numpy()
        df = pd.DataFrame({"bin": bins, "n": 1.0, "e": ee}).groupby("bin").sum()
        df["group"] = group
        out[name] = df.reset_index()

    add("species", _species(sec.pdg.to_numpy()))
    add("eion_keV", _bin(e, EION_EDGES))
    add("len_mm", _bin(sec.len.to_numpy(), LEN_EDGES))
    add("location", sec["loc"].to_numpy())
    add("wire_distance_mm", _bin(sec.r.to_numpy(), R_EDGES))
    add("along_wire_over_half", _bin(sec.uf.to_numpy(), U_EDGES))
    add("muon_p_first_straw", sec.pb.to_numpy())
    ins = sec[sec["loc"] == 0]
    add("instraw_by_muon_len", _bin(ins.mulen.to_numpy(), MULEN_EDGES), ins)
    out_ = sec[sec["loc"] > 0]
    add("plane_distance", _bin(out_.dplmin.to_numpy(), DPL_EDGES), out_)
    add("along_wire_by_location", sec["loc"].to_numpy() * 10 + _bin(sec.uf.to_numpy(), U_EDGES))
    # secondary steps per straw (straws holding >= 1 secondary step), muon straws (class 0) and others (class 1)
    ps = sec.assign(cls=(sec["loc"] > 0).astype(np.int64)).groupby(T.KEY + ["sid"]).agg(n=("eion", "size"), eion=("eion", "sum"),
                                                                                    cls=("cls", "max"))
    add("sec_steps_per_straw", ps.cls.to_numpy() * 10 + _bin(ps.n.to_numpy(), MULT_EDGES), ps)
    if g4:
        add("g4_creation_code", sec.ccode.fillna(-1).astype(np.int64).to_numpy())
        m = np.where(sec.pdg.abs().to_numpy() == 11, ME, 0.0)
        pp = np.sqrt(sec.spx ** 2 + sec.spy ** 2 + sec.spz ** 2).to_numpy()
        add("g4_start_ke_MeV", _bin(np.sqrt(pp ** 2 + m ** 2) - m, KE_EDGES))
    den = {"muon_p_first_straw": pd.DataFrame({"bin": mu.pb.to_numpy(), "n": 1.0}).groupby("bin").sum().reset_index(),
           "instraw_by_muon_len": pd.DataFrame({"bin": _bin(mu.groupby(T.KEY + ["sid"]).len.sum().to_numpy(), MULEN_EDGES),
                                                "n": 1.0}).groupby("bin").sum().reset_index()}
    for v in den.values():
        v["group"] = group
    tot = {"group": group, "events": int(s[T.KEY].drop_duplicates().shape[0]), "mu_steps": float(len(mu)),
           "mu_eion": float(1e3 * mu.eion.sum()), "sec_steps": float(len(sec)), "sec_eion": float(e.sum()),
           "sec_pdg13": float((sec.pdg.abs() == 13).sum())}
    return out, den, tot


def _straw_eion(s):
    return 1e3 * s.groupby(T.KEY + ["sid"]).eion.sum().to_numpy()


def _occ(s, accs):
    """Step-level panel occupancy: distinct straws with a step per event and uniquePanel (all particles; muon straws)."""
    # IDs are local to this accumulation call; no bit packing or cross-file pairing is needed.
    ev, _ = pd.factorize(pd.MultiIndex.from_frame(s[T.KEY]), sort=False)
    straw_keys = pd.MultiIndex.from_frame(s[T.KEY + ["sid"]])
    mu = s.mu.to_numpy()
    secondary_only = ~straw_keys.isin(straw_keys[mu])
    for name, m in (("all", np.ones(len(s), bool)), ("muon_straws", mu), ("secondary_only", secondary_only)):
        d = pd.DataFrame({"ev": ev[m], "sid": s.sid.to_numpy()[m], "up": s.upanel.to_numpy()[m]}).drop_duplicates(["ev", "sid"])
        accs[name] = O.accumulate(d.ev.to_numpy(), d.up.to_numpy(), acc=accs.get(name))


def _jack(num_f, den_f, num_r, den_r):
    """Ratio (num_f/den_f) / (num_r/den_r) - 1 over NG groups with its delete-one-group jackknife error."""
    nf, df_, nr, dr = (np.asarray(a, float) for a in (num_f, den_f, num_r, den_r))
    tot = lambda a, k=None: a.sum() - (0 if k is None else a[k])
    if tot(df_) == 0 or tot(dr) == 0 or tot(nr) == 0:
        return float("nan"), float("nan")
    R = (tot(nf) / tot(df_)) / (tot(nr) / tot(dr))
    rk = []
    for k in range(len(nf)):
        if tot(df_, k) > 0 and tot(dr, k) > 0 and tot(nr, k) > 0:
            rk.append((tot(nf, k) / tot(df_, k)) / (tot(nr, k) / tot(dr, k)))
    rk = np.array(rk)
    g = len(rk)
    err = float(np.sqrt((g - 1) / g * ((rk - rk.mean()) ** 2).sum())) if g > 1 else float("nan")
    return float(R - 1), err


def _by_group(df, col, bins):
    """(bin x group) matrix of column col."""
    m = np.zeros((len(bins), NG))
    if df is None or not len(df):
        return m
    grouped = df.groupby(["bin", "group"])[col].sum().unstack("group", fill_value=0)
    return grouped.reindex(index=bins, columns=range(NG), fill_value=0).to_numpy(dtype=float)


def _paired_files(ff, fr):
    """Match disjoint, identical (run, subrun) coverage without splitting or reusing a reference file."""
    if not ff or not fr:
        raise ValueError("paired seccensus requires input files in both arms")
    owners = {}
    for arm, files in (("fast", ff), ("g4", fr)):
        owner = {}
        for i, path in enumerate(files):
            keys = T.read(path, "subruns", ["run", "subrun"])
            if keys.empty or keys.isna().any().any():
                raise ValueError(f"paired seccensus: missing (run, subrun) coverage in {arm} file {path}")
            for key in keys[["run", "subrun"]].drop_duplicates().itertuples(index=False, name=None):
                if key in owner:
                    raise ValueError(f"paired seccensus: overlapping (run, subrun) {key} in {arm} files "
                                     f"{files[owner[key]]} and {path}")
                owner[key] = i
        owners[arm] = owner
    fast, ref = owners["fast"], owners["g4"]
    if fast.keys() != ref.keys():
        raise ValueError(f"paired seccensus: unequal (run, subrun) coverage; "
                         f"fast-only {sorted(fast.keys() - ref.keys())}; g4-only {sorted(ref.keys() - fast.keys())}")
    ref_groups = [set() for _ in fr]
    for key, i in ref.items():
        ref_groups[i].add(fast[key])
    refs = [[] for _ in ff]
    for path, groups in zip(fr, ref_groups):
        if len(groups) != 1:
            raise ValueError(f"paired seccensus: reference file {path} straddles multiple fast file groups")
        refs[next(iter(groups))].append(path)
    return [(path, sorted(refs[i]), i % NG) for i, path in enumerate(ff)]


def _report(C, D, TT, names):
    """Per breakdown and bin: fast and G4 rates per primary-muon step (or per the breakdown's own denominator), their
    share of all secondary steps, and fast/G4 - 1 with the jackknife error; steps and eIon."""
    rep = {}
    tf, tr = TT["fast"], TT["g4"]
    muf = np.array([tf.get(g, {}).get("mu_steps", 0.0) for g in range(NG)])
    mur = np.array([tr.get(g, {}).get("mu_steps", 0.0) for g in range(NG)])
    for name in names:
        cf, cr = C["fast"].get(name), C["g4"].get(name)
        bins = sorted(set((cf["bin"].tolist() if cf is not None else []) + (cr["bin"].tolist() if cr is not None else [])),
                      key=lambda b: (isinstance(b, str), b))
        rows = []
        for q in ("n", "e"):
            mf, mr = _by_group(cf, q, bins), _by_group(cr, q, bins)
            for i, b in enumerate(bins):
                if name in D["fast"]:
                    df_, dr = _by_group(D["fast"][name], "n", bins)[i], _by_group(D["g4"][name], "n", bins)[i]
                else:
                    df_, dr = muf, mur
                r, e = _jack(mf[i], df_, mr[i], dr)
                if q == "n":
                    rows.append({"bin": b if isinstance(b, str) else int(b), "steps": [float(mf[i].sum()), float(mr[i].sum())],
                                 "per_den": [float(mf[i].sum() / max(df_.sum(), 1)), float(mr[i].sum() / max(dr.sum(), 1))],
                                 "share": [float(mf[i].sum() / max(mf.sum(), 1)), float(mr[i].sum() / max(mr.sum(), 1))],
                                 "ratio_minus_1": r, "err": e})
                else:
                    rows[i].update({"eion_keV": [float(mf[i].sum()), float(mr[i].sum())], "eion_ratio_minus_1": r, "eion_err": e})
        rep[name] = rows
    return rep


def run(inputs, outdir, compare=None, gdml=None, paired=False, **kw):
    if not compare:
        raise SystemExit("seccensus FAST --compare G4 [--paired]")
    ff, fr = T.expand(inputs), T.expand(compare)
    if paired:
        pairs = _paired_files(ff, fr)
    else:
        pairs = [(f, None, i % NG) for i, f in enumerate(ff)] + [(None, [f], i % NG) for i, f in enumerate(fr)]
    from .steps import default_gdml
    geom = geometry.from_gdml(gdml or default_gdml())
    os.makedirs(outdir, exist_ok=True)
    C = {"fast": {}, "g4": {}}
    D = {"fast": {}, "g4": {}}
    TT = {"fast": {}, "g4": {}}
    straw = {"fast": [], "g4": [], "g4_group": []}
    occ = {"fast": {}, "g4": {}, "g4_even": {}, "g4_odd": {}}
    nsel = {"fast": 0, "g4": 0, "both": 0}
    lt = {"fast": 0.0, "g4": 0.0}
    for f_fast, f_g4, g in pairs:
        arms = {}
        for lab, f in (("fast", [f_fast] if f_fast else None), ("g4", f_g4)):
            if f is None:
                continue
            s = pd.concat([_load(x, lab == "g4") for x in f], ignore_index=True)
            arms[lab] = (s, _selected_keys(s))
            lt[lab] += T.livetime(f)
            nsel[lab] += len(arms[lab][1])
        if paired:
            both = arms["fast"][1].merge(arms["g4"][1], on=T.KEY)
            nsel["both"] += len(both)
            arms = {k: (v[0], both) for k, v in arms.items()}
        for lab, (s, keys) in arms.items():
            s = s.merge(keys, on=T.KEY)
            s = _classify(s, geom)
            c, d, t = _counts(s, g, lab == "g4")
            for k, v in c.items():
                C[lab][k] = pd.concat([C[lab].get(k), v]) if k in C[lab] else v
            for k, v in d.items():
                D[lab][k] = pd.concat([D[lab].get(k), v]) if k in D[lab] else v
            prev = TT[lab].get(g, {})
            TT[lab][g] = {k: prev.get(k, 0) + v for k, v in t.items() if k != "group"}
            se = _straw_eion(s)
            straw[lab].append(se)
            if lab == "g4":
                straw["g4_group"].append(np.full(len(se), g))
            _occ(s, occ[lab])
            if lab == "g4":
                _occ(s, occ["g4_even" if g % 2 == 0 else "g4_odd"])
        print(f"# group {g}: {f_fast or ''} {len(f_g4 or [])} G4 files; selected so far {nsel}", flush=True)
    for lab in C:
        for k in list(C[lab]):
            C[lab][k] = C[lab][k].groupby(["group", "bin"], as_index=False).sum()
        for k in list(D[lab]):
            D[lab][k] = D[lab][k].groupby(["group", "bin"], as_index=False).sum()
    names = sorted(set(C["fast"]) | set(C["g4"]))
    rep = _report(C, D, TT, [n for n in names if not n.startswith("g4_")])
    g4only = {}
    for n in [n for n in names if n.startswith("g4_")]:
        df = C["g4"][n].groupby("bin")[["n", "e"]].sum()
        g4only[n] = [{"bin": int(b), **({"process": PROCESS.get(int(b), "other")} if n == "g4_creation_code" else {}),
                      "steps": float(r.n), "share": float(r.n / df.n.sum()), "eion_keV": float(r.e)}
                     for b, r in df.iterrows()]
    G = lambda lab, k: np.array([TT[lab].get(g, {}).get(k, 0.0) for g in range(NG)])
    row_steps = _jack(G("fast", "sec_steps"), G("fast", "mu_steps"), G("g4", "sec_steps"), G("g4", "mu_steps"))
    # eIon share: ratio of shares with the jackknife (share = sec / (sec + mu))
    row_share = _jack(G("fast", "sec_eion"), G("fast", "sec_eion") + G("fast", "mu_eion"),
                      G("g4", "sec_eion"), G("g4", "sec_eion") + G("g4", "mu_eion"))
    row_eion_per_mu = _jack(G("fast", "sec_eion"), G("fast", "mu_steps"), G("g4", "sec_eion"), G("g4", "mu_steps"))
    sf_, sr_ = np.concatenate(straw["fast"]), np.concatenate(straw["g4"])
    grp = np.concatenate(straw["g4_group"])
    band = stats.ratio_band(sf_, sr_)
    floor = stats.ratio_band(sr_[grp % 2 == 0], sr_[grp % 2 == 1])
    res = {"paired": bool(paired),
           "statistics": {"occupancy": "independent-arm covariance; diagnostic only for paired/shared-event inputs",
                          "energy_band": "legacy independent-entry variance, not event-clustered; G4 split floor reported",
                          "breakdown_errors": "delete-one-file-group jackknife; paired groups only with --paired"},
           "input_files": {"fast": ff, "g4": fr},
           "selection": {"primary_muon": "isprim == 1 and abs(pdg) == 13 (not ismcprim)",
                         "event_filter": "PrimaryFilter tracker term passed by any one primary muon (steps grouped by SimParticle)",
                         "event_keys": "intersection of selected (run, subrun, event) keys" if paired else "independent per arm",
                         "file_pairing": "equal, disjoint (run, subrun) coverage; whole reference files assigned once to a fast file"
                                         if paired else "none",
                         "momentum_bin": "event-level |p| at earliest primary-muon straw step by t; input order breaks ties; "
                                         "all muon and secondary steps in the event share this bin",
                         "groups": "fast file index mod 20 with whole G4 partner files" if paired else "file index mod 20 independently per arm"},
           "file_groups": [{"fast": f, "g4": r, "group": g} for f, r, g in pairs],
           "livetime_s": lt, "selected_events": nsel,
           "totals": {lab: {k: float(G(lab, k).sum()) for k in ("events", "mu_steps", "mu_eion", "sec_steps", "sec_eion", "sec_pdg13")}
                      for lab in TT},
           "tv1b": {"sec_steps_per_mu_step": {"value": [float(G("fast", "sec_steps").sum() / G("fast", "mu_steps").sum()),
                                                        float(G("g4", "sec_steps").sum() / G("g4", "mu_steps").sum())],
                                              "ratio_minus_1": row_steps[0], "err": row_steps[1], "threshold": 0.10,
                                              "pass": bool(abs(row_steps[0]) <= 0.10)},
                    "sec_eion_share": {"value": [float(G("fast", "sec_eion").sum() / (G("fast", "sec_eion").sum() + G("fast", "mu_eion").sum())),
                                                 float(G("g4", "sec_eion").sum() / (G("g4", "sec_eion").sum() + G("g4", "mu_eion").sum()))],
                                       "ratio_minus_1": row_share[0], "err": row_share[1], "threshold": 0.10,
                                       "pass": bool(abs(row_share[0]) <= 0.10)},
                    "sec_eion_per_mu_step": {"ratio_minus_1": row_eion_per_mu[0], "err": row_eion_per_mu[1]}},
           "eion_per_straw": {"mean": [float(sf_.mean()), float(sr_.mean())],
                              "quantiles_q25_q50_q75_q90": [np.quantile(sf_, [0.25, 0.5, 0.75, 0.9]).tolist(),
                                                            np.quantile(sr_, [0.25, 0.5, 0.75, 0.9]).tolist()],
                              "band_minus_2sigma": band["max_abs_dev_minus_2sigma"], "band": band["max_abs_dev"],
                              "g4_half_vs_half_band_minus_2sigma": floor["max_abs_dev_minus_2sigma"],
                              "g4_half_vs_half_band": floor["max_abs_dev"],
                              "bins": {"edges": band["edges"].tolist(), "fast": band["h_fast"].tolist(), "g4": band["h_ref"].tolist()}},
           "breakdowns": rep, "g4_only": g4only,
           "occupancy_steps": {k: O.compare(occ["fast"][k], occ["g4"][k]) for k in occ["fast"]},
           "occupancy_steps_g4_floor": {k: O.compare(occ["g4_even"][k], occ["g4_odd"][k]) for k in occ["g4_even"]}}
    with open(os.path.join(outdir, "seccensus.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    _print(res)


def _print(res):
    print(json.dumps({k: res[k] for k in ("paired", "input_files", "selection", "file_groups", "livetime_s", "selected_events", "totals", "tv1b")}, indent=1, default=float))
    e = res["eion_per_straw"]
    print(f"eion per straw: mean {e['mean']}, band-2s {e['band_minus_2sigma']}, G4 half/half {e['g4_half_vs_half_band_minus_2sigma']}")
    for name, rows in res["breakdowns"].items():
        print(f"== {name}")
        for r in rows:
            print(f"  {str(r['bin']):>10s}  steps {r['steps'][0]:9.0f} {r['steps'][1]:9.0f}  per-den {r['per_den'][0]:.5f} "
                  f"{r['per_den'][1]:.5f}  share {r['share'][0]:.3f} {r['share'][1]:.3f}  ratio-1 {r['ratio_minus_1']:+.3f} "
                  f"+- {r['err']:.3f}  eion ratio-1 {r.get('eion_ratio_minus_1', float('nan')):+.3f} +- {r.get('eion_err', float('nan')):.3f}")
    for name, rows in res["g4_only"].items():
        print(f"== {name} (G4)")
        for r in rows:
            print(f"  {r['bin']:>6d} {r.get('process', ''):>12s}  steps {r['steps']:9.0f}  share {r['share']:.3f}  eion {r['eion_keV']:.0f}")
    for k, v in [(f"{a}", b) for a, b in res["occupancy_steps"].items()] + \
            [(f"{a} G4 even/odd", b) for a, b in res["occupancy_steps_g4_floor"].items()]:
        print(f"== step occupancy {k}: chi2/ndf Poisson {v['chi2ndf_poisson']:.2f} clustered {v['chi2ndf_clustered']:.2f} "
              f"overdisp {v['overdispersion_median']:.2f} max|pull| {v['max_abs_pull']:.1f} >5s {v['panels_beyond_5sigma']}")
