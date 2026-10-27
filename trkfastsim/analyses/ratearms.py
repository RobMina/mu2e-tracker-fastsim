"""ratearms -- PrimaryFilter tracker-term rate of the transport-ledger arms, paired with G4 on the same CRY primaries.

Three steps, each writing new files only (an existing output is an error):

  python3 -m trkfastsim.analyses.ratearms g4 G4_SGSTEPS... --out G4.parquet
      per primary muon of every G4-kept event: good steps, crossings, |p| at the first/last/slowest step, steps below
      the 50 MeV/c filter momentum, the G4 end point and energy; the generation index links it to the fast muon.
  python3 -m trkfastsim.analyses.ratearms fast GEN... --gdml GDML --arm PRESET --seed 1 --index0 0 --g4 G4.parquet
          --outdir DIR [--history-steps]
      stage C fast transport with the current defaults (sliced MSC, interior, kink splitting, tail) and the ledger arm
      (trkfastsim.ledger.PRESETS); file j of the call uses the generator seed (seed, index0 + j), so arms and file
      subsets with the same global index are paired draw for draw up to the first arm difference. Writes per-muon rows
      for every event that is G4-kept or has a fast good step (muons.<n>.parquet), the per-step ledger of those events
      with --history-steps (segments.<n>.parquet), and per-file ledger closure summaries (files.jsonl).
  python3 -m trkfastsim.analyses.ratearms compare --g4 G4.parquet --gdml GDML --arm NAME=DIR [...] --base NAME
          --out REPORT.json
      paired tables per arm against G4 (event-level tracker term, any primary muon; and the first-muon-only convention
      of fatecheck), by energy margin (paired_budget.py: p_gen / (2.1 MeV cm2/g x world column), first muon) and
      generated |p|; each arm against the base arm (same primaries and seeds); and the replay inventory of the G4-only
      and fast-only events (fast fate and ledger state of the muon that passes in the other arm).

Errors are the paired ones, sqrt(only_a + only_b) / n_ref. Muon identity: a G4 primary's SimParticle key is its index
among ALL the event's generated particles + 1 (CRY also makes p, n, e, gamma primaries; the particles' `genidx` column
is 0 throughout), while the fast key `sp` is the muon's rank among the generated muons. compare maps every G4 muon to
the fast key through the generator index (g4_fast_keys) and matches by (event, fast key); "first muon" is fast key 1.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from mu2efastsim import eloss
from mu2efastsim.tables import all_primary_muons
from .. import geometry, ledger, steps as st, tables as T

MIN = T.FILTER_MIN_TRK_STEPS
MARGIN_EDGES = np.array([0, 1, 1.25, 1.5, 2, 3, 5, 1e9])
P_EDGES = np.array([0.0, 300.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 10000.0, 1e9])
IDM = T.KEY + ["sp"]


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _src_identity():
    here = Path(__file__).resolve().parents[1]
    files = ["steps.py", "ledger.py", "analyses/ratearms.py", "crossings.py", "geometry.py", "tables.py"]
    out = {f: _sha(here / f) for f in files}
    import mu2efastsim
    core = Path(mu2efastsim.__file__).resolve().parent
    out.update({f"core/{f}": _sha(core / f) for f in ("transport.py", "eloss.py", "materials.py", "world.py", "tables.py")})
    try:
        out["tracker_git_head"] = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True,
                                                 text=True, timeout=20).stdout.strip()
    except Exception:
        out["tracker_git_head"] = None
    return out


def _new(path):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _step_summary(s, key_cols):
    """Per muon over its steps: crossings, good steps, |p| first/last/min (path order = time order), steps below the
    filter momentum that have the filter eIon."""
    if not len(s):
        return pd.DataFrame(columns=key_cols + ["nsteps", "ngood", "p_first", "p_last", "p_min", "n_lowp"])
    s = s.sort_values(key_cols + ["t"], kind="stable")
    p = np.sqrt(s.px ** 2 + s.py ** 2 + s.pz ** 2)
    s = s.assign(_p=p.to_numpy(), _good=T.good_steps(s).to_numpy(),
                 _low=((s.eion > T.FILTER_MIN_STEP_EION) & (p <= T.FILTER_MIN_MOM)).to_numpy())
    g = s.groupby(key_cols, sort=False)
    return pd.DataFrame({"nsteps": g.size(), "ngood": g._good.sum(), "p_first": g._p.first(), "p_last": g._p.last(),
                         "p_min": g._p.min(), "n_lowp": g._low.sum()}).reset_index()


# ---------------------------------------------------------------------------------------------------------------- g4
def run_g4(inputs, out):
    out = _new(out)
    rows, files = [], []
    for f in T.expand(inputs):
        s = T.read(f, "sgsteps")
        p = T.read(f, "particles")
        kept = T.read(f, "gen", T.KEY).drop_duplicates()
        pm = p[(p.ismcprim == 1) & (p.pdg.abs() == 13)][T.KEY + ["key", "genidx", "ex", "ey", "ez", "ee", "et"]]
        pm = pm.merge(kept, on=T.KEY)
        s = s.merge(pm[T.KEY + ["key"]].rename(columns={"key": "sp"}), on=IDM)
        summ = _step_summary(s, IDM)
        mm = pm.rename(columns={"key": "sp"}).merge(summ, on=IDM, how="left")
        for c in ("nsteps", "ngood", "n_lowp"):
            mm[c] = mm[c].fillna(0).astype(np.int64)
        mm["g4_stopped"] = (mm.ee - eloss.MMU) < 1.0
        rows.append(mm)
        files.append({"file": f, "sha256": _sha(f), "kept_events": int(len(kept)), "primary_muons": int(len(pm))})
        print(f"# {f}: {len(kept):,} kept events, {len(pm):,} primary muons, "
              f"{int((summ.ngood >= MIN).sum()) if len(summ) else 0:,} muons passing", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out, index=False)
    json.dump({"files": files, "source": _src_identity()}, open(str(out) + ".json", "w"), indent=1)
    print(f"wrote {out}: {len(df):,} muons")


# -------------------------------------------------------------------------------------------------------------- fast
def run_fast(inputs, outdir, gdml, arm, seed, index0, g4, history_steps=False, tag=None):
    outdir = Path(outdir)
    tag = tag or f"{index0:04d}"
    m_out, s_out, j_out = (_new(outdir / f"muons.{tag}.parquet"), outdir / f"segments.{tag}.parquet",
                           _new(outdir / f"files.{tag}.jsonl"))
    if history_steps:
        _new(s_out)
    arms = ledger.preset(arm)
    geom = geometry.from_gdml(gdml)
    world = st.load_world(gdml, geom)
    interior = st.TrackerInterior(gdml, geom)
    y_pl = st.plane_y(geom)
    g4k = pd.read_parquet(g4, columns=T.KEY).drop_duplicates()
    from .worldcheck import _column
    settings = {"arm": arm, **arms.as_dict(), **ledger.core_world_convention(), "seed": seed, "gdml_sha256": _sha(gdml),
                "source": _src_identity(), "g4": str(g4), "history_steps": history_steps,
                "platform": {"system": platform.system(), "machine": platform.machine(),
                             "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
    mrows, srows = [], []
    with open(j_out, "w") as jf:
        jf.write(json.dumps({"settings": settings}) + "\n")
        for j, f in enumerate(T.expand(inputs)):
            i = index0 + j
            gen = T.read(f, "gen")
            # energy writer (writer-only diagnostic) is defined without exit scattering, which does not change steps
            sgs, _ = st.make_steps(gen, geom, np.random.default_rng([int(seed), i]), stage="C", world=world,
                                   sliced_msc=True, interior=interior, arms=arms, history=True,
                                   exit_scatter=arms.writer != "energy")
            mu = st.make_steps.muon_ledger.copy()
            seg = st.make_steps.segment_ledger
            allmu = all_primary_muons(gen)
            if not (allmu[T.KEY].to_numpy() == mu[T.KEY].to_numpy()).all():
                raise RuntimeError("muon order of the ledger differs from all_primary_muons")
            mu["genidx"] = allmu.idx.to_numpy() if "idx" in allmu.columns else -1
            o = allmu[["x", "y", "z"]].to_numpy()
            dvec = allmu[["px", "py", "pz"]].to_numpy() / mu.p_gen_MeV_c.to_numpy()[:, None]
            mu["near"] = st.near_tracker(o, dvec, geom, y_pl)
            summ = _step_summary(sgs[sgs.sp < st.SECONDARY_KEY0], IDM)
            mu = mu.merge(summ, on=IDM, how="left")
            for c in ("nsteps", "ngood", "n_lowp"):
                mu[c] = mu[c].fillna(0).astype(np.int64)
            ev = mu.groupby(T.KEY).ngood.max().rename("nmax").reset_index()
            keep = pd.concat([ev[ev.nmax > 0][T.KEY], g4k.merge(ev[T.KEY], on=T.KEY)]).drop_duplicates()
            mk = mu.merge(keep, on=T.KEY)
            gm = allmu.merge(keep, on=T.KEY)
            mk["gcm2"], _ = _column(gm, world, y_pl) if len(gm) else (np.zeros(0), None)
            mk["file_index"] = i
            mrows.append(mk)
            if history_steps:
                srows.append(seg.merge(keep, on=T.KEY).assign(file_index=i))
            q = lambda x: np.quantile(x, [0, .01, .5, .99, 1]).tolist() if len(x) else []
            last = seg.groupby(IDM, sort=False).tail(1)
            rec = {"file": f, "sha256": _sha(f), "index": i, "gen_events": int(len(gen[T.KEY].drop_duplicates())),
                   "muons": int(len(mu)), "steps": int(len(seg)), "stats": getattr(st.make_steps, "interior_stats", {}),
                   "writer_debit_MeV_q": q(seg.writer_debit_MeV), "escaping_gas_MeV_q": q(seg.escaping_gas_MeV),
                   "deposited_gas_MeV_sum": float(seg.deposited_gas_MeV.sum()),
                   "gas_mean_loss_MeV_sum": float((seg.escaping_gas_MeV + seg.deposited_gas_MeV).sum()),
                   "lost_thin_last_MeV_q": q(last.lost_thin_MeV) if "lost_thin_MeV" in last else [],
                   "l_thin_last_mm_q": q(last.l_thin_mm) if "l_thin_mm" in last else [],
                   "t_int_minus_gen_ns_q": q((seg.t_integrated_ns - seg.t_generated_beta_ns).dropna()),
                   "events_kept": int(len(keep)), "fast_pass_any": int((ev.nmax >= MIN).sum())}
            jf.write(json.dumps(rec) + "\n")
            jf.flush()
            print(f"# [{i}] {f}: {rec['muons']:,} muons, {rec['fast_pass_any']:,} passes, {rec['events_kept']:,} kept",
                  flush=True)
    pd.concat(mrows, ignore_index=True).to_parquet(m_out, index=False)
    if history_steps:
        pd.concat(srows, ignore_index=True).to_parquet(s_out, index=False)
    print(f"wrote {m_out}")


# ----------------------------------------------------------------------------------------------------------- compare
def _events(mu, prefix):
    """Event-level pass flags from per-muon rows: any primary muon, and the first muon (key 1; 0 steps if not stored)."""
    mu = mu.sort_values(IDM, kind="stable")
    g = mu.groupby(T.KEY, sort=False)
    ev = pd.DataFrame({f"{prefix}_nmax": g.ngood.max(), f"{prefix}_n1": mu.ngood.where(mu.sp == 1, 0).groupby(
                           [mu[k] for k in T.KEY], sort=False).max(),
                       f"{prefix}_nmu_pass": (mu.ngood >= MIN).groupby([mu[k] for k in T.KEY], sort=False).sum()
                       }).reset_index()
    ev[f"{prefix}_any"] = ev[f"{prefix}_nmax"] >= MIN
    ev[f"{prefix}_mu1"] = ev[f"{prefix}_n1"] >= MIN
    return ev


def _paired(a, b):
    """a, b boolean arrays (arm, reference): counts and paired relative difference with error."""
    na, nb = int(a.sum()), int(b.sum())
    oa, ob = int((a & ~b).sum()), int((b & ~a).sum())
    return {"ref": nb, "arm": na, "arm_only": oa, "ref_only": ob, "diff": na - nb,
            "rel": (na / nb - 1) if nb else None, "err": (np.sqrt(oa + ob) / nb) if nb else None}


def _binned(df, a, b, col, edges):
    k = np.clip(np.digitize(df[col].to_numpy(), edges) - 1, 0, len(edges) - 2)
    return [{"bin": [float(edges[i]), float(edges[i + 1])], **_paired(df[a].to_numpy()[k == i], df[b].to_numpy()[k == i])}
            for i in range(len(edges) - 1)]


def _fmt(r):
    rel = "" if r["rel"] is None else f"{100 * r['rel']:+7.2f} % +- {100 * r['err']:.2f}"
    return f"ref {r['ref']:6d} arm {r['arm']:6d}  {rel}  (arm-only {r['arm_only']}, ref-only {r['ref_only']})"


FAST_COLS = ["near", "alive_plane", "p_plane_MeV_c", "nsteps", "ngood", "p_first", "p_last", "p_min", "n_lowp",
             "stopped_interior", "ke_final_MeV", "path_interior_mm", "lost_support_MeV", "lost_thin_MeV"]
G4_COLS = ["nsteps", "ngood", "p_first", "p_last", "p_min", "n_lowp", "ey", "ee", "g4_stopped"]


def _fast_fate(f):
    """Fast fate of the matched muon: not transported (outside the footprint), stopped above the plane, no straw,
    1-11 good steps split by why (ranged out in the interior, steps below 50 MeV/c, left the straw volume), pass."""
    few = f.ngood_fast < MIN
    return np.select([f.near_fast.isna(), ~f.near_fast.fillna(False).astype(bool),
                      ~f.alive_plane_fast.fillna(False).astype(bool), f.nsteps_fast == 0,
                      few & (f.stopped_interior_fast == 1), few & (f.n_lowp_fast > 0), few],
                     ["absent", "not_near", "stopped_above", "missed", "few_stopped_interior", "few_below_50MeV",
                      "few_left_volume"], "pass")


def _g4_fate(g):
    few = g.ngood_g4 < MIN
    return np.select([g.ngood_g4.isna(), g.nsteps_g4 == 0, few & g.g4_stopped_g4.fillna(False).astype(bool), few],
                     ["g4_not_kept", "g4_missed", "g4_few_stopped", "g4_few_through"], "pass")


def _replay(ev, fm, gm, a, b):
    """Who is lost: per event passing in G4 only (fast only), the passing muon (lowest generation index) with the
    other arm's state of the same generated muon."""
    out = {}
    ctx = ev[T.KEY + ["margin", "p_gen_MeV_c"]]
    for name, sel in (("g4_only", ev[b] & ~ev[a]), ("fast_only", ev[a] & ~ev[b])):
        x = ev[sel][T.KEY]
        src = gm if name == "g4_only" else fm
        pm = src[src.ngood >= MIN].merge(x, on=T.KEY).sort_values(IDM).drop_duplicates(T.KEY)[IDM]
        f = fm[IDM + FAST_COLS].rename(columns={c: c + "_fast" for c in FAST_COLS})
        g = gm[IDM + G4_COLS].rename(columns={c: c + "_g4" for c in G4_COLS})
        j = pm.merge(f, on=IDM, how="left").merge(g, on=IDM, how="left")
        j = j.merge(ctx, on=T.KEY, how="left")
        j["fate"] = _fast_fate(j) if name == "g4_only" else _g4_fate(j)
        tab = {}
        k = np.clip(np.digitize(j.margin.to_numpy(), MARGIN_EDGES) - 1, 0, len(MARGIN_EDGES) - 2)
        for i in range(len(MARGIN_EDGES) - 1):
            y = j[k == i]
            tab[f"{MARGIN_EDGES[i]:g}-{MARGIN_EDGES[i + 1]:g}"] = {"n": int(len(y)),
                                                                     **{q: int(v) for q, v in y.fate.value_counts().items()}}
        out[name] = {"total": int(len(j)), "fates": {q: int(v) for q, v in j.fate.value_counts().items()},
                     "by_margin": tab}
        out[name + "_rows"] = j
    return out


def g4_fast_keys(gm, fm):
    """G4 per-muon rows with `sp` replaced by the fast muon key: G4 key - 1 is the generator index (`genidx` of the
    fast rows). Returns (rows, number of G4 muons without a generated muon at that index, dropped)."""
    j = gm.drop(columns=["genidx"], errors="ignore").assign(genidx=gm.sp - 1).rename(columns={"sp": "sp_g4"})
    j = j.merge(fm[T.KEY + ["genidx", "sp"]].drop_duplicates(), on=T.KEY + ["genidx"], how="left")
    bad = int(j.sp.isna().sum())
    j = j[j.sp.notna()].astype({"sp": np.int64})
    if j.duplicated(IDM).any():
        raise RuntimeError("two G4 muons map to the same fast key")
    return j, bad


def load_arm(directory, tag=None):
    """Read complete chunks, rejecting missing manifests, duplicate identities and inconsistent settings."""
    directory = Path(directory)
    paths = sorted(directory.glob("muons.*.parquet")) if tag is None else [directory / f"muons.{tag}.parquet"]
    if not paths:
        raise ValueError(f"{directory}: no muons.*.parquet")
    if tag is None:
        expected = {p.name.replace("muons.", "files.").replace(".parquet", ".jsonl") for p in paths}
        actual = {p.name for p in directory.glob("files.*.jsonl")}
        if expected != actual:
            raise ValueError(f"{directory}: incomplete chunk outputs/manifests")
    rows, records, settings = [], [], []
    for path in paths:
        manifest = path.with_name(path.name.replace("muons.", "files.").replace(".parquet", ".jsonl"))
        with open(manifest) as stream:
            entries = [json.loads(line) for line in stream]
        if not entries or "settings" not in entries[0] or len(entries) == 1:
            raise ValueError(f"{manifest}: missing settings or file records")
        config = entries[0]["settings"]
        files = entries[1:]
        indices = [r["index"] for r in files]
        if len(set(indices)) != len(indices):
            raise ValueError(f"{manifest}: duplicate file indices")
        m = pd.read_parquet(path)
        if not set(m.file_index.unique()).issubset(indices):
            raise ValueError(f"{path}: muon rows outside the manifest")
        counts = _events(m, "fast").fast_any.sum()
        if int(counts) != sum(r["fast_pass_any"] for r in files):
            raise ValueError(f"{path}: pass count differs from completed-file records")
        if len(m[T.KEY].drop_duplicates()) != sum(r["events_kept"] for r in files):
            raise ValueError(f"{path}: retained event count differs from completed-file records")
        segment_path = path.with_name(path.name.replace("muons.", "segments."))
        if config.get("history_steps") and not segment_path.exists():
            raise ValueError(f"{path}: requested segment history is missing")
        rows.append(m)
        records.extend(files)
        settings.append(config)
    if any(s != settings[0] for s in settings[1:]):
        raise ValueError(f"{directory}: mixed settings/source identities across chunks")
    if len({r["index"] for r in records}) != len(records):
        raise ValueError(f"{directory}: duplicate file indices across chunks")
    m = pd.concat(rows, ignore_index=True)
    if m.duplicated(IDM).any():
        raise ValueError(f"{directory}: duplicate muon identities")
    return m, records, settings


def _input_signature(records):
    return sorted((r["index"], r["sha256"], r["gen_events"]) for r in records)


def check_arm(directory, inputs, index0, arm, seed, gdml, g4, tag=None):
    """Validate a resumable chunk against its exact input content and requested physics configuration."""
    _, records, settings = load_arm(directory, tag)
    expected = [(index0 + i, _sha(f)) for i, f in enumerate(T.expand(inputs))]
    if sorted((r["index"], r["sha256"]) for r in records) != expected:
        raise ValueError(f"{directory}: incomplete or mismatched input file coverage")
    config = settings[0]
    requested = {"arm": arm, "seed": seed, "gdml_sha256": _sha(gdml), "g4": str(g4),
                 **ledger.preset(arm).as_dict(), **ledger.core_world_convention()}
    if any(config.get(k) != v for k, v in requested.items()):
        raise ValueError(f"{directory}: requested settings differ from saved settings")
    source = _src_identity()
    physics = [k for k in source if k not in ("tracker_git_head", "analyses/ratearms.py")]
    if any(config["source"].get(k) != source[k] for k in physics):
        raise ValueError(f"{directory}: transport source differs from saved source")
    print(f"validated {directory} tag={tag}: {len(records)} files, {sum(r['gen_events'] for r in records)} events")


def run_compare(g4, gdml, arm_dirs, base, out, csv_dir=None):
    out = _new(out)
    gm = pd.read_parquet(g4)
    arms = {}
    for spec in arm_dirs:
        name, d = spec.split("=", 1)
        if name in arms:
            raise ValueError(f"duplicate arm name {name}")
        arms[name] = load_arm(d)
    first = next(iter(arms.values()))
    signature = _input_signature(first[1])
    with open(str(g4) + ".json") as stream:
        g4_manifest = json.load(stream)
    if len(g4_manifest["files"]) != len(signature):
        raise ValueError("G4 and fast input file counts differ")
    if sum(r["primary_muons"] for r in g4_manifest["files"]) != len(gm) or gm.duplicated(IDM).any():
        raise ValueError("G4 table count or identities disagree with its manifest")
    for name, (m, records, configs) in arms.items():
        if _input_signature(records) != signature:
            raise ValueError(f"{name}: input hashes, seed indices or generated event counts differ between arms")
        for key in ("seed", "gdml_sha256"):
            if configs[0].get(key) != first[2][0].get(key):
                raise ValueError(f"{name}: {key} differs between arms")
    gm, unmapped = g4_fast_keys(gm, first[0])
    if unmapped:
        raise ValueError(f"{unmapped} G4 muons have no generated muon identity; refusing biased comparison")
    gev = _events(gm, "g4")
    rep = {"g4": str(g4), "g4_events_kept": int(len(gev)), "g4_muons_unmapped": unmapped,
           "input_signature": signature, "comparison_source": _src_identity(),
           "scope": "primary-muon tracker term only; not the any-particle PrimaryFilter OR",
           "binning": "truth diagnostic: generated momentum and column of the first primary muon",
           "arms": {}}
    base_ev, arm_ev = None, {}
    for name, (m, files, sets) in arms.items():
        idx = sorted(r["index"] for r in files)
        if len(set(idx)) != len(idx):
            raise SystemExit(f"{name}: duplicate file indices")
        fev = _events(m, "fast")
        mu1 = m.sort_values(IDM).drop_duplicates(T.KEY)[T.KEY + ["p_gen_MeV_c", "gcm2"]]
        ev = fev.merge(gev, on=T.KEY, how="outer").merge(mu1, on=T.KEY, how="left")
        for c in ("fast_any", "fast_mu1", "g4_any", "g4_mu1"):
            ev[c] = ev[c].fillna(False).astype(bool)
        if ev.p_gen_MeV_c.isna().any():
            raise SystemExit(f"{name}: {int(ev.p_gen_MeV_c.isna().sum())} G4-kept events without a fast muon row "
                             "(the arm does not cover the G4 files)")
        ev["margin"] = ev.p_gen_MeV_c / (2.1 * np.maximum(ev.gcm2, 1e-3))
        r = {"settings": sets[0] if sets else {}, "files": len(files),
             "gen_events": int(sum(x["gen_events"] for x in files)),
             "any_muon": {"all": _paired(ev.fast_any.to_numpy(), ev.g4_any.to_numpy()),
                          "by_margin": _binned(ev, "fast_any", "g4_any", "margin", MARGIN_EDGES),
                          "by_p_gen": _binned(ev, "fast_any", "g4_any", "p_gen_MeV_c", P_EDGES)},
             "first_muon": {"all": _paired(ev.fast_mu1.to_numpy(), ev.g4_mu1.to_numpy()),
                            "by_margin": _binned(ev, "fast_mu1", "g4_mu1", "margin", MARGIN_EDGES)},
             "fatecheck_convention": {"all": _paired(ev.fast_mu1.to_numpy(), ev.g4_any.to_numpy()),
                                      "by_margin": _binned(ev, "fast_mu1", "g4_any", "margin", MARGIN_EDGES)},
             "g4_multi_muon": {"g4_pass_not_first_muon": int((ev.g4_any & ~ev.g4_mu1).sum()),
                               "fast_pass_not_first_muon": int((ev.fast_any & ~ev.fast_mu1).sum())},
             "ledger": _ledger_summary(files)}
        rp = _replay(ev, m, gm, "fast_any", "g4_any")
        r["replay"] = {k: v for k, v in rp.items() if not k.endswith("_rows")}
        if csv_dir:
            Path(csv_dir).mkdir(parents=True, exist_ok=True)
            for k in ("g4_only", "fast_only"):
                rp[k + "_rows"].to_csv(Path(csv_dir) / f"{name}.{k}.csv.gz", index=False)
        print(f"== {name}: {r['settings'].get('arm')} {r['settings'].get('world_convention')} files {len(files)}")
        print(f"   any muon   {_fmt(r['any_muon']['all'])}")
        for b in r["any_muon"]["by_margin"]:
            print(f"     margin {b['bin'][0]:g}-{b['bin'][1]:g}: {_fmt(b)}")
        print(f"   first muon {_fmt(r['first_muon']['all'])}")
        print(f"   fatecheck  {_fmt(r['fatecheck_convention']['all'])}")
        print(f"   replay G4-only {r['replay']['g4_only']['fates']}  fast-only {r['replay']['fast_only']['fates']}")
        rep["arms"][name] = r
        arm_ev[name] = ev[T.KEY + ["margin", "p_gen_MeV_c"]]
        if name == base:
            base_ev = ev
    if base is not None:
        if base_ev is None:
            raise SystemExit(f"--base {base} is not among the arms")
        rep["vs_base"] = {}
        for name, (m, files, sets) in arms.items():
            if name == base:
                continue
            fev = _events(m, "fast")
            x = base_ev[T.KEY + ["fast_any"]].rename(columns={"fast_any": "base_any"}).merge(
                fev[T.KEY + ["fast_any"]], on=T.KEY, how="outer")
            for c in ("base_any", "fast_any"):
                x[c] = x[c].fillna(False).astype(bool)
            # margin and p_gen belong to the generated muon: from whichever arm holds the event
            ctx = pd.concat([base_ev[T.KEY + ["margin", "p_gen_MeV_c"]], arm_ev[name]]).drop_duplicates(T.KEY)
            x = x.merge(ctx, on=T.KEY, how="left")
            x = x[x.margin.notna() | x.fast_any]
            rr = {"all": _paired(x.fast_any.to_numpy(), x.base_any.to_numpy()),
                  "by_margin": _binned(x[x.margin.notna()], "fast_any", "base_any", "margin", MARGIN_EDGES),
                  "by_p_gen": _binned(x[x.margin.notna()], "fast_any", "base_any", "p_gen_MeV_c", P_EDGES)}
            rep["vs_base"][name] = rr
            print(f"-- {name} vs {base}: {_fmt(rr['all'])}")
    json.dump(rep, open(out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {out}")


def _ledger_summary(files):
    if not files:
        return {}
    w = [r["writer_debit_MeV_q"] for r in files if r["writer_debit_MeV_q"]]
    th = [r["lost_thin_last_MeV_q"] for r in files if r.get("lost_thin_last_MeV_q")]
    lt = [r["l_thin_last_mm_q"] for r in files if r.get("l_thin_last_mm_q")]
    dt = [r["t_int_minus_gen_ns_q"] for r in files if r.get("t_int_minus_gen_ns_q")]
    med = lambda v, k: float(np.median([x[k] for x in v])) if v else None
    return {"steps": int(sum(r["steps"] for r in files)),
            "writer_debit_MeV_median_of_file_medians": med(w, 2), "writer_debit_MeV_max": max((x[4] for x in w), default=None),
            "lost_thin_per_muon_MeV_median": med(th, 2), "lost_thin_per_muon_MeV_q99_median": med(th, 3),
            "thin_path_per_muon_mm_median": med(lt, 2),
            "gas_deposited_over_mean_loss": float(sum(r["deposited_gas_MeV_sum"] for r in files)
                                                  / max(sum(r["gas_mean_loss_MeV_sum"] for r in files), 1e-30)),
            "t_integrated_minus_generated_ns_median": med(dt, 2), "t_integrated_minus_generated_ns_q99": med(dt, 3)}


def main(argv=None):
    p = argparse.ArgumentParser(prog="ratearms", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("g4")
    a.add_argument("inputs", nargs="+")
    a.add_argument("--out", required=True)
    b = sub.add_parser("fast")
    b.add_argument("inputs", nargs="+")
    b.add_argument("--gdml", required=True)
    b.add_argument("--arm", required=True, choices=sorted(ledger.PRESETS))
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--index0", type=int, required=True, help="global index of the first input (generator seed (seed, i))")
    b.add_argument("--g4", required=True)
    b.add_argument("--outdir", required=True)
    b.add_argument("--history-steps", action="store_true")
    v = sub.add_parser("check")
    v.add_argument("directory")
    v.add_argument("--inputs", nargs="+", required=True)
    v.add_argument("--index0", type=int, required=True)
    v.add_argument("--arm", choices=sorted(ledger.PRESETS), required=True)
    v.add_argument("--seed", type=int, default=1)
    v.add_argument("--gdml", required=True)
    v.add_argument("--g4", required=True)
    v.add_argument("--tag", default=None)
    c = sub.add_parser("compare")
    c.add_argument("--g4", required=True)
    c.add_argument("--gdml", default=None)
    c.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    c.add_argument("--base", default=None)
    c.add_argument("--out", required=True)
    c.add_argument("--csv-dir", default=None, help="write the replay rows (G4-only / fast-only events) here")
    x = p.parse_args(argv)
    if x.cmd == "g4":
        run_g4(x.inputs, x.out)
    elif x.cmd == "fast":
        run_fast(x.inputs, x.outdir, x.gdml, x.arm, x.seed, x.index0, x.g4, x.history_steps)
    elif x.cmd == "check":
        check_arm(x.directory, x.inputs, x.index0, x.arm, x.seed, x.gdml, x.g4, x.tag)
    else:
        run_compare(x.g4, x.gdml, x.arm, x.base, x.out, x.csv_dir)


if __name__ == "__main__":
    main()
