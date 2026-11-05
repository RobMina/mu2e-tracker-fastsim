"""Saved cosmic exact-primary diagnostic; no reconstruction or transport is rerun.

The persisted valid rank-zero contributor is joined ONLY by (run, subrun, event,
SimParticle id).  Identity-corrected first-straw proxy, not same-location resolution,
no production/physics closure. All reference-clock step events remain in the ledger.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from trkfastsim.validation import cecontract, cereco, fieldquantiles, fieldreco

KEY = fieldreco.KEY
TRACK = KEY + ["sp"]
FILES = {"g4_all": "g4prim_field.root", "g4_clock": "g4fp_field.root", "fast": "fasttrk_field.root"}
ARMS = ("g4", "fast")
DRAWS = 500
SEED = 20261103
SCOPE = "identity-corrected first-straw proxy, not same-location resolution, no production/physics closure"


def _ids(values, label, missing=False):
    """Never accept float IDs, including apparently integral floats above 2**53."""
    s = pd.Series(values)
    if not pd.api.types.is_integer_dtype(s.dtype) or (s.dropna() < 0).any():
        raise ValueError(f"noninteger/negative particle IDs: {label}")
    if not missing and s.isna().any():
        raise ValueError(f"missing particle IDs: {label}")
    return s.astype("UInt64")


def _tracks(d, label):
    d = cecontract.integer_keys(d, label)
    d["sp"] = _ids(d.sp, label).array
    return d


def read_arm(path):
    """Vectorized first-good-track read in precisely fieldreco's track/surface order.

    trk is a record of jagged leaves, whereas trkmcsim is event->track->list
    of records. No trksegsmc/simid, proximity or per-event Python conversion is used.
    Files are closed on return; uproot's unqualified tree name selects highest cycle.
    """
    with uproot.open(path) as root:
        t = root[fieldreco.TREE]
        cols = {k: ak.to_numpy(t[f"evtinfo/{k}"].array(library="ak")) for k in KEY}
        trk = {k: t[f"trk/trk.{k}"].array(library="ak") for k in fieldreco.TRK}
        good = trk["status"] > 0
        ntrk = ak.to_numpy(ak.sum(good, axis=1))
        has = ntrk > 0
        first = ak.argmax(good, axis=1, keepdims=True)
        d = pd.DataFrame(cols)
        d["ntrk"] = ntrk
        d["ntracks_total"] = ak.to_numpy(ak.num(good, axis=1))
        d["ntracks_not_good"] = d.ntracks_total - d.ntrk
        index = ak.firsts(ak.local_index(good, axis=1)[first])
        d["selected_track_index"] = np.where(has, ak.to_numpy(ak.fill_none(index, -1)), -1)
        for k in fieldreco.TRK:
            v = ak.to_numpy(ak.fill_none(ak.firsts(trk[k][first]), 0)).astype(float)
            d[k] = np.where(has, v, np.nan)
        d["chi2ndf"] = d.chisq / np.maximum(d.ndof, 1)
        seg = ak.fill_none(ak.firsts(t["trksegs"].array(library="ak")[first], axis=1), [], axis=0)
        seg = seg[(seg.sid >= 0) & (seg.sid <= 4) & seg.inbounds]
        s0 = ak.firsts(seg[ak.argmin(seg.time, axis=1, keepdims=True)])
        d["sid"] = np.where(has, ak.to_numpy(ak.fill_none(s0.sid, -1)), -1)
        pr = np.full(len(d), np.nan)
        if ak.any(~ak.is_none(s0)):
            pv = fieldreco._vec(ak.fill_none(s0.mom, {"fCoordinates": {"fX": np.nan, "fY": np.nan, "fZ": np.nan}}))
            pr = np.linalg.norm(pv, axis=1)
        d["p_reco"] = np.where(has, pr, np.nan)
        # Saved pilot has no surface truth; this legacy column is not used.
        d["p_true"] = np.nan
        count = np.zeros(len(d), dtype=np.int64)
        ids = pd.Series(pd.NA, index=d.index, dtype="UInt64")
        raw_ids = pd.Series(pd.NA, index=d.index, dtype="UInt64")
        invalid = np.zeros(len(d), dtype=bool)
        unavailable = "trkmcsim" not in t
        if not unavailable:
            allmc = t["trkmcsim"].array(library="ak")
            if not ak.all(ak.num(allmc, axis=1) == ak.num(good, axis=1)):
                raise ValueError("trkmcsim/track multiplicity mismatch")
            mc = ak.fill_none(ak.firsts(allmc[first], axis=1), [], axis=0)
            if not {"valid", "rank", "id"}.issubset(ak.fields(mc)):
                raise ValueError("trkmcsim missing valid/rank/id fields")
            candidates = mc[(mc.valid != 0) & (mc.rank == 0)]
            count = np.where(has, ak.to_numpy(ak.num(candidates, axis=1)), 0)
            raw_option = ak.to_numpy(ak.firsts(candidates.id), allow_missing=True)
            raw = np.ma.getdata(raw_option)
            missing_id = np.ma.getmaskarray(raw_option)
            native = pd.api.types.is_integer_dtype(raw.dtype) and raw.dtype != np.dtype(bool)
            if native:
                positive = (raw > 0) & ~missing_id
                take = (count == 1) & positive
                diagnostic_take = (count == 1) & (raw >= 0)
                raw_ids.loc[diagnostic_take] = raw[diagnostic_take].astype(np.uint64)
                ids.loc[take] = raw[take].astype(np.uint64)
                invalid = (count == 1) & ~positive
            else:
                invalid = count == 1
        d["mc_id"] = ids
        d["mc_raw_rank0_id"] = raw_ids
        d["mc_rank0_count"] = count
        d["mc_missing"] = has & (count == 0)
        d["mc_ambiguous"] = has & (count > 1)
        d["mc_invalid_id"] = invalid
        d["mc_unavailable"] = unavailable
        if len(d) != t.num_entries:
            raise ValueError("ntuple event count mismatch")
        return cecontract.unique(d, str(path))


def _momentum(d):
    # Same expression (and rounding) as legacy fieldreco --events/--arm-tables.
    return np.sqrt(d.px ** 2 + d.py ** 2 + d.pz ** 2)


def _step_views(steps, particles, label):
    s = _tracks(steps, label)
    member = s[TRACK].drop_duplicates().merge(particles[TRACK], on=TRACK, how="left", indicator=True)
    if (member._merge != "both").any():
        raise ValueError(f"step IDs outside particle table: {label}")
    if not np.isfinite(s[["t", "px", "py", "pz"]].to_numpy(float)).all():
        raise ValueError(f"nonfinite step time/momentum: {label}")
    # Reproduce the old event-first proxy exactly, separately from the stable own-ID first step.
    event = s.sort_values(KEY + ["t"]).groupby(KEY, sort=False).first().reset_index()
    event["p_event_first"] = _momentum(event)
    event = event.rename(columns={"sp": "event_first_id"})
    own = s.sort_values(TRACK + ["t"], kind="stable").drop_duplicates(TRACK).copy()
    own["p_own_id"] = _momentum(own)
    return event, own[TRACK + ["p_own_id"]]


def assemble(particles, steps, frames):
    """Build a left-joined event ledger; failed links remain flagged, never guessed.

    particles is the verified common entry table (sp column), steps has all three
    saved injector arms, frames has g4/fast read_arm results. Foreign contributors
    are diagnostic rows, unlike foreign injector steps which violate the input contract.
    """
    p = _tracks(particles, "particles")
    if p.empty or p.duplicated(TRACK).any():
        raise ValueError("empty or duplicate particle table")
    if set(steps) != set(FILES) or set(frames) != set(ARMS):
        raise ValueError("require all three injector tables and g4/fast ntuples")
    views = {n: _step_views(s, p, n) for n, s in steps.items()}
    ref = views["g4_clock"][0]
    if ref.empty:
        raise ValueError("empty reference-clock event denominator")
    den = ref[KEY].copy()
    den["p_first"] = ref.p_event_first
    if "plane" in ref:
        den["first_plane"] = ref.plane
    counts = p.groupby(KEY).size().rename("primary_count").reset_index()
    d = den.merge(counts, on=KEY, validate="one_to_one")
    d["multiprimary"] = d.primary_count > 1
    for arm in ARMS:
        event, own = views["g4_clock" if arm == "g4" else "fast"]
        event = event[KEY + ["p_event_first", "event_first_id"]].rename(
            columns={"p_event_first": "p_first_own"})
        d = d.merge(event.rename(columns={c: f"{arm}_{c}" for c in event if c not in KEY}),
                    on=KEY, how="left", validate="one_to_one")
        f = cecontract.unique(frames[arm], arm)
        f["mc_id"] = _ids(f.mc_id, arm, missing=True).array
        f["ntuple_present"] = True
        f = f.merge(p[TRACK].rename(columns={"sp": "mc_id"}).assign(in_particles=True),
                    on=KEY + ["mc_id"], how="left", validate="many_to_one")
        f["matched_id"] = f.mc_id.where(f.in_particles.fillna(False)).astype("UInt64")
        f = f.merge(own.rename(columns={"sp": "matched_id"}), on=KEY + ["matched_id"],
                    how="left", validate="many_to_one")
        f = f.drop(columns="in_particles")
        d = d.merge(f.rename(columns={c: f"{arm}_{c}" for c in f if c not in KEY}),
                    on=KEY, how="left", validate="one_to_one")
        for c in ("mc_id", "matched_id", "event_first_id"):
            d[f"{arm}_{c}"] = d[f"{arm}_{c}"].astype("UInt64")
        for c in ("ntuple_present", "mc_missing", "mc_ambiguous", "mc_invalid_id", "mc_unavailable"):
            d[f"{arm}_{c}"] = d[f"{arm}_{c}"].fillna(False).astype(bool)
        present = d[f"{arm}_ntuple_present"]
        has = d[f"{arm}_ntrk"].fillna(0) > 0
        known = d[f"{arm}_mc_id"].notna()
        matched = d[f"{arm}_matched_id"].notna()
        d[f"{arm}_ntuple_missing"] = ~present
        d[f"{arm}_no_good_track"] = present & ~has
        d[f"{arm}_foreign_id"] = known & ~matched
        d[f"{arm}_no_own_steps"] = matched & d[f"{arm}_p_own_id"].isna()
        d[f"{arm}_invalid_own_proxy"] = matched & ~(np.isfinite(d[f"{arm}_p_own_id"]) & (d[f"{arm}_p_own_id"] > 0))
        d[f"{arm}_event_first_id_mismatch"] = (known & d[f"{arm}_event_first_id"].notna()
            & (d[f"{arm}_mc_id"] != d[f"{arm}_event_first_id"])).fillna(False).astype(bool)
    both = (d.g4_ntrk.fillna(0) > 0) & (d.fast_ntrk.fillna(0) > 0)
    same_sid = both & (d.g4_sid == d.fast_sid) & d.g4_sid.isin(fieldreco.SIDS)
    known = d.g4_mc_id.notna() & d.fast_mc_id.notna()
    d["crossarm_selected_id_mismatch"] = (both & known & (d.g4_mc_id != d.fast_mc_id)).fillna(False).astype(bool)
    same_id = (known & (d.g4_mc_id == d.fast_mc_id)).fillna(False).astype(bool)
    d["valid_same_id_paired_same_sid"] = (same_sid & same_id & d.g4_matched_id.notna() & d.fast_matched_id.notna())
    old = [_relative(d, a, "p_first_own") for a in ARMS]
    own = [_relative(d, a, "p_own_id") for a in ARMS]
    d["original_proxy_pair"] = same_sid & np.isfinite(old[0]) & np.isfinite(old[1])
    d["same_id_proxy_pair"] = (d.valid_same_id_paired_same_sid & d.original_proxy_pair
                                & np.isfinite(own[0]) & np.isfinite(own[1]))
    # Inclusive reason flags plus one disjoint reason partition explain every excluded row.
    reasons = [(f"{a}_{c}", d[f"{a}_{c}"]) for a in ARMS for c in (
        "ntuple_missing", "no_good_track", "mc_unavailable", "mc_missing", "mc_ambiguous", "mc_invalid_id", "foreign_id")]
    reasons += [("crossarm_selected_id_mismatch", d.crossarm_selected_id_mismatch),
                ("different_or_missing_sid", ~same_sid)]
    reasons += [(f"{a}_{c}", d[f"{a}_{c}"]) for a in ARMS for c in ("no_own_steps", "invalid_own_proxy")]
    reasons += [("nonfinite_original_or_own_proxy", ~d.same_id_proxy_pair)]
    d["excluded_reason"] = np.select([v.to_numpy(bool) for _, v in reasons],
                                      [k for k, _ in reasons], default="included")
    if not (d.excluded_reason.eq("included") == d.same_id_proxy_pair).all():
        raise ValueError("exclusion partition failed")
    return cecontract.unique(d, "identity ledger")


def _relative(d, arm, truth):
    p = d[f"{arm}_{truth}"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (d[f"{arm}_p_reco"] - p) / p
    return r.where((p > 0) & np.isfinite(p))


def _paired(a, b):
    r = cereco.paired_core(a, b, seed=SEED, draws=DRAWS)
    r.update(metrics=["relative_median", "relative_IQR_sigma"], bootstrap_draws=DRAWS,
             bootstrap_seed=SEED, candidate_arm="fast", reference_arm="g4")
    for k in ("candidate", "reference", "difference", "paired_error"):
        r.setdefault(k, [None, None])
    return r


def _bin_report(d):
    old = {a: _relative(d, a, "p_first_own") for a in ARMS}
    own = {a: _relative(d, a, "p_own_id") for a in ARMS}
    views = {}
    for name, mask, values in (
        ("original_proxy_all_pairs", d.original_proxy_pair, old),
        ("original_proxy_same_id_subset", d.same_id_proxy_pair, old),
        ("own_particle_proxy_same_id_subset", d.same_id_proxy_pair, own)):
        views[name] = _paired(values["fast"][mask], values["g4"][mask])
    orig, cond, corr = (views[k] for k in views)

    def shift(a, b):
        return a - b if a is not None and b is not None else None

    widths = {n: v["difference"][1] for n, v in views.items()}
    singleton = d.original_proxy_pair & (d.primary_count == 1)
    return {"denominator_events": len(d), **views,
            "original_proxy_single_entry_particle_events": _paired(old["fast"][singleton], old["g4"][singleton]),
            "single_entry_scope": "multiplicity-restricted legacy proxy only; no compressed-ID remapping inferred",
            "width_attribution": {
                "quantity": "fast minus g4 relative IQR sigma; fractions, not percent",
                "width_differences": widths,
                "conditioning_change": shift(cond["difference"][1], orig["difference"][1]),
                "association_change_same_subset": shift(corr["difference"][1], cond["difference"][1]),
                "total_change": shift(corr["difference"][1], orig["difference"][1]),
                "arm_width_changes_same_subset": {a: shift(corr[k][1], cond[k][1])
                    for a, k in (("fast", "candidate"), ("g4", "reference"))},
                "interpretation": "descriptive decomposition, not a causal or physics-closure verdict"}}


def report(d):
    """Fixed legacy reference-event-first-G4 bins; 500 predeclared paired bootstraps."""
    d = cecontract.unique(d, "identity ledger")
    bins = [{"p_lo": float(lo), "p_hi": float(hi),
             **_bin_report(d[(d.p_first >= lo) & (d.p_first < hi)])}
            for lo, hi in zip(fieldreco.P_EDGES[:-1], fieldreco.P_EDGES[1:])]
    flags = [c for c in d if pd.api.types.is_bool_dtype(d[c])]
    raw_counts = {a: {str(k): int(v) for k, v in d[f"{a}_mc_raw_rank0_id"].value_counts(dropna=False).items()}
                  for a in ARMS if f"{a}_mc_raw_rank0_id" in d}
    return finite_json({"scope": SCOPE, "events": len(d), "raw_rank0_id_counts": raw_counts,
        "denominator": "all original g4_clock sgstep EVENT keys, including no-reco events; not generated cosmics",
        "association": "exact persisted positive native-integer valid rank0 contributor -> (event, sp); no proximity guesses",
        "reference_clock": "saved paired fast exit + 1 ns, not independent G4 first passage",
        "bins": bins, "binning": "fixed fieldreco.P_EDGES using old event-first G4 momentum, never selected-ID momentum",
        "whole": _bin_report(d), "two_to_five_GeV": bins[3],
        "flag_counts": {c: int(d[c].sum()) for c in flags},
        "excluded_reason_counts": d.excluded_reason.value_counts().to_dict(),
        "uncertainty": "shared-index 500-draw event bootstrap, seed 20261103; one saved sample, not parent/family-wise errors",
        "change_uncertainty": "view errors are paired fast-g4 errors; shifts between views are descriptive, not independent-error significance",
        "legacy_fieldreco_compare": fieldreco.compare(d, "g4", "fast")})


def finite_json(value):
    """Strict JSON nulls for unavailable/nonfinite numerics, recursively."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(v) for v in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def check_baseline(d, baseline):
    """Fail closed if any saved legacy column or recomputed old metric changed."""
    b = cecontract.unique(baseline, "saved fieldreco baseline")
    x = cecontract.unique(d, "rebuilt ledger")
    required = {"p_first", *(f"{a}_{c}" for a in ARMS for c in
                ("ntrk", "sid", "nactive", "p_reco", "p_first_own"))}
    if not required.issubset(b) or not set(b).issubset(x):
        raise ValueError("incomplete or unsupported baseline columns")
    serialized = pd.read_csv(io.StringIO(x[list(b)].to_csv(index=False)),
                             dtype={k: "uint64" for k in KEY}, float_precision="round_trip")
    try:
        pd.testing.assert_frame_equal(serialized, b, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError("saved fieldreco events changed") from exc
    old = finite_json(fieldreco.compare(b, "g4", "fast"))
    quantiles = finite_json(fieldquantiles.report(b))
    if old != finite_json(fieldreco.compare(serialized, "g4", "fast")) or quantiles != finite_json(fieldquantiles.report(serialized)):
        raise ValueError("saved old metrics changed")
    return {"unchanged": True, "columns": list(b), "legacy_fieldreco_compare": old,
            "legacy_fieldquantiles": quantiles,
            "comparison_representation": "rebuilt native ROOT values serialized by the original to_csv pipeline then parsed as saved CSV; exact decimal-payload equality, not float32-to-float64 binary equality"}


def _flat_table(tree):
    """Vectorized scalar table read for either TTree or RNTuple; keep native integers."""
    a = tree.arrays(library="ak")
    return pd.DataFrame({k: ak.to_numpy(a[k]) for k in ak.fields(a)})


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(g4_ntuple, fast_ntuple, field_dir, outdir, baseline_events=None):
    """Read only closed saved inputs, verify their before/after hashes, write a fresh output directory."""
    out = Path(outdir)
    if out.exists():
        raise FileExistsError(f"outdir must be fresh: {out}")
    directory = Path(field_dir)
    paths = {a: directory / filename for a, filename in FILES.items()}
    paths.update(g4_ntuple=Path(g4_ntuple), fast_ntuple=Path(fast_ntuple),
                 fieldcompare=directory / "fieldcompare.json")
    if baseline_events:
        paths["baseline_events"] = Path(baseline_events)
    paths = {k: p.resolve(strict=True) for k, p in paths.items()}
    before = {k: _sha256(p) for k, p in paths.items()}
    # Bind the saved fieldcompare input without interpreting its exiting-track count as our denominator.
    saved_compare = json.loads(paths["fieldcompare"].read_text())
    if not isinstance(saved_compare, dict):
        raise ValueError("fieldcompare.json must contain a report object")
    particles, steps, cycles = None, {}, {}
    for arm in FILES:
        with uproot.open(paths[arm]) as root:
            pt = root["dump/particles"]
            st = root["dump/sgsteps"]
            cycles[arm] = {"particles": pt.object_path, "sgsteps": st.object_path}
            p = _tracks(_flat_table(pt).rename(columns={"key": "sp"}), arm)
            p = p.sort_values(TRACK).reset_index(drop=True)
            if particles is not None:
                pd.testing.assert_frame_equal(particles, p, check_exact=True)
            else:
                particles = p
            steps[arm] = _flat_table(st)
    frames = {}
    for arm in ARMS:
        path = paths[f"{arm}_ntuple"]
        with uproot.open(path) as root:
            tree = root[fieldreco.TREE]
            cycles[f"{arm}_ntuple"] = {"tree": tree.object_path, "entries": int(tree.num_entries)}
        frames[arm] = read_arm(path)
    d = assemble(particles, steps, frames)
    result = report(d)
    result["ntuple_events"] = {a: len(f) for a, f in frames.items()}
    result["ntuple_events_outside_denominator"] = {
        a: int((f[KEY].merge(d[KEY], on=KEY, how="left", indicator=True)._merge == "left_only").sum())
        for a, f in frames.items()}
    if baseline_events:
        b = pd.read_csv(paths["baseline_events"], dtype={k: "uint64" for k in KEY}, float_precision="round_trip")
        result["baseline_check"] = check_baseline(d, b)
    after = {k: _sha256(p) for k, p in paths.items()}
    if before != after:
        raise ValueError("closed input changed during diagnostic")
    result["inputs"] = {k: {"path": str(p), "sha256_before": before[k], "sha256_after": after[k],
                            **cycles.get(k, {})} for k, p in paths.items()}
    result = finite_json(result)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    out.mkdir(parents=True, exist_ok=False)
    d.to_csv(out / "events.csv.gz", index=False)
    (out / "fieldidentity.json").write_text(payload)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g4-ntuple", required=True)
    ap.add_argument("--fast-ntuple", required=True)
    ap.add_argument("--field-dir", required=True, help="saved three injectors plus fieldcompare.json")
    ap.add_argument("--outdir", required=True, help="must not exist")
    ap.add_argument("--baseline-events", help="optional saved fieldreco events.csv.gz; exact old columns/metrics must agree")
    args = ap.parse_args(argv)
    print(json.dumps(run(**vars(args)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
