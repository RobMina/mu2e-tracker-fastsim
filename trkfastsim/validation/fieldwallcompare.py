"""Saved-only complete-wall cosmic pilot; descriptive paired proxies, not physics closure.

Legacy/G4 transport and reconstruction are retained inputs, never executed here.
"""
import argparse
import io
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from trkfastsim.validation import cecontract, cereco, fieldidentity, fieldquantiles, fieldreco

KEY = fieldreco.KEY
TRACK = KEY + ["sp"]
ARMS = ("g4", "legacy", "candidate")
FILES = {"legacy": "fasttrk_field.root", "g4_clock": "g4fp_field.root", "g4_all": "g4prim_field.root"}
BASELINE_EVENTS = 40991
ENTRY_PARTICLES = 43156
DRAWS = 500
SEED = 20261103
PAIRS = (("candidate", "g4"), ("legacy", "g4"), ("candidate", "legacy"))
finite_json = fieldidentity.finite_json
_sha256 = fieldidentity._sha256


def _tracks(d, label):
    d = cecontract.integer_keys(d, label)
    if "sp" not in d or not pd.api.types.is_integer_dtype(d.sp.dtype) or d.sp.isna().any() or (d.sp < 0).any():
        raise ValueError(f"invalid native integer particle keys: {label}")
    d["sp"] = d.sp.astype(np.uint64)
    return d


def _particles(d):
    d = _tracks(d, "particles")
    if d.empty or d.duplicated(TRACK).any():
        raise ValueError("empty or duplicate particle keys")
    return d


def _steps(particles, steps, label):
    s = _tracks(steps, label)
    membership = s[TRACK].drop_duplicates().merge(particles[TRACK], on=TRACK, how="left", indicator=True)
    if (membership._merge != "both").any():
        raise ValueError(f"foreign step particle keys: {label}")
    return s


def _first(steps):
    # Preserve the original sort/groupby.first and float32 arithmetic before CSV serialization.
    d = steps.sort_values(KEY + ["t"]).groupby(KEY, sort=False).first().reset_index()
    d["p_first_own"] = np.sqrt(d.px ** 2 + d.py ** 2 + d.pz ** 2)
    return d


def _selected_straws(hits, has, selected):
    chosen = hits[selected]
    counts = ak.to_numpy(ak.num(chosen, axis=1))
    flat = {c: ak.to_numpy(ak.flatten(chosen[c])) for c in ("plane", "panel", "straw")}
    flat["entry"] = np.repeat(np.arange(len(has)), counts)
    unique = pd.DataFrame(flat).drop_duplicates(["entry", "plane", "panel", "straw"])
    nstraw = np.bincount(unique.entry.to_numpy(dtype=int), minlength=len(has))
    return np.where(has, nstraw, np.nan), np.where(has, counts, np.nan)


def _active_straws(hits, has):
    """KinKal active includes left(-1), null(0) and right(1); inactive is -2."""
    return _selected_straws(hits, has, hits.state > -2)


def read_arm(path):
    """Read first good track, earliest in-bounds SID 0..4, and its distinct active straws."""
    with uproot.open(path) as root:
        t = root[fieldreco.TREE]
        d = pd.DataFrame({k: ak.to_numpy(t[f"evtinfo/{k}"].array(library="ak")) for k in KEY})
        trk = {k: t[f"trk/trk.{k}"].array(library="ak") for k in fieldreco.TRK}
        good = trk["status"] > 0
        first = ak.argmax(good, axis=1, keepdims=True)
        d["ntrk"] = ak.to_numpy(ak.sum(good, axis=1))
        d["ntracks_total"] = ak.to_numpy(ak.num(good, axis=1))
        has = d.ntrk.to_numpy() > 0
        for k in fieldreco.TRK:
            values = ak.to_numpy(ak.fill_none(ak.firsts(trk[k][first]), 0)).astype(float)
            d[k] = np.where(has, values, np.nan)
        d["chi2ndf"] = d.chisq / np.maximum(d.ndof, 1)
        seg = ak.fill_none(ak.firsts(t["trksegs"].array(library="ak")[first], axis=1), [], axis=0)
        seg = seg[(seg.sid >= 0) & (seg.sid <= 4) & seg.inbounds]
        s0 = ak.firsts(seg[ak.argmin(seg.time, axis=1, keepdims=True)])
        d["sid"] = np.where(has, ak.to_numpy(ak.fill_none(s0.sid, -1)), -1)
        momentum = np.full(len(d), np.nan)
        if ak.any(~ak.is_none(s0)):
            pv = fieldreco._vec(ak.fill_none(s0.mom, {"fCoordinates": {"fX": np.nan, "fY": np.nan, "fZ": np.nan}}))
            momentum = np.linalg.norm(pv, axis=1)
        d["p_reco"] = np.where(has, momentum, np.nan)
        d["p_true"] = np.nan
        hits = ak.fill_none(ak.firsts(t["trkhits"].array(library="ak")[first], axis=1), [], axis=0)
        d["nstraw"], d["active_hit_rows"] = _active_straws(hits, has)
        d["state_positive_nstraw"], d["state_positive_hit_rows"] = _selected_straws(hits, has, hits.state > 0)
        if not np.array_equal(d.active_hit_rows.to_numpy()[has], d.nactive.to_numpy()[has]):
            raise ValueError("KinKal active-state hit count differs from persisted nactive")
        if len(d) != t.num_entries:
            raise ValueError("highest-cycle ntuple entry/frame count mismatch")
        return cecontract.unique(d, str(path))


def step_summary(particles, steps):
    """One row per original ENTRY particle, including zeros; rows and distinct straws are different measures."""
    p = _particles(particles)
    s = _steps(p, steps, "step summary")
    out = p[TRACK].copy()
    counts = s.groupby(TRACK, sort=False).size().rename("step_rows").reset_index()
    distinct = s.drop_duplicates(TRACK + ["plane", "panel", "straw"])
    unique = distinct.groupby(TRACK, sort=False).size().rename("distinct_straws").reset_index()
    out = out.merge(counts, on=TRACK, how="left", validate="one_to_one")
    out = out.merge(unique, on=TRACK, how="left", validate="one_to_one")
    out[["step_rows", "distinct_straws"]] = out[["step_rows", "distinct_straws"]].fillna(0).astype(np.int64)
    return out


def _frame_exposure(f, denominator, entry_events):
    membership = f.merge(denominator[KEY].assign(in_baseline=True), on=KEY, how="left", validate="one_to_one")
    membership = membership.merge(entry_events.assign(in_entries=True), on=KEY, how="left", validate="one_to_one")
    outside = membership.in_baseline.isna()
    foreign = membership.in_entries.isna()
    good = membership.ntrk.fillna(0) > 0
    return dict(highest_cycle_entries=len(f), selected_ntuple_rows=len(f), good_events=int(good.sum()),
                good_tracks=int(f.ntrk.fillna(0).sum()), baseline_present=int((~outside).sum()),
                baseline_missing=int(len(denominator) - (~outside).sum()), outside_baseline_events=int(outside.sum()),
                outside_baseline_good_events=int((outside & good).sum()), foreign_entry_events=int(foreign.sum()),
                foreign_entry_good_events=int((foreign & good).sum()),
                outside_baseline_keys=membership.loc[outside, KEY].to_dict("records"),
                foreign_entry_keys=membership.loc[foreign, KEY].to_dict("records"))


def assemble(particles, steps, frames, baseline):
    """Left-join to the full fixed saved denominator; foreign NT rows remain in exposure diagnostics."""
    p = _particles(particles)
    b = cecontract.unique(baseline, "baseline")
    if b.empty or set(steps) != {*FILES, "candidate"} or set(frames) != set(ARMS):
        raise ValueError("require nonempty baseline, all four step tables and three ntuples")
    entries = p[KEY].drop_duplicates()
    cecontract.subset(b[KEY], entries, "baseline")
    views = {}
    summaries = {}
    for arm, s in steps.items():
        s = _steps(p, s, arm)
        if not np.isfinite(s[["t", "px", "py", "pz"]].to_numpy(float)).all():
            raise ValueError(f"nonfinite step time/momentum: {arm}")
        views[arm] = _first(s)
        summaries[arm] = step_summary(p, s)
    clock = cecontract.unique(views["g4_clock"][KEY], "G4 clock")
    if not clock.equals(b[KEY]):
        raise ValueError("G4 clock event keys differ from fixed baseline")
    d = b[KEY].merge(views["g4_clock"][KEY + ["p_first_own", "plane"]].rename(
        columns={"p_first_own": "p_first", "plane": "first_plane"}), on=KEY, how="left", validate="one_to_one")
    exposure = {}
    for arm in ARMS:
        f = cecontract.unique(frames[arm], arm)
        required = {"ntrk", "sid", "nactive", "nstraw", "p_reco"}
        if not required.issubset(f):
            raise ValueError(f"missing frame columns: {arm}")
        exposure[arm] = _frame_exposure(f, b, entries)
        own = views["g4_clock" if arm == "g4" else arm][KEY + ["p_first_own"]]
        d = d.merge(own.rename(columns={"p_first_own": f"{arm}_p_first_own"}), on=KEY, how="left", validate="one_to_one")
        f = f.assign(ntuple_present=True)
        d = d.merge(f.rename(columns={c: f"{arm}_{c}" for c in f if c not in KEY}), on=KEY, how="left", validate="one_to_one")
        d[f"{arm}_ntuple_present"] = d[f"{arm}_ntuple_present"].fillna(False).astype(bool)
    particle_steps = p[TRACK].copy()
    for arm, summary in summaries.items():
        particle_steps = particle_steps.merge(summary.rename(columns={c: f"{arm}_{c}" for c in
            ("step_rows", "distinct_straws")}), on=TRACK, how="left", validate="one_to_one")
    d.attrs["particle_steps"] = particle_steps
    d.attrs["ntuple_exposure"] = exposure
    d.attrs["entry_exposure"] = _entry_report(p, summaries, frames, b)
    return d


def _mean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(mean=float(x.mean()) if len(x) else None,
                sem=float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None, events=len(x))


def _paired_mean(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return dict(candidate=_mean(a), reference=_mean(b), difference=_mean(a - b), paired_events=len(a))


def _entry_report(p, summaries, frames, baseline):
    events = p[KEY].drop_duplicates()
    outside = events.merge(baseline[KEY], on=KEY, how="left", indicator=True)._merge.eq("left_only")
    out = dict(particles=len(p), entry_events=len(events), entry_events_outside_baseline=int(outside.sum()),
               population="identical original dump/particles ordered rows; no gas-step selection", steps={},
               comparison={}, reconstruction={})
    for arm, summary in summaries.items():
        zero = summary.step_rows == 0
        gas_events = summary.loc[~zero, KEY].drop_duplicates()
        event_gas = events.merge(gas_events.assign(has_gas=True), on=KEY, how="left", validate="one_to_one")
        event_gas = event_gas.merge(baseline[KEY].assign(in_baseline=True), on=KEY, how="left", validate="one_to_one")
        outside_rows = summary.merge(baseline[KEY], on=KEY, how="left", indicator=True)._merge.eq("left_only")
        out["steps"][arm] = dict(particles=len(summary), total_step_rows=int(summary.step_rows.sum()),
            total_particle_distinct_straws=int(summary.distinct_straws.sum()), zero_gas_particles=int(zero.sum()),
            zero_gas_entry_events=int(len(events) - len(gas_events)), particles_outside_baseline=int(outside_rows.sum()),
            zero_gas_entry_events_outside_baseline=int((event_gas.has_gas.isna() & event_gas.in_baseline.isna()).sum()),
            zero_gas_particles_outside_baseline=int((zero.to_numpy() & outside_rows.to_numpy()).sum()),
            step_rows_outside_baseline=int(summary.loc[outside_rows.to_numpy(), "step_rows"].sum()),
            particle_distinct_straws_outside_baseline=int(summary.loc[outside_rows.to_numpy(), "distinct_straws"].sum()),
            step_rows=_mean(summary.step_rows), distinct_straws=_mean(summary.distinct_straws))
    for ref in ("legacy", "g4_clock"):
        out["comparison"][f"candidate_vs_{ref}"] = {c: _paired_mean(summaries["candidate"][c], summaries[ref][c])
                                                      for c in ("step_rows", "distinct_straws")}
    full = events.copy()
    for arm, frame in frames.items():
        frame = cecontract.unique(frame, arm)
        full = full.merge(frame[KEY + ["ntrk"]].rename(columns={"ntrk": f"{arm}_ntrk"}), on=KEY, how="left", validate="one_to_one")
    out["reconstruction"] = {f"{a}_vs_{r}": _efficiency(full, a, r) for a, r in PAIRS}
    return out


def _good(d, arm):
    return d[f"{arm}_ntrk"].fillna(0).to_numpy() > 0


def _efficiency(d, arm, ref):
    a, b = _good(d, arm), _good(d, ref)
    return dict(**_paired_mean(a, b), gained=int((a & ~b).sum()), lost=int((~a & b).sum()),
                denominator_events=len(d), candidate_good=int(a.sum()), reference_good=int(b.sum()))


def _relative(d, arm):
    p = d[f"{arm}_p_first_own"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (d[f"{arm}_p_reco"] - p) / p
    return r.where(np.isfinite(p) & (p > 0))


def _mask(d, arms):
    valid = np.ones(len(d), dtype=bool)
    sid = d[f"{arms[0]}_sid"]
    for arm in arms:
        valid &= _good(d, arm) & d[f"{arm}_sid"].isin(fieldreco.SIDS).to_numpy()
        valid &= (d[f"{arm}_sid"] == sid).to_numpy() & np.isfinite(_relative(d, arm)).to_numpy()
    return valid


def _core(a, b):
    out = cereco.paired_core(a, b, draws=DRAWS, seed=SEED)
    out.update(metrics=["relative_median", "relative_IQR_sigma"], bootstrap_draws=DRAWS, bootstrap_seed=SEED)
    for k in ("candidate", "reference", "difference", "paired_error"):
        out.setdefault(k, [None, None])
    return out


def _momentum(d, arm, ref, mask):
    a, b = (_relative(d, n)[mask] for n in (arm, ref))
    return dict(core=_core(a, b), descriptive_tails={str(cut): _paired_mean(np.abs(a) > cut, np.abs(b) > cut)
                                                 for cut in (0.01, 0.05, 0.10)})


def _comparison(d, arm, ref):
    both = _good(d, arm) & _good(d, ref)
    paired = _mask(d, (arm, ref))
    return dict(candidate_arm=arm, reference_arm=ref, efficiency=_efficiency(d, arm, ref),
        both_good_events=int(both.sum()), momentum_pair_events=int(paired.sum()),
        active_hits_both_good=_paired_mean(d.loc[both, f"{arm}_nactive"], d.loc[both, f"{ref}_nactive"]),
        distinct_active_straws_both_good=_paired_mean(d.loc[both, f"{arm}_nstraw"], d.loc[both, f"{ref}_nstraw"]),
        momentum=_momentum(d, arm, ref, paired),
        active_hit_rows_both_good=(_paired_mean(d.loc[both, f"{arm}_active_hit_rows"], d.loc[both, f"{ref}_active_hit_rows"])
                                  if all(f"{n}_active_hit_rows" in d for n in (arm, ref)) else None))


def _sample(d):
    common = _mask(d, ARMS)
    return dict(denominator_events=len(d), comparisons={f"{a}_vs_{r}": _comparison(d, a, r) for a, r in PAIRS},
        three_way_common_diagnostic=dict(label="diagnostic only, not an exclusion or corrected headline; all three good, same SID and finite own proxies",
            events=int(common.sum()), comparisons={f"{a}_vs_{r}": _momentum(d, a, r, common) for a, r in PAIRS}))


def report(d):
    """Full fixed denominator, fixed G4-first-straw bins, and separately labelled common-cohort diagnostic."""
    cecontract.unique(d, "comparison ledger")
    out = dict(scope="development-only completewall cosmic pilot; descriptive first-straw proxies, no equivalence or closure claim",
        events=len(d), denominator="all unchanged saved baseline reference-clock step events, including missing/no-good reco",
        proxy="each arm's event-first straw, original float32 expression; not truth matching or same-location resolution",
        pairing="each pair has its own both-good, same earliest in-bounds SID 0..4, finite own-proxy cohort; cohorts can differ",
        uncertainty="paired mean SEM; shared-index 500 event bootstraps, seed 20261103; one saved sample, not family-wise inference",
        tails="predeclared descriptive abs(relative residual) > 0.01, 0.05, 0.10; no tuning or equivalence gate",
        active_straw_definition="KinKal hit state > -2 includes left(-1), null(0), right(1); count verified against persisted nactive; state>0 is right-only diagnostic, not active",
        whole=_sample(d), bins=[], ntuple_exposure=d.attrs.get("ntuple_exposure", {}),
        entry_exposure=d.attrs.get("entry_exposure", {}))
    for lo, hi in zip(fieldquantiles.EDGES[:-1], fieldquantiles.EDGES[1:]):
        sel = (d.p_first >= lo) & (d.p_first < hi)
        out["bins"].append(dict(reference_first_straw_p_lo=lo, reference_first_straw_p_hi=hi, **_sample(d[sel])))
    return finite_json(out)


def check_baseline(d, baseline):
    """Reproduce old columns and metrics through the exact original CSV/default-parser representation."""
    b = cecontract.unique(baseline, "saved baseline")
    x = d.rename(columns={c: c.replace("legacy_", "fast_", 1) for c in d if c.startswith("legacy_")})
    required = {"p_first", "first_plane", *(f"{a}_{c}" for a in ("g4", "fast") for c in
                 ("ntrk", "sid", "nactive", "p_reco", "p_first_own", "p_true"))}
    if not required.issubset(b) or not set(b).issubset(x):
        raise ValueError("incomplete or unsupported baseline columns")
    serialized = pd.read_csv(io.StringIO(x[list(b)].to_csv(index=False)))
    serialized = cecontract.unique(serialized, "serialized baseline reproduction")
    try:
        pd.testing.assert_frame_equal(serialized, b, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError("retained G4/legacy baseline columns changed") from exc
    old = finite_json(fieldreco.compare(b, "g4", "fast"))
    quantiles = finite_json(fieldquantiles.report(b))
    if old != finite_json(fieldreco.compare(serialized, "g4", "fast")) or quantiles != finite_json(fieldquantiles.report(serialized)):
        raise ValueError("retained old metrics changed")
    return dict(unchanged=True, columns=list(b), legacy_fieldreco_compare=old, legacy_fieldquantiles=quantiles,
        comparison_representation="original to_csv then default pd.read_csv; exact parsed decimal representation, NOT raw-bit G4 equality")


def _read_table(path):
    with uproot.open(path) as root:
        pt, st = root["dump/particles"], root["dump/sgsteps"]
        p = fieldidentity._flat_table(pt).rename(columns={"key": "sp"})
        s = fieldidentity._flat_table(st)
        if len(p) != pt.num_entries or len(s) != st.num_entries:
            raise ValueError("table highest-cycle entry/frame count mismatch")
        return _particles(p), s, dict(particles=pt.object_path, sgsteps=st.object_path,
                                     particle_entries=int(pt.num_entries), step_entries=int(st.num_entries))


def run(candidate_table, legacy_dir, candidate_ntuple, legacy_ntuple, g4_ntuple, baseline_events, outdir):
    """Hash closed saved inputs before/after; exclusively create output and persist FAILED on any exception."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=False)
    paths, before, cycles = {}, {}, {}
    try:
        directory = Path(legacy_dir)
        paths = {a: directory / name for a, name in FILES.items()}
        paths.update(candidate=Path(candidate_table), candidate_ntuple=Path(candidate_ntuple),
                     legacy_ntuple=Path(legacy_ntuple), g4_ntuple=Path(g4_ntuple),
                     baseline_events=Path(baseline_events), fieldcompare=directory / "fieldcompare.json")
        paths = {k: p.resolve(strict=True) for k, p in paths.items()}
        before = {k: _sha256(p) for k, p in paths.items()}
        if not isinstance(json.loads(paths["fieldcompare"].read_text()), dict):
            raise ValueError("fieldcompare.json must be a saved report object")
        baseline = pd.read_csv(paths["baseline_events"])
        baseline = cecontract.unique(baseline, "saved baseline")
        if len(baseline) != BASELINE_EVENTS:
            raise ValueError(f"fixed baseline must contain {BASELINE_EVENTS} events")
        particles, steps = None, {}
        for arm in (*FILES, "candidate"):
            p, steps[arm], cycles[arm] = _read_table(paths[arm])
            if particles is None:
                particles = p
            else:
                try:
                    pd.testing.assert_frame_equal(particles, p, check_exact=True)
                except AssertionError as exc:
                    raise ValueError(f"ordered all-entry particle records changed: {arm}") from exc
        if len(particles) != ENTRY_PARTICLES:
            raise ValueError(f"fixed entry exposure must contain {ENTRY_PARTICLES} particles")
        frames = {}
        for arm in ARMS:
            path = paths[f"{arm}_ntuple"]
            with uproot.open(path) as root:
                tree = root[fieldreco.TREE]
                cycles[f"{arm}_ntuple"] = dict(tree=tree.object_path, entries=int(tree.num_entries))
            frames[arm] = read_arm(path)
            if len(frames[arm]) != cycles[f"{arm}_ntuple"]["entries"]:
                raise ValueError(f"highest-cycle ntuple entry/frame count mismatch: {arm}")
        d = assemble(particles, steps, frames, baseline)
        checked = check_baseline(d, baseline)
        for column in baseline:
            if column not in KEY:
                name = column.replace("fast_", "legacy_", 1) if column.startswith("fast_") else column
                d[name] = baseline[column].to_numpy()
        result = report(d)
        result.update(status="OK", baseline_check=checked,
                      retained_numeric_representation="G4/legacy common columns are the unchanged default-parser baseline after exact serialization validation; candidate proxy retains native float32 arithmetic",
                      finished_log_validation="external launcher responsibility; no finished-log event total inferred from NT")
        after = {k: _sha256(p) for k, p in paths.items()}
        if before != after:
            raise ValueError("closed input changed during comparison")
        result["inputs"] = {k: dict(path=str(p), sha256_before=before[k], sha256_after=after[k], **cycles.get(k, {}))
                            for k, p in paths.items()}
        result = finite_json(result)
        d.to_csv(out / "events.csv.gz", index=False)
        d.attrs["particle_steps"].to_csv(out / "particle_steps.csv.gz", index=False)
        (out / "fieldwallcompare.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    except Exception as exc:
        after = {k: _sha256(p) if p.is_file() else None for k, p in paths.items()}
        failure = dict(status="FAILED", error_type=type(exc).__name__, error=str(exc),
                       inputs={k: dict(path=str(p), sha256_before=before.get(k), sha256_after=after.get(k)) for k, p in paths.items()})
        (out / "fieldwallcompare.json").write_text(json.dumps(finite_json(failure), indent=2, allow_nan=False) + "\n")
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-table", "legacy-dir", "candidate-ntuple", "legacy-ntuple", "g4-ntuple", "baseline-events", "outdir"):
        ap.add_argument(f"--{name}", required=True)
    print(json.dumps(run(**vars(ap.parse_args(argv))), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
