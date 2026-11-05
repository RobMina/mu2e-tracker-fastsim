"""Saved-only missing-wall exposure and legacy event-first reco observational diagnostic.

Collision means are the retained mean-model geometry diagnostic, not realized loss,
response correction, causal attribution, or a physical expected effect. No replay,
raw EventNtuple MC links, proximity association, revised efficiency or cores.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mu2efastsim import tables
from trkfastsim.validation import fieldcrossing as fc, fieldparents
from trkfastsim.validation import fieldmaterialparticles as mp, fieldpathaudit as pa

KEY = tables.KEY
TRACK = KEY + ["sp"]
ORIGINAL = KEY + ["originalsp"]
EDGES = (0., 300., 1000., 2000., 5000., 1e9)
EXPOSURE = ("missing_wall_mm", "consumed_wall_mm", "geometric_wall_mm",
            "missing_wall_x0", "consumed_wall_x0", "geometric_wall_x0",
            "mean_missing_collision_loss_MeV")
SCOPE = (__doc__.strip() + " Event totals include every retained entry particle; "
         "legacy first-good tracks are not primary-matched, especially in multi-entry events.")


def identities(frame, columns, label):
    """Accept only native integers or decimal CSV text, never floating identities."""
    d = frame.copy()
    for col in columns:
        if col not in d or d[col].isna().any():
            raise ValueError(f"missing identity {label}/{col}")
        s = d[col]
        if pd.api.types.is_bool_dtype(s.dtype) or pd.api.types.is_float_dtype(s.dtype):
            raise ValueError(f"noninteger identity {label}/{col}")
        if not s.astype(str).str.fullmatch(r"[0-9]+").all():
            raise ValueError(f"invalid identity {label}/{col}")
        try:
            d[col] = s.astype("uint64")
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"out-of-range identity {label}/{col}") from exc
    return d


def read_csv(path, columns, *, legacy=False):
    """Read native keys as text first, avoiding pandas mixed uint64/float inference."""
    options = {} if legacy else dict(float_precision="round_trip")
    d = pd.read_csv(path, dtype={c: "string" for c in columns}, **options)
    return identities(d, columns, str(path))


def _receipt_paths(report, kind, prefix, paths):
    receipts = report.get(kind)
    if not isinstance(receipts, dict) or not receipts:
        raise ValueError(f"missing {prefix}/{kind} receipts")
    for name, receipt in receipts.items():
        if not isinstance(receipt, dict) or not receipt.get("path"):
            raise ValueError(f"invalid receipt {prefix}/{kind}/{name}")
        path = Path(receipt["path"]).resolve(strict=True)
        expected = receipt.get("sha256_before") if kind == "inputs" else receipt.get("sha256")
        if kind == "inputs" and receipt.get("sha256_after") != expected:
            raise ValueError(f"open before/after receipt {prefix}/{name}")
        if not expected or pa.sha256(path) != expected:
            raise ValueError(f"receipt hash mismatch {prefix}/{name}")
        paths[f"{prefix}/{name}"] = path


def evidence(particle, source, particle_path, source_path, paths):
    """Bind all inherited receipt paths and hashes, not just selected convenient inputs."""
    if (particle.get("passed") is not True or particle.get("interpretation_allowed") is not True
            or particle.get("inputs_unchanged") is not True
            or particle.get("reproduction", {}).get("reproduced") is not True
            or particle.get("global_material_comparison", {}).get("equal") is not True
            or particle.get("particle_material_stats", {}).get("consistent") is not True):
        raise ValueError("particle report not passed/interpretation_allowed/consistent")
    required = {f"{arm}/{tree}" for arm in fc.FILES for tree in ("particles", "sgsteps")}
    checks = particle["reproduction"].get("tables", {})
    if set(checks) != required or any(v.get("equal") is not True for v in checks.values()):
        raise ValueError("all six exact old saved payload gates required")
    before, after = particle.get("hashes_before"), particle.get("hashes_after")
    if not before or before != after:
        raise ValueError("particle before/after receipts not closed")
    _receipt_paths(particle, "hashes_before", "particle", paths)
    if source.get("passed") is not True or source.get("inputs_unchanged") is not True:
        raise ValueError("source audit not passed/inputs_unchanged")
    _receipt_paths(source, "inputs", "source/input", paths)
    _receipt_paths(source, "outputs", "source/output", paths)
    if not {"renumber_map", "all_outliers", "art_inventory"}.issubset(source["outputs"]):
        raise ValueError("source audit required output receipts missing")
    if (source.get("g4all_ordered_payload_exact") is not True
            or source.get("clock_payload_exact_except_event_nsgs") is not True
            or source.get("all_outlier_csv_payloads_and_recomputed_selection_exact") is not True):
        raise ValueError("source exact saved payload gates required")
    field = Path(particle.get("original_field_dir", "")).resolve(strict=True)
    expected = {"original/fieldcompare.json": field / "fieldcompare.json"}
    expected.update({f"original/{filename}": field / filename for filename in fc.FILES.values()})
    for name, path in expected.items():
        if paths.get("particle/" + name) != path:
            raise ValueError(f"particle original field path mismatch: {name}")
    for label, filename in fc.FILES.items():
        if paths.get("source/input/voids/inputs/" + label) != field / filename:
            raise ValueError("source audit field_dir differs")
    if paths.get("source/input/voids/inputs/fieldcompare") != field / "fieldcompare.json":
        raise ValueError("source audit field_dir differs")
    if ("particle/input_table" not in paths or paths.get("particle/input_table")
            != paths.get("source/input/input_table")):
        raise ValueError("original input table path differs")
    if before["input_table"]["sha256"] != source["inputs"]["input_table"]["sha256_before"]:
        raise ValueError("original input table sha256 differs")
    ledger = particle_path.parent / "particles.csv"
    manifest = particle_path.parent.parent / "closed.sha256"
    declared = {}
    for line in manifest.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        target = (particle_path.parent / relative).resolve(strict=True)
        if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or target in declared or not target.is_relative_to(particle_path.parent)):
            raise ValueError("invalid/duplicate particle closed-manifest entry")
        if pa.sha256(target) != digest:
            raise ValueError("particle closed-manifest hash mismatch")
        declared[target] = digest
        paths["particle/closed/" + relative] = target
    if particle_path not in declared or ledger.resolve(strict=True) not in declared:
        raise ValueError("particle closed manifest must bind report and per-particle ledger")
    paths["particle_closed_manifest"] = manifest.resolve(strict=True)
    renumber = source_path.parent / "fieldsourceaudit_renumber.csv.gz"
    if paths.get("source/output/renumber_map") != renumber:
        raise ValueError("source renumber output path differs")
    paths.update(particles_csv=ledger.resolve(strict=True), renumber_map=renumber.resolve(strict=True))
    return field


def join_particles(ledger, genealogy, expected_entries):
    """Require the full exact original KEY/sp set, with one-to-one genealogy."""
    p = identities(ledger, TRACK, "material ledger")
    g = identities(genealogy, ORIGINAL + ["injectorkey"], "source genealogy")
    if (len(p) != expected_entries or len(g) != expected_entries
            or p.duplicated(TRACK).any() or g.duplicated(ORIGINAL).any()
            or g.duplicated(KEY + ["injectorkey"]).any()
            or (g.injectorkey == 0).any() or (g.originalsp == 0).any() or (p.sp == 0).any()):
        raise ValueError("incomplete/duplicate/zero original entry genealogy")
    needed = ("pentry_MeV_c",) + mp.COUNTS + mp.LENGTHS + mp.MODELS
    if not set(needed).issubset(p) or "entryp" not in g:
        raise ValueError("incomplete particle exposure or source entryp")
    values = p[list(needed)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("nonfinite/negative particle exposure")
    if not np.isfinite(g.entryp.to_numpy(float)).all() or (g.entryp < 0).any():
        raise ValueError("invalid source entryp")
    if not np.equal(p[list(mp.COUNTS)], np.floor(p[list(mp.COUNTS)])).all().all():
        raise ValueError("noninteger particle counters")
    residual = p.geometric_wall_mm - p.consumed_wall_mm - p.missing_wall_mm
    if (np.abs(residual) > mp.ATOL_MM).any():
        raise ValueError("particle wall partition inconsistent")
    g = g.rename(columns={"entryp": "source_entryp_MeV_c"})
    d = p.rename(columns={"sp": "originalsp"}).merge(
        g, on=ORIGINAL, how="left", sort=False, validate="one_to_one", indicator=True)
    if (d._merge != "both").any():
        raise ValueError("foreign original KEY/sp: full genealogy set differs")
    d = d.drop(columns="_merge")
    d["sp"] = d.originalsp
    d["source_minus_material_entryp_MeV_c"] = d.source_entryp_MeV_c - d.pentry_MeV_c
    d["has_missing_wall"] = d.missing_wall_mm > 0
    return d


def distribution(values):
    """All retained rows are the denominator, including zero exposures."""
    x = np.asarray(values, dtype=float)
    q = np.quantile(x, [.0, .1, .25, .5, .75, .9, .95, .99, 1.]).tolist() if len(x) else [None] * 9
    return dict(n=len(x), zero=int((x == 0).sum()), nonzero=int((x != 0).sum()),
                total=float(x.sum()), mean=float(x.mean()) if len(x) else None,
                max=float(x.max()) if len(x) else None,
                quantiles=dict(zip(("q0", "q10", "q25", "q50", "q75", "q90", "q95", "q99", "q100"), q)))


def exposure_summary(d):
    missing, geometric = float(d.missing_wall_mm.sum()), float(d.geometric_wall_mm.sum())
    return dict(rows=len(d), zero_missing=int((d.missing_wall_mm == 0).sum()),
                nonzero_missing=int((d.missing_wall_mm > 0).sum()),
                missing_fraction_of_total_geometric_wall=missing / geometric if geometric else None,
                distributions={c: distribution(d[c]) for c in EXPOSURE})


def particle_summary(d):
    result = dict(denominator="all retained original entry particles, including zero-chord/zero-exposure entries",
                  entryp="material ledger pentry_MeV_c, not source float32 sqrt or event p_first",
                  edges=list(EDGES), interval="lo <= material pentry < hi", population=exposure_summary(d), bins=[])
    total_missing, total_wall = float(d.missing_wall_mm.sum()), float(d.geometric_wall_mm.sum())
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        part = d[(d.pentry_MeV_c >= lo) & (d.pentry_MeV_c < hi)]
        row = dict(lo=lo, hi=hi, **exposure_summary(part))
        row["fraction_population_geometric_wall"] = float(part.geometric_wall_mm.sum()) / total_wall if total_wall else None
        row["fraction_population_missing_wall"] = float(part.missing_wall_mm.sum()) / total_missing if total_missing else None
        result["bins"].append(row)
    result["outside_fixed_bins"] = int(((d.pentry_MeV_c < EDGES[0]) | (d.pentry_MeV_c >= EDGES[-1])).sum())
    return result


def join_events(particles, baseline):
    """Vectorized event totals left-joined to the untouched legacy event order/denominator."""
    b = identities(baseline, KEY, "baseline")
    if b.duplicated(KEY).any() or not set(fieldparents.REQUIRED).issubset(b):
        raise ValueError("duplicate or incomplete legacy baseline")
    total = particles.groupby(KEY, sort=False)[list(EXPOSURE)].sum()
    total["entry_particles"] = particles.groupby(KEY, sort=False).size()
    total = total.reset_index()
    membership = pd.MultiIndex.from_frame(total[KEY]).isin(pd.MultiIndex.from_frame(b[KEY]))
    outside = int((~membership).sum())
    names = {c: "material_" + c for c in list(EXPOSURE) + ["entry_particles"]}
    if set(names.values()).intersection(b):
        raise ValueError("baseline contains reserved material columns")
    d = b.merge(total.rename(columns=names), on=KEY, how="left", validate="one_to_one", sort=False)
    d[list(names.values())] = d[list(names.values())].fillna(0)
    d["material_entry_particles"] = d.material_entry_particles.astype("uint64")
    d["material_has_missing_wall"] = d.material_missing_wall_mm > 0
    d["material_zero_missing_wall"] = ~d.material_has_missing_wall
    d["material_no_entry_particles"] = d.material_entry_particles == 0
    ia, ir = (d[f"{arm}_ntrk"].fillna(0) > 0 for arm in ("fast", "g4"))
    d["legacy_fast_firstgood"] = ia
    d["legacy_g4_firstgood"] = ir
    d["legacy_good_category"] = np.select([ia & ~ir, ~ia & ir, ia & ir], ["gained", "lost", "both"], default="neither")
    ra = (d.fast_p_reco - d.fast_p_first_own) / d.fast_p_first_own
    rr = (d.g4_p_reco - d.g4_p_first_own) / d.g4_p_first_own
    pair = ia & ir & (d.fast_sid == d.g4_sid) & np.isfinite(ra) & np.isfinite(rr)
    d["legacy_paired_2to5GeV_ownfirststraw_proxy"] = pair & (d.p_first >= 2000.) & (d.p_first < 5000.)
    pd.testing.assert_frame_equal(d[list(b.columns)], b.reset_index(drop=True), check_exact=True)
    return d, outside


def event_summary(d):
    def describe(frame):
        exposure = frame[["material_" + c for c in EXPOSURE]].rename(columns=lambda c: c[len("material_"):])
        return dict(events=len(frame), single_entry=int((frame.material_entry_particles == 1).sum()),
                    multi_entry=int((frame.material_entry_particles > 1).sum()),
                    no_entry=int((frame.material_entry_particles == 0).sum()), exposure=exposure_summary(exposure))

    selected = d[d.legacy_paired_2to5GeV_ownfirststraw_proxy]
    return dict(label="observational diagnostic; not primary-matched in multi-entry events",
                denominator="unchanged original baseline event rows/order; no efficiency or core recalculation",
                all_baseline=describe(d),
                legacy_firstgood_joint={name: describe(d[d.legacy_good_category == name])
                                        for name in ("gained", "lost", "both", "neither")},
                paired_2to5GeV_ownfirststraw_proxy=dict(
                    gate="fieldparents: both first-good, equal reco sid, finite own-first-straw relative residuals; 2000 <= p_first < 5000",
                    label="observational event-total omitted-wall and mean-model diagnostic; not a core exclusion/recalculation",
                    all=describe(selected), no_omitted=describe(selected[~selected.material_has_missing_wall]),
                    has_omitted=describe(selected[selected.material_has_missing_wall])))


def run(particle_report, source_report, baseline_events, outdir):
    """Admit only closed saved evidence and write new artifacts exclusively."""
    particle_path, source_path, baseline_path, out = map(
        lambda p: Path(p).resolve(), (particle_report, source_report, baseline_events, outdir))
    if out.exists():
        raise FileExistsError(out)
    paths = dict(particle_report=particle_path, source_report=source_path, baseline_events=baseline_path,
                 code=Path(__file__).resolve())
    initial = pa.receipts(paths)
    particle, source = (json.loads(p.read_text()) for p in (particle_path, source_path))
    field = evidence(particle, source, particle_path, source_path, paths)
    if any(out == p or out in p.parents for p in paths.values()) or any(
            p == out or p in out.parents for p in (field, particle_path.parent, source_path.parent)):
        raise ValueError("outdir must be exclusive and separate from evidence")
    before = pa.receipts(paths)
    evidence(particle, source, particle_path, source_path, paths)
    if any(before[k] != v for k, v in initial.items()):
        raise ValueError("input changed during evidence admission")
    old = json.loads((field / "fieldcompare.json").read_text())
    n = old.get("entries")
    if (isinstance(n, bool) or not isinstance(n, int) or n <= 0
            or source.get("entry_particles") != n
            or particle["particle_material_stats"].get("entries") != n):
        raise ValueError("original full entry denominator counts differ")
    p = join_particles(read_csv(paths["particles_csv"], TRACK),
                       read_csv(paths["renumber_map"], ORIGINAL + ["injectorkey"]), n)
    totals = particle["particle_material_stats"].get("totals", {})
    for col in mp.COUNTS:
        expected = totals.get(col)
        values = p[col]
        if (isinstance(expected, bool) or not isinstance(expected, int)
                or not pd.api.types.is_integer_dtype(values.dtype)
                or np.sum(values.to_numpy(dtype=object)) != expected):
            raise ValueError(f"particle ledger/report exact count differs: {col}")
    for col in mp.LENGTHS + mp.MODELS:
        expected = totals.get(col)
        if (isinstance(expected, bool) or not isinstance(expected, (int, float))
                or not np.isfinite(expected) or abs(float(p[col].sum()) - expected) > 1e-7):
            raise ValueError(f"particle ledger/report floating total differs: {col}")
    baseline = read_csv(baseline_path, KEY, legacy=True)
    events, outside = join_events(p, baseline)
    result = dict(passed=True, interpretation_allowed=True, label="OBSERVATIONAL_MEANMODELGEOMETRYDIAGNOSTIC",
                  scope=SCOPE, original_field_dir=str(field), entry_particles=n,
                  particle_ledger_binding="report and particles.csv both bound by upstream wrapper closed.sha256; all declared outputs rehashed",
                  total_comparison="integer counters exact; floating lengths/X0/MeV use absolute 1e-7 in their respective stored units, rtol=0",
                  momentum_scope="source entryp and material pentry retained separately; float32 sqrt versus float64 transport norm need not be identical",
                  particles=particle_summary(p), baseline=event_summary(events), eventoutsidebaseline=outside,
                  hashes_before=before)
    if pa.receipts(paths) != before:
        raise ValueError("input changed during saved summary")
    out.mkdir(parents=True, exist_ok=False)
    outputs = dict(particles=out / "particles.csv.gz", events=out / "events.csv.gz")
    p.to_csv(outputs["particles"], index=False, compression={"method": "gzip", "mtime": 0})
    events.to_csv(outputs["events"], index=False, compression={"method": "gzip", "mtime": 0})
    after = pa.receipts(paths)
    if before != after:
        raise ValueError("input changed during output publication")
    result.update(hashes_after=after, inputs_unchanged=True, outputs=pa.receipts(outputs))
    with (out / "fieldmaterialsummary.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("particle-report", "source-report", "baseline-events", "outdir"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(run(**vars(parser.parse_args(argv))), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
