"""Report-only own-SID finite endpoint-secant wire-void proxy; never replay transport."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from mu2efastsim import tables
from trkfastsim import geometry
from trkfastsim.validation import fieldcrossing as fc, fieldgasledger as fg, fieldmaterial as fm

FILES, TRACK, STRAW = fc.FILES, fg.TRACK, fg.STRAW
ENDPOINTS = fg.ENDPOINTS
THRESHOLDS = (.001, .01, .1, 1., 10., 100.)
METRICS = ("rowcount", "savedlen_mm", "chord_mm", "inner_mm", "wire_mm", "gasonly_mm",
           "short_by1_rows", "residual_short_by1_rows")
LIMITATIONS = (
    "Geometry-only gap proxy on saved SGS endpoint secants, not physical G4 trajectories or a gas-length correction.",
    "Raw gas path excludes daughter wire volumes; subtracting the finite wire OUTER cylinder from a straight gas secant is only a geometric comparison.",
    "Agreement cannot prove wire is the actual trajectory cause; disagreement cannot recover curvature, constituent ordering or multiple passages.",
    "Gas-only secant = max(own finite inner-cylinder overlap - own finite wire overlap, 0); not full chord minus wire when endpoints lie outside the cylinder.",
    "Wire overlap is intersected with both the finite endpoint segment and inner gas interval; outside-cylinder loss is not subtracted twice.",
    "All saved rows and complete entry-particle denominator retained, including zero secants and zero-step particles; ratios use nonzero chords only.",
    "The saved clock arm uses paired fast exit + 1 ns, not physical first passage; no closure, correction, acceptance or causal attribution.",
)


def distribution(values):
    out = fg.distribution(values)
    out["max"] = float(np.max(values)) if len(values) else None
    return out


def secant_lengths(steps, geom, origin, batch=65536):
    """Paired assigned-straw intersections only; callers must first pass fg.audit_arm.

    Zero secants stay zero. Intersecting wire and inner intervals before measuring
    avoids double subtraction even for endpoints outside finite radial/axial bounds.
    """
    fc._positive_integer(batch, "secant batch")
    wire_r = np.asarray(geom.wire_r)
    if (wire_r.shape != () or not np.isfinite(wire_r) or not 0 < wire_r < geom.radius):
        raise ValueError("invalid wire outer radius: require 0 < wire_r < inner gas radius")
    mapping = fg._geometry(geom)
    assigned = pd.Index(mapping.sid).get_indexer(pd.Index(steps.sid))
    if (assigned < 0).any():
        raise ValueError("foreign saved SID")
    start = steps[["x0", "y0", "z0"]].to_numpy(float) + origin
    end = steps[["x1", "y1", "z1"]].to_numpy(float) + origin
    delta = end - start
    chord = np.linalg.norm(delta, axis=1)
    inner, wire = np.zeros(len(steps)), np.zeros(len(steps))
    # Batches are row batches, never per-event or lane-by-all-straw searches.
    for first in range(0, len(steps), batch):
        rows = np.arange(first, min(first + batch, len(steps)))
        rows = rows[chord[rows] > 0]
        own = assigned[rows]
        direction = delta[rows] / chord[rows, None]
        args = (start[rows] - geom.mid[own], direction, geom.dir[own], geom.halflen[own])
        gas0, gas1 = fm.cylinder_intervals(*args, geom.radius)
        wire0, wire1 = fm.cylinder_intervals(*args, float(wire_r))
        gas0, gas1 = np.maximum(gas0, 0), np.minimum(gas1, chord[rows])
        wire0, wire1 = np.maximum(wire0, gas0), np.minimum(wire1, gas1)
        inner[rows] = np.maximum(gas1 - gas0, 0)
        wire[rows] = np.maximum(wire1 - wire0, 0)
    gasonly = np.maximum(inner - wire, 0)
    if not all(np.isfinite(v).all() for v in (chord, inner, wire, gasonly)):
        raise ValueError("nonfinite derived finite-cylinder lengths")
    return chord, inner, wire, gasonly


def audit_arm(steps, denominator, geom, origin, label="saved arm"):
    """Validate exact keys/labels with fieldgasledger; return per-particle, ALL outliers, report."""
    # Includes all numeric payloads, integer identities, denominator membership,
    # unique geometry mapping, unit axes, native origin and finite endpoints.
    _, _, validation = fg.audit_arm(steps, denominator, geom, origin, label)
    s = fg.track_keys(steps, label).reset_index(drop=True)
    s["sid"] = s.sid.astype(np.uint64)
    chord, inner, wire, gasonly = secant_lengths(s, geom, np.asarray(origin, float))
    length = s["len"].to_numpy(float)
    raw, residual = chord - length, gasonly - length
    short, remaining = raw > 1., residual > 1.
    s["originalrowindex"] = np.arange(len(s), dtype=np.int64)
    for name, values in (("savedlen_mm", length), ("chord_mm", chord), ("inner_mm", inner),
                         ("wire_mm", wire), ("gasonly_mm", gasonly), ("raw_shortfall_mm", raw),
                         ("residual_mm", residual), ("outside_inner_mm", chord - inner)):
        s[name] = values
    s["rowcount"] = np.ones(len(s), dtype=np.int64)
    s["short_by1_rows"], s["residual_short_by1_rows"] = short.astype(np.int64), remaining.astype(np.int64)
    den = fg.track_keys(denominator, "denominator")[TRACK]
    per = s.groupby(TRACK, sort=True, observed=True)[list(METRICS)].sum()
    per = per.reindex(pd.MultiIndex.from_frame(den), fill_value=0).reset_index()
    for name in ("rowcount", "short_by1_rows", "residual_short_by1_rows"):
        per[name] = per[name].astype(np.int64)
    momenta = [c for c in steps.columns if c in {f"p{a}{suffix}" for a in "xyz" for suffix in ("", "0", "1")}]
    columns = TRACK + ["sid"] + STRAW + ["originalrowindex", "savedlen_mm", "chord_mm", "inner_mm",
        "wire_mm", "gasonly_mm", "raw_shortfall_mm", "residual_mm", "outside_inner_mm"] + ENDPOINTS + momenta
    outliers = s.loc[short | remaining, columns].copy()
    positive = chord > 0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = length[positive] / chord[positive]
    report = {
        "saved_rows": len(s), "particle_denominator": len(den),
        "particles_with_rows": int((per.rowcount > 0).sum()),
        "zero_step_particles": int((per.rowcount == 0).sum()),
        "zero_secants": int((~positive).sum()), "nonzero_secants": int(positive.sum()),
        "zero_secants_positive_len": int(((~positive) & (length > 0)).sum()),
        "lengths": {name: distribution(s[name].to_numpy()) for name in (
            "savedlen_mm", "chord_mm", "inner_mm", "wire_mm", "gasonly_mm", "outside_inner_mm",
            "raw_shortfall_mm", "residual_mm")},
        "len_over_chord": distribution(ratio),
        "ratio_below": {str(t): int((ratio < t).sum()) for t in (.99, .9, .5, .1)},
        "shortfall_beyond_mm": {kind: {str(t): int((values > t).sum()) for t in THRESHOLDS}
                                 for kind, values in (("raw", raw), ("residual", residual))},
        "raw_short_by1_rows": int(short.sum()),
        "raw_short_by1_with_residual_by1_rows": int((short & remaining).sum()),
        "raw_short_by1_without_residual_by1_rows": int((short & ~remaining).sum()),
        "positive_shortfalls_mm": {
            "before_full_chord": distribution(np.maximum(raw, 0)),
            "after_wire_only": distribution(np.maximum(raw - wire, 0)),
            "after_inner_and_wire": distribution(np.maximum(residual, 0))},
        "wire_overlap_rows": int((wire > 0).sum()), "outside_wire_rows": int((wire == 0).sum()),
        "all_outlier_rows": len(outliers), "endpoint_residuals": validation["endpoint_residuals"],
        "per_particle": {name: distribution(per[name].to_numpy()) for name in METRICS},
    }
    return per, outliers, report


def assemble(particles, arms, geom, origin):
    """Use fc.verify_particles; preserve exact uint64 identities and all entry particles."""
    for label, frame in particles.items():
        fg._finite(frame, f"{label} particles")
    denominator = fc.verify_particles(particles)
    if set(arms) != set(FILES):
        raise ValueError("all three saved step arms required")
    ledger, outliers, reports = denominator.copy(), {}, {}
    for label in FILES:
        per, outliers[label], reports[label] = audit_arm(arms[label], denominator, geom, origin, label)
        ledger = ledger.merge(per.rename(columns={m: f"{label}_{m}" for m in METRICS}),
                              on=TRACK, how="left", validate="one_to_one")
    return ledger, outliers, {"denominator_particles": len(denominator), "arms": reports}


def run(field_dir, gdml, outdir):
    """Read only frozen tables; hash inputs/helpers before/after and publish exclusively."""
    field_dir, gdml, outdir = Path(field_dir), Path(gdml), Path(outdir)
    if outdir.exists():
        raise FileExistsError(outdir)
    from trkfastsim.validation import cecontract, fieldmultiplicity, fieldreco
    from mu2efastsim import world
    modules = {"fieldgasledger": fg, "fieldmaterial": fm, "fieldcrossing": fc,
               "geometry": geometry, "tables": tables, "cecontract": cecontract,
               "fieldmultiplicity": fieldmultiplicity, "fieldreco": fieldreco, "world": world}
    paths = {"fieldcompare": field_dir / "fieldcompare.json", "gdml": gdml,
             **{label: field_dir / filename for label, filename in FILES.items()},
             "code": Path(__file__), **{f"code/{name}": Path(mod.__file__) for name, mod in modules.items()}}
    before = {name: fg._sha256(path) for name, path in paths.items()}
    provenance = json.loads(paths["fieldcompare"].read_text())
    origin = np.asarray(provenance["origin_mm"], float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("invalid saved origin_mm")
    particles, arms, tree_present = {}, {}, {}
    for label in FILES:
        particles[label] = tables.read(paths[label], "particles").rename(columns={"key": "sp"})
        with uproot.open(paths[label]) as root:
            tree_present[label] = f"{tables.DIR}/sgsteps" in root
            arms[label] = (root[f"{tables.DIR}/sgsteps"].arrays(library="pd") if tree_present[label]
                           else pd.DataFrame({k: pd.Series(dtype="uint64" if k in fg.GROUP + STRAW else "float64")
                                              for k in fg.GROUP + STRAW + ENDPOINTS + ["len"]}))
    ledger, outliers, result = assemble(particles, arms, geometry.from_gdml(str(gdml)), origin)
    if "entries" in provenance and provenance["entries"] != len(ledger):
        raise ValueError("fieldcompare entries disagree with complete saved denominator")
    result.update(limitations=list(LIMITATIONS), verdict="report-only geometry gap proxy; no causal attribution or physics closure",
                  origin_mm=origin.tolist(),
                  origin_source="unaltered native origin_mm from hashed saved fieldcompare.json; add to saved detector endpoints for GDML Mu2e frame",
                  outlier_coordinates="saved detector-frame x0/y0/z0/x1/y1/z1; add origin_mm; momenta copied where available, not asserted to be entry momenta",
                  identity="exact KEY+sp+sid and matching plane/panel/straw; originalrowindex is zero-based saved row position; no proximity",
                  outlier_selection="ALL rows with chord-savedlen > 1 mm OR gasonly-savedlen > 1 mm, no bounded examples",
                  ratios="len/chord only for nonzero chords",
                  particle_keys_sha256=hashlib.sha256(ledger[TRACK].to_numpy(dtype="<u8").tobytes()).hexdigest())
    for label in FILES:
        result["arms"][label]["saved_step_tree_present"] = tree_present[label]
    json.dumps(result, allow_nan=False)  # Fail before publishing on nonfinite derived JSON.
    fg._check_hashes(paths, before)
    outdir.mkdir(parents=True, exist_ok=False)
    outputs = {"per_particle": outdir / "fieldgasvoids_particles.csv.gz"}
    ledger.to_csv(outputs["per_particle"], index=False, compression={"method": "gzip", "mtime": 0})
    for label in FILES:
        outputs[f"{label}_outliers"] = outdir / f"fieldgasvoids_{label}_outliers.csv.gz"
        outliers[label].to_csv(outputs[f"{label}_outliers"], index=False, compression={"method": "gzip", "mtime": 0})
    after = fg._check_hashes(paths, before)
    result["inputs"] = {name: {"path": str(path.resolve()), "sha256_before": before[name], "sha256_after": after[name]}
                        for name, path in paths.items()}
    result["outputs"] = {name: {"path": str(path.resolve()), "sha256": fg._sha256(path)} for name, path in outputs.items()}
    with (outdir / "fieldgasvoids.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-dir", required=True)
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--outdir", required=True)
    print(json.dumps(run(**vars(parser.parse_args(argv))), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
