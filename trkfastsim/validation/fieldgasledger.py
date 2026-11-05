"""Report-only saved SGS gas geometry/grouping ledger; no transport, wall paths or passage inference."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from mu2efastsim import tables
from trkfastsim import geometry
from trkfastsim.validation.fieldcrossing import FILES, verify_particles
from trkfastsim.validation.fieldmultiplicity import STRAW, TRACK, track_keys

QUANTILES = (0, .01, .1, .5, .9, .99, 1)
ENDPOINTS = [f"{a}{end}" for end in (0, 1) for a in "xyz"]
GROUP = TRACK + ["sid"]
METRICS = ("rows", "distinct_straws", "repeat_rows", "gaslength_mm", "chordlength_mm",
           "len_minus_chord_mm", "zero_secants", "shorter_than_chord_rows")
GROUP_METRICS = ("rows", "gaslength_mm", "chordlength_mm", "len_minus_chord_mm",
                 "zero_secants", "shorter_than_chord_rows")
LIMITATIONS = (
    "Saved gas-grouped SGS endpoints only, not a wall ledger or physical first-passage audit.",
    "Gas-grouped endpoints lack wall paths, constituent ordering and exact entry/exit momenta; returns cannot be reconstructed.",
    "Length/chord excess does NOT prove returns: curvature alone can produce excess.",
    "The saved whole-SGS clock cut uses paired fast exit + 1 ns; it is not physical passage and may retain a whole grouped SGS.",
    "All three arms are treated symmetrically; paired differences are descriptive only, not closure, correction or acceptance.",
)


def _finite(frame, label):
    """Validate every saved numeric column without coercing identities through float."""
    if not frame.columns.is_unique:
        raise ValueError(f"duplicate payload columns: {label}")
    for name in frame.columns:
        values = frame[name]
        if (not pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype)
                or pd.api.types.is_complex_dtype(values.dtype)):
            raise ValueError(f"malformed nonnumeric payload {label}/{name}")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"nonfinite payload rejected: {label}/{name}; no rows dropped")


def _identities(frame, columns, label):
    if (frame[columns].isna().any().any() or (frame[columns] < 0).any().any()
            or not all(pd.api.types.is_integer_dtype(frame[k].dtype)
                       and not pd.api.types.is_bool_dtype(frame[k].dtype) for k in columns)):
        raise ValueError(f"invalid integer identities: {label}")


def distribution(values):
    """Strict-JSON distribution, including explicit empty populations."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("nonfinite/malformed derived distribution")
    total = float(x.sum())
    if not np.isfinite(total):
        raise ValueError("nonfinite derived total")
    return {"n": len(x), "total": total, "mean": float(x.mean()) if len(x) else None,
            "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, np.quantile(x, QUANTILES))}
            if len(x) else {str(q): None for q in QUANTILES}}


def length_metrics(length, chord):
    """Ratios are undefined for zero secants; zero rows remain in all other populations."""
    length, chord = np.asarray(length, float), np.asarray(chord, float)
    positive = chord > 0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = length[positive] / chord[positive]
        difference = length - chord
    return {"len_mm": distribution(length), "chord_mm": distribution(chord),
            "len_minus_chord_mm": distribution(difference), "len_over_chord": distribution(ratio),
            "ratio_population": "nonzero secants only; zero secants not discarded from ledger",
            "zero_secants": int((~positive).sum()), "nonzero_secants": int(positive.sum()),
            "zero_secants_positive_len": int(((~positive) & (length > 0)).sum()),
            "ratio_above": {str(t): int((ratio > t).sum()) for t in (1.01, 1.1, 2)},
            "len_shorter_than_chord_beyond_0.001mm": int((difference < -1e-3).sum())}


def _geometry(geom):
    mapping = pd.DataFrame({"sid": geom.sid, **{k: getattr(geom, k) for k in STRAW}})
    _identities(mapping, ["sid"] + STRAW, "geometry")
    if mapping.empty or mapping.sid.duplicated().any() or mapping.duplicated(STRAW).any():
        raise ValueError("empty or duplicate geometry SID/label mapping")
    n = len(mapping)
    if (np.shape(geom.mid) != (n, 3) or np.shape(geom.dir) != (n, 3)
            or np.shape(geom.halflen) != (n,) or np.shape(geom.radius) != ()):
        raise ValueError("malformed geometry arrays")
    if not all(np.isfinite(v).all() for v in (geom.mid, geom.dir, geom.halflen, geom.radius)):
        raise ValueError("nonfinite geometry")
    if (geom.radius <= 0 or (geom.halflen <= 0).any()
            or not np.allclose(np.linalg.norm(geom.dir, axis=1), 1, atol=1e-12, rtol=0)):
        raise ValueError("invalid finite-cylinder geometry")
    mapping["sid"] = mapping.sid.astype(np.uint64)
    return mapping


def audit_arm(steps, denominator, geom, origin, label="saved arm"):
    """Vectorized exact assigned-straw diagnostics; never search neighboring straws."""
    required = GROUP + STRAW + ENDPOINTS + ["len"]
    if not set(required).issubset(steps.columns):
        raise ValueError(f"missing required SGS columns: {label}")
    _finite(steps, label)
    s = track_keys(steps, label).reset_index(drop=True)
    den = track_keys(denominator, "denominator")[TRACK]
    if den.empty or den.duplicated(TRACK).any():
        raise ValueError("empty or duplicate particle denominator")
    if not pd.MultiIndex.from_frame(s[TRACK]).isin(pd.MultiIndex.from_frame(den)).all():
        raise ValueError(f"step particle outside entry denominator: {label}")
    _identities(s, ["sid"] + STRAW, label)
    s["sid"] = s.sid.astype(np.uint64)
    mapping = _geometry(geom)
    assigned = pd.Index(mapping.sid).get_indexer(pd.Index(s.sid))
    if (assigned < 0).any():
        raise ValueError(f"foreign saved SID: {label}")
    if not np.array_equal(s[STRAW].to_numpy(), mapping.iloc[assigned][STRAW].to_numpy()):
        raise ValueError(f"SID plane/panel/straw labels disagree: {label}")
    origin = np.asarray(origin, float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("invalid saved origin_mm")
    length = s["len"].to_numpy(dtype=float)
    if (length < 0).any():
        raise ValueError(f"negative saved gas length: {label}")
    start = s[["x0", "y0", "z0"]].to_numpy(dtype=float) + origin
    end = s[["x1", "y1", "z1"]].to_numpy(dtype=float) + origin
    with np.errstate(over="ignore", invalid="ignore"):
        chord = np.linalg.norm(end - start, axis=1)
    if not np.isfinite(start).all() or not np.isfinite(end).all() or not np.isfinite(chord).all():
        raise ValueError("nonfinite derived endpoints/secants")
    residuals = {}
    for name, points in (("entry", start), ("exit", end)):
        displacement = points - geom.mid[assigned]
        axial = np.einsum("ij,ij->i", displacement, geom.dir[assigned])
        transverse = displacement - axial[:, None] * geom.dir[assigned]
        radial = np.linalg.norm(transverse, axis=1) - geom.radius
        axial = np.abs(axial) - geom.halflen[assigned]
        residuals[name] = {
            "radial_mm": {**distribution(radial), "outside_beyond_0.01mm": int((radial > .01).sum())},
            "axial_mm": {**distribution(axial), "outside_beyond_0.01mm": int((axial > .01).sum())},
            "outside_finite_cylinder_beyond_0.01mm": int(((radial > .01) | (axial > .01)).sum())}
    s["gaslength_mm"], s["chordlength_mm"] = length, chord
    s["len_minus_chord_mm"] = length - chord
    s["zero_secants"] = (chord == 0).astype(np.int64)
    s["shorter_than_chord_rows"] = (length - chord < -1e-3).astype(np.int64)
    sums = ["gaslength_mm", "chordlength_mm", "len_minus_chord_mm", "zero_secants", "shorter_than_chord_rows"]
    grouped = s.groupby(GROUP, sort=True, observed=True)[sums].sum()
    grouped["rows"] = s.groupby(GROUP, sort=True, observed=True).size()
    grouped["repeat_rows"] = grouped.rows - 1
    grouped = grouped.reset_index()
    per = s.groupby(TRACK, sort=True, observed=True)[sums].sum()
    per["rows"] = s.groupby(TRACK, sort=True, observed=True).size()
    per["distinct_straws"] = grouped.groupby(TRACK, sort=True, observed=True).size()
    per["repeat_rows"] = per.rows - per.distinct_straws
    per = per.reindex(pd.MultiIndex.from_frame(den), fill_value=0).reset_index()
    for metric in ("rows", "distinct_straws", "repeat_rows", "zero_secants", "shorter_than_chord_rows"):
        per[metric] = per[metric].astype(np.int64)
    report = {
        "saved_rows": len(s), "particle_denominator": len(den),
        "particles_with_rows": int((per.rows > 0).sum()), "zero_step_particles": int((per.rows == 0).sum()),
        "distinct_particle_sid_pairs": len(grouped), "repeat_rows": int(grouped.repeat_rows.sum()),
        "repeated_particle_sid_pairs": int((grouped.rows > 1).sum()),
        "saved_row_lengths": length_metrics(length, chord), "endpoint_residuals": residuals,
        "residual_convention": "signed rho-inner_radius and abs(axis_coordinate)-halflength; positive = outside, negative = interior; not distance to nearest surface",
        "per_particle": {m: distribution(per[m]) for m in METRICS},
        "per_particle_straw": {"lengths": length_metrics(grouped.gaslength_mm, grouped.chordlength_mm),
                               "rows": distribution(grouped.rows), "repeat_rows": distribution(grouped.repeat_rows)},
    }
    return per, grouped, report


def paired(left, right, keys, metrics):
    """Align exact integer identities before numeric differences; absent straw groups contribute zeros."""
    a, b = left.set_index(keys), right.set_index(keys)
    index = a.index.union(b.index, sort=False)
    a, b = a.reindex(index, fill_value=0), b.reindex(index, fill_value=0)
    return {"paired_identities": len(index), "left_only": int((~index.isin(right.set_index(keys).index)).sum()),
            "right_only": int((~index.isin(left.set_index(keys).index)).sum()),
            "difference_left_minus_right": {m: distribution(a[m].to_numpy() - b[m].to_numpy()) for m in metrics}}


def assemble(particles, arms, geom, origin):
    """Preserve the complete common saved particle denominator, including empty arms."""
    denominator = verify_particles(particles)
    if set(arms) != set(FILES):
        raise ValueError("all three saved step arms required")
    ledger, groups, reports, per = denominator.copy(), {}, {}, {}
    for label in FILES:
        per[label], groups[label], reports[label] = audit_arm(arms[label], denominator, geom, origin, label)
        ledger = ledger.merge(per[label].rename(columns={m: f"{label}_{m}" for m in METRICS}),
                              on=TRACK, how="left", validate="one_to_one")
    differences = {}
    for left, right in itertools.combinations(FILES, 2):
        name = f"{left}_minus_{right}"
        differences[name] = {"per_particle": paired(per[left], per[right], TRACK, METRICS),
                             "per_particle_sid": paired(groups[left], groups[right], GROUP, GROUP_METRICS)}
        for metric in METRICS:
            ledger[f"{name}_{metric}"] = ledger[f"{left}_{metric}"] - ledger[f"{right}_{metric}"]
    return ledger, groups, {"denominator_particles": len(denominator), "arms": reports, "paired": differences}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _check_hashes(paths, before):
    after = {name: _sha256(path) for name, path in paths.items()}
    if after != before:
        raise ValueError("saved inputs/code changed during gas ledger audit; no success report")
    return after


def run(field_dir, gdml, outdir):
    """Read frozen payloads, verify provenance before/after, publish into an exclusively new directory."""
    field_dir, gdml, outdir = Path(field_dir), Path(gdml), Path(outdir)
    if outdir.exists():
        raise FileExistsError(outdir)
    paths = {"fieldcompare": field_dir / "fieldcompare.json", "gdml": gdml,
             **{label: field_dir / filename for label, filename in FILES.items()},
             "code": Path(__file__), "geometry_code": Path(geometry.__file__)}
    from trkfastsim.validation import fieldcrossing, fieldmultiplicity, cecontract
    paths.update({"fieldcrossing_code": Path(fieldcrossing.__file__),
                  "fieldmultiplicity_code": Path(fieldmultiplicity.__file__), "cecontract_code": Path(cecontract.__file__),
                  "tables_code": Path(tables.__file__)})
    before = {name: _sha256(path) for name, path in paths.items()}
    provenance = json.loads(paths["fieldcompare"].read_text())
    origin = np.asarray(provenance["origin_mm"], float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("invalid saved origin_mm")
    particles, arms, tree_present = {}, {}, {}
    for label in FILES:
        particles[label] = tables.read(paths[label], "particles").rename(columns={"key": "sp"})
        _finite(particles[label], f"{label} particles")
        with uproot.open(paths[label]) as root:
            tree_present[label] = f"{tables.DIR}/sgsteps" in root
            arms[label] = (root[f"{tables.DIR}/sgsteps"].arrays(library="pd") if tree_present[label]
                           else pd.DataFrame({k: pd.Series(dtype="uint64" if k in GROUP + STRAW else "float64")
                                              for k in GROUP + STRAW + ENDPOINTS + ["len"]}))
    ledger, groups, result = assemble(particles, arms, geometry.from_gdml(str(gdml)), origin)
    if "entries" in provenance and provenance["entries"] != len(ledger):
        raise ValueError("fieldcompare entries disagree with complete saved denominator")
    result.update(limitations=list(LIMITATIONS), verdict="report-only saved gas geometry/grouping diagnostics; no physics closure",
                  origin_mm=origin.tolist(), origin_source="unaltered native detector-frame origin_mm from hashed old fieldcompare.json; endpoints translated to GDML Mu2e frame",
                  identity="exact full KEY+sp+sid; SID must map uniquely to geom.sid and agree with plane/panel/straw; never proximity",
                  momentum_bins="omitted: fieldcompare write_tables particles contain KEY,key,pdg,parent,isprim,ismcprim, not entry px/py/pz; SGS momenta are not entry momentum",
                  particle_keys_sha256=hashlib.sha256(ledger[TRACK].to_numpy(dtype="<u8").tobytes()).hexdigest())
    for label in FILES:
        result["arms"][label]["saved_step_tree_present"] = tree_present[label]
    _check_hashes(paths, before)
    outdir.mkdir(parents=True, exist_ok=False)
    outputs = {"per_particle": outdir / "fieldgasledger_particles.csv.gz"}
    ledger.to_csv(outputs["per_particle"], index=False, compression={"method": "gzip", "mtime": 0})
    for label in FILES:
        outputs[f"{label}_per_straw"] = outdir / f"fieldgasledger_{label}_straws.csv.gz"
        groups[label].to_csv(outputs[f"{label}_per_straw"], index=False, compression={"method": "gzip", "mtime": 0})
    after = _check_hashes(paths, before)
    result["inputs"] = {name: {"path": str(path.resolve()), "sha256_before": before[name], "sha256_after": after[name]}
                        for name, path in paths.items()}
    result["outputs"] = {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in outputs.items()}
    with (outdir / "fieldgasledger.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-dir", required=True)
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.field_dir, args.gdml, args.outdir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
