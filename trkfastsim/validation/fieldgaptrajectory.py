"""Saved original-dumper trajectory polylines for ALL source-audit large-gap targets.

No transport, native ART decoding, proximity association, tuning, or physical-passage
claim. Sparse sampled polylines cannot recover absent raw steps, causality, or a
same-truth-surface comparison. The clock arm is not stock-G4 first passage.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from mu2efastsim import tables
from trkfastsim import geometry
from trkfastsim.validation import fieldsourceaudit as source
from trkfastsim.validation import fieldgasledger as fg, fieldmaterial as fm

KEY = tables.KEY
TRACK = KEY + ["originalsp"]
TARGET = TRACK + ["sid", "sourceoriginalrowindex"]
POINT_COLUMNS = KEY + ["sp", "ipt", "x", "y", "z", "t", "ke"]
ARMS = ("g4_all", "g4_clock")
LIMITATIONS = [
    "Sampled-polyline finite-cylinder geometry only, NOT physical G4 gas paths or G4 passages.",
    "Sparse points and secants cannot recover absent raw steps, constituent ordering, curvature, causality, or same-truth-surface closure.",
    "Full retained original particle trajectory is NOT the SGS span; after_sgs_entry uses only points t >= raw SGS t, without interpolation or an invented exit clock.",
    "No SGS last time is available: neither fixed view isolates an SGS range. Clock is not stock-G4 first passage.",
    "Disconnected runs are purely polyline geometric runs, split at wire gaps, segment-interior gaps, or nonconsecutive native ipt; not physical passage counts.",
    "Stored SGS length versus modeled polyline gas length is descriptive only: no causal ratio, correction, or acceptance claim.",
]


def integer_ids(frame, columns, label):
    """Never coerce native identities through float, including uint64 beyond 2**53.

    Native point indices and event coordinates can be zero; SimParticle keys must
    be positive. Original row indices are zero-based positional indices.
    """
    fg._identities(frame, columns, label)
    for name in ("sp", "originalsp"):
        if name in columns and (frame[name] == 0).any():
            raise ValueError(f"nonpositive SimParticle key: {label}")


def verify_source(report):
    """Require every receipt and mandatory certification; return all frozen paths."""
    if report.get("passed") is not True or report.get("inputs_unchanged") is not True:
        raise ValueError("source audit must have passed with inputs_unchanged true")
    paths = {}
    for kind in ("inputs", "outputs"):
        receipts = report.get(kind)
        if not isinstance(receipts, dict) or not receipts:
            raise ValueError(f"missing source {kind} receipts")
        for name, receipt in receipts.items():
            path = Path(receipt["path"]).resolve()
            expected = receipt["sha256_before"] if kind == "inputs" else receipt["sha256"]
            if kind == "inputs" and receipt["sha256_after"] != expected:
                raise ValueError(f"source input changed: {name}")
            if fg._sha256(path) != expected:
                raise ValueError(f"source hash mismatch: {kind}/{name}")
            paths[f"source/{kind}/{name}"] = path
    for name in ("input_table", "source_art", "voids/inputs/gdml", "voids/inputs/fieldcompare"):
        if f"source/inputs/{name}" not in paths:
            raise ValueError(f"missing source input receipt: {name}")
    for name in ("all_outliers", "renumber_map", "art_inventory"):
        if f"source/outputs/{name}" not in paths:
            raise ValueError(f"missing source output receipt: {name}")
    inventory = json.loads(paths["source/outputs/art_inventory"].read_text())
    if inventory != report.get("art_inventory"):
        raise ValueError("typed source ART inventory differs from source report")
    if (Path(inventory["path"]).resolve() != paths["source/inputs/source_art"]
            or inventory.get("raw_payload_read") is not False
            or inventory.get("raw_decoded") is not False
            or inventory.get("native_art_exact_link_certified") is not False
            or not isinstance(inventory.get("branches"), list)
            or not all(isinstance(b.get("name"), str) and isinstance(b.get("typename"), str)
                       for b in inventory["branches"])
            or not inventory.get("presence_scope")):
        raise ValueError("invalid typed source ART branch-presence-only inventory scope")
    return paths


def targets_from_source(frame, report):
    """Deduplicate exact original SGS rows shared by both arms, keeping both flags."""
    required = TARGET + ["arm", "sp", "savedlen_mm"] + fg.ENDPOINTS
    if not set(required).issubset(frame):
        raise ValueError("missing source outlier identity/payload columns")
    integer_ids(frame, TARGET + ["sp"], "source outliers")
    if not frame.arm.isin(ARMS).all() or frame.duplicated(TARGET + ["arm"]).any():
        raise ValueError("foreign arm or duplicate source target within arm")
    for arm in ARMS:
        if int((frame.arm == arm).sum()) != report["outlier_rows"][arm]:
            raise ValueError("incomplete source outlier inventory")
    # Same original row must never be associated with a different original particle/SID.
    unique = frame.drop_duplicates(TARGET).copy()
    if unique.sourceoriginalrowindex.duplicated().any():
        raise ValueError("original row assigned conflicting exact identities")
    payload = ["sp", "savedlen_mm"] + fg.ENDPOINTS
    reference = unique[TARGET + payload]
    bound = frame.merge(reference, on=TARGET, suffixes=("", "_ref"), validate="many_to_one")
    for name in payload:
        if not bound[name].equals(bound[name + "_ref"].rename(name)):
            raise ValueError(f"both-arm source payload mismatch: {name}")
    for arm in ARMS:
        members = pd.MultiIndex.from_frame(frame.loc[frame.arm == arm, TARGET])
        unique[arm] = pd.MultiIndex.from_frame(unique[TARGET]).isin(members)
    unique = unique[TARGET + ["sp", "savedlen_mm"] + fg.ENDPOINTS + list(ARMS)].reset_index(drop=True)
    unique["target"] = np.arange(len(unique), dtype=np.int64)
    return unique


def bind_sgs(targets, raw, origin):
    """Bind positional original dumper rows, exact IDs and original CSV serialization.

    Float32 endpoints were serialized to short decimals by sourceaudit; compare
    that same serialization, not exact float32-to-float64 values or a tolerance.
    """
    integer_ids(raw, KEY + ["sp", "sid"], "original sgsteps")
    indices = targets.sourceoriginalrowindex.to_numpy(dtype=np.uint64)
    if (indices >= len(raw)).any():
        raise ValueError("sourceoriginalrowindex outside original input table")
    rows = raw.iloc[indices.astype(np.int64)].reset_index(drop=True)
    integer_ids(rows, KEY + ["sp", "sid"], "bound original SGS")
    fg._finite(rows[KEY + ["sp", "sid", "t", "len"] + fg.ENDPOINTS], "bound original SGS")
    for name in KEY + ["sid", "sp"]:
        expected = targets["originalsp" if name == "sp" else name].to_numpy()
        if not np.array_equal(rows[name].to_numpy(), expected):
            raise ValueError(f"original SGS exact identity mismatch: {name}")
    serialized = source.csv_representation(rows)
    if not np.array_equal(serialized[fg.ENDPOINTS].to_numpy(float), targets[fg.ENDPOINTS].to_numpy(float)):
        raise ValueError("original SGS serialized endpoint payload mismatch")
    if (rows["len"] < 0).any() or not np.array_equal(rows["len"].to_numpy(float), targets.savedlen_mm.to_numpy(float)):
        raise ValueError("original SGS saved length mismatch")
    out = targets.copy()
    out["raw_sgs_t"] = rows.t.to_numpy()
    # Save original Mu2e-frame SGS endpoints as context, never use them to bind points.
    for end in (0, 1):
        for axis, offset in zip("xyz", origin):
            out[f"raw_sgs_{axis}{end}_mu2e_mm"] = rows[f"{axis}{end}"].to_numpy(float) + offset
    return out


def read_points(path):
    """An absent tree is a recorded availability outcome, not a failed audit."""
    with uproot.open(path) as root:
        present = f"{tables.DIR}/trajpts" in root
    if not present:
        empty = pd.DataFrame({c: pd.Series(dtype="uint64" if c in KEY + ["sp", "ipt"] else "float64")
                              for c in POINT_COLUMNS})
        return empty, "original dump/trajpts tree absent"
    points = tables.read(path, "trajpts", POINT_COLUMNS)
    return points, "original dump/trajpts tree present but zero points" if points.empty else "available"


def validate_points(points):
    integer_ids(points, KEY + ["sp", "ipt"], "native trajectory points")
    fg._finite(points[POINT_COLUMNS], "native trajectory points")
    if (points.ke < 0).any():
        raise ValueError("negative native trajectory kinetic energy")
    if points.duplicated(KEY + ["sp", "ipt"]).any():
        raise ValueError("duplicate native trajectory KEY/sp/ipt")


def segments(points):
    """Consecutive saved samples by exact particle, never a cross-particle gap."""
    p = points.sort_values(TRACK + ["ipt"], kind="stable").reset_index(drop=True).copy()
    group = p.groupby(TRACK, sort=False, observed=True)
    # fill_value avoids the float cast of uint64 indices caused by shift + NaN.
    p["nextipt"] = group.ipt.shift(-1, fill_value=0)
    for name in ("x", "y", "z", "t"):
        p["next" + name] = group[name].shift(-1)
    p = p.loc[group.cumcount() < group.ipt.transform("size") - 1].copy()
    delta = p[["nextx", "nexty", "nextz"]].to_numpy(float) - p[["x", "y", "z"]].to_numpy(float)
    length = np.linalg.norm(delta, axis=1)
    if not np.isfinite(length).all():
        raise ValueError("nonfinite trajectory segment length")
    p["segment_mm"] = length
    for i, axis in enumerate("xyz"):
        p["c" + axis] = np.divide(delta[:, i], length, out=np.zeros(len(p)), where=length > 0)
    return p


def intervals(candidates, geom):
    """Paired finite gas-inner minus wire-outer intervals; no all-straw search."""
    mapping = fg._geometry(geom)
    if not np.isfinite(geom.wire_r) or not 0 < geom.wire_r < geom.radius:
        raise ValueError("invalid wire outer radius")
    assigned = pd.Index(mapping.sid).get_indexer(candidates.sid)
    if (assigned < 0).any():
        raise ValueError("unknown exact target SID")
    own = assigned
    args = (candidates[["x", "y", "z"]].to_numpy(float) - geom.mid[own],
            candidates[["cx", "cy", "cz"]].to_numpy(float), geom.dir[own], geom.halflen[own])
    g0, g1 = fm.cylinder_intervals(*args, geom.radius)
    w0, w1 = fm.cylinder_intervals(*args, geom.wire_r)
    length = candidates.segment_mm.to_numpy(float)
    g0, g1 = np.maximum(g0, 0), np.minimum(g1, length)
    w0, w1 = np.maximum(w0, g0), np.minimum(w1, g1)
    haswire = w1 > w0
    # Without wire, all gas is the left interval. With wire, up to two pieces.
    starts = np.column_stack((g0, w1))
    ends = np.column_stack((np.where(haswire, w0, g1), g1))
    keep = (ends > starts) & np.column_stack((np.ones(len(length), dtype=bool), haswire))
    row, piece = np.nonzero(keep)
    result = candidates.iloc[row][["target", "ipt", "nextipt", "segment_mm"]].reset_index(drop=True)
    result["lo"], result["hi"] = starts[row, piece], ends[row, piece]
    if not np.isfinite(result[["lo", "hi"]].to_numpy()).all():
        raise ValueError("nonfinite derived gas-only intervals")
    return result


def summarize(points, targets, geom, prefix, after_entry=False, batch=32768):
    """Vectorized key-merge segment/target evaluation in bounded row batches.

    Only fixed-view and batch loops; no event, track, trajectory or straw loops.
    """
    tracks = targets[TRACK].drop_duplicates()
    p = points.rename(columns={"sp": "originalsp"}).merge(tracks, on=TRACK, validate="many_to_one")
    s = segments(p)
    # A different SGS time may exist for each targeted row of a single particle.
    count_rows = p.merge(targets[TRACK + ["target", "raw_sgs_t"]], on=TRACK, validate="many_to_many")
    if after_entry:
        count_rows = count_rows.loc[count_rows.t >= count_rows.raw_sgs_t]
    counts = count_rows.groupby("target", sort=False).agg(
        points=("ipt", "size"), min_ipt=("ipt", "min"), max_ipt=("ipt", "max"),
        min_t=("t", "min"), max_t=("t", "max"))
    for col in ("min_ipt", "max_ipt"):
        counts[col] = counts[col].astype("UInt64")
    parts, candidate_count, segment_counts = [], 0, []
    for first in range(0, len(s), batch):
        c = s.iloc[first:first + batch].merge(targets[TRACK + ["target", "sid", "raw_sgs_t"]],
                                             on=TRACK, validate="many_to_many", sort=False)
        if after_entry:
            c = c.loc[(c.t >= c.raw_sgs_t) & (c.nextt >= c.raw_sgs_t)]
        candidate_count += len(c)
        segment_counts.append(c.groupby("target").size().rename("segments"))
        c = c.loc[c.segment_mm > 0]
        if len(c):
            parts.append(intervals(c, geom))
    base = targets[["target"]].merge(counts, on="target", how="left", validate="one_to_one")
    base["points"] = base.points.fillna(0).astype(np.int64)
    nseg = pd.concat(segment_counts).groupby(level=0).sum() if segment_counts else pd.Series(dtype="int64", name="segments")
    nseg.index.name = "target"
    base = base.merge(nseg, on="target", how="left", validate="one_to_one")
    base["segments"] = base.segments.fillna(0).astype(np.int64)
    if parts and any(len(part) for part in parts):
        hits = pd.concat(parts, ignore_index=True).sort_values(["target", "ipt", "lo"], kind="stable")
        hits["gas_mm"] = hits.hi - hits.lo
        group = hits.groupby("target", sort=False)
        # Adjacent retained samples are not necessarily adjacent native ipt.
        previous_end_ipt = group.nextipt.shift(1, fill_value=0)
        adjacent_native = ((hits.nextipt - hits.ipt) == 1)
        previous_adjacent = adjacent_native.groupby(hits.target).shift(1, fill_value=False)
        connected = ((group.cumcount() > 0) & (previous_end_ipt == hits.ipt)
                     & adjacent_native & previous_adjacent & (hits.lo == 0)
                     & (group.hi.shift(1) == group.segment_mm.shift(1)))
        hits["runs"] = (~connected).astype(np.int64)
        stats = hits.groupby("target").agg(polyline_gas_mm=("gas_mm", "sum"),
            positive_intervals=("gas_mm", "size"), gas_segments=("ipt", "nunique"), disconnected_runs=("runs", "sum"))
        if not np.isfinite(stats.to_numpy(float)).all():
            raise ValueError("nonfinite aggregate polyline metrics")
        base = base.merge(stats, on="target", how="left", validate="one_to_one")
    for name in ("polyline_gas_mm", "positive_intervals", "gas_segments", "disconnected_runs"):
        if name not in base:
            base[name] = 0
        base[name] = base[name].fillna(0).astype(float if name == "polyline_gas_mm" else np.int64)
    # 0/1 point targets are unavailable, not an asserted zero physical gas path.
    base["availability"] = np.select([base.points == 0, base.points == 1],
        ["no exact-particle points in this fixed view", "one exact-particle point; no segment"], default="sampled polyline available")
    base["unresolved"] = True  # Geometry availability never resolves the physical gap's cause.
    return base.rename(columns={name: prefix + "_" + name for name in base if name != "target"}), candidate_count


def run(source_report, outdir):
    source_report, outdir = Path(source_report).resolve(), Path(outdir).resolve()
    if outdir.exists():
        raise FileExistsError(outdir)
    initial = fg._sha256(source_report)
    report = json.loads(source_report.read_text())
    paths = verify_source(report)
    paths.update(source_report=source_report, code=Path(__file__).resolve())
    for name, module in (("tables", tables), ("geometry", geometry), ("sourceaudit", source),
                         ("fieldgasledger", fg), ("fieldmaterial", fm)):
        paths[f"code/{name}"] = Path(module.__file__).resolve()
    if any(p == outdir or outdir in p.parents for p in paths.values()):
        raise ValueError("output directory overlaps frozen inputs")
    before = {name: fg._sha256(path) for name, path in paths.items()}
    if before["source_report"] != initial:
        raise ValueError("source report changed during receipt verification")
    verify_source(report)
    input_table = paths["source/inputs/input_table"]
    origin = np.asarray(report["origin_mm"], float)
    frozen = json.loads(paths["source/inputs/voids/inputs/fieldcompare"].read_text())
    if origin.shape != (3,) or not np.isfinite(origin).all() or not np.array_equal(origin, frozen["origin_mm"]):
        raise ValueError("source original origin mismatch")
    dtype = {k: "uint64" for k in TARGET + ["sp"]}
    saved = pd.read_csv(paths["source/outputs/all_outliers"], dtype=dtype, float_precision="round_trip")
    targets = targets_from_source(saved, report)
    raw = tables.read(input_table, "sgsteps", KEY + ["sp", "sid", "t", "len"] + fg.ENDPOINTS)
    targets = bind_sgs(targets, raw, origin)
    points, availability = read_points(input_table)
    validate_points(points)
    geom = geometry.from_gdml(str(paths["source/inputs/voids/inputs/gdml"]))
    # Validate even when no segments exist.
    mapping = fg._geometry(geom)
    if np.shape(geom.wire_r) != () or not np.isfinite(geom.wire_r) or not 0 < geom.wire_r < geom.radius:
        raise ValueError("invalid wire outer radius")
    if not targets.sid.isin(mapping.sid).all():
        raise ValueError("foreign exact target SID")
    evaluations = {}
    for prefix, after in (("full", False), ("after_sgs_entry", True)):
        per, evaluations[prefix] = summarize(points, targets, geom, prefix, after)
        targets = targets.merge(per, on="target", validate="one_to_one")
    for prefix in evaluations:
        targets[prefix + "_polyline_minus_stored_sgs_mm"] = targets[prefix + "_polyline_gas_mm"] - targets.savedlen_mm
    result = dict(passed=True, report_only=True, all_outliers_unresolved=True,
        trajectory_availability=availability, trajectory_tree=f"{tables.DIR}/trajpts",
        original_input_table=str(input_table), trajectory_total_points=len(points),
        source_outlier_rows=len(saved), unique_targets=len(targets),
        original_exact_particles=len(targets[TRACK].drop_duplicates()),
        arm_memberships={arm: int(targets[arm].sum()) for arm in ARMS},
        targets_with_zero_points=int((targets.full_points == 0).sum()),
        targets_with_one_point=int((targets.full_points == 1).sum()),
        targets_with_sampled_polyline=int((targets.full_points >= 2).sum()),
        segment_target_evaluations=evaluations, limitations=LIMITATIONS,
        raw_sgs_entry_times=targets[TARGET + ["raw_sgs_t"]].astype(object).to_dict("records"),
        identity="exact original uint64 KEY+originalsp only; no proximity; unique original KEY+originalsp+sid+sourceoriginalrowindex with both arm flags",
        coordinates="native original trajpts already Mu2e frame; raw detector SGS endpoints add unaltered original origin_mm",
        origin_mm=origin.tolist(), native_art_inventory_scope=report["art_inventory"]["presence_scope"],
        raw_art_decoded=False, physical_passages_certified=False, same_truth_surface_certified=False,
        point_index_scope="native nonnegative integer ipt, sorted by exact original particle; missing ipt gaps remain sampled secants but split geometric runs",
        time_scope="two fixed untuned views: full retained exact particle and points t>=raw SGS entry t; no exit time or SGS-span isolation")
    json.dumps(result, allow_nan=False)
    fg._check_hashes(paths, before)
    outdir.mkdir(parents=True, exist_ok=False)
    output = outdir / "fieldgaptrajectory_targets.csv.gz"
    targets.to_csv(output, index=False, compression={"method": "gzip", "mtime": 0})
    after = fg._check_hashes(paths, before)
    result.update(inputs_unchanged=before == after,
        inputs={k: dict(path=str(p), sha256_before=before[k], sha256_after=after[k]) for k, p in paths.items()},
        outputs={"targets": dict(path=str(output), sha256=fg._sha256(output))})
    with (outdir / "fieldgaptrajectory.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--outdir", required=True)
    print(json.dumps(run(**vars(parser.parse_args(argv))), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
