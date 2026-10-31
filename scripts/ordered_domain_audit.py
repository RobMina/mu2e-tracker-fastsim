"""Saved-only domain diagnostic; selection bounds are not transport admission boundaries.

All forward rays are retained separately from overlaps with actual recorded spans.
No propagation, geometry repair, energy correction or acceptance inference is made.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trkfastsim import crossings, geometry, surfaces

KEY = ["run", "subrun", "event", "sp"]
POS = ["x", "y", "z"]
DIR = ["ux", "uy", "uz"]
DOMAINS = ("loader_aabb", "vd_cylinder")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _rays(origin, direction):
    o, d = np.asarray(origin, float), np.asarray(direction, float)
    if o.ndim != 2 or o.shape[1:] != (3,) or o.shape != d.shape:
        raise ValueError("rays require matching (N,3) arrays")
    if not np.isfinite(o).all() or not np.isfinite(d).all() or (np.linalg.norm(d, axis=1) == 0).any():
        raise ValueError("finite nonzero ray directions required")
    return o, d


def _forward(lo, hi, possible):
    entry = np.maximum(lo, 0.)
    hit = possible & (hi >= entry)
    return np.where(hit, entry, np.nan), np.where(hit, hi, np.nan), hit


def ray_box(origin, direction, lower, upper):
    """Closed AABB forward intervals, including tangencies; misses have NaN endpoints."""
    o, d = _rays(origin, direction)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all() or (upper <= lower).any():
        raise ValueError("finite positive box extents required")
    parallel = d == 0
    a, b = (lower - o) / np.where(parallel, 1., d), (upper - o) / np.where(parallel, 1., d)
    lo = np.max(np.where(parallel, -np.inf, np.minimum(a, b)), axis=1)
    hi = np.min(np.where(parallel, np.inf, np.maximum(a, b)), axis=1)
    possible = ~np.any(parallel & ((o < lower) | (o > upper)), axis=1)
    return _forward(lo, hi, possible)


def ray_cylinder(origin, direction, radius, z_front, z_back):
    """Closed finite solid cylinder in detector frame; stable quadratic roots, no tail cap."""
    o, d = _rays(origin, direction)
    if not np.isfinite([radius, z_front, z_back]).all() or radius <= 0 or z_back <= z_front:
        raise ValueError("finite positive cylinder extents required")
    a = np.sum(d[:, :2] ** 2, axis=1)
    h = np.sum(o[:, :2] * d[:, :2], axis=1)
    c = np.sum(o[:, :2] ** 2, axis=1) - radius ** 2
    cross = o[:, 0] * d[:, 1] - o[:, 1] * d[:, 0]
    discriminant = a * radius ** 2 - cross ** 2
    radial = a != 0
    q = -h - np.copysign(np.sqrt(np.maximum(discriminant, 0.)), h)
    root1 = q / np.where(radial, a, 1.)
    root2 = np.divide(c, q, out=np.zeros_like(c), where=q != 0)
    root2 = np.where(q == 0, -h / np.where(radial, a, 1.), root2)
    lo = np.where(radial, np.minimum(root1, root2), -np.inf)
    hi = np.where(radial, np.maximum(root1, root2), np.inf)
    axial = d[:, 2] != 0
    z0 = (z_front - o[:, 2]) / np.where(axial, d[:, 2], 1.)
    z1 = (z_back - o[:, 2]) / np.where(axial, d[:, 2], 1.)
    lo = np.maximum(lo, np.where(axial, np.minimum(z0, z1), -np.inf))
    hi = np.minimum(hi, np.where(axial, np.maximum(z0, z1), np.inf))
    possible = np.where(radial, discriminant >= 0, c <= 0)
    possible &= axial | ((o[:, 2] >= z_front) & (o[:, 2] <= z_back))
    return _forward(lo, hi, possible)


def split_interval(start, end, entry, exit, hit):
    """Lengths only: intersect supplied local span with the complete forward domain interval."""
    start, end, entry, exit = np.broadcast_arrays(start, end, entry, exit)
    if not np.isfinite(start).all() or not np.isfinite(end).all() or (start < 0).any() or (end < start).any():
        raise ValueError("finite ordered forward span required")
    hit = np.broadcast_to(np.asarray(hit, bool), start.shape)
    inside = np.where(hit, np.maximum(0., np.minimum(end, exit) - np.maximum(start, entry)), 0.)
    before = np.where(hit, np.maximum(0., np.minimum(end, entry) - start), 0.)
    after = np.where(hit, np.maximum(0., end - np.maximum(start, exit)), 0.)
    return dict(inside_mm=inside, outside_mm=end - start - inside,
                outside_before_entry_mm=before, outside_after_exit_mm=after,
                outside_no_forward_hit_mm=np.where(hit, 0., end - start))


def _identities(frame):
    for name in KEY:
        if name not in frame or frame[name].isna().any():
            raise ValueError("missing full run/subrun/event/sp identity")
        if len(frame) and frame[name].dtype.kind not in "iu":
            raise ValueError("identity columns must be integers, never rounded floats")


def prepare_vertices(vertices):
    _identities(vertices)
    v = vertices.sort_values(KEY + ["path"], kind="stable").reset_index(drop=True).copy()
    if v.duplicated(KEY + ["path"]).any():
        raise ValueError("duplicate full-identity vertex path")
    if not np.isfinite(v[["path"] + POS + DIR].to_numpy(float)).all() or (v.path < 0).any():
        raise ValueError("invalid vertex state")
    o, d = _rays(v[POS].to_numpy(float), v[DIR].to_numpy(float))
    if not np.allclose(np.linalg.norm(d, axis=1), 1., rtol=0, atol=1e-10):
        raise ValueError("unit directions required for mm intervals")
    g = v.groupby(KEY, sort=False)
    length = (g.path.shift(-1) - v.path).fillna(0.).to_numpy()
    next_pos = g[POS].shift(-1).to_numpy(float)
    use = length > 0
    if not np.allclose(next_pos[use], (o + d * length[:, None])[use], rtol=0, atol=1e-7):
        raise ValueError("vertex endpoints disagree with recorded arc/direction")
    v["vertex_index"] = np.arange(len(v))
    v["traversed_mm"] = length
    return v


def domain_columns(v, lower, upper, cylinder):
    """All vertices, including terminal/singleton vertices; terminal ray is diagnostic only."""
    out = v.copy()
    o, d = v[POS].to_numpy(float), v[DIR].to_numpy(float)
    intervals = (ray_box(geometry.to_mu2e(o), d, lower, upper), ray_cylinder(o, d, *cylinder))
    inside_origin = (np.all((geometry.to_mu2e(o) >= lower) & (geometry.to_mu2e(o) <= upper), axis=1),
                     (np.hypot(o[:, 0], o[:, 1]) <= cylinder[0]) & (o[:, 2] >= cylinder[1]) & (o[:, 2] <= cylinder[2]))
    for name, (entry, exit, hit), inside in zip(DOMAINS, intervals, inside_origin):
        out[f"{name}_origin_inside"] = inside
        out[f"{name}_forward_hit"] = hit
        out[f"{name}_entry_mm"] = entry
        out[f"{name}_exit_mm"] = exit
        out[f"{name}_forward_inside_mm"] = exit - entry
        out[f"{name}_tangent"] = hit & (entry == exit)
        for field, value in split_interval(np.zeros(len(v)), v.traversed_mm.to_numpy(), entry, exit, hit).items():
            out[f"{name}_{field}"] = value
    return out


def omission_splits(witnesses, v):
    """Use general cell_start_path minus slice start, not an assumed path-zero witness."""
    required = KEY + ["path", "cell_start_path_mm", "cell_end_path_mm", "state", "material", "source",
                      "ancestors_verified", "replaces_material"]
    if witnesses.empty:
        return pd.DataFrame(columns=required + [f"{n}_{f}" for n in DOMAINS for f in
                            ("inside_mm", "outside_mm", "outside_before_entry_mm", "outside_after_exit_mm",
                             "outside_no_forward_hit_mm")])
    _identities(witnesses)
    if not set(required).issubset(witnesses):
        raise ValueError("actual omission-witness schema required")
    joined = witnesses.merge(v, on=KEY + ["path"], how="left", validate="many_to_one", suffixes=("", "_vertex"))
    if joined.vertex_index.isna().any():
        raise ValueError("omission witness has no exact full-identity slice start")
    for name in POS + DIR:
        if name in witnesses and not np.allclose(joined[name], joined[f"{name}_vertex"], rtol=0, atol=1e-7):
            raise ValueError("omission witness state differs from saved vertex")
    start = joined.cell_start_path_mm.to_numpy(float) - joined.path.to_numpy(float)
    end = joined.cell_end_path_mm.to_numpy(float) - joined.path.to_numpy(float)
    if (end > joined.traversed_mm.to_numpy()).any() or (joined.traversed_mm <= 0).any():
        raise ValueError("omission cell extends outside its actual traversed slice")
    joined["cell_local_start_mm"], joined["cell_local_end_mm"] = start, end
    joined["cell_length_mm"] = end - start
    for name in DOMAINS:
        split = split_interval(start, end, joined[f"{name}_entry_mm"].to_numpy(),
                               joined[f"{name}_exit_mm"].to_numpy(), joined[f"{name}_forward_hit"].to_numpy())
        for field, value in split.items():
            joined[f"{name}_{field}"] = value
    return joined


def gas_entries(segments, v):
    _identities(segments)
    entries = segments.copy()
    entries["gas_entry_index"] = np.arange(len(entries))
    selected = KEY + ["path"] + POS + [f"{n}_origin_inside" for n in DOMAINS]
    entries = entries.merge(v[selected], left_on=KEY + ["path_mm"], right_on=KEY + ["path"],
                            how="left", validate="many_to_one", indicator=True)
    if (entries._merge != "both").any():
        raise ValueError("gas entry has no exact full-identity saved vertex")
    entries = entries.drop(columns="_merge")
    for name in DOMAINS:
        entries[f"{name}_outside"] = ~entries[f"{name}_origin_inside"].astype(bool)
    return entries


def lane_coverage(muons, v, entries, cells):
    _identities(muons)
    if muons.duplicated(KEY).any():
        raise ValueError("duplicate full muon identity")
    base = muons[KEY + ["path_interior_mm"]].copy()
    for frame in (v, entries, cells):
        unknown = frame[KEY].drop_duplicates().merge(base[KEY], on=KEY, how="left", indicator=True)
        if (unknown._merge != "both").any():
            raise ValueError("saved evidence identity absent from muons.csv")
    totals = v.assign(_positive_slice=v.traversed_mm > 0).groupby(KEY, sort=False).agg(vertices=("path", "size"), slices=("_positive_slice", "sum"),
                first_path_mm=("path", "min"), last_path_mm=("path", "max"), traversed_mm=("traversed_mm", "sum"))
    out = base.merge(totals.reset_index(), on=KEY, how="left", validate="one_to_one")
    for field in ("vertices", "slices", "traversed_mm"):
        out[field] = out[field].fillna(0)
    out["has_vertices"] = out.vertices > 0
    out["interior_recorded"] = out.path_interior_mm.notna()
    positive = out.path_interior_mm > 0
    if (positive & ~out.has_vertices).any() or not np.allclose(out.loc[positive, "last_path_mm"],
                out.loc[positive, "path_interior_mm"], rtol=0, atol=1e-8):
        raise ValueError("positive interior path lacks complete saved vertex coverage")
    for name in DOMAINS:
        sums = v.groupby(KEY)[[f"{name}_inside_mm", f"{name}_outside_mm"]].sum().reset_index()
        gas = entries.groupby(KEY).agg(**{f"{name}_gas_entries_outside": (f"{name}_outside", "sum")}).reset_index()
        out = out.merge(sums, on=KEY, how="left", validate="one_to_one").merge(gas, on=KEY, how="left", validate="one_to_one")
        for field in (f"{name}_inside_mm", f"{name}_outside_mm", f"{name}_gas_entries_outside"):
            out[field] = out[field].fillna(0)
    count = entries.groupby(KEY).size().rename("gas_entries").reset_index()
    out = out.merge(count, on=KEY, how="left", validate="one_to_one")
    out["gas_entries"] = out.gas_entries.fillna(0)
    if not cells.empty:
        fields = [f"{n}_{f}" for n in DOMAINS for f in ("inside_mm", "outside_mm")]
        sums = cells.groupby(KEY)[fields].sum().add_prefix("omitted_cell_").reset_index()
        count = cells.groupby(KEY).size().rename("omitted_cells").reset_index()
        unresolved = cells.assign(_unresolved=cells.state == "unresolved").groupby(KEY)._unresolved.sum()
        out = out.merge(sums, on=KEY, how="left", validate="one_to_one")
        out = out.merge(count, on=KEY, how="left", validate="one_to_one")
        out = out.merge(unresolved.rename("unresolved_omitted_cells").reset_index(), on=KEY,
                        how="left", validate="one_to_one")
    for field in ["omitted_cells", "unresolved_omitted_cells"] + [f"omitted_cell_{n}_{f}" for n in DOMAINS
                                                                  for f in ("inside_mm", "outside_mm")]:
        out[field] = out[field].fillna(0) if field in out else 0
    return out


def _check_witness_hashes(witness, directory, before):
    recorded = witness.get("input_sha256", {})
    for name in ("vertices.csv", "segments.csv"):
        matches = [value for path, value in recorded.items() if Path(path).name == name]
        if len(matches) != 1 or matches[0] != before[str(directory / name)]:
            raise ValueError(f"omission witness does not bind current {name}")


def run(directory, omission_witness, gdml, outdir):
    directory, omission_witness, gdml, outdir = map(lambda p: Path(p).resolve(),
                                                   (directory, omission_witness, gdml, outdir))
    if outdir.exists():
        raise FileExistsError(outdir)
    inputs = [directory / name for name in ("complete.json", "vertices.csv", "segments.csv", "muons.csv")]
    inputs += [omission_witness, gdml]
    sources = [Path(__file__).resolve(), *sorted(Path(geometry.__file__).parent.glob("*.py"))]
    before = {str(p): digest(p) for p in inputs}
    source_before = {str(p): digest(p) for p in sources}
    receipt = json.loads(inputs[0].read_text())
    if receipt.get("status") != "bounded_full_chain_complete":
        raise ValueError("completed full-chain directory required")
    witness = json.loads(omission_witness.read_text())
    _check_witness_hashes(witness, directory, before)
    gdml_receipt_hashes = [h for p, h in receipt.get("input_sha256", {}).items() if Path(p).suffix.lower() == ".gdml"]
    if gdml_receipt_hashes and (len(set(gdml_receipt_hashes)) != 1 or gdml_receipt_hashes[0] != before[str(gdml)]):
        raise ValueError("explicit GDML differs from producer receipt")
    geom = geometry.from_gdml(gdml)
    lo, hi = crossings.panel_boxes(geom)
    lower, upper = lo.min(0) - 300., hi.max(0) + 300.
    cylinder = (surfaces.R_OUT, surfaces.Z_FRONT, surfaces.Z_BACK)
    v = domain_columns(prepare_vertices(pd.read_csv(inputs[1])), lower, upper, cylinder)
    entries = gas_entries(pd.read_csv(inputs[2]), v)
    cells = omission_splits(pd.DataFrame(witness["witnesses"]), v)
    lanes = lane_coverage(pd.read_csv(inputs[3]), v, entries, cells)
    length_fields = ["cell_length_mm"] + [f"{n}_{f}" for n in DOMAINS for f in ("inside_mm", "outside_mm")]
    omission_totals = [] if cells.empty else cells.groupby(["state", "material", "ancestors_verified"], dropna=False)[
        length_fields].sum().reset_index().to_dict("records")
    result = dict(status="saved_domain_audit_complete", admitted_for_production=False,
        scope="saved full-chain vertices, emitted gas entries and actual omission cells only; no transport clipping",
        warning="AABB selects loader geometry, not transport extent/admission. VD cylinder is a named diagnostic domain. "
                "Forward terminal rays are hypothetical, never traversed. Unresolved cells remain unresolved. "
                "Omission sums are per witness/placement and may overlap; no material/debit correction or rate inference.",
        domains=dict(loader_aabb=dict(frame="Mu2e", lower_mm=lower.tolist(), upper_mm=upper.tolist(),
                                      definition="panel_boxes min/max +/-300 mm, exact default TrackerInterior loader selection"),
                     vd_cylinder=dict(frame="detector", radius_mm=cylinder[0], z_front_mm=cylinder[1], z_back_mm=cylinder[2],
                                      definition="solid cylinder from named surfaces.R_OUT/Z_FRONT/Z_BACK; not annulus")),
        boundary_policy="closed domains; tangencies have zero length; no tolerance padding or arbitrary ray horizon",
        coverage=dict(muon_lanes=len(lanes), lanes_with_vertices=int(lanes.has_vertices.sum()),
                      lanes_without_vertices=int((~lanes.has_vertices).sum()), vertices=len(v),
                      traversed_slices=int((v.traversed_mm > 0).sum()), gas_entries=len(entries), omission_cells=len(cells)),
        gas_entries_outside={n: int(entries[f"{n}_outside"].sum()) for n in DOMAINS},
        omitted_cell_totals_by_state_material_ancestry=omission_totals,
        producer_receipt=receipt, omission_witness_provenance={k: value for k, value in witness.items() if k != "witnesses"},
        gdml_bound_by_producer_receipt=bool(gdml_receipt_hashes), input_sha256_before=before,
        source_sha256_before=source_before)
    outdir.mkdir(parents=True, exist_ok=False)
    for name, frame in (("vertices", v), ("lanes", lanes), ("gas_entries", entries), ("omitted_cells", cells)):
        with (outdir / f"{name}.csv").open("x") as stream:
            frame.to_csv(stream, index=False)
    after = {str(p): digest(p) for p in inputs}
    source_after = {str(p): digest(p) for p in sources}
    if before != after or source_before != source_after:
        raise ValueError("saved inputs or diagnostic sources changed; output is incomplete")
    result.update(input_sha256_after=after, source_sha256_after=source_after,
                  output_sha256={p.name: digest(p) for p in sorted(outdir.glob("*.csv"))})
    with (outdir / "complete.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="completed bounded_full_chain_complete saved directory")
    parser.add_argument("--omission-witness", required=True, help="explicit ordered_omission_witness JSON")
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--outdir", required=True, help="new directory; existing paths refused")
    args = parser.parse_args()
    print(json.dumps(run(args.directory, args.omission_witness, args.gdml, args.outdir), indent=2, allow_nan=False))
