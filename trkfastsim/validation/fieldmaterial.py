"""Report-only wall coverage on actual fast chords and optional saved SGS endpoint secants.

No RNG, cuts, material corrections, or physics closure. Saved fast/G4 endpoint secants
are incomplete/nonphysical path coverage, not actual G4 trajectories or wall exposure.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from mu2efastsim import tables
from trkfastsim import crossings, geometry
from trkfastsim.sim import fieldsteps as fs
from trkfastsim.validation import fieldcompare, fieldcrossing as fc, fieldpathaudit as pa

_CHORD_INDEX = fs.ChordIndex
_STRAW_PIECES = fs._straw_pieces
ATOL_MM = 1e-7
MAX_ORACLE = 256
COVERAGE = ("actual stochastic fast chords observed at _straw_pieces; finite outer-minus-inner cylinder length "
            "versus wall length actually returned for consumption; report only, no correction or physics closure")
SECANT_COVERAGE = ("saved SGS endpoint secants only: incomplete/nonphysical path coverage, not actual stochastic "
                   "fast chords, actual G4 trajectories, or G4 wall exposure")
SEARCH_CAVEAT = ("outer candidates use production ChordIndex on the same geometry with radius replaced by r_out "
                 "and expanded panel bounds; not an independent geometry/index search; bounded all-straw oracle only")
IDENTITY = "call/local lane/straw index only; original event/particle IDs unavailable"


def cylinder_intervals(displacement, direction, axis, half, radius):
    """Vectorized full-line intervals inside finite cylinders, including axis-parallel rays and caps.

    Empty intervals have lower >= upper. Tangencies of zero length are empty.
    Inputs are paired candidate arrays, not a lane-by-all-straw allocation.
    """
    axial_x = np.einsum("ij,ij->i", displacement, axis)
    axial_c = np.einsum("ij,ij->i", direction, axis)
    transverse_x = displacement - axial_x[:, None] * axis
    transverse_c = direction - axial_c[:, None] * axis
    a = np.einsum("ij,ij->i", transverse_c, transverse_c)
    b = np.einsum("ij,ij->i", transverse_c, transverse_x)
    q = np.einsum("ij,ij->i", transverse_x, transverse_x) - radius ** 2
    parallel = a <= 1e-15
    disc = b * b - a * q
    valid = np.where(parallel, q <= 0, disc > 0)
    root = np.sqrt(np.maximum(disc, 0))
    denom = np.where(parallel, 1., a)
    lo = np.where(parallel, -np.inf, (-b - root) / denom)
    hi = np.where(parallel, np.inf, (-b + root) / denom)
    stationary = np.abs(axial_c) <= 1e-12
    cap_denom = np.where(stationary, 1., axial_c)
    cap0, cap1 = (-half - axial_x) / cap_denom, (half - axial_x) / cap_denom
    lo = np.maximum(lo, np.where(stationary, -np.inf, np.minimum(cap0, cap1)))
    hi = np.minimum(hi, np.where(stationary, np.inf, np.maximum(cap0, cap1)))
    valid &= ~stationary | (np.abs(axial_x) <= half)
    return np.where(valid, lo, np.inf), np.where(valid, hi, -np.inf)


def geometric_wall(x, c, length, geom, r_out, *, outer_geom=None, index=None, batch=65536):
    """Enumerate outer hits, then independently solve both finite cylinders on bounded candidate batches.

    Returned intervals are clamped to [0,L]; line_has_gas refers to the unbounded
    line intersecting the finite straw gas, NOT an infinite axial cylinder.
    """
    x, c, length = fc.validate_rays(x, c, length)
    fc._positive_integer(batch, "candidate batch")
    if not np.isfinite(r_out) or not 0 < geom.radius <= r_out:
        raise ValueError("invalid inner/outer radius")
    outer_geom = replace(geom, radius=float(r_out)) if outer_geom is None else outer_geom
    if outer_geom.radius != r_out:
        raise ValueError("outer geometry radius mismatch")
    index = _CHORD_INDEX(outer_geom) if index is None else index
    candidates = fc.canonical(index.crossings(x, c, length, outer_geom), length, geom.nstraws, "outer candidates")
    parts = []
    for start in range(0, len(candidates), batch):
        rows = candidates[start:start + batch]
        lane, straw = rows[:, 0].astype(np.int64), rows[:, 1].astype(np.int64)
        displacement = x[lane] - geom.mid[straw]
        args = (displacement, c[lane], geom.dir[straw], geom.halflen[straw])
        outer0, outer1 = cylinder_intervals(*args, r_out)
        gas0, gas1 = cylinder_intervals(*args, geom.radius)
        line_has_gas = gas1 > gas0
        outer0, outer1 = np.clip(outer0, 0, length[lane]), np.clip(outer1, 0, length[lane])
        gas0, gas1 = np.clip(gas0, 0, length[lane]), np.clip(gas1, 0, length[lane])
        gas_mm = np.maximum(gas1 - gas0, 0)
        outer_mm = np.maximum(outer1 - outer0, 0)
        parts.append(pd.DataFrame(dict(lane=lane, straw_index=straw, outer_in=outer0, outer_out=outer1,
                                       gas_in=gas0, gas_out=gas1, outer_mm=outer_mm, gas_mm=gas_mm,
                                       wall_mm=np.maximum(outer_mm - gas_mm, 0), line_has_gas=line_has_gas)))
    columns = ["lane", "straw_index", "outer_in", "outer_out", "gas_in", "gas_out", "outer_mm",
               "gas_mm", "wall_mm", "line_has_gas"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns).astype(
        {"lane": "int64", "straw_index": "int64", **{k: "float64" for k in columns[2:-1]}, "line_has_gas": "bool"})


def consumed_rows(sp, length, nstraws):
    """Read actual sp[4] without modifying or substituting the returned transport object."""
    columns = ["lane", "straw_index", "consumed_in", "consumed_out", "consumed_wall_mm", "consumed_gas_mm"]
    if sp is None:
        return pd.DataFrame(columns=columns).astype({k: "int64" if k in columns[:2] else "float64" for k in columns})
    if len(sp) != 7:
        raise ValueError("invalid _straw_pieces tuple")
    arrays = [np.asarray(v) for v in sp]
    if any(v.ndim != 1 or v.shape != arrays[0].shape or not np.isfinite(v).all() for v in arrays):
        raise ValueError("invalid/nonfinite consumed pieces")
    raw = np.column_stack(arrays[:4])
    canonical = fc.canonical(raw, length, nstraws, "consumed gas")
    if len(canonical) != len(raw):
        raise ValueError("nonpositive consumed gas")
    lane = arrays[0].astype(np.int64)
    if ((arrays[2] < 0).any() or (arrays[3] > length[lane]).any() or (arrays[4] < 0).any()):
        raise ValueError("consumed pieces outside finite chord or negative wall")
    return pd.DataFrame(dict(lane=lane, straw_index=arrays[1].astype(np.int64), consumed_in=arrays[2],
                             consumed_out=arrays[3], consumed_wall_mm=arrays[4], consumed_gas_mm=arrays[3] - arrays[2]))


class WallObserver:
    """Aggregate material ledger with bounded examples; inputs and original pieces remain untouched."""

    def __init__(self, oracle_chords=MAX_ORACLE):
        fc._positive_integer(oracle_chords, "oracle chords", allow_zero=True)
        if oracle_chords > MAX_ORACLE:
            raise ValueError("oracle chords must be <= 256")
        self.limit = oracle_chords
        self._outer_cache = {}
        self.stats = dict(calls=0, lanes=0, outer_hit_lanes=0, zero_outer_hit_lanes=0, outer_rows=0,
                          consumed_rows=0, consumed_wall_mm=0., geometric_outer_mm=0., geometric_gas_mm=0.,
                          geometric_wall_mm=0., consumed_gas_mm=0., matched_rows=0, matched_geometric_wall_mm=0.,
                          matched_consumed_wall_mm=0., missing_rows=0, missing_lanes=0, missing_wall_mm=0.,
                          matched_wall_mismatches=0, matched_wall_abs_discrepancy_mm=0.,
                          gas_mismatches=0, extra_consumed_rows=0, extra_consumed_wall_mm=0.,
                          oracle_chords=0, oracle_positive_lanes=0, oracle_rows=0, oracle_mismatches=0,
                          examples=[])
        for category in ("finite_chord_boundary", "gas_free_line_graze", "unexpected_missing_gas"):
            self.stats[category] = dict(rows=0, lanes=0, wall_mm=0.)

    def _example(self, frame, kind, call, x, c, length):
        room = fc.EXAMPLES - len(self.stats["examples"])
        if room <= 0 or frame.empty:
            return
        example = frame.head(room).copy()
        lanes = example.lane.to_numpy(dtype=np.int64)
        example["kind"] = kind
        example["call"] = call
        example["identity_scope"] = IDENTITY
        example["L_mm"] = length[lanes]
        for j, axis in enumerate("xyz"):
            example[f"x_{axis}"] = x[lanes, j]
            example[f"c_{axis}"] = c[lanes, j]
            example[f"end_{axis}"] = x[lanes, j] + c[lanes, j] * length[lanes]
        self.stats["examples"].extend(example.astype(object).where(pd.notna(example), None).to_dict("records"))

    def observe(self, x, c, length, geom, r_out, sp):
        x, c, length = fc.validate_rays(x, c, length)
        key = (id(geom), float(r_out))
        if key not in self._outer_cache:
            outer = replace(geom, radius=float(r_out))
            self._outer_cache[key] = (geom, outer, _CHORD_INDEX(outer))
        _, outer, index = self._outer_cache[key]
        g = geometric_wall(x, c, length, geom, r_out, outer_geom=outer, index=index)
        actual = consumed_rows(sp, length, geom.nstraws)
        d = g.merge(actual, on=["lane", "straw_index"], how="outer", indicator=True, validate="one_to_one")
        s = self.stats
        call = s["calls"]
        s["calls"] += 1
        s["lanes"] += len(x)
        s["outer_rows"] += len(g)
        s["consumed_rows"] += len(actual)
        hits = np.unique(g.lane.to_numpy(dtype=np.int64))
        s["outer_hit_lanes"] += len(hits)
        s["zero_outer_hit_lanes"] += len(x) - len(hits)
        for key_, column in (("geometric_outer_mm", "outer_mm"), ("geometric_gas_mm", "gas_mm"),
                              ("geometric_wall_mm", "wall_mm")):
            s[key_] += float(g[column].sum())
        s["consumed_wall_mm"] += float(actual.consumed_wall_mm.sum())
        s["consumed_gas_mm"] += float(actual.consumed_gas_mm.sum())
        both, missing, extra = d._merge == "both", d._merge == "left_only", d._merge == "right_only"
        s["matched_rows"] += int(both.sum())
        s["matched_geometric_wall_mm"] += float(d.loc[both, "wall_mm"].sum())
        s["matched_consumed_wall_mm"] += float(d.loc[both, "consumed_wall_mm"].sum())
        s["missing_rows"] += int(missing.sum())
        s["missing_lanes"] += int(d.loc[missing, "lane"].nunique())
        s["missing_wall_mm"] += float(d.loc[missing, "wall_mm"].sum())
        s["extra_consumed_rows"] += int(extra.sum())
        s["extra_consumed_wall_mm"] += float(d.loc[extra, "consumed_wall_mm"].sum())
        wall_error = np.abs(d.consumed_wall_mm - d.wall_mm)
        wall_bad = both & (wall_error > ATOL_MM)
        gas_bad = both & ((d.gas_mm <= 0) | (np.abs(d.consumed_gas_mm - d.gas_mm) > ATOL_MM) |
                          (np.abs(d.consumed_in - d.gas_in) > ATOL_MM) |
                          (np.abs(d.consumed_out - d.gas_out) > ATOL_MM))
        s["matched_wall_mismatches"] += int(wall_bad.sum())
        s["matched_wall_abs_discrepancy_mm"] += float(wall_error[both].sum())
        s["gas_mismatches"] += int(gas_bad.sum() + extra.sum() + (missing & (d.gas_mm > 0)).sum())
        for mask, kind in ((wall_bad, "matched_wall_discrepancy"), (gas_bad, "gas_discrepancy"),
                           (extra, "extra_consumed")):
            self._example(d.loc[mask].drop(columns="_merge"), kind, call, x, c, length)
        for category, mask in (("finite_chord_boundary", missing & (d.gas_mm == 0) & (d.line_has_gas == True)),
                               ("gas_free_line_graze", missing & (d.gas_mm == 0) & (d.line_has_gas == False)),
                               ("unexpected_missing_gas", missing & (d.gas_mm > 0))):
            stat = s[category]
            stat["rows"] += int(mask.sum())
            stat["lanes"] += int(d.loc[mask, "lane"].nunique())
            stat["wall_mm"] += float(d.loc[mask, "wall_mm"].sum())
            self._example(d.loc[mask].drop(columns="_merge"), category, call, x, c, length)
        selected = hits[:max(0, self.limit - s["oracle_chords"])]
        if len(selected):
            oracle = fc.oracle_crossings(x[selected], c[selected], length[selected], outer)
            s["oracle_chords"] += len(selected)
            s["oracle_positive_lanes"] += len(np.unique(oracle[:, 0]))
            s["oracle_rows"] += len(oracle)
            subset = g[g.lane.isin(selected)][["lane", "straw_index", "outer_in", "outer_out"]].to_numpy(float)
            subset[:, 0] = np.searchsorted(selected, subset[:, 0])
            diff = fc.compare_rows(subset, oracle, ["indexed_outer", "oracle_outer"])
            s["oracle_mismatches"] += len(diff)
            if len(diff):
                diff["lane"] = selected[diff.lane.to_numpy(dtype=np.int64)]
                self._example(diff, "outer_oracle_discrepancy", call, x, c, length)

    def report(self):
        s = self.stats
        partition = sum(s[k]["wall_mm"] for k in ("finite_chord_boundary", "gas_free_line_graze", "unexpected_missing_gas"))
        errors = dict(geometric_wall_mm=s["geometric_wall_mm"] - s["matched_geometric_wall_mm"] - s["missing_wall_mm"],
                      missing_wall_mm=s["missing_wall_mm"] - partition,
                      consumed_wall_mm=s["consumed_wall_mm"] - s["matched_consumed_wall_mm"] - s["extra_consumed_wall_mm"],
                      outer_minus_gas_mm=s["geometric_outer_mm"] - s["geometric_gas_mm"] - s["geometric_wall_mm"])
        consistent = (s["matched_wall_mismatches"] == s["gas_mismatches"] == s["oracle_mismatches"] == 0
                      and all(abs(v) <= ATOL_MM for v in errors.values())
                      and s["outer_rows"] == s["matched_rows"] + s["missing_rows"]
                      and s["consumed_rows"] == s["matched_rows"] + s["extra_consumed_rows"]
                      and s["missing_rows"] == sum(s[k]["rows"] for k in (
                          "finite_chord_boundary", "gas_free_line_graze", "unexpected_missing_gas")))
        return {**s, "coverage": COVERAGE, "candidate_search_caveat": SEARCH_CAVEAT,
                "oracle_selection": "deterministic first requested OUTER-HIT lanes in call/local-lane order; no vacuum-only prefix",
                "oracle_chords_requested": self.limit, "oracle_unchecked_outer_hit_lanes": s["outer_hit_lanes"] - s["oracle_chords"],
                "identity_scope": IDENTITY, "length_atol_mm": ATOL_MM, "length_rtol": 0,
                "lane_count_scope": "summed within-call unique lanes, not unique original particles",
                "partition_residuals_mm": errors, "consistent": consistent,
                "missing_wall_is_diagnostic_not_failure": True}


@contextmanager
def observing_material(observer):
    """Temporarily wrap only _straw_pieces; call it once and return its exact object, restoring even on error.

    Process-local; do not run concurrent transports or stack other material wrappers.
    ChordIndex is never subclassed/replaced, preserving production isinstance dispatch.
    """
    original = fs._straw_pieces

    def observed(x, c, L, geom, r_out, rw, rw_in, index=None):
        sp = original(x, c, L, geom, r_out, rw, rw_in, index)
        observer.observe(x, c, L, geom, r_out, sp)
        return sp

    try:
        fs._straw_pieces = observed
        yield observer
    finally:
        fs._straw_pieces = original


def audit_saved_secants(batches, geom, origin, *, oracle_chords=256, batch=512):
    """Offline endpoint geometry and original _straw_pieces comparison, never a G4 material measurement."""
    fc._positive_integer(batch, "secant batch")
    observer = WallObserver(oracle_chords)
    index = _CHORD_INDEX(geom)
    r_out = geom.wall_rmax if geom.wall_rmax is not None else geom.radius + .015
    rows, zero = 0, 0
    for frame in batches:
        for start in range(0, len(frame), batch):
            block = frame.iloc[start:start + batch]
            x, c, length, keep = fc.endpoint_secants(block[[f"{a}0" for a in "xyz"]],
                                                    block[[f"{a}1" for a in "xyz"]], origin)
            rows += len(block)
            zero += int((~keep).sum())
            if len(x):
                sp = _STRAW_PIECES(x, c, length, geom, r_out, geom.wire_r,
                                   max(0., geom.wire_r - geom.wire_plate_t), index)
                observer.observe(x, c, length, geom, r_out, sp)
    return {**observer.report(), "coverage": SECANT_COVERAGE, "saved_rows": rows, "zero_length_rows": zero}


def audit_saved_tables(field_dir, geom, origin, *, batch=512, oracle_chords=256):
    """Read all three saved arms without interpreting secant length as their physical material path."""
    return {label: audit_saved_secants([tables.read(str(Path(field_dir) / filename), "sgsteps")], geom, origin,
                                        batch=batch, oracle_chords=oracle_chords)
            for label, filename in fc.FILES.items()}


def run(field_dir, input_table, gdml, field_map, outdir, *, entry, oracle_chords=256):
    """Replay the frozen original first-entry injector, gate attribution on exact saved payload equality."""
    observer = WallObserver(oracle_chords)
    field_dir, input_table, gdml, field_map, outdir = map(Path, (field_dir, input_table, gdml, field_map, outdir))
    field_dir, outdir = field_dir.resolve(), outdir.resolve()
    if outdir == field_dir or field_dir in outdir.parents:
        raise ValueError("output must be separate from retained original directory")
    old = json.loads((field_dir / "fieldcompare.json").read_text())
    settings = pa.frozen_settings(old, entry)
    if Path(old["field"]).name != field_map.name:
        raise ValueError("map name differs from frozen fieldcompare report")
    paths = {"input_table": input_table, "gdml": gdml, "map_header": field_map}
    binary = field_map.with_suffix(".bin")
    if field_map.suffix != ".header" or not binary.is_file():
        raise ValueError("original G4BL map .header and adjacent .bin required")
    paths["map_adjacent_binary"] = binary
    for path in sorted(field_dir.rglob("*")):
        if path.is_file():
            paths[f"original/{path.relative_to(field_dir)}"] = path
    for filename in fc.FILES.values():
        if not (field_dir / filename).is_file():
            raise ValueError(f"missing original injector table: {filename}")
    modules = dict(observer=sys.modules[__name__], fieldsteps=fs, fieldcompare=fieldcompare,
                   fieldcrossing=fc, fieldpathaudit=pa, crossings=crossings, geometry=geometry, core_tables=tables,
                   core_ehelix=fs.ehelix, core_eloss=fs.eloss, core_transport=fs.tr,
                   core_materials=fs.materials, steps=fs.st)
    paths.update({f"code/{name}": Path(module.__file__) for name, module in modules.items()})
    before = pa.receipts(paths)
    outdir.mkdir(parents=True, exist_ok=False)
    result = dict(passed=False, coverage=COVERAGE, saved_secant_analysis="not run by CLI; offline APIs available",
                  original_field_dir=str(field_dir), frozen_settings=settings, explicit_original_entry_cli=entry,
                  runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, pandas=pd.__version__),
                  environment_scope="freeze original code/environment externally; hashes are partial receipts, not historical attestation",
                  final_state_comparison="not performed: saved injector has no final tree",
                  hashes_before=before, reproduction=dict(reproduced=False, tables={}))
    try:
        with observing_material(observer):
            fieldcompare.run([str(input_table)], str(outdir / "replay"), gdml=str(gdml),
                             field_map=str(field_map), **settings)
        checks = pa.compare_saved_tables(field_dir, outdir / "replay")
        result["reproduction"] = dict(reproduced=all(v["equal"] for v in checks.values()), tables=checks,
                                      scope="all three saved injector particles and sgsteps, EXACT ordered payloads")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        after = {}
        for name, path in paths.items():
            try:
                after[name] = dict(path=str(path.resolve()), sha256=pa.sha256(path))
            except OSError as exc:
                after[name] = dict(path=str(path.resolve()), error=str(exc))
        result["hashes_after"] = after
        result["inputs_unchanged"] = before == after
        result["material_stats"] = observer.report()
        result["comparisons_diagnostic_only"] = not (result["inputs_unchanged"] and result["reproduction"]["reproduced"])
        result["passed"] = (not result["comparisons_diagnostic_only"] and "error" not in result
                            and observer.stats["lanes"] > 0 and result["material_stats"]["consistent"])
        result["verdict"] = "report-only wall coverage; missing wall diagnostic, no tuning, correction, or physics closure"
        with (outdir / "fieldmaterial.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        if not result["passed"]:
            (outdir / "FAILED").write_text("Audit failed; retain replay and diagnostic material report.\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field-dir", required=True)
    ap.add_argument("--input-table", required=True)
    ap.add_argument("--gdml", required=True)
    ap.add_argument("--map", dest="field_map", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--entry", required=True, choices=["first"])
    ap.add_argument("--oracle-chords", type=int, default=256)
    result = run(**vars(ap.parse_args(argv)))
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
