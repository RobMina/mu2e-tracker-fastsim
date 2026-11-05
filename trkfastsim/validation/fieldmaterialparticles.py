"""Frozen-code diagnostic of wall exposure on actual fast chords, by ORIGINAL entry KEY/sp.

MEANMODELGEOMETRYDIAGNOSTIC: collision means conditioned on current truth-fast-chord
momentum only; NOT realized sampled loss, G4 energy, a correction, or a cause.
Excludes radiative losses and every material other than Mylar/Al6061 straw wall.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from mu2efastsim import tables
from trkfastsim import crossings, geometry
from trkfastsim.sim import fieldsteps as fs
from trkfastsim.validation import cecontract, fieldcompare, fieldcrossing as fc
from trkfastsim.validation import fieldmaterial as fm, fieldmultiplicity, fieldpathaudit as pa

TRACK = tables.KEY + ["sp"]
ATOL_MM = 1e-7
LABEL = "MEANMODELGEOMETRYDIAGNOSTIC"
SCOPE = (__doc__.strip() + " Original final state unavailable; no final-state equality claim. "
         "Outer search shares production ChordIndex; not independent navigation or an all-material ledger.")
COUNTS = ("nactualchords", "gas_rows", "geometric_rows", "missing_rows", "boundary_rows", "graze_rows")
LENGTHS = ("geometric_outer_mm", "geometric_gas_mm", "consumed_gas_mm", "geometric_wall_mm",
           "consumed_wall_mm", "missing_wall_mm", "boundary_wall_mm", "graze_wall_mm",
           "unexpected_missing_wall_mm")
MODELS = ("geometric_wall_x0", "consumed_wall_x0", "missing_wall_x0",
          "mean_missing_collision_loss_MeV", "mean_consumed_collision_loss_MeV")


def _indices(act, n, lanes):
    a = np.asarray(act)
    if (a.ndim != 1 or len(a) != lanes or not np.issubdtype(a.dtype, np.integer)
            or np.issubdtype(a.dtype, np.bool_) or (a < 0).any() or (a >= n).any()
            or len(np.unique(a)) != len(a)):
        raise ValueError("invalid/duplicate/foreign active indices; cannot infer act")
    return a.astype(np.int64, copy=True)


def _kinematics(pa_, ma, lanes):
    p, m = np.array(pa_, dtype=float, copy=True), np.array(ma, dtype=float, copy=True)
    if (p.shape != (lanes,) or m.shape != (lanes,) or not np.isfinite(p).all()
            or not np.isfinite(m).all() or (p <= 0).any() or (m <= 0).any()):
        raise ValueError("invalid/nonfinite current pa/ma")
    return p, m


class ParticleWallObserver:
    """Array-only per-entry counters; observe is usable without frame introspection."""

    def __init__(self, oracle_chords=256):
        self.global_observer = fm.WallObserver(oracle_chords)
        self._outer_cache = self.global_observer._outer_cache
        self._denominator = None
        self._values = {}
        self.transport_calls = 0
        self.options = None

    @property
    def ledger(self):
        if self._denominator is None:
            return None
        return pd.concat([self._denominator, pd.DataFrame(self._values)], axis=1)

    def begin(self, start, options=None):
        if self._denominator is not None:
            raise ValueError("exactly one original entry transport required")
        den = fieldmultiplicity.track_keys(start, "ORIGINAL transport start")
        if (den.empty or den.duplicated(TRACK).any() or (den.sp == 0).any()
                or (den[tables.KEY] == 0).all(axis=1).any()):
            raise ValueError("empty/duplicate/zero synthetic original identities forbidden")
        momentum = den[["px", "py", "pz"]].to_numpy(dtype=float, copy=True)
        pentry = np.linalg.norm(momentum, axis=1)
        if not np.isfinite(pentry).all():
            raise ValueError("nonfinite original entry momentum")
        self._denominator = den[TRACK].reset_index(drop=True).copy()
        self._denominator["pentry_MeV_c"] = pentry
        for name in COUNTS:
            self._values[name] = np.zeros(len(den), dtype=np.int64)
        for name in LENGTHS + MODELS:
            self._values[name] = np.zeros(len(den))
        self.options = options
        self.transport_calls += 1

    def observe(self, act, pa_, ma, x, c, length, geom, r_out, sp):
        if self._denominator is None:
            raise ValueError("missing original start denominator")
        x, c, length = (np.array(v, dtype=float, copy=True) for v in (x, c, length))
        x, c, length = fc.validate_rays(x, c, length)
        act = _indices(act, len(self._denominator), len(x))
        p, m = _kinematics(pa_, ma, len(x))
        pieces = None if sp is None else tuple(np.array(v, copy=True) for v in sp)
        key = (id(geom), float(r_out))
        if key not in self._outer_cache:
            outer = replace(geom, radius=float(r_out))
            self._outer_cache[key] = (geom, outer, fm._CHORD_INDEX(outer))
        _, outer, index = self._outer_cache[key]
        g = fm.geometric_wall(x, c, length, geom, r_out, outer_geom=outer, index=index)
        a = fm.consumed_rows(pieces, length, geom.nstraws)
        d = g.merge(a, on=["lane", "straw_index"], how="outer", indicator=True, validate="one_to_one")
        missing = d._merge == "left_only"
        boundary = missing & (d.gas_mm == 0) & (d.line_has_gas == True)
        graze = missing & (d.gas_mm == 0) & (d.line_has_gas == False)
        unexpected = missing & (d.gas_mm > 0)
        # Material clones isolate cached density properties from production material objects.
        mylar, metal = replace(fs.materials.MYLAR), replace(fs.materials.ALUMINIUM_6061)
        split = min(max(geom.metal_t / max(r_out - geom.radius, 1e-9), 0.), 1.)
        x0rate = (1 - split) / fs.eloss.radiation_length_mm(mylar) + split / fs.eloss.radiation_length_mm(metal)
        meanrate = ((1 - split) * fs.eloss.dedx_restricted(mylar, p, 1e9, m)
                    + split * fs.eloss.dedx_restricted(metal, p, 1e9, m))
        if not np.isfinite(meanrate).all() or not np.isfinite(x0rate):
            raise ValueError("nonfinite pure mean wall model")

        def add(name, lane, values):
            np.add.at(self._values[name], act[np.asarray(lane, dtype=np.int64)], np.asarray(values))

        add("nactualchords", np.arange(len(x)), 1)
        add("gas_rows", a.lane, 1)
        add("geometric_rows", g.lane, 1)
        for name, mask in (("missing", missing), ("boundary", boundary), ("graze", graze)):
            add(name + "_rows", d.loc[mask, "lane"], 1)
        for name, frame, column in (("geometric_outer_mm", g, "outer_mm"), ("geometric_gas_mm", g, "gas_mm"),
                                     ("geometric_wall_mm", g, "wall_mm"), ("consumed_wall_mm", a, "consumed_wall_mm"),
                                     ("consumed_gas_mm", a, "consumed_gas_mm")):
            add(name, frame.lane, frame[column])
        for name, mask in (("missing", missing), ("boundary", boundary), ("graze", graze), ("unexpected_missing", unexpected)):
            add(name + "_wall_mm", d.loc[mask, "lane"], d.loc[mask, "wall_mm"])
        for name, frame, column in (("geometric", g, "wall_mm"), ("consumed", a, "consumed_wall_mm"),
                                     ("missing", d.loc[missing], "wall_mm")):
            lanes = frame.lane.to_numpy(dtype=np.int64)
            wall = frame[column].to_numpy(dtype=float)
            add(name + "_wall_x0", lanes, wall * x0rate)
            if name != "geometric":
                add("mean_" + name + "_collision_loss_MeV", lanes, wall * meanrate[lanes])
        self.global_observer.observe(x, c, length, geom, r_out, pieces)

    def report(self):
        global_ = self.global_observer.report()
        if self._denominator is None:
            return dict(consistent=False, entries=0, global_material=global_)
        d = self.ledger
        residuals = dict(wall=d.geometric_wall_mm - d.consumed_wall_mm - d.missing_wall_mm,
                         missing=d.missing_wall_mm - d.boundary_wall_mm - d.graze_wall_mm - d.unexpected_missing_wall_mm,
                         outer=d.geometric_outer_mm - d.geometric_gas_mm - d.geometric_wall_mm)
        maxima = {k: float(np.max(np.abs(v))) for k, v in residuals.items()}
        sums = {k: float(v.sum()) for k, v in residuals.items()}
        totals = {k: int(d[k].sum()) if k in COUNTS else float(d[k].sum()) for k in COUNTS + LENGTHS + MODELS}
        global_residuals = {k: totals[k] - global_[k] for k in LENGTHS if k in global_}
        global_residuals.update({name + "_wall_mm": totals[name + "_wall_mm"] - global_[category]["wall_mm"]
                                 for name, category in (("boundary", "finite_chord_boundary"),
                                                        ("graze", "gas_free_line_graze"),
                                                        ("unexpected_missing", "unexpected_missing_gas"))})
        consistent = (global_["consistent"] and self.transport_calls == 1
                      and all(abs(v) <= ATOL_MM for v in global_residuals.values())
                      and totals["geometric_rows"] == global_["outer_rows"]
                      and totals["missing_rows"] == global_["missing_rows"]
                      and totals["boundary_rows"] == global_["finite_chord_boundary"]["rows"]
                      and totals["graze_rows"] == global_["gas_free_line_graze"]["rows"]
                      and all(v <= ATOL_MM for v in maxima.values())
                      and all(abs(v) <= ATOL_MM for v in sums.values())
                      and totals["nactualchords"] == global_["lanes"]
                      and totals["gas_rows"] == global_["consumed_rows"])
        return dict(consistent=bool(consistent), entries=len(d), zero_chord_entries=int((d.nactualchords == 0).sum()),
                    zero_consumed_wall_entries=int((d.consumed_wall_mm == 0).sum()), totals=totals,
                    partition_max_abs_mm=maxima, partition_sum_mm=sums, global_sum_residuals_mm=global_residuals,
                    global_material=global_,
                    transport_calls=self.transport_calls, transport_options=self.options, label=LABEL, scope=SCOPE,
                    identity="ORIGINAL native uint64 KEY/sp, not injector renumbering; no rmap inferred",
                    length_atol_mm=ATOL_MM, length_rtol=0)


@contextmanager
def observing_particles(observer, expected_settings=None):
    """Fragile frozen-code frame diagnostic; fail closed, never infer active lane identities."""
    original_transport, original_straw = fs.transport, fs._straw_pieces
    frozen_code = original_transport.__code__
    signature = inspect.signature(original_transport)
    active_start = None

    def transport(*args, **kwargs):
        nonlocal active_start
        if active_start is not None:
            raise ValueError("nested/concurrent transport unsupported")
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        opts = bound.arguments
        start = opts["start"]
        gas = fs.st.STRAW_GAS if opts["gas"] is None else opts["gas"]
        frozen = dict(chord_mm=float(opts["chord_mm"]), sagitta_mm=float(opts["sagitta_mm"]),
                      radiative=bool(opts["radiative"]), gas_state=gas.state, gas_name=gas.name)
        for name, default in (("max_path_mm", 30000.), ("max_chords", 20000), ("wire", True),
                              ("tail", True), ("tail_k", None), ("msc", True), ("straws", True)):
            frozen[name] = opts.get(name, default)
        frozen["custom_loss"] = opts["loss"] is not None
        frozen["stop_present"] = opts.get("stop") is not None
        if "origin" in opts:
            frozen["origin_mm"] = np.asarray(opts["origin"], dtype=float).tolist()
        if opts.get("bnom") is not None:
            frozen["bnom_T"] = np.asarray(opts["bnom"], dtype=float).tolist()
        if expected_settings is not None:
            expected_gas = fs.STRAW_GAS_G4STATE if expected_settings["gas_state"] == "gas" else fs.st.STRAW_GAS
            if (opts["loss"] is not None or gas is not expected_gas
                    or frozen["chord_mm"] != expected_settings["chord_mm"]
                    or frozen["sagitta_mm"] != expected_settings["sagitta_mm"]
                    or frozen["radiative"] != expected_settings["mu_radiative"]):
                raise ValueError("transport options/gas differ from frozen replay")
            defaults = dict(max_path_mm=30000., max_chords=20000, wire=True, tail=True,
                            tail_k=None, msc=True, straws=True, stop_present=False)
            if (any(frozen[k] != v for k, v in defaults.items())
                    or ("origin_mm" in frozen and frozen["origin_mm"] != expected_settings["origin"])):
                raise ValueError("non-default transport controls differ from frozen fieldcompare flow")
        observer.begin(start, frozen)
        active_start = start
        try:
            return original_transport(*args, **kwargs)
        finally:
            active_start = None

    def straw(x, c, L, geom, r_out, rw, rw_in, index=None):
        sp = original_straw(x, c, L, geom, r_out, rw, rw_in, index)
        frame = inspect.currentframe()
        caller = None
        try:
            caller = frame.f_back
            if caller is None or caller.f_code is not frozen_code or active_start is None:
                raise ValueError("unexpected caller context; frozen original transport frame required")
            loc = caller.f_locals
            if (loc.get("xa") is not x or loc.get("c") is not c or loc.get("L") is not L
                    or loc.get("start") is not active_start):
                raise ValueError("unexpected caller locals/start identity")
            observer.observe(loc.get("act"), loc.get("pa"), loc.get("ma"), x, c, L, geom, r_out, sp)
        finally:
            del caller, frame
        return sp

    try:
        fs.transport, fs._straw_pieces = transport, straw
        yield observer
    finally:
        fs.transport, fs._straw_pieces = original_transport, original_straw


def compare_material_totals(current, prior):
    """Same global wall gate, absolute 1e-7 mm only; counts and schema exact."""
    template = {k: v for k, v in fm.WallObserver(0).stats.items() if k != "examples"}
    template["oracle_chords_requested"] = 0
    failures = {}

    def safe(value):
        return value if isinstance(value, (int, float)) and np.isfinite(value) else repr(value)

    def compare(left, right, schema, prefix=""):
        if not isinstance(left, dict) or not isinstance(right, dict):
            failures[prefix] = "missing/non-object material totals"
            return
        for key, expected in schema.items():
            name = prefix + key
            if key not in left or key not in right:
                failures[name] = "missing material total"
                continue
            x, y = left[key], right[key]
            if isinstance(expected, dict):
                compare(x, y, expected, name + "/")
            elif isinstance(expected, int):
                if (isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, int)
                        or not isinstance(y, int) or x != y):
                    failures[name] = dict(prior=safe(y), current=safe(x))
            elif (isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float))
                  or not isinstance(y, (int, float)) or not np.isfinite(x) or not np.isfinite(y)
                  or abs(x - y) > ATOL_MM):
                failures[name] = dict(prior=safe(y), current=safe(x))

    compare(current, prior, template)
    return dict(equal=not failures and current.get("consistent") is True and prior.get("consistent") is True,
                mismatches=failures, length_atol_mm=ATOL_MM, length_rtol=0)


def _prior_evidence(prior, paths, field_dir):
    if (prior.get("passed") is not True or prior.get("inputs_unchanged") is not True
            or prior.get("reproduction", {}).get("reproduced") is not True
            or prior.get("material_stats", {}).get("consistent") is not True
            or prior.get("comparisons_diagnostic_only") is not False
            or Path(prior.get("original_field_dir", "")).resolve() != field_dir):
        raise ValueError("prior fieldmaterial report not passed/consistent/closed")
    checks = prior.get("reproduction", {}).get("tables", {})
    required_checks = {f"{arm}/{tree}" for arm in fc.FILES for tree in ("particles", "sgsteps")}
    if set(checks) != required_checks or any(v.get("equal") is not True for v in checks.values()):
        raise ValueError("prior material report requires all six exact payload equality checks")
    if prior.get("frozen_settings") != pa.frozen_settings(
            json.loads((field_dir / "fieldcompare.json").read_text()), "first"):
        raise ValueError("prior material frozen settings differ from original fieldcompare")
    before, after = prior.get("hashes_before", {}), prior.get("hashes_after", {})
    if not before or before != after:
        raise ValueError("prior material before/after hash evidence not closed")
    required = {k for k in paths if k.startswith("original/") or not k.startswith("code/")}
    required.discard("material_report")
    required.update("code/" + k for k in ("observer", "fieldsteps", "fieldcompare", "fieldcrossing", "fieldpathaudit",
                                        "crossings", "geometry", "core_tables", "core_ehelix", "core_eloss",
                                        "core_transport", "core_materials", "steps"))
    for name in required:
        expected = Path(fm.__file__) if name == "code/observer" else paths[name]
        receipt = before.get(name, {})
        same_path = Path(receipt.get("path", "")).resolve() == expected.resolve()
        if ((not name.startswith("code/") and not same_path)
                or receipt.get("sha256") != pa.sha256(expected)):
            raise ValueError(f"prior material evidence path/hash mismatch: {name}")
    for name, receipt in before.items():
        path = Path(receipt["path"])
        if receipt.get("sha256") != pa.sha256(path):
            raise ValueError(f"prior material receipt changed: {name}")
        paths["prior_receipt/" + name] = path


def run(field_dir, input_table, gdml, field_map, material_report, outdir, *, entry):
    field_dir, input_table, gdml, field_map, material_report, outdir = (
        Path(v).resolve() for v in (field_dir, input_table, gdml, field_map, material_report, outdir))
    sources = (field_dir, input_table, gdml, field_map, material_report)
    if any(outdir == p or p in outdir.parents or outdir in p.parents for p in sources):
        raise ValueError("output must be exclusive and separate from retained evidence")
    if outdir.exists():
        raise FileExistsError(outdir)
    old = json.loads((field_dir / "fieldcompare.json").read_text())
    settings = pa.frozen_settings(old, entry)
    if Path(old["field"]).name != field_map.name:
        raise ValueError("map name differs from frozen fieldcompare report")
    if (not isinstance(old.get("entries"), int) or isinstance(old["entries"], bool) or old["entries"] <= 0):
        raise ValueError("original fieldcompare entry denominator count required")
    binary = field_map.with_suffix(".bin")
    if field_map.suffix != ".header" or not binary.is_file():
        raise ValueError("original map .header and adjacent .bin required")
    prior = json.loads(material_report.read_text())
    observer = ParticleWallObserver(prior.get("material_stats", {}).get("oracle_chords_requested", 256))
    paths = dict(input_table=input_table, gdml=gdml, map_header=field_map,
                 map_adjacent_binary=binary, material_report=material_report)
    for path in sorted(field_dir.rglob("*")):
        if path.is_file():
            paths[f"original/{path.relative_to(field_dir)}"] = path
    for filename in fc.FILES.values():
        if not (field_dir / filename).is_file():
            raise ValueError(f"missing original injector table: {filename}")
    modules = dict(observer=sys.modules[__name__], fieldmaterial=fm, fieldsteps=fs, fieldcompare=fieldcompare,
                   fieldcrossing=fc, fieldpathaudit=pa, fieldmultiplicity=fieldmultiplicity, cecontract=cecontract,
                   crossings=crossings, geometry=geometry, core_tables=tables, core_ehelix=fs.ehelix,
                   core_eloss=fs.eloss, core_transport=fs.tr, core_materials=fs.materials, steps=fs.st)
    paths.update({f"code/{name}": Path(module.__file__) for name, module in modules.items()})
    outdir.mkdir(parents=True, exist_ok=False)
    result = dict(passed=False, label=LABEL, scope=SCOPE, original_field_dir=str(field_dir),
                  material_report=str(material_report), frozen_settings=settings, explicit_original_entry_cli=entry,
                  final_state_comparison="not performed: ORIGINAL final state unavailable",
                  runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, pandas=pd.__version__),
                  environment_scope="partial local receipts, not historical code/environment attestation",
                  reproduction=dict(reproduced=False, tables={}))
    before = {}
    try:
        before = pa.receipts(paths)
        result["hashes_before"] = before
        _prior_evidence(prior, paths, field_dir)
        before = pa.receipts(paths)
        result["hashes_before"] = before
        result["prior_material_evidence_verified"] = True
        prior_gate = compare_material_totals(prior["material_stats"], prior["material_stats"])
        if not prior_gate["equal"]:
            result["global_material_comparison"] = prior_gate
            raise ValueError("prior material totals incomplete/nonfinite")
        with observing_particles(observer, settings):
            fieldcompare.run([str(input_table)], str(outdir / "replay"), gdml=str(gdml), field_map=str(field_map), **settings)
        if observer.ledger is None or observer.report()["entries"] != old.get("entries"):
            raise ValueError("observed entry denominator count differs from original fieldcompare entries")
        checks = pa.compare_saved_tables(field_dir, outdir / "replay")
        result["reproduction"] = dict(reproduced=len(checks) == 6 and all(v["equal"] for v in checks.values()),
                                      tables=checks, scope="EXACT ordered all six particles/sgsteps saved payloads")
        result["global_material_comparison"] = compare_material_totals(observer.global_observer.report(), prior["material_stats"])
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
        result["particle_material_stats"] = observer.report()
        result["passed"] = bool("error" not in result and result["inputs_unchanged"]
                                and result["reproduction"]["reproduced"]
                                and result.get("global_material_comparison", {}).get("equal")
                                and result["particle_material_stats"]["consistent"]
                                and observer.global_observer.stats["lanes"] > 0)
        result["interpretation_allowed"] = result["passed"]
        result["verdict"] = ("report-only geometry and pure collision mean diagnostic; no correction or causal claim"
                             if result["passed"] else "FAILED: retain artifacts; no interpretation")
        ledger = observer.ledger if observer.ledger is not None else pd.DataFrame(columns=TRACK + ["pentry_MeV_c"] + list(COUNTS + LENGTHS + MODELS))
        with (outdir / "particles.csv").open("x") as stream:
            ledger.to_csv(stream, index=False)
        with (outdir / "fieldmaterialparticles.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        if not result["passed"]:
            (outdir / "FAILED").write_text("No interpretation; retain replay, ledger and failed evidence.\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ("field-dir", "input-table", "gdml", "material-report", "outdir"):
        ap.add_argument("--" + flag, required=True)
    ap.add_argument("--map", dest="field_map", required=True)
    ap.add_argument("--entry", required=True, choices=["first"])
    result = run(**vars(ap.parse_args(argv)))
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
