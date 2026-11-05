"""Development-only enabled-wall cosmic pilot on frozen original entry states.

Exactly one candidate transport, no G4/reco/default changes. The retained G4 clock
arm stays fixed: candidate exit clocks never redefine the reconstruction denominator.
Direct consumer observation is passive, but its timings are not production throughput.
"""
import argparse
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from mu2efastsim import tables
from trkfastsim import geometry
from trkfastsim.sim import fieldsteps as fs, fieldwall as fw
from trkfastsim.validation import fieldcompare as fc, fieldpathaudit as pa, fieldsourceaudit as sa

TRACK = tables.KEY + ["sp"]
ATOL = 1e-7
PHYSICS_SHA = {"fieldsteps": "6bc76fbcc44a2ea1d070d34bac9e58294ede77173ad607421c4bd81e73e6ccb0",
               "fieldwall": "2c446b8c855cd375eab19516300e8f61f480b320f7b815444130c0753096ae1a"}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def equal(a, b, label):
    if not np.allclose(a, b, rtol=0, atol=ATOL, equal_nan=False):
        raise ValueError(f"enabled consumer mismatch: {label}")


class ConsumerLedger:
    """Per-original-entry wall exposure plus direct loss-column/MSC/tail consumption checks."""

    def __init__(self, start):
        if start.empty or start.duplicated(TRACK).any():
            raise ValueError("empty/duplicate original entry states")
        self.start = start
        self.data = start[TRACK].reset_index(drop=True).copy()
        self.values = {k: np.zeros(len(start)) for k in ("wall_mm", "wall_x0", "wall_S1_mm2", "wall_S2_mm3")}
        self.counts = {k: np.zeros(len(start), dtype=np.int64) for k in
                       ("chords", "wall_chords", "wall_only_chords", "msc_chords", "loss_column_checks", "tail_checks")}
        self.pending = None
        self.column_calls = self.msc_calls = self.tail_calls = 0

    def wall(self, loc, columns):
        act = np.asarray(loc["act"])
        if (act.dtype.kind not in "iu" or len(np.unique(act)) != len(act) or
                np.any(act < 0) or np.any(act >= len(self.start))):
            raise ValueError("invalid native active-entry indices")
        if self.pending is not None:
            raise ValueError("wall columns not consumed before next chord")
        self.pending = tuple(np.array(v, copy=True) for v in columns)
        ln, m1, m2 = self.pending
        if any(v.shape != (len(act),) or not np.isfinite(v).all() or (v < 0).any() for v in self.pending):
            raise ValueError("invalid wall columns")
        inv = ((1-loc["f_metal"])/fs.eloss.radiation_length_mm(fs.materials.MYLAR)
               + loc["f_metal"]/fs.eloss.radiation_length_mm(fs.materials.ALUMINIUM_6061))
        for name, value in (("wall_mm", ln), ("wall_x0", ln*inv), ("wall_S1_mm2", m1), ("wall_S2_mm3", m2)):
            np.add.at(self.values[name], act, value)
        gas = np.zeros(len(act), dtype=bool)
        if loc["sp"] is not None:
            gas[loc["sp"][0]] = True
        np.add.at(self.counts["chords"], act, 1)
        np.add.at(self.counts["wall_chords"], act, ln > 0)
        np.add.at(self.counts["wall_only_chords"], act, (ln > 0) & ~gas)

    def columns(self, loc, thin):
        if self.pending is None:
            raise ValueError("loss-column consumption without wall ledger")
        ln, _, _ = self.pending
        fraction = loc["f_metal"]
        for mat, split in ((fs.materials.MYLAR, 1-fraction), (fs.materials.ALUMINIUM_6061, fraction)):
            observed = thin.get(mat.name, (mat, np.zeros(len(ln))))[1]
            equal(observed, ln*split, mat.name + " loss/tail material column")
        np.add.at(self.counts["loss_column_checks"], loc["act"], 1)
        self.pending = None
        self.column_calls += 1

    def scattering(self, loc, before, after):
        if self.pending is not None:
            raise ValueError("MSC before loss columns verified")
        X, S1, S2 = (np.array(loc[k], copy=True) for k in ("X", "S1", "S2"))
        inv = ((1-loc["f_metal"])/fs.eloss.radiation_length_mm(fs.materials.MYLAR)
               + loc["f_metal"]/fs.eloss.radiation_length_mm(fs.materials.ALUMINIUM_6061))
        expected = [loc[k]*inv for k in ("wall_ln", "wall_s1", "wall_s2")]
        if loc["sp"] is not None:
            lane, _, s0, s1, _, wr, pl = loc["sp"]
            for mat, ln in ((loc["gas"], s1-s0), (fs.materials.TUNGSTEN, wr), (fs.materials.GOLD, pl)):
                xx = ln/fs.eloss.radiation_length_mm(mat)
                lever = loc["L"][lane] - .5*(s0+s1)
                for out, value in zip(expected, (xx, xx*lever, xx*lever**2)):
                    np.add.at(out, lane, value)
        for name, (ln, first, second) in loc["mom"].items():
            x0 = fs.eloss.radiation_length_mm(loc["getm"](name))
            L = loc["L"]
            for out, value in zip(expected, (ln/x0, (ln*L-first)/x0, (ln*L**2-2*L*first+second)/x0)):
                out += value
        for actual, wanted, name in zip((X, S1, S2), expected, ("X", "S1", "S2")):
            equal(actual, wanted, name)
        kk = loc["kk"]
        equal(after-before, X[kk], "Highland increment")
        np.add.at(self.counts["msc_chords"], loc["act"][kk], 1)
        self.msc_calls += 1

    def tails(self, loc, parts):
        kk = loc["kk"]
        expected = [(m, ln[kk]) for m, ln in loc["thin"].values()]
        expected += [(loc["getm"](name), ln[kk]) for name, ln in loc["sup"].items()]
        if len(parts) != len(expected):
            raise ValueError("tail material column count changed")
        for (mat, actual), (wanted, ln) in zip(parts, expected):
            if mat is not wanted:
                raise ValueError("tail material/order changed")
            equal(actual, ln, "tail "+mat.name)
        np.add.at(self.counts["tail_checks"], loc["act"][kk], 1)
        self.tail_calls += 1

    def report(self):
        if self.pending is not None or not self.column_calls or self.msc_calls != self.tail_calls:
            raise ValueError("incomplete direct consumer observations")
        return dict(entries=len(self.data), column_calls=self.column_calls, msc_calls=self.msc_calls,
                    tail_calls=self.tail_calls, totals={k: float(v.sum()) for k, v in self.values.items()},
                    counters={k: int(v.sum()) for k, v in self.counts.items()},
                    scope="actual enabled wall columns and loss/MSC/tail inputs, not G4 material or realized loss",
                    throughput_scope="passively observed transport timing, not unmodified production throughput")

    def frame(self):
        return pd.concat([self.data, pd.DataFrame(self.counts), pd.DataFrame(self.values)], axis=1)


@contextmanager
def observing_consumers(observer):
    """Frozen-code process-local hooks call originals exactly once and return their original objects."""
    code = fs.transport.__code__
    original_columns, original_highland, original_tail = fs._columns, fs.tr.highland_var_increment, fs.st._group_tail
    original_wall = fw.WallIndex.columns

    def context(frame):
        if frame is None or frame.f_code is not code or frame.f_locals.get("start") is not observer.start:
            raise ValueError("unexpected enabled-consumer caller/start")
        loc = frame.f_locals
        if loc.get("complete_outer_wall") is not True or loc.get("wall_index") is None:
            raise ValueError("complete wall ledger is not enabled")
        return loc

    def wall(self, x, c, length):
        result = original_wall(self, x, c, length)
        frame = inspect.currentframe()
        try:
            loc = context(frame.f_back)
            if loc["xa"] is not x or loc["c"] is not c or loc["L"] is not length:
                raise ValueError("wall call arguments differ from actual transport chord")
            observer.wall(loc, result)
        finally:
            del frame
        return result

    def columns(n, parts):
        result = original_columns(n, parts)
        frame = inspect.currentframe()
        try:
            observer.columns(context(frame.f_back), result)
        finally:
            del frame
        return result

    def highland(p, before, after, mass):
        result = original_highland(p, before, after, mass)
        frame = inspect.currentframe()
        try:
            observer.scattering(context(frame.f_back), before, after)
        finally:
            del frame
        return result

    def tails(rng, parts, *args, **kwargs):
        result = original_tail(rng, parts, *args, **kwargs)
        frame = inspect.currentframe()
        try:
            observer.tails(context(frame.f_back), parts)
        finally:
            del frame
        return result

    try:
        fw.WallIndex.columns, fs._columns = wall, columns
        fs.tr.highland_var_increment, fs.st._group_tail = highland, tails
        yield observer
    finally:
        fw.WallIndex.columns, fs._columns = original_wall, original_columns
        fs.tr.highland_var_increment, fs.st._group_tail = original_highland, original_tail


def export_candidate(fast, ent, particles):
    rk = ent[TRACK + ["pdg"]].sort_values(TRACK).copy()
    rk["key"] = rk.groupby(tables.KEY).cumcount()+1
    rmap = rk[TRACK + ["key"]]
    generated = rk[tables.KEY+["key", "pdg"]].assign(parent=-1, isprim=1, ismcprim=1).reset_index(drop=True)
    sa.exact(generated, particles.reset_index(drop=True))
    out = fast.merge(rmap, on=TRACK).drop(columns="sp").rename(columns={"key": "sp"})
    return out[sa.EXPORT].reset_index(drop=True)


def verify_source_receipts(report, input_table, legacy_dir, gdml=None, field_map=None):
    """Rehash saved source binding; do not inherit a report's success flag without its files."""
    if not report.get("passed") or not report.get("g4all_ordered_payload_exact"):
        raise ValueError("prior source audit did not pass")
    receipts = report["inputs"]
    for name, receipt in receipts.items():
        if pa.sha256(receipt["path"]) != receipt["sha256_before"] or receipt["sha256_after"] != receipt["sha256_before"]:
            raise ValueError("source receipt changed: "+name)
    for name, receipt in report.get("outputs", {}).items():
        if pa.sha256(receipt["path"]) != receipt["sha256"]:
            raise ValueError("source output receipt changed: "+name)
    if Path(receipts["input_table"]["path"]).resolve() != input_table:
        raise ValueError("source input table path mismatch")
    for name, filename in (("g4_all", "g4prim_field.root"), ("g4_clock", "g4fp_field.root"), ("fast", "fasttrk_field.root")):
        found = [r for r in receipts.values() if Path(r["path"]).resolve() == legacy_dir/filename]
        if not found:
            raise ValueError("source audit lacks legacy arm binding: "+name)
    for label, path in (("gdml", gdml), ("map_header", field_map),
                         ("map_binary", field_map.with_suffix(".bin") if field_map else None)):
        if path is not None and not any(Path(r["path"]).resolve() == path for r in receipts.values()):
            raise ValueError("source audit lacks supplied geometry/map binding: "+label)
    return {"prior/inputs/"+k: Path(r["path"]) for k, r in receipts.items()} | {
        "prior/outputs/"+k: Path(r["path"]) for k, r in report.get("outputs", {}).items()}


def verify_path_receipts(report, input_table, legacy_dir, gdml, field_map):
    before, after = report.get("hashes_before", {}), report.get("hashes_after", {})
    if not report.get("passed") or before != after:
        raise ValueError("prior path audit failed or input receipts changed")
    expected = dict(input_table=input_table, gdml=gdml, map_header=field_map,
                    map_adjacent_binary=field_map.with_suffix(".bin"),
                    **{"original/"+name: legacy_dir/name for name in
                       ("fasttrk_field.root", "g4prim_field.root", "g4fp_field.root", "fieldcompare.json")})
    for name, path in expected.items():
        receipt = before.get(name, {})
        if Path(receipt.get("path", "")).resolve() != path or pa.sha256(path) != receipt.get("sha256"):
            raise ValueError("prior path-audit binding mismatch: "+name)
    for name, receipt in before.items():
        if pa.sha256(receipt["path"]) != receipt["sha256"]:
            raise ValueError("prior path-audit receipt changed: "+name)
    return {"path_audit/"+name: Path(receipt["path"]) for name, receipt in before.items()}


def run(input_table, legacy_dir, source_report, gdml, field_map, outdir, path_report=None):
    input_table, legacy_dir, source_report, gdml, field_map, outdir = map(
        lambda p: Path(p).resolve(), (input_table, legacy_dir, source_report, gdml, field_map, outdir))
    if outdir.exists():
        raise FileExistsError(outdir)
    source = json.loads(source_report.read_text())
    prior_paths = verify_source_receipts(source, input_table, legacy_dir, gdml,
                                         field_map if path_report is None else None)
    if path_report is not None:
        path_report = Path(path_report).resolve()
        path_audit = json.loads(path_report.read_text())
        prior_paths.update(verify_path_receipts(path_audit, input_table, legacy_dir, gdml, field_map))
        prior_paths["path_report"] = path_report
    old = json.loads((legacy_dir/"fieldcompare.json").read_text())
    opts = pa.frozen_settings(old, "first")
    for name, mod in (("fieldsteps", fs), ("fieldwall", fw)):
        if pa.sha256(mod.__file__) != PHYSICS_SHA[name]:
            raise ValueError("frozen candidate physics changed: "+name)
    paths = dict(input_table=input_table, gdml=gdml, map_header=field_map, map_binary=field_map.with_suffix(".bin"),
                 source_report=source_report, legacy_report=legacy_dir/"fieldcompare.json",
                 legacy_fast=legacy_dir/"fasttrk_field.root", g4_clock=legacy_dir/"g4fp_field.root",
                 g4_all=legacy_dir/"g4prim_field.root")
    modules = {"driver": sys.modules[__name__], "fieldsteps": fs, "fieldwall": fw, "fieldcompare": fc,
               "sourceaudit": sa, "geometry": geometry, "core_tables": tables, "core_eloss": fs.eloss,
               "core_transport": fs.tr, "core_materials": fs.materials, "ehelix": fs.ehelix, "steps": fs.st}
    paths.update({"code/"+k: Path(v.__file__) for k, v in modules.items()})
    paths.update(prior_paths)
    before = pa.receipts(paths)
    outdir.mkdir()
    result = dict(passed=False, label="development-only explored cosmic pilot; not untouched confirmation",
                  hashes_before=before, settings=opts, complete_outer_wall=True,
                  legacy_reference="retained original G4 clock and legacy fast outputs, never regenerated",
                  geometry_scope="same geometry/map/constants; synthetic tests are not runtime geometry certification")
    try:
        geom = geometry.from_gdml(gdml)
        p, s, vd, surf, census = fc.primaries([str(input_table)], 13)
        raw = tables.read(input_table, "sgsteps").reset_index(drop=True)
        rows = raw.assign(sourceoriginalrowindex=np.arange(len(raw), dtype=np.int64)).merge(p, on=TRACK, how="inner")
        built = sa.reconstruct(p, s, vd, surf, np.asarray(opts["origin"]), opts["entry_vds"], rows)
        particles = tables.read(legacy_dir/"g4prim_field.root", "particles")
        sa.exact(built["particles"], particles)
        sa.exact(built["sgsteps"], tables.read(legacy_dir/"g4prim_field.root", "sgsteps"))
        ent = built["entry"]
        if len(ent) != source["entry_particles"] or len(ent) != old["entries"]:
            raise ValueError("original entry exposure changed")
        org = np.asarray(opts["origin"])
        field = fs.ehelix.grid_field(*fs.ehelix.read_g4bl(str(field_map)))
        env = fs.Envelope(geom, axis_xy=org[:2])
        inter = fs.interior(gdml, geom, env, origin=org)
        bnom = fs.ehelix.field_at(field, org[None])[0]
        obs = ConsumerLedger(ent)
        rng = np.random.default_rng(opts["seed"])
        initial_rng = clean(rng.bit_generator.state)
        with observing_consumers(obs):
            fast, final = fs.transport(rng, ent, geom, field, inter, chord_mm=opts["chord_mm"],
                                       sagitta_mm=opts["sagitta_mm"], envelope=env, origin=org, bnom=bnom,
                                       radiative=opts["mu_radiative"], gas=fs.STRAW_GAS_G4STATE if opts["gas_state"] == "gas" else None,
                                       complete_outer_wall=True)
        ledger = obs.report()
        exported = export_candidate(fast, ent, particles)
        tables.write(outdir/"fasttrk_field.root", {"sgsteps": exported, "particles": particles}, as_ttree=True)
        tables.write(outdir/"states.root", {"entry": ent, "final": final, "fast": fast}, as_ttree=True)
        obs.frame().to_csv(outdir/"wall_consumers.csv.gz", index=False)
        built["genealogy"].to_csv(outdir/"genealogy.csv.gz", index=False)
        gclock = tables.read(legacy_dir/"g4fp_field.root", "sgsteps")
        result.update(entries=len(ent), original_dts_census=census, consumers=ledger,
                      steps={"candidate": len(fast), "fixed_g4_clock": len(gclock)},
                      final_status=final.status.value_counts().sort_index().to_dict(),
                      rng_initial=initial_rng, rng_final=clean(rng.bit_generator.state),
                      passed=True, frozen_reference_clock="candidate exit clock never changes retained G4 rows")
        after = pa.receipts(paths)
        if before != after:
            raise ValueError("input/code/map changed during candidate transport")
        result["hashes_after"] = after
        result["outputs"] = pa.receipts({p.name: p for p in outdir.iterdir() if p.is_file()})
    except Exception as exc:
        result.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        (outdir/"fieldwallpilot.json").write_text(json.dumps(clean(result), indent=2, allow_nan=False)+"\n")
        raise
    (outdir/"fieldwallpilot.json").write_text(json.dumps(clean(result), indent=2, allow_nan=False)+"\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("input-table", "legacy-dir", "source-report", "gdml", "map", "outdir"):
        ap.add_argument("--"+name, required=True)
    ap.add_argument("--path-report", help="closed original replay audit binding the supplied map header/binary")
    a = ap.parse_args(argv)
    result = run(a.input_table, a.legacy_dir, a.source_report, a.gdml, a.map, a.outdir, a.path_report)
    print(json.dumps(clean(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
