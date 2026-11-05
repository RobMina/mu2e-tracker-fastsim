"""Validation-only observation of actual stochastic chords during an exact saved-arm replay.

Run with the original code/environment snapshot and original dumper, GDML and map.
This does not execute Geant4, reconstruction, tuning, or a final-state equality test.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from mu2efastsim import tables
from trkfastsim import crossings, geometry
from trkfastsim.sim import fieldsteps as fs
from trkfastsim.validation import fieldcompare, fieldcrossing as fc

MAX_ORACLE_CHORDS = 256
COVERAGE = ("actual fs.ChordIndex calls on stochastic fast transport chords, including zero-hit lanes and gaps; "
            "gas crossing search only, not material navigation, physical G4 first passage or deficit closure")


class CrossingObserver:
    """Bounded examples and aggregate complete-set comparisons; no RNG or transport state access."""

    def __init__(self, oracle_chords=MAX_ORACLE_CHORDS):
        fc._positive_integer(oracle_chords, "oracle chords", allow_zero=True)
        if oracle_chords > MAX_ORACLE_CHORDS:
            raise ValueError("oracle chords must be <= 256")
        self.limit = oracle_chords
        self.stats = dict(calls=0, lanes=0, zero_hit_lanes=0, oracle_chords=0,
                          rows=dict(A_raw=0, A=0, B=0, C=0), comparisons={}, examples=[])
        for pair in ("A_B", "A_C", "B_C"):
            self.stats["comparisons"][pair] = dict(lanes=0, rows_left=0, rows_right=0,
                                                    mismatches=0, mismatch_lanes=0, kinds={})

    def observe(self, raw, x, c, length, geom, panel_index):
        x, c, length = fc.validate_rays(x, c, length)
        stat = self.stats
        call = stat["calls"]
        stat["calls"] += 1
        stat["lanes"] += len(x)
        a = fc.canonical(raw, length, geom.nstraws, "A")
        b = fc.canonical(crossings.straw_crossings(x, c, geom, index=panel_index),
                         length, geom.nstraws, "B")
        n = min(self.limit - stat["oracle_chords"], len(x))
        oracle = fc.oracle_crossings(x[:n], c[:n], length[:n], geom) if n else np.empty((0, 4))
        stat["oracle_chords"] += n
        stat["zero_hit_lanes"] += len(x) - len(np.unique(a[:, 0]))
        for name, rows in (("A_raw", raw), ("A", a), ("B", b), ("C", oracle)):
            stat["rows"][name] += len(rows)
        for pair, left, right, count in (("A_B", a, b, len(x)),
                                         ("A_C", a[a[:, 0] < n], oracle, n),
                                         ("B_C", b[b[:, 0] < n], oracle, n)):
            diff = fc.compare_rows(left, right, pair.split("_"))
            comp = stat["comparisons"][pair]
            comp["lanes"] += count
            comp["rows_left"] += len(left)
            comp["rows_right"] += len(right)
            comp["mismatches"] += len(diff)
            comp["mismatch_lanes"] += int(diff.lane.nunique())
            for kind, total in diff.kind.value_counts().items():
                comp["kinds"][kind] = comp["kinds"].get(kind, 0) + int(total)
            room = fc.EXAMPLES - len(stat["examples"])
            if room > 0 and len(diff):
                example = diff.head(room).copy()
                lanes = example.lane.to_numpy(dtype=np.int64)
                example["chord_call"] = call
                example["comparison"] = pair
                example["identity_scope"] = "chord call and local lane only; original event/particle unavailable"
                example["length_mm"] = length[lanes]
                for j, axis in enumerate("xyz"):
                    example[f"x_{axis}"] = x[lanes, j]
                    example[f"c_{axis}"] = c[lanes, j]
                stat["examples"].extend(example.astype(object).where(pd.notna(example), None).to_dict("records"))

    def report(self):
        return {**self.stats, "coverage": COVERAGE, "oracle_chords_requested": self.limit,
                "oracle_selection": "first requested actual chords in call order, then local lane order; no sampling",
                "oracle_unchecked_chords": self.stats["lanes"] - self.stats["oracle_chords"],
                "interval_atol_mm": fc.ATOL_MM, "interval_rtol": 0,
                "mismatch_unit": "lane/straw disagreement per comparison, not unique chords across comparisons",
                "total_mismatches": sum(s["mismatches"] for s in self.stats["comparisons"].values())}


@contextmanager
def observing_crossings(observer):
    """Process-local temporary class replacement; do not use concurrently with other transports."""
    original = fs.ChordIndex

    class ObservedChordIndex(original):
        def __init__(self, geom):
            super().__init__(geom)
            self.audit_panel_index = crossings.panel_index(geom)

        def crossings(self, x, c, L, geom):
            rows = super().crossings(x, c, L, geom)
            observer.observe(rows, x, c, L, geom, self.audit_panel_index)
            return rows

    try:
        fs.ChordIndex = ObservedChordIndex
        yield observer
    finally:
        fs.ChordIndex = original


def compare_saved_tables(old_dir, replay_dir):
    """Exact ordered payload equality, without converting integer identities through float."""
    checks = {}
    for label, filename in fc.FILES.items():
        for tree in ("particles", "sgsteps"):
            key = f"{label}/{tree}"
            try:
                old = tables.read(str(Path(old_dir) / filename), tree)
                new = tables.read(str(Path(replay_dir) / filename), tree)
                pd.testing.assert_frame_equal(old, new, check_exact=True)
            except Exception as exc:
                checks[key] = dict(equal=False, error=f"{type(exc).__name__}: {exc}")
            else:
                checks[key] = dict(equal=True, rows=len(old))
    return checks


def frozen_settings(old, entry):
    """Reject ambiguous provenance instead of silently filling missing physics/RNG settings."""
    required = ("species", "inputs", "seed", "chord_mm", "sagitta_mm", "gas_state", "mu_radiative",
                "origin_mm", "entry_vds", "field")
    if any(k not in old for k in required):
        raise ValueError(f"missing frozen fieldcompare settings: {[k for k in required if k not in old]}")
    if old["species"] != 13 or old["inputs"] != 1:
        raise ValueError("only species 13 and exactly one original input are supported")
    if entry != "first" or old.get("entry", entry) != entry:
        raise ValueError("explicit original --entry first required")
    fc._positive_integer(old["seed"], "seed", allow_zero=True)
    for name in ("chord_mm", "sagitta_mm"):
        if isinstance(old[name], bool) or not np.isfinite(old[name]) or old[name] <= 0:
            raise ValueError(f"invalid frozen {name}")
    if old["gas_state"] not in ("solid", "gas") or not isinstance(old["mu_radiative"], bool):
        raise ValueError("invalid frozen gas_state/mu_radiative")
    origin = np.asarray(old["origin_mm"], float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("invalid frozen origin_mm")
    vds = old["entry_vds"]
    if not isinstance(vds, list) or not vds or len(set(vds)) != len(vds):
        raise ValueError("invalid frozen entry_vds")
    for vd in vds:
        fc._positive_integer(vd, "entry VD")
    if not isinstance(old["field"], str) or old["field"] == "none" or old["field"].startswith("bz="):
        raise ValueError("original comparison must use a field map")
    if any(k in old.get("census", {}) for k in ("unbiased", "self_pass")):
        raise ValueError("selected/truncated replay options require separate provenance; unsupported")
    return dict(species=13, seed=old["seed"], chord_mm=old["chord_mm"], sagitta_mm=old["sagitta_mm"],
                gas_state=old["gas_state"], mu_radiative=old["mu_radiative"], origin=origin.tolist(),
                entry_vds=vds, entry=entry, write_tables=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipts(paths):
    return {name: dict(path=str(path.resolve()), sha256=sha256(path)) for name, path in paths.items()}


def run(field_dir, input_table, gdml, field_map, outdir, *, entry, oracle_chords=256):
    observer = CrossingObserver(oracle_chords)
    field_dir, input_table, gdml, field_map, outdir = map(Path, (field_dir, input_table, gdml, field_map, outdir))
    field_dir, outdir = field_dir.resolve(), outdir.resolve()
    if outdir == field_dir or field_dir in outdir.parents:
        raise ValueError("output must be separate from retained original directory")
    old = json.loads((field_dir / "fieldcompare.json").read_text())
    settings = frozen_settings(old, entry)
    if Path(old["field"]).name != field_map.name:
        raise ValueError("map name differs from frozen fieldcompare report")
    paths = {"input_table": input_table, "gdml": gdml, "map_header": field_map}
    binary = field_map.with_suffix(".bin")
    if binary.is_file():
        paths["map_adjacent_binary"] = binary
    for path in sorted(field_dir.rglob("*")):
        if path.is_file():
            paths[f"original/{path.relative_to(field_dir)}"] = path
    for filename in fc.FILES.values():
        if not (field_dir / filename).is_file():
            raise ValueError(f"missing original injector table: {filename}")
    modules = dict(observer=sys.modules[__name__], fieldsteps=fs, fieldcompare=fieldcompare,
                   fieldcrossing=fc, crossings=crossings, geometry=geometry, core_tables=tables,
                   core_ehelix=fs.ehelix, core_eloss=fs.eloss, core_transport=fs.tr,
                   core_materials=fs.materials, steps=fs.st)
    paths.update({f"code/{name}": Path(module.__file__) for name, module in modules.items()})
    before = receipts(paths)
    outdir.mkdir(parents=True, exist_ok=False)
    result = dict(passed=False, coverage=COVERAGE, original_field_dir=str(field_dir),
                  parent_source_paths={k: str(v.resolve()) for k, v in paths.items() if not k.startswith("code/")},
                  frozen_settings=settings, explicit_original_entry_cli=entry,
                  runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, pandas=pd.__version__),
                  environment_scope="original code/environment and all stream settings must be frozen externally; local hashes are partial receipts, not historical attestation",
                  final_state_comparison="not performed: original injector outputs contain no saved final tree",
                  hashes_before=before, reproduction=dict(reproduced=False, tables={}))
    try:
        with observing_crossings(observer):
            fieldcompare.run([str(input_table)], str(outdir / "replay"), gdml=str(gdml),
                             field_map=str(field_map), **settings)
        checks = compare_saved_tables(field_dir, outdir / "replay")
        result["reproduction"] = dict(reproduced=all(v["equal"] for v in checks.values()), tables=checks,
                                      scope="all three saved injector particles and sgsteps, exact ordered payloads only")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        after = {}
        for name, path in paths.items():
            try:
                after[name] = dict(path=str(path.resolve()), sha256=sha256(path))
            except OSError as exc:
                after[name] = dict(path=str(path.resolve()), error=str(exc))
        result["hashes_after"] = after
        result["inputs_unchanged"] = before == after
        result["crossing_stats"] = observer.report()
        result["comparisons_diagnostic_only"] = not (result["inputs_unchanged"] and result["reproduction"]["reproduced"])
        result["passed"] = (not result["comparisons_diagnostic_only"] and "error" not in result
                            and observer.stats["lanes"] > 0 and observer.report()["total_mismatches"] == 0)
        result["verdict"] = "report-only crossing search agreement; no physics closure or correction"
        with (outdir / "fieldpathaudit.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        if not result["passed"]:
            (outdir / "FAILED").write_text("Audit failed; retain replay and diagnostic mismatch report.\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field-dir", required=True)
    ap.add_argument("--input-table", required=True)
    ap.add_argument("--gdml", required=True)
    ap.add_argument("--map", dest="field_map", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--entry", required=True, choices=["first"], help="explicit original invocation; not inferred from report")
    ap.add_argument("--oracle-chords", type=int, default=256)
    result = run(**vars(ap.parse_args(argv)))
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
