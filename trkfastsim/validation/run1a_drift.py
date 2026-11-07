"""Audit the independent stock drift oracle, optionally checking a source-level evaluator.

The oracle is validation data, not an interpolation model. Model construction uses
resolved FHiCL and exported geometry/origin field only, never stock response values.
This does not validate a track fit, fitted timing, field-map tracking or multi-IOV use.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from trkfastsim.digi.digicond import parse_fhicl_dump

SCOPE = "stock_query_oracle_only_not_interpolation_conditions"
QUERIES = ("dt", "phi", "used_phi", "c_drift", "r_drift", "drift_velocity",
           "signed_error", "unsigned_error")
SCALARS = ("schema", "run", "subrun", "event", "reference_straw", "iov_start_run",
           "iov_start_subrun", "iov_end_run", "iov_end_subrun", "angle_bins", "extra_probes",
           "cc", "wire_radius", "outer_radius", "origin_bz_float", "time_min", "time_max", "time_step")
VECTORS = QUERIES + ("detector_origin_mu2e", "origin_field", "response_cids", "tracker_cids")
FIELDS = SCALARS + VECTORS + ("scope",)


def read_oracle(path):
    """Read the single first-IOV row without iterating over queries or events."""
    if Path(path).stat().st_size > 128 * 1024**2:
        raise ValueError("oracle file exceeds bounded reader size")
    with uproot.open(path) as root:
        tree = root["driftoracle/oracle"]
        if tree.classname != "TTree" or tree.num_entries != 1 or not set(FIELDS).issubset(tree.keys()):
            raise ValueError("expected one complete first-IOV oracle row in a TTree")
        if tree.uncompressed_bytes > 96 * 1024**2:
            raise ValueError("oracle uncompressed baskets exceed reader bound")
        rows = tree.arrays(list(FIELDS), library="ak")
    out = {k: np.asarray(ak.to_numpy(rows[k][0])) for k in VECTORS}
    out.update({k: ak.to_numpy(rows[k])[0].item() for k in SCALARS})
    out["scope"] = str(rows["scope"][0])
    return out


def audit(data, config):
    """Validate schema, full configured query census, metadata and analyzer-only scheduling."""
    if set(data) != set(FIELDS) or data["schema"] != 1 or data["scope"] != SCOPE:
        raise ValueError("unsupported oracle schema/scope")
    for key in SCALARS:
        if not np.isscalar(data[key]) or not np.isfinite(data[key]):
            raise ValueError(f"invalid scalar {key}")
    for key in SCALARS[:SCALARS.index("cc")]:
        value = data[key]
        limit = 65535 if key == "reference_straw" else np.iinfo(np.uint32).max
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value <= limit:
            raise ValueError(f"invalid integer metadata {key}")
    for key in VECTORS:
        a = np.asarray(data[key])
        if a.ndim != 1 or len(a) > 1_000_000 or a.dtype.kind not in "fiu" or not np.all(np.isfinite(a)):
            raise ValueError(f"invalid vector {key}")
    for key in ("response_cids", "tracker_cids"):
        a = np.asarray(data[key])
        if a.dtype.kind not in "iu" or len(a) > 4096 or np.any(a < 0) or np.any(a > np.iinfo(np.int32).max):
            raise ValueError(f"invalid CIDs {key}")
    for key in ("detector_origin_mu2e", "origin_field"):
        if np.asarray(data[key]).shape != (3,):
            raise ValueError(f"invalid three-vector {key}")
    if not 0 < data["wire_radius"] < data["outer_radius"]:
        raise ValueError("invalid nominal radii")
    if data["origin_bz_float"] != float(np.float32(data["origin_field"][2])):
        raise ValueError("origin Bz rounding differs")
    first = (data["run"], data["subrun"])
    if not (data["iov_start_run"], data["iov_start_subrun"]) <= first <= (data["iov_end_run"], data["iov_end_subrun"]):
        raise ValueError("first event outside exported IOV")
    physics = config["physics"]
    if (physics["trigger_paths"] != [] or physics["RecoPath"] != []
            or physics["end_paths"] != ["DumpPath"] or physics["DumpPath"] != ["driftoracle"]
            or config["outputs"] != {}):
        raise ValueError("not an oracle-only schedule")
    query = physics["analyzers"]["driftoracle"]
    if query["module_type"] != "FieldOnDriftOracle":
        raise ValueError("wrong oracle module")
    for key, cfgkey in (("time_min", "TimeMin"), ("time_max", "TimeMax"), ("time_step", "TimeStep"),
                        ("angle_bins", "AngleBins"), ("extra_probes", "ExtraProbes")):
        if data[key] != query[cfgkey]:
            raise ValueError(f"query config/metadata differs: {key}")
    lo, hi, step = (data[k] for k in ("time_min", "time_max", "time_step"))
    na, nextra, limit = data["angle_bins"], data["extra_probes"], query["MaxQueries"]
    if (step <= 0 or lo >= hi or any(isinstance(x, bool) or int(x) != x for x in (na, nextra, limit))
            or na < 2 or nextra < 0 or not 0 < limit <= 1_000_000):
        raise ValueError("invalid bounded query grid")
    intervals = (hi - lo) / step
    if not np.isfinite(intervals) or abs(intervals - round(intervals)) > 1e-8:
        raise ValueError("invalid integral time intervals")
    nt = int(round(intervals)) + 1
    tp, pp = np.asarray(query["TimeProbes"], float), np.asarray(query["AngleProbes"], float)
    if tp.ndim != 1 or tp.shape != pp.shape or not np.all(np.isfinite(tp)) or not np.all(np.isfinite(pp)):
        raise ValueError("invalid paired probes")
    expected = nt * int(na) + int(nextra) + len(tp)
    if expected > limit or any(np.asarray(data[k]).shape != (expected,) for k in QUERIES):
        raise ValueError("query bound/census differs")
    j = np.arange(int(nextra), dtype=float) + .5
    halfpi = np.pi / 2
    times = np.concatenate((np.repeat(lo + step*np.arange(nt), int(na)),
                            lo + (hi-lo)*np.fmod(j*.6180339887498948, 1), tp))
    angles = np.concatenate((np.tile(halfpi*np.arange(int(na))/(na-1), nt),
                             (8*np.fmod(j*.4142135623730950, 1)-4)*halfpi, pp))
    for key, expected_array in (("dt", times), ("phi", angles)):
        if not np.allclose(data[key], expected_array, rtol=0, atol=1e-12):
            raise ValueError(f"configured query coordinates differ: {key}")
    response = config["services"]["ProditionsService"]["strawResponse"]
    used = np.zeros(expected) if response["driftIgnorePhi"] else data["phi"]
    if not np.array_equal(data["used_phi"], used):
        raise ValueError("used_phi differs from stock ignore-phi semantics")
    metadata = {k: data[k] for k in SCALARS}
    metadata.update({k: np.asarray(data[k]).tolist() for k in VECTORS if k not in QUERIES})
    return dict(scope=SCOPE, queries=expected, rectangular_queries=nt*int(na),
                off_grid_queries=int(nextra), explicit_queries=len(tp), finite=True,
                configured_coordinates_atol=1e-12, oracle_only_schedule=True, metadata=metadata,
                ranges={k: [float(np.min(data[k])), float(np.max(data[k]))] for k in QUERIES})


def compare_model(data, config):
    """Compare all queries with tolerances fixed before the real-oracle comparison."""
    from trkfastsim.reco.fielddrift import build_model, drift_info
    pro = config["services"]["ProditionsService"]
    model = build_model(pro["strawDrift"], pro["strawResponse"], wire_radius=data["wire_radius"],
                        outer_radius=data["outer_radius"], bz=data["origin_field"][2])
    predicted = drift_info(model, data["dt"], data["phi"])
    tolerances = {"used_phi": 0., "c_drift": 1e-10, "r_drift": 1e-10,
                  "drift_velocity": 1e-10, "signed_error": 1e-12, "unsigned_error": 1e-12}
    checks = {}
    for key, atol in tolerances.items():
        value = np.asarray(predicted[key])
        if value.shape != np.asarray(data[key]).shape or not np.all(np.isfinite(value)):
            raise ValueError(f"model shape/finite failure: {key}")
        delta = np.abs(value - data[key])
        checks[key] = dict(atol=atol, max_abs=float(delta.max()), rms=float(np.sqrt(np.mean(delta**2))),
                           outside_tolerance=int(np.count_nonzero(delta > atol)))
    cc = float(model.cc)
    return dict(passed=cc == data["cc"] and all(v["outside_tolerance"] == 0 for v in checks.values()),
                model_cc=cc, stock_cc=data["cc"], cc_exact=cc == data["cc"], quantities=checks,
                caveat=__doc__)


def _sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--compare-oracle", type=Path)
    parser.add_argument("--check-model", action="store_true")
    args = parser.parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    data = read_oracle(args.oracle)
    config = parse_fhicl_dump(args.config.read_text())
    report = audit(data, config)
    paths = {"oracle": args.oracle, "config": args.config, "validator": Path(__file__)}
    log = args.oracle.parent / "export.log"
    manifest = args.oracle.parent / "source-sha256.txt"
    done = args.oracle.parent / "export.done"
    if not all(p.is_file() for p in (log, manifest, done)) or done.read_text().strip() != "finished":
        raise ValueError("required successful export sidecars missing or incomplete")
    text = log.read_text()
    census = re.search(r"TrigReport Events total = (\d+) passed = (\d+) failed = (\d+)", text)
    if census is None or "Art has completed and will exit with status 0." not in text:
        raise ValueError("missing successful art event census")
    total, passed, failed = map(int, census.groups())
    if total < 1 or total != passed or failed != 0:
        raise ValueError("failed/empty art event census")
    report["art_events"] = dict(total=total, passed=passed, failed=failed,
                                caveat="one oracle row per first IOV, not one row per event")
    paths.update(export_log=log, export_done=done, source_manifest=manifest)
    records = [line.split(maxsplit=1) for line in manifest.read_text().splitlines()]
    names = ("expanded.fcl", "FieldOnDriftOracle_module.cc", "StrawDrift.cc", "StrawDriftMaker.cc", "StrawResponse.cc")
    if (len(records) != 5 or any(len(r) != 2 or re.fullmatch(r"[0-9a-f]{64}", r[0]) is None for r in records)
            or tuple(Path(r[1]).name for r in records) != names):
        raise ValueError("malformed export source manifest")
    if records[0] != [_sha(args.config), "expanded.fcl"]:
        raise ValueError("expanded FHiCL hash differs from export manifest")
    report["export_source_manifest"] = {name: digest for digest, name in records}
    report["provenance_scope"] = "recorded source digests and verified config; not a cryptographic binding of ROOT to log or loaded binary"
    if args.compare_oracle:
        previous = read_oracle(args.compare_oracle)
        different = [k for k in FIELDS if not np.array_equal(data[k], previous[k])]
        report["repeat"] = dict(exact=not different, different_fields=different)
        paths["compare_oracle"] = args.compare_oracle
    if args.check_model:
        report["model"] = compare_model(data, config)
        from trkfastsim.reco import fielddrift
        paths["model_source"] = Path(fielddrift.__file__)
    report["inputs"] = {k: dict(path=str(p.resolve()), sha256=_sha(p)) for k, p in paths.items()}
    report["passed"] = report.get("repeat", {}).get("exact", True) and report.get("model", {}).get("passed", True)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.outdir.mkdir(parents=True, exist_ok=False)
    (args.outdir / "report.json").write_text(text)
    print(text, end="")
    if not report["passed"]:
        raise RuntimeError("oracle/model comparison failed; report preserved")
    return report


if __name__ == "__main__":
    main()
