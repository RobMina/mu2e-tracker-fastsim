"""Initial unbiased-CA/CAD reconstruction at captured seed inputs; no refit or native-intermediate claim."""
import argparse
import json
import os
from pathlib import Path
import re

import awkward as ak
import numpy as np
import uproot

import f1_initial_residual_audit as R
import f1_captured_ca_population as P
import f1_native_algebra_audit as ALG
from f1_captured_ca_audit import A, C, NATIVE_CA_PRECISION, fixed_state_rows, snapshot_state
from f1_native_decision_audit import KEYS, array, column, digest, keys, match_rows, read_tree, require
from f1_hit_operand_check import HIT_KEYS, boolean
from f1_trace_compare import _dense, _metric
from f1_initial_drift import CALIBRATION_METHOD, drift_configuration, build_zero_field, drift_info

RESIDUAL_REPORT_SHA256 = "1539f808dd7aeab96042dac29023aed33e6688349686d6abb2c804a305bd3219"
ARMS = ("chol_factor_roundtrip", "numpy_lu_roundtrip", "ideal_zero_weight_control")
PROBE_FIELDS = ("dtgrid", "rdrift", "dvel", "sderr", "uderr", "strawOuterRadius")
NATIVE_PATHS = (
    "KinKal/Fit/Track.hh", "KinKal/Fit/Measurement.hh", "KinKal/Detector/ResidualHit.hh",
    "KinKal/Detector/Hit.hh", "KinKal/General/FitData.hh", "KinKal/General/Weights.hh",
    "KinKal/General/Weights.cc", "KinKal/General/Parameters.cc", "KinKal/Trajectory/ClosestApproach.hh",
    "Offline/Mu2eKinKal/inc/KKTrack.hh", "Offline/Mu2eKinKal/inc/KKFit.hh",
    "Offline/Mu2eKinKal/inc/KKStrawHit.hh", "Offline/Mu2eKinKal/src/CADSHU.cc",
    "Offline/Mu2eKinKal/inc/CADSHU.hh", "Offline/Mu2eKinKal/inc/WireHitState.hh",
    "Offline/Mu2eKinKal/inc/StrawHitUpdaters.hh", "Offline/Mu2eKinKal/src/KKSHFlag.cc",
    "Offline/Mu2eKinKal/src/KKFitUtilities.cc", "Offline/TrackerConditions/src/StrawDriftMaker.cc",
    "Offline/TrackerConditions/src/StrawDrift.cc", "Offline/TrackerConditions/inc/StrawDrift.hh",
    "Offline/TrackerConditions/src/StrawResponse.cc", "Offline/TrackerConditions/src/StrawResponseMaker.cc",
    "Offline/TrackerConfig/inc/StrawDriftConfig.hh", "Offline/GeometryService/src/TrackerMaker.cc",
    "TrkFastSim/src/KinematicLineFitTrace_module.cc", "TrkFastSim/src/StrawHitTableDumper_module.cc")
LIMITS = [
    "All 147 captured initial targets / 2538 hits, no success selection or exclusions; no later iterations.",
    "Captured seed/reference parameters, covariance and mass remain inputs, not F1-generated seed closure.",
    "Initial zero cached weight follows inspected constructor/update ordering and ROOT zero initialization, not a pre-update weight snapshot.",
    "Independent CAD outputs are compared with captured state; unbiased q/C/CA/DriftInfo intermediates are NOT captured.",
    "chol-factor is the existing lower-storage pivot-based diagnostic, not ROOT InvertChol/Bunch-Kaufman parity.",
    "NumPy LU and ideal zero-weight controls are predeclared arithmetic sensitivity arms, not extra native observations.",
    "Physical magnetic field is zero; nominal seed bnom=(0,0,0.001) is retained and only supplies the Lorentz-angle axis.",
    "All native drift phi slices coincide at zero physical B; no field-dependent drift claim.",
    "Full drift calibration is regenerated from source coefficients and geometry, then checked against all saved native drift probes.",
    "Calibration uses source-oriented NumPy online regression; agreement with saved probes is not universal GSL/compiler parity.",
    "Expanded FCL/source hashes establish continuity with prior audits, not an original capture-generation hash receipt.",
    "No native-inversion parity, full state-policy, later-history, fit/status/acceptance or full numerical closure claim.",
    "No regularization, ULP search, fitted correction, production default or tolerance changes; no fits/builds/native capture.",
]


def receipt(path, watched):
    watched[str(Path(path).resolve())] = RESIDUAL_REPORT_SHA256
    require(digest(path) == RESIDUAL_REPORT_SHA256, "residual receipt hash differs")
    data = json.loads(Path(path).read_text())
    before = data.get("before_sha256")
    require(isinstance(before, dict) and bool(before), "missing residual manifest")
    watched.update(before)
    require(data.get("completed") is True and data.get("hashes_unchanged") is True
            and before == data.get("end_sha256"), "incomplete residual receipt")
    census = data.get("population", {})
    require(all(type(census.get(k)) is int for k in ("n_hits", "n_targets", "n_excluded_hits")), "invalid receipt census")
    require((census["n_hits"], census["n_targets"], census["n_excluded_hits"]) == (2538, 147, 0), "receipt census differs")
    require(data.get("rtol") == R.RTOL and data.get("atol") == R.ATOL, "receipt tolerances differ")
    for p, sha in watched.items():
        require(isinstance(sha, str) and re.fullmatch(r"[a-f0-9]{64}", sha) is not None, "invalid receipt digest")
        require(digest(p) == sha, "receipt input/source changed: " + p)
    return data


def source_geometry(area, expanded, watch):
    """Read the actual text include chain for the two radii; zero-field map is mandatory."""
    text = Path(expanded).read_text()
    blocks = re.findall(r"(?ms)^([ ]*)GeometryService\s*:\s*\{(.*?)^\1\}", text)
    require(len(blocks) == 1, "requires unique expanded GeometryService")
    geometry = re.findall(r'\binputFile\s*:\s*"([^"]+)"', blocks[0][1])
    field = re.findall(r'\bbFieldFile\s*:\s*"([^"]+)"', blocks[0][1])
    require(geometry == ["Offline/Mu2eG4/geom/geom_common_extracted_v04.txt"]
            and field == ["Offline/Mu2eG4/geom/bfgeom_no_field.txt"], "unsupported geometry/field configuration")
    radii, chain = {}, []

    def visit(relative, stack):
        path = (area/relative).resolve()
        require(path.is_relative_to(area.resolve()) and path not in stack, "unsafe/cyclic geometry include")
        watch(path)
        chain.append(str(path))
        content = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
        content = re.sub(r"//[^\n]*", "", content)
        pattern = r'#include\s+"([^"]+)"|double\s+tracker\.(wireRadius|strawOuterRadius)\s*=\s*([^;]+);'
        for m in re.finditer(pattern, content):
            if m[1]:
                visit(m[1], (*stack, path))
            else:
                require(re.fullmatch(R"[0-9eE.+-]+", m[3].strip()) is not None, "nonliteral tracker radius")
                radii[m[2]] = float(m[3])

    visit(geometry[0], ())
    require(set(radii) == {"wireRadius", "strawOuterRadius"}, "missing tracker radii")
    require(np.isfinite(list(radii.values())).all() and 0 < radii["wireRadius"] < radii["strawOuterRadius"], "invalid tracker radii")
    bpath = area/field[0]
    watch(bpath)
    btext = re.sub(r"//[^\n]*", "", bpath.read_text())
    for name in ("innerMaps", "outerMaps"):
        entries = re.findall(r"vector<string>\s+bfield\."+name+r"\s*=\s*\{([^}]*)\}", btext)
        require(entries == [""], "nonempty/ambiguous magnetic field maps")
    scale = re.findall(r"double\s+bfield\.scaleFactor\s*=\s*([^;]+);", btext)
    require(len(scale) == 1 and float(scale[0]) == 0, "nonzero field scale")
    cluster = re.findall(r'\bStrawHitClusterLevel\s*:\s*"([^"]+)"', text)
    require(bool(cluster) and set(cluster) == {"none"}, "initial hit clustering unsupported")
    return dict(radii=radii, geometry_include_chain=chain, bfield_path=str(bpath), zero_field=True,
                drift_angle_scope="B=0: native gamma=1 for every phi bin, no angle dependence")


def reference_covariance(hits, pieces, seeds):
    wanted = keys(hits, HIT_KEYS)[:, :4]
    seed = seeds[match_rows(wanted, keys(seeds))]
    old = pieces[(column(pieces, "iteration_ordinal") == 0) & (column(pieces, "owner") == 0)]
    old = old[match_rows(wanted, keys(old))]
    cov = _dense(hits, "reference_covariance", (6, 6))
    require(np.isfinite(cov), "nonfinite covariance")
    require(np.array_equal(cov, np.swapaxes(cov, -1, -2)), "nonsymmetric reference covariance")
    require(np.array_equal(cov, _dense(seed, "covariance", (6, 6)))
            and np.array_equal(cov, _dense(old, "covariance", (6, 6))), "seed/old/reference covariance differs")
    for table, name in ((hits, "reference_bnom"), (old, "bnom"), (seed, "bnom")):
        require(_dense(table, name, (3,)) == np.array([0., 0., .001]), "captured nominal field differs from seed convention")
    return cov


@A._checked
def unbiased_roundtrip(q, covariance, mode):
    require(mode in ARMS, "undeclared unbiased arithmetic arm")
    q, covariance = array(q).astype(float), array(covariance).astype(float)
    require(q.ndim == 2 and q.shape[1] == 6 and covariance.shape == q.shape[:1]+(6, 6), "unbiased operand shape")
    require(np.isfinite(q) & (q[:, -1:] > 0), "invalid seed parameters")
    require(np.isfinite(covariance), "nonfinite covariance")
    require(np.array_equal(covariance, covariance.swapaxes(-1, -2)), "nonsymmetric covariance")
    if mode == "ideal_zero_weight_control":
        return q.copy(), covariance.copy(), dict(scope="mathematical zero-weight identity, no native roundtrip", n_inversions=0)
    before = dict(ALG.INVERSE_STATS)
    try:
        if mode == "chol_factor_roundtrip":
            inverse = ALG._inverse_chol_factor
        else:
            inverse = np.linalg.inv
        weight = inverse(covariance)
        vector = A._matvec6(weight, q)
        outcov = inverse(weight)
        outq = A._matvec6(outcov, vector)
    except np.linalg.LinAlgError as exc:
        raise ValueError("singular initial unbiased roundtrip; no regularization") from exc
    finally:
        stats = {k: ALG.INVERSE_STATS[k]-before[k] for k in before}
        ALG.INVERSE_STATS.update(before)
    require(np.isfinite(outq) & (outq[:, -1:] > 0), "invalid unbiased parameters")
    require(np.isfinite(outcov), "nonfinite unbiased covariance")
    return outq, outcov, dict(scope="independent zero-measurement-weight roundtrip, not native exact intermediate", n_inversions=2*len(q), **stats)


@A._checked
def similarity(derivative, covariance):
    derivative, covariance = array(derivative).astype(float), array(covariance).astype(float)
    require(derivative.ndim == 2 and derivative.shape[1] == 6
            and covariance.shape == derivative.shape[:1]+(6, 6), "similarity operand shape")
    require(np.isfinite(derivative), "nonfinite similarity derivative")
    require(np.isfinite(covariance), "nonfinite similarity covariance")
    projected = A._matvec6(covariance, derivative)
    product = derivative*projected
    value = product[:, 0].copy()
    for i in range(1, 6):
        value += product[:, i]
    require(np.isfinite(value), "nonfinite projected variance")
    return value


@A._checked
def lorentz_angle(state):
    delta = array(state["delta"]).astype(float)
    sdir = array(state["sensor_direction"]).astype(float)
    require(delta.ndim == 2 and delta.shape[1] == 3 and sdir.shape == delta.shape, "Lorentz vectors shape differs")
    require(np.isfinite(delta) & np.isfinite(sdir), "nonfinite Lorentz vectors")
    tperp = -delta
    bdir = np.array([0., 0., 1.])
    radial = np.cross(sdir, bdir)
    radial = np.where((A._dot3(radial, tperp) < 0.)[:, None], -radial, radial)
    angle = np.arctan2(A._dot3(tperp, radial), A._dot3(tperp, np.broadcast_to(bdir, tperp.shape)))
    return np.where(angle > np.pi/2., np.pi-angle, angle)


def cad_initial(doca, docavar, rdrift, usable, config):
    """Initial unfrozen null-state CAD transition, including native strict cut boundaries."""
    require(config == [400, 800000, 50, 50, "TOT:LongVal", "Null:Inactive", "", 0], "unsupported initial CAD config")
    doca, docavar, rdrift = (array(x).astype(float) for x in (doca, docavar, rdrift))
    usable = array(usable)
    require(doca.ndim == 1 and doca.shape == docavar.shape == rdrift.shape == usable.shape, "CAD shapes differ")
    require(np.isfinite(doca) & np.isfinite(docavar) & np.isfinite(rdrift), "nonfinite CAD operand")
    require(usable.dtype.kind in "biu" and np.all((usable == 0) | (usable == 1)), "invalid CA usable flag")
    usable = usable.astype(bool)
    maxdoca, maxerr, minr, maxr = np.asarray(config[:4], dtype=np.float32).astype(float)
    maxvar = maxerr*maxerr
    cuts = dict(drift_upper=rdrift < maxr, doca_upper=np.abs(doca) < maxdoca,
                variance_positive=docavar > 0., variance_upper=docavar < maxvar)
    eligible = np.logical_and.reduce(list(cuts.values())) & usable
    updated = usable & ((~eligible) | (rdrift <= minr))
    state = np.where(~usable, -3, np.where(eligible, 0, -2))
    flags = np.where(updated, 513, 0)
    return dict(state=state, active=state > -2, constrain_tot=(flags & 1) != 0,
                constrain_long=(flags & 512) != 0, constrain_absdrift=np.zeros(len(state), dtype=bool),
                algorithm=np.where(~usable, -1, np.where(updated, 1, 0)), frozen=np.zeros(len(state), dtype=bool),
                updated=updated, cuts=cuts, margins=dict(drift_upper=maxr-rdrift, doca_upper=maxdoca-np.abs(doca),
                variance_positive=docavar, variance_upper=maxvar-docavar))


def probe_comparison(probes, table, config):
    require(all(len(probes[k]) == 1 for k in PROBE_FIELDS), "requires one calibration probe row")
    times = array(probes["dtgrid"][0])
    require(times.shape == (2001,) and np.isfinite(times).all(), "requires all 2001 native calibration probes")
    generated = drift_info(times, table, config)
    out = {}
    for name, stored in (("rdrift", "rdrift"), ("velocity", "dvel"), ("sderr", "sderr"), ("uderr", "uderr")):
        target = array(probes[stored][0])
        require(target.shape == times.shape and np.isfinite(target).all(), "invalid native drift probe")
        out[name] = dict(criterion=_metric(generated[name], target), exact=_metric(generated[name], target, exact=True))
    return dict(n_probes=len(times), range=[float(times.min()), float(times.max())], metrics=out,
                passed=all(x["criterion"]["passed"] for x in out.values()))


def compare_population(hits, rows, qseed, qref, covariance, mass, ctime, measured, speed, table, config, policy):
    ids = keys(hits, HIT_KEYS)
    selected = rows[match_rows(ids, keys(rows, HIT_KEYS))]
    for name in ("shindex", "strawid", "input_index", "miter", "iter"):
        require(np.array_equal(column(selected, name), column(hits, name)), "sidecar identity differs: " + name)
    require(column(selected, "precision") == NATIVE_CA_PRECISION, "native precision differs")
    line = A.sensor(measured["early"], measured["late"], measured["measurement_time"], speed)
    hints = C.constructor_hint(qseed, line, measured["corrected_time"], mass)
    built = C.solve(qseed, line, *hints, precision=NATIVE_CA_PRECISION, mass=mass)
    require(built["usable"], "unusable constructor; unsupported initial branch, no exclusions")
    reference = C.solve(qref, line, built["particle_time"], built["sensor_time"], precision=NATIVE_CA_PRECISION, mass=mass)
    require(reference["usable"], "unusable reference; unsupported initial branch, no exclusions")
    require(np.abs(reference["particle_poca"][:, 2]-line["mid"][:, 2]) <= 100, "unsupported initial hit retry")
    refrows = fixed_state_rows(qref, line, reference, mass)
    compare = lambda a, b: P.comparison(a, b, ids)
    out = dict(n_hits=len(hits), n_targets=len(np.unique(ids[:, :4], axis=0)), n_excluded_hits=0,
               hit_key_fields=list(HIT_KEYS), hit_keys=ids.tolist(),
               constructor_time=compare(built["particle_time"], ctime),
               reference_geometry={k: compare(reference[k], snapshot_state(selected)[k]) for k in P.STATE_FIELDS},
               reference_doca_variance=compare(similarity(refrows["H"][:, 0], covariance), column(selected, "doca_variance")),
               arms={})
    for arm in ARMS:
        uq, uc, stats = unbiased_roundtrip(qref, covariance, arm)
        state = C.solve(uq, line, reference["particle_time"], reference["sensor_time"], precision=NATIVE_CA_PRECISION, mass=mass)
        require(state["usable"], "unusable unbiased CA; no lane exclusions")
        generated = fixed_state_rows(uq, line, state, mass)
        variance = similarity(generated["H"][:, 0], uc)
        dt = state["sensor_time"]-state["particle_time"]
        drift = drift_info(dt, table, config)
        drift["lorentz_angle"] = lorentz_angle(state)
        selection = cad_initial(generated["doca"], variance, drift["rdrift"], state["usable"], policy["cad"])
        comparisons = {"hit_state": compare(selection["state"], column(selected, "hit_state"))}
        for name in ("active", "constrain_tot", "constrain_long", "constrain_absdrift"):
            comparisons[name] = compare(selection[name], boolean(selected, name))
        out["arms"][arm] = dict(inversion=stats,
            reference_roundtrip=dict(parameters=compare(uq, qref), covariance=compare(uc, covariance)),
            reconstructed_unbiased=dict(parameters=uq.tolist(), covariance=uc.tolist(),
                state={k: state[k].tolist() for k in P.STATE_FIELDS}, doca=generated["doca"].tolist(),
                doca_variance=variance.tolist(), delta_time=dt.tolist(), drift={k: v.tolist() for k, v in drift.items()}),
            cad_outputs={k: v.tolist() for k, v in selection.items() if k not in ("cuts", "margins")},
            cad_cuts={k: v.tolist() for k, v in selection["cuts"].items()},
            margins={k: dict(minimum=float(v.min()), maximum=float(v.max()), values=v.tolist()) for k, v in selection["margins"].items()},
            state_vs_capture=comparisons, state_agreement=all(m["exact"]["passed"] for m in comparisons.values()),
            native_unbiased_intermediates_compared=False)
    return out


def run(args):
    with args.out.open("x", encoding="utf-8") as stream:
        report = dict(completed=False, full_closure=False, full_numerical_validation=False,
                      native_unbiased_intermediates_compared=False, full_state_policy_closure=False,
                      rtol=R.RTOL, atol=R.ATOL, limits=LIMITS, declared_arms=list(ARMS))
        watched = {}

        def watch(path):
            path = str(Path(path).resolve())
            sha = digest(path)
            require(path not in watched or watched[path] == sha, "source/input changed: " + path)
            watched[path] = sha
            return sha

        try:
            require(not any(k.startswith("F1_") and v for k, v in os.environ.items()), "requires default F1 environment")
            previous = receipt(args.residual_report, watched)
            here = Path(__file__).resolve().parent
            for p in [Path(__file__), here/"f1_initial_drift.py", here/"run_f1_initial_cad.sh", here/"f1_native_algebra_audit.py"]:
                watch(p)
            for name in NATIVE_PATHS:
                watch(args.native_source/name)
            report["executing_modules"] = R.executing_sources(watched)
            report["residual_receipt"] = dict(path=str(args.residual_report), sha256=RESIDUAL_REPORT_SHA256)
            prior_path = previous["ca_receipt"]["path"]
            prior = json.loads(Path(prior_path).read_text())
            old, new = prior["frozen_gate"], prior["operand_gate"]
            expanded = Path(prior["speed_configuration"]["path"])
            require(digest(expanded) == R.FROZEN_EXPANDED_SHA256, "expanded FCL changed")
            geo = source_geometry(args.native_source, expanded, watch)
            report["geometry_configuration"] = geo
            roots = [Path(p).parent for p in prior["root_source_sha256"] if p.endswith("DisplacementVector3D.h")]
            require(len(roots) == 1, "ambiguous ROOT source path")
            root = roots[0].parent
            for name in ("SMatrix.icc", "SVector.icc", "GenVector/DisplacementVector3D.h"):
                watch(root/name)
            module_source = (args.native_source/"TrkFastSim/src/KinematicLineFitTrace_module.cc").read_text()
            require("VEC3 bnom(0.0, 0.0, 0.001);" in module_source, "native seed bnom source differs")
            report["nominal_field_contract"] = dict(seed_bnom=[0., 0., .001], physical_field=0.,
                scope="KinematicLine copy constructor retains seed bnom; direction sets angle, physical field sets drift table")
            report["initial_weight_contract"] = dict(value="zero", basis="source lifecycle and ROOT default-zero SMatrix/SVector; no native pre-update snapshot",
                order="fresh KKStrawHit(null) -> Measurement.updateReference only -> first updateWHS.unbiasedParameters before updateWeight", clusters="none")
            config = drift_configuration(expanded)
            policy = R.initial_policy(expanded, args.native_source/"Offline/TrackerConditions/src/DriftInfo.cc")
            policy = dict(policy, state_eligibility_reconstructed=True,
                          scope="Independent CAD eligibility at reconstructed unbiased inputs; native unbiased intermediates not captured")
            report.update(drift_configuration=config, calibration_method=CALIBRATION_METHOD, policy=policy)
            with uproot.open(old["inputs"]["capture"]["path"]) as f:
                tables = {name: read_tree(f, "kktrace", name, old["highest_cycles"]["capture"])
                          for name in ("iterations", "pieces", "effects", "seed_contract")}
            hits, qseed, qref, mass, ctime, hkeys = P.select_initial(tables["iterations"], tables["pieces"], tables["effects"], tables["seed_contract"])
            P.population_guard(tables["iterations"], hkeys)
            covariance = reference_covariance(hits, tables["pieces"], tables["seed_contract"])
            with uproot.open(new["inputs"]["capture"]["path"]) as f:
                rows = read_tree(f, "kktrace", "hit_operands", new["highest_cycles"]["capture"])
            with uproot.open(old["inputs"]["baseline"]["path"]) as f:
                events = read_tree(f, "trkdump", "events", old["highest_cycles"]["baseline"], [*KEYS[:3], *["s_"+k for k in R.EVENT_FIELDS]])
                allgeom = read_tree(f, "trkdump", "conditions", old["highest_cycles"]["baseline"], [*R.GEOMETRY_FIELDS, *PROBE_FIELDS])
                probes = {k: array(allgeom[k]) for k in PROBE_FIELDS}
                geom = {k: array(allgeom[k]) for k in R.GEOMETRY_FIELDS}
            require(np.shape(probes["strawOuterRadius"]) == (1,) and probes["strawOuterRadius"][0] == geo["radii"]["strawOuterRadius"], "source/dumped straw radius differs")
            measured, _ = R.measured_inputs(events, geom, hkeys[:, [0, 1, 2, 4]], column(hits, "strawid"))
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                table = build_zero_field(config, geo["radii"]["strawOuterRadius"], geo["radii"]["wireRadius"])
                report["native_probe_validation"] = probe_comparison(probes, table, config)
                require(report["native_probe_validation"]["passed"], "source drift model fails native probe criterion")
                speed = P.source_signal_speed(measured["energy"], prior["speed_configuration"])
                report["population"] = compare_population(hits, rows, qseed, qref, covariance, mass, ctime, measured, speed, table, config, policy)
            report["completed"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            end = {}
            for p in watched:
                try:
                    end[p] = digest(p)
                except OSError:
                    end[p] = None
            report.update(before_sha256=watched, end_sha256=end, hashes_unchanged=bool(watched) and watched == end)
            if not report["hashes_unchanged"]:
                report.update(completed=False, integrity_error="missing/changed source/input hashes")
            stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("residual-report", "native-source", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps({k: report[k] for k in ("completed", "hashes_unchanged", "error", "integrity_error") if k in report}))
    raise SystemExit(0 if report["completed"] else 1)


if __name__ == "__main__":
    main()
