"""Measured timing profiles and opt-in joint drift fitting; not tracking efficiency.

Default profiles freeze geometry; --fit-drift jointly minimizes geometry and t0.
Both flight directions retain every selected hit. The conditional radial score
uses signed drift errors; it is not chi-square, a direction probability, or a
covariance estimate. No truth, stock-track membership, TOT drift radius or stored
propagation correction enters the timing measurement. Full event/group censuses
and unsuccessful wire-fit lanes remain explicit.
"""
import argparse
import json
import time
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from trkfastsim.digi.digicond import parse_fhicl_dump
from trkfastsim.reco.fielddrift import build_model
from trkfastsim.reco.hitreco import load_conditions, unique_straw, _pieceline
from trkfastsim.reco.stereoseed import _integer
from .run1a_drift import read_oracle, audit, compare_model
from .run1a_wirefit import prepare, _quantiles, _sha256

END_FIELDS = ("sid", "midx", "midy", "midz", "dirx", "diry", "dirz", "hlen",
              "wcalx", "wcaly", "wcalz", "whvx", "whvy", "whvz", "hpv2", "kgrid", "hpv_s0")


def load_geometry(path, keys, sh):
    """Read public wire endpoints/propagation probes, never stock track products."""
    with uproot.open(path) as root:
        tree = root["trkdump/conditions"]
        if tree.num_entries != 1:
            raise ValueError("expected exactly one reference conditions row")
        row = tree.arrays(list(END_FIELDS), library="ak")
        geom = {k: np.asarray(ak.to_numpy(row[k][0])) for k in END_FIELDS}
        events = root["trkdump/events"].arrays(["run", "subrun", "event", "s_sid"], library="ak")
    actual = np.column_stack([ak.to_numpy(events[k]) for k in ("run", "subrun", "event")])
    ev = np.repeat(np.arange(len(actual)), ak.to_numpy(ak.num(events["s_sid"])))
    if (not np.array_equal(actual, keys) or not np.array_equal(ev, sh["ev"])
            or not np.array_equal(ak.to_numpy(ak.flatten(events["s_sid"])), sh["sid"])):
        raise ValueError("reference event/hit census differs")
    return geom


def measurement_arrays(data, geom, response):
    """Bind explicit endpoints to the saved geometry and use resolved energy calibration."""
    sh, st = data["sh"], data["conditions"]["straw"]
    nh = len(sh["sid"])
    for key in ("et0", "et1", "edep", "eend"):
        a = np.asarray(sh[key])
        if a.shape != (nh,) or a.dtype.kind not in "fiu" or not np.all(np.isfinite(a)):
            raise ValueError(f"invalid timing measurement {key}")
    if (sh["eend"].dtype.kind not in "iu" or np.any((sh["eend"] < 0) | (sh["eend"] > 1))
            or np.any(sh["edep"] < 0)):
        raise ValueError("invalid early end or energy deposit")
    if not np.array_equal(sh["eend"], (sh["et1"] < sh["et0"]).astype(int)):
        raise ValueError("early-end identity inconsistent with calibrated end times")
    ids = np.asarray(geom["sid"])
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids > 65535)
            or np.any((ids & 127) >= 96) or np.any(((ids >> 7) & 7) >= 6)):
        raise ValueError("invalid reference geometry straw identities")
    us = unique_straw(ids.astype(np.int64))
    if len(np.unique(us)) != len(us) or np.any(us >= len(st["hlen"])):
        raise ValueError("duplicate/out-of-range reference geometry straw")
    for key in END_FIELDS[1:-2]:
        a = np.asarray(geom[key])
        if a.shape != ids.shape or a.dtype.kind not in "fiu" or not np.all(np.isfinite(a)):
            raise ValueError(f"invalid reference geometry {key}")
    for key in ("midx", "midy", "midz", "dirx", "diry", "dirz", "hlen"):
        if not np.array_equal(np.asarray(geom[key]), np.asarray(st[key])[us]):
            raise ValueError(f"endpoint reference differs from saved geometry: {key}")
    cal = np.column_stack([geom[k] for k in ("wcalx", "wcaly", "wcalz")])
    hv = np.column_stack([geom[k] for k in ("whvx", "whvy", "whvz")])
    mid = np.column_stack([geom[k] for k in ("midx", "midy", "midz")])
    direction = np.column_stack([geom[k] for k in ("dirx", "diry", "dirz")])
    if np.any(np.asarray(geom["hlen"]) <= 0) or not np.allclose(np.linalg.norm(direction, axis=1), 1., rtol=0, atol=1e-12):
        raise ValueError("endpoint geometry requires unit wire directions and positive half-lengths")
    extent = np.asarray(geom["hlen"])[:, None] * direction
    if (not np.allclose(cal, mid+extent, rtol=0, atol=1e-10)
            or not np.allclose(hv, mid-extent, rtol=0, atol=1e-10)):
        raise ValueError("cal/HV endpoints inconsistent with saved wire axis")
    nbins = _integer("eBins", response["eBins"], 2)
    width = float(response["eBinWidth"])
    scale = np.asarray(response["eHalfPVScale"], float)
    if (nbins > 100000 or not np.isfinite(width) or width <= 0 or scale.shape != (nbins,)
            or not np.all(np.isfinite(scale)) or np.any(scale <= 0)):
        raise ValueError("invalid resolved propagation calibration")
    egrid = np.arange(nbins)*width
    at_two = _pieceline(egrid, scale, np.asarray(2.))
    if not np.isfinite(at_two) or at_two <= 0 or np.any(np.asarray(geom["hpv2"]) <= 0):
        raise ValueError("invalid propagation reference speed")
    base = np.asarray(geom["hpv2"]) / at_two
    k, probes = np.asarray(geom["kgrid"]), np.asarray(geom["hpv_s0"])
    if (k.ndim != 1 or probes.shape != k.shape or not len(k) or not len(ids)
            or not np.all(np.isfinite(k)) or not np.all(np.isfinite(probes))):
        raise ValueError("invalid propagation calibration probes")
    if not np.allclose(_pieceline(egrid, scale, k)*base[0], probes, rtol=1e-12, atol=1e-10):
        raise ValueError("resolved propagation calibration differs from stock probes")
    lookup = np.full(len(st["hlen"]), -1, np.int64)
    lookup[us] = np.arange(len(us))
    index = lookup[unique_straw(sh["sid"])]
    if np.any(index < 0):
        raise ValueError("hit missing from endpoint geometry")
    speed = 2*base[index]*_pieceline(egrid, scale, 1000.*sh["edep"].astype(float))
    if not np.all(np.isfinite(speed)) or np.any(speed <= 0):
        raise ValueError("nonpositive/nonfinite energy-dependent signal speed")
    return dict(wire_cal=cal[index], wire_hv=hv[index], signal_speed=speed,
                early_time=np.where(sh["eend"] == 0, sh["et0"], sh["et1"]).astype(float),
                early_end=sh["eend"])


def match_conditions_config(reference, oracle):
    """Match configured conditions/geometry services without claiming a database snapshot hash."""
    for service in ("ProditionsService", "GeometryService", "DbService", "GlobalConstantsService"):
        if service not in reference["services"] or reference["services"][service] != oracle["services"].get(service):
            raise ValueError(f"reference/oracle conditions configuration differs: {service}")


def prepare_saved(keys, sh, cond, association, wirefit):
    """Validate original support and the saved wire-fit census without refitting."""
    data = prepare(keys, sh, cond, association)
    data["conditions"] = cond
    ng, nh = len(data["gev"]), len(data["hg"])
    identity = {"evkey": keys, "source_hit_ev": data["sh"]["ev"], "source_hit_sid": data["sh"]["sid"],
                "hit_group": data["hg"], "group_ev": data["gev"], "group_support": data["count"],
                "source_selected": association["association_source_used"]}
    for key, value in identity.items():
        kind = "b" if key == "source_selected" else "iu"
        if (key not in wirefit or np.asarray(wirefit[key]).dtype.kind not in kind
                or not np.array_equal(wirefit[key], value)):
            raise ValueError(f"wire-fit census/support differs: {key}")
    for key in ("fit_success", "fit_converged", "fit_physical_identifiable"):
        a = np.asarray(wirefit[key])
        if a.shape != (ng,) or a.dtype.kind != "b":
            raise ValueError(f"invalid wire-fit flag {key}")
    if (not np.array_equal(wirefit["fit_success"], wirefit["fit_converged"])
            or np.any(wirefit["fit_physical_identifiable"] & ~wirefit["fit_success"])
            or np.any(wirefit["fit_success"] & (data["count"] < 8))):
        raise ValueError("inconsistent wire-fit success flags")
    ok = wirefit["fit_success"] & wirefit["fit_physical_identifiable"]
    for key, shape in (("fit_local_parameters", (ng, 5)), ("fit_z_ref", (ng,)),
                       ("fit_p_abs", (ng,)), ("fit_status", (ng,))):
        a = np.asarray(wirefit[key])
        if a.shape != shape or (a.dtype.kind not in "US" if key == "fit_status" else
                                a.dtype.kind not in "fiu" or not np.all(np.isfinite(a[ok]))):
            raise ValueError(f"invalid saved fit column {key}")
    if not np.array_equal(wirefit["fit_status"] == "converged", wirefit["fit_success"]):
        raise ValueError("wire-fit status/success mismatch")
    selected = data["selected"]
    hit_ok = ok[data["hg"][selected]]
    eligible = selected[hit_ok]
    for key in ("source_poca_z", "source_doca", "source_wire_u", "source_wire_pull"):
        a = np.asarray(wirefit[key])
        if a.shape != (nh,) or a.dtype.kind not in "fiu" or not np.all(np.isfinite(a[eligible])):
            raise ValueError(f"invalid saved POCA diagnostic {key}")
    for key in ("source_poca_valid", "source_poca_converged", "source_poca_inside", "source_poca_boundary"):
        a = np.asarray(wirefit[key])
        if a.shape != (nh,) or a.dtype.kind != "b" or not np.all(a[eligible] == (key != "source_poca_boundary")):
            raise ValueError(f"unusable saved POCA {key}")
    data.update(eligible=ok, wirefit=wirefit)
    return data


def run(keys, sh, cond, association, wirefit, geom, response, model, *, bz,
        group_batch=32, max_query_cells=262144, mass=105.6583755, fit_drift=False,
        drift_max_iterations=40, **profile_settings):
    """Profile both directions in bounded batches, retaining fixed full-width reductions."""
    from trkfastsim.reco.helixtiming import prepare_timing, profile_t0
    if not isinstance(fit_drift, (bool, np.bool_)):
        raise ValueError("fit_drift must be boolean")
    drift_max_iterations = _integer("drift_max_iterations", drift_max_iterations, 0)
    data = prepare_saved(keys, sh, cond, association, wirefit)
    profile_settings = {"offset_min": -200., "offset_max": 50., "grid_step": .5, "refinements": 2, **profile_settings}
    measurements = measurement_arrays(data, geom, response)
    ng, nh, ne = len(data["gev"]), len(data["hg"]), len(keys)
    group_batch = _integer("group_batch", group_batch, 1)
    max_query_cells = _integer("max_query_cells", max_query_cells, 1)
    count, selected = data["count"], data["selected"]
    width = int(count.max(initial=0))
    if width > 256 or 2*width > max_query_cells:
        raise ValueError("timing row/query-cell bound exceeded")
    batch = min(group_batch, 512, max(1, min(max_query_cells, 65536) // (2*max(1, width))))
    starts = np.r_[0, np.cumsum(count)]
    payload = dict(evkey=np.asarray(keys), source_hit_ev=data["sh"]["ev"], source_hit_sid=data["sh"]["sid"],
                   hit_group=data["hg"], group_ev=data["gev"], source_selected=np.asarray(association["association_source_used"]),
                   group_support=data["count"])
    payload.update({key: np.asarray(wirefit[key]) for key in ("fit_success", "fit_physical_identifiable", "fit_status")})
    row_result, hit_result = {}, {}
    directions = np.array([-1, 1])
    for begin in range(0, max(1, ng), batch):
        groups = np.arange(begin, min(begin+batch, ng))
        valid = np.arange(width)[None, :] < count[groups, None]
        ranks = starts[groups, None] + np.arange(width)[None, :]
        idx = np.zeros(valid.shape, np.int64)
        if len(selected):
            idx[valid] = selected[ranks[valid]]
        padded = {}
        for key, a in measurements.items():
            padded[key] = np.zeros(valid.shape + a.shape[1:], dtype=a.dtype)
            padded[key][valid] = a[idx[valid]]
        poca_z = np.zeros(valid.shape)
        poca_z[valid] = wirefit["source_poca_z"][idx[valid]]
        t = prepare_timing(wirefit["fit_local_parameters"][groups], wirefit["fit_z_ref"][groups],
                           poca_z, **padded, valid=valid, bz=bz, mass=mass,
                           seed_valid=data["eligible"][groups])
        check = valid & data["eligible"][groups, None]
        fitted_u = np.zeros(valid.shape)
        fitted_u[valid] = np.sum((t["sensor_poca"][valid] - data["mid"][idx[valid]]) *
                                 data["direction"][idx[valid]], axis=1)
        fitted_pull = np.zeros(valid.shape)
        fitted_pull[valid] = (fitted_u[valid] - data["u"][idx[valid]]) / np.sqrt(data["uvar"][idx[valid]])
        for key, actual in (("source_doca", t["doca"]), ("source_wire_u", fitted_u), ("source_wire_pull", fitted_pull)):
            if not np.allclose(actual[check], wirefit[key][idx[check]], rtol=1e-10, atol=1e-8):
                raise ValueError("saved fitted POCA residual replay differs")
        pars = np.where(data["eligible"][groups, None], wirefit["fit_local_parameters"][groups], 0.)
        speed = np.hypot(1., np.hypot(pars[:, 2], pars[:, 3]))
        p_abs = .299792458*abs(bz)*speed/np.where(data["eligible"][groups], np.abs(pars[:, 4]), 1.)
        if not np.allclose(p_abs[data["eligible"][groups]], wirefit["fit_p_abs"][groups][data["eligible"][groups]],
                           rtol=1e-12, atol=1e-9):
            raise ValueError("wire-fit momentum/field hypothesis mismatch")
        prof = profile_t0(model, t, valid, max_query_cells=max_query_cells, **profile_settings)
        if not np.array_equal(prof["direction"], directions):
            raise RuntimeError("unexpected direction order")
        if fit_drift:
            from trkfastsim.reco.helixdriftfit import fit_drift_helix
            repeated = {k: np.repeat(a, 2, axis=0) for k, a in padded.items()}
            u, uv = np.zeros(valid.shape), np.ones(valid.shape)
            u[valid], uv[valid] = data["u"][idx[valid]], data["uvar"][idx[valid]]
            result = fit_drift_helix(np.repeat(wirefit["fit_local_parameters"][groups], 2, axis=0),
                np.repeat(wirefit["fit_z_ref"][groups], 2), prof["t0"].reshape(-1),
                np.tile(directions, len(groups)), **repeated, measured_u=np.repeat(u, 2, axis=0),
                uvar=np.repeat(uv, 2, axis=0), valid=np.repeat(valid, 2, axis=0), model=model,
                bz=bz, mass=mass, seed_valid=prof["profile_valid"].reshape(-1), max_iterations=drift_max_iterations)
            hit_names = {"initial_LR", "doca", "signed_doca", "phi", "delta_t", "r_drift", "signed_error",
                         "drift_residual", "pull", "long_pull", "wire_u", "poca_z", "poca_valid", "poca_iterations",
                         "poca_converged", "poca_inside", "poca_boundary", "mask", "valid", "residuals",
                         "timing_valid", "drift_valid", "inside", "stationary", "parallel", "angle_valid", "sign_valid",
                         "outside_oracle_mask"}
            for key, value in result.items():
                if key in ("status_counts", "objective_label"):
                    continue
                a = np.asarray(value)
                if a.shape[:1] != (2*len(groups),):
                    raise RuntimeError(f"unexpected drift-fit output {key} shape {a.shape}")
                a = a.reshape((len(groups), 2) + a.shape[1:])
                if key in hit_names:
                    name = "drift_"+key
                    hit_result.setdefault(name, np.full((nh, 2) + a.shape[3:], np.nan if a.dtype.kind == "f" else 0, a.dtype))
                    hit_result[name][idx[valid]] = np.moveaxis(a, 1, 2)[valid]
                else:
                    row_result.setdefault("drift_"+key, []).append(a)
        for key, a in t.items():
            a = np.asarray(a)
            if a.shape[:2] == valid.shape and a.ndim >= 2:
                name = "timing_" + key
                hit_result.setdefault(name, np.full((nh,) + a.shape[2:], np.nan if a.dtype.kind == "f" else 0, a.dtype))
                hit_result[name][idx[valid]] = a[valid]
            elif a.shape[:1] == groups.shape:
                row_result.setdefault("timing_"+key, []).append(a)
        for key, a in prof.items():
            a = np.asarray(a)
            if key == "direction":
                continue
            if a.shape == (len(groups), 2, width):
                name = "profile_" + key
                hit_result.setdefault(name, np.full((nh, 2), np.nan if a.dtype.kind == "f" else 0, a.dtype))
                hit_result[name][idx[valid]] = np.moveaxis(a, 1, 2)[valid]
            elif a.shape[:1] == groups.shape:
                row_result.setdefault("profile_"+key, []).append(a)
            elif a.ndim == 0:
                payload["profile_"+key] = a
            else:
                raise RuntimeError(f"unexpected profile output {key} shape {a.shape}")
    payload.update({k: np.concatenate(v) for k, v in row_result.items()})
    payload.update({"source_"+k: v for k, v in hit_result.items()})
    payload["profile_direction"] = directions
    pair = np.all(payload["profile_profile_valid"], axis=1)
    score = payload["profile_score"]
    payload["score_negative_minus_positive"] = np.where(pair, score[:, 0]-score[:, 1], np.nan)
    payload["event_timing_profiles"] = np.bincount(data["gev"][pair], minlength=ne)
    report = dict(events=ne, groups=ng, hits=nh, zero_hit_events=int(np.sum(np.bincount(data["sh"]["ev"], minlength=ne) == 0)),
                  fixed_selected_hits=len(selected), successful_wirefits=int(np.sum(wirefit["fit_success"])),
                  eligible_wirefits=int(np.sum(data["eligible"])), timing_geometry_valid=int(np.sum(payload["timing_row_valid"])),
                  both_directions_profiled=int(np.sum(pair)), profiled_source_hits=int(np.sum(count[pair])),
                  events_profiled=int(np.sum(payload["event_timing_profiles"] > 0)),
                  geometry_status={str(k): int(v) for k, v in zip(*np.unique(payload["timing_status"], return_counts=True))},
                  direction_order=directions.tolist(), mass_hypothesis_MeV=float(mass), nominal_axial_bz_T=float(bz),
                  origin_drift_bz_T=float(model.bz), group_batch=group_batch, max_query_cells=max_query_cells,
                  profile_settings={k: np.asarray(v).item() for k, v in profile_settings.items()}, selected_fit_residuals_replayed=True,
                  original_association_inputs_hash_bound=False, assigned_directions=0,
                  score_difference_negative_minus_positive=_quantiles(payload["score_negative_minus_positive"][pair]),
                  caveat=__doc__)
    report["score_ordering_diagnostic_only"] = dict(negative_lower=int(np.sum(pair & (score[:, 0] < score[:, 1]))),
                                                    positive_lower=int(np.sum(pair & (score[:, 1] < score[:, 0]))),
                                                    exact_ties=int(np.sum(pair & (score[:, 0] == score[:, 1]))))
    for i, label in enumerate(("negative_z", "positive_z")):
        good = payload["profile_profile_valid"][:, i]
        source_good = np.zeros(nh, bool)
        source_good[selected] = good[data["hg"][selected]]
        report[label] = dict(profiles=int(np.sum(good)), boundary=int(np.sum(payload["profile_boundary"][:, i])),
                             status={str(k): int(v) for k, v in zip(*np.unique(payload["profile_status"][:, i], return_counts=True))},
                             t0_ns=_quantiles(payload["profile_t0"][good, i]),
                             offset_from_measured_anchor_ns=_quantiles(payload["profile_offset"][good, i]),
                             conditional_score=_quantiles(score[good, i]),
                             delta_t_ns=_quantiles(payload["source_profile_delta_t"][source_good, i]),
                             residual_mm=_quantiles(payload["source_profile_drift_residual"][source_good, i]),
                             abs_residual_mm=_quantiles(np.abs(payload["source_profile_drift_residual"][source_good, i])),
                             abs_pull=_quantiles(np.abs(payload["source_profile_pull"][source_good, i])),
                             outside_oracle_hits=int(np.sum(payload["profile_outside_oracle"][good, i])))
    if fit_drift:
        success = payload["drift_success"]
        attempted = payload["profile_profile_valid"]
        report["drift_fit"] = dict(max_iterations=drift_max_iterations,
            numerical_settings=dict(condition_limit=1e12, gradient_tolerance=1e-6, fd_step=1e-3,
                                    initial_damping=1e-3, max_parameter_step=10., poca_iterations=20,
                                    poca_tolerance=1e-7, poca_max_step=50., poca_max_offset=500.),
            physical_identifiable_hypotheses=int(np.sum(payload["drift_physical_identifiable"])),
            attempted_hypotheses=int(attempted.sum()),
            successful_hypotheses=int(success.sum()), groups_any_success=int(np.any(success, axis=1).sum()),
            groups_both_success=int(np.all(success, axis=1).sum()), direction_order=directions.tolist(),
            caveat="Joint constant-axial-field geometry/t0 minimization with frozen support and initial LR; "
                   "no direction decision, material, field-map propagation, outlier handling, or covariance.")
        payload["event_drift_fit_hypotheses"] = np.bincount(data["gev"], weights=success.sum(axis=1), minlength=ne).astype(np.int64)
        report["drift_fit"]["events_any_converged"] = int(np.count_nonzero(payload["event_drift_fit_hypotheses"]))
        for i, label in enumerate(("negative_z", "positive_z")):
            ok = success[:, i]
            eligible = attempted[:, i]
            source_ok = np.zeros(nh, bool)
            source_ok[selected] = ok[data["hg"][selected]]
            obj_valid = payload["drift_objective_valid"][:, i]
            initial, final = payload["drift_initial_objective"][:, i], payload["drift_final_objective"][:, i]
            if np.any(obj_valid & (final > initial + 1e-8)):
                raise RuntimeError("drift fit increased fixed-membership objective")
            current_signed = payload["source_drift_signed_doca"][source_ok, i]
            initial_lr = payload["source_drift_initial_LR"][source_ok, i]
            entry = dict(converged=int(ok.sum()), attempted=int(eligible.sum()),
                         spatial_sign_crossings_converged=int(np.sum(np.sign(current_signed) != initial_lr)),
                         outside_oracle_converged=int(np.sum(payload["drift_outside_oracle"][ok, i])),
                         status={str(k): int(v) for k, v in zip(*np.unique(payload["drift_status"][:, i], return_counts=True))},
                         converged_hits=int(source_ok.sum()), objective_decreased=int(np.sum(obj_valid & (final < initial))))
            for name in ("initial_objective", "final_objective", "condition", "gradient_norm", "iterations"):
                entry[name+"_converged"] = _quantiles(payload["drift_"+name][ok, i])
            entry["initial_abs_radial_residual_mm_same_hits"] = _quantiles(np.abs(payload["source_profile_drift_residual"][source_ok, i]))
            for name in ("drift_residual", "pull", "long_pull"):
                entry[name+"_abs_converged_hits"] = _quantiles(np.abs(payload["source_drift_"+name][source_ok, i]))
            report["drift_fit"][label] = entry
    return report, payload


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("hits", "conditions", "association", "wirefit", "reference", "oracle", "config", "drift-check", "outdir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--group-batch", type=int, default=32)
    p.add_argument("--max-query-cells", type=int, default=262144)
    p.add_argument("--compare-saved", type=Path)
    p.add_argument("--fit-drift", action="store_true", help="opt-in joint six-parameter fixed-membership fit")
    p.add_argument("--drift-max-iterations", type=int, default=40)
    args = p.parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    started = time.perf_counter()
    with np.load(args.hits, allow_pickle=False) as src:
        keys = src["evkey"]
        sh = {k[2:]: src[k] for k in src.files if k.startswith("s_")}
    with np.load(args.association, allow_pickle=False) as src:
        association = dict(src)
    with np.load(args.wirefit / "diagnostics.npz", allow_pickle=False) as src:
        wirefit = dict(src)
    wire_report = json.loads((args.wirefit / "report.json").read_text())
    for key in ("hits", "conditions", "association"):
        if wire_report["inputs"][key]["sha256"] != _sha256(getattr(args, key)):
            raise ValueError(f"wire-fit recorded input hash differs: {key}")
    cond = load_conditions(args.conditions)
    geom = load_geometry(args.reference, keys, sh)
    config = parse_fhicl_dump(args.config.read_text())
    reference_config = args.reference.parent / "expanded.fcl"
    match_conditions_config(parse_fhicl_dump(reference_config.read_text()), config)
    oracle = read_oracle(args.oracle)
    checked = json.loads(args.drift_check.read_text())
    if not checked.get("passed") or not checked.get("model", {}).get("passed"):
        raise ValueError("a successful drift-model audit report is required")
    for key, path in (("oracle", args.oracle), ("config", args.config),
                      ("export_log", args.oracle.parent / "export.log"),
                      ("export_done", args.oracle.parent / "export.done"),
                      ("source_manifest", args.oracle.parent / "source-sha256.txt")):
        if checked["inputs"][key]["sha256"] != _sha256(path):
            raise ValueError(f"drift audit recorded input differs: {key}")
    oracle_audit = audit(oracle, config)
    if not compare_model(oracle, config)["passed"]:
        raise ValueError("source drift evaluator no longer passes this stock oracle")
    if len(keys):
        runsub = keys[:, :2]
        low, high = (oracle["iov_start_run"], oracle["iov_start_subrun"]), (oracle["iov_end_run"], oracle["iov_end_subrun"])
        inside = ((runsub[:, 0] > low[0]) | ((runsub[:, 0] == low[0]) & (runsub[:, 1] >= low[1]))) & (
            (runsub[:, 0] < high[0]) | ((runsub[:, 0] == high[0]) & (runsub[:, 1] <= high[1])))
        if not np.all(inside):
            raise ValueError("input events outside drift oracle IOV")
    pro = config["services"]["ProditionsService"]
    model = build_model(pro["strawDrift"], pro["strawResponse"], wire_radius=oracle["wire_radius"],
                        outer_radius=oracle["outer_radius"], bz=oracle["origin_field"][2])
    loaded = time.perf_counter()
    report, payload = run(keys, sh, cond, association, wirefit, geom, pro["strawResponse"], model,
                          bz=wire_report["bz_tesla"], group_batch=args.group_batch, max_query_cells=args.max_query_cells,
                          fit_drift=args.fit_drift, drift_max_iterations=args.drift_max_iterations)
    report.update(load_and_oracle_check_seconds=loaded-started, processing_seconds=time.perf_counter()-loaded,
                  processing_scope="timing profiles plus joint drift fit" if args.fit_drift else "timing profiles only",
                  oracle_queries=oracle_audit["queries"], wirefit_recorded_input_hashes_verified=True,
                  provenance_caveat="Wire-fit report hashes bind its recorded inputs, not the original association "
                    "or its output NPZ; identities and successful fitted residuals are replayed. Current files are hashed.")
    paths = {name: getattr(args, name) for name in ("hits", "conditions", "association", "reference", "oracle", "config")}
    paths.update(wirefit=args.wirefit / "diagnostics.npz", wirefit_report=args.wirefit / "report.json",
                 drift_check=args.drift_check, reference_config=reference_config, driver=Path(__file__))
    from trkfastsim.reco import helixtiming, fielddrift
    paths.update(timing_source=Path(helixtiming.__file__), drift_source=Path(fielddrift.__file__))
    if args.fit_drift:
        from trkfastsim.reco import helixdriftfit
        paths["drift_fit_source"] = Path(helixdriftfit.__file__)
    report["inputs"] = {k: dict(path=str(path.resolve()), sha256=_sha256(path)) for k, path in paths.items()}
    if args.compare_saved:
        with np.load(args.compare_saved, allow_pickle=False) as previous:
            if set(previous.files) != set(payload):
                raise ValueError("timing repeat schema differs")
            different = [k for k, a in payload.items() if not np.array_equal(previous[k], a, equal_nan=a.dtype.kind in "fc")]
        report["saved_comparison"] = dict(exact=not different, different_arrays=different)
        if different:
            raise ValueError(f"timing repeat differs: {different}")
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.outdir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.outdir / "diagnostics.npz", **payload)
    (args.outdir / "report.json").write_text(text)
    print(text, end="")
    return report


if __name__ == "__main__":
    main()
