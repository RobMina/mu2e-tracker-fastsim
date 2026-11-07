"""Full-census saved-wire timing driver tests; no stock-track or truth inputs."""
import copy
import json

import numpy as np
import pytest

from trkfastsim.reco.fielddrift import build_model
from trkfastsim.reco.helixfit import _trajectory
from trkfastsim.reco.helixseed import fit_helix_seed
from trkfastsim.reco.helixwire import wire_residuals
from trkfastsim.reco.hitreco import FLAG
from trkfastsim.validation.run1a_timing import measurement_arrays, prepare_saved, run


def fixture():
    n = 12
    z = np.linspace(-800, 800, n)
    local = np.array([[20., -10., .4, -.2, .0006]])
    pos, tangent, _ = _trajectory(local, np.zeros(1), z[None, :])
    pos, tangent = pos[0], tangent[0]
    a = np.arange(n)*1.11
    d = np.column_stack((np.cos(a), np.sin(a), .001*np.sin(2*a)))
    d /= np.linalg.norm(d, axis=1)[:, None]
    normal = np.cross(tangent, d)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    doca = .7 + .1*np.sin(a)
    sensor = pos-doca[:, None]*normal
    mid = sensor-10*d
    hlen = np.full(n, 1000.)
    direction = (np.arange(n) % 2).astype(np.int64)
    early = np.full(n, 100.) + (hlen + (2*direction-1)*10)/200.
    sh = dict(ev=np.ones(n, np.int64), sid=np.arange(n), eend=direction,
              et0=np.where(direction == 0, early, early+.2), et1=np.where(direction == 1, early, early+.2),
              edep=np.full(n, .002), time=np.full(n, 100.), timevar=np.ones(n),
              uvar=np.full(n, 100.), flag=np.full(n, 1 << FLAG["tdiv"], np.uint32),
              **dict(zip(("x", "y", "z"), sensor.T)))
    st = dict(zip(("midx", "midy", "midz"), mid.T))
    st.update(zip(("dirx", "diry", "dirz"), d.T))
    st["hlen"] = hlen
    cond = {"straw": st}
    keys = np.array([[1430, 0, 21], [1430, 0, 23], [1430, 0, 29]])
    seed = fit_helix_seed(pos[None, :, :], np.ones((1, n), bool), bz=1.0117)
    base = {k: seed[k] for k in ("success", "center_xy", "radius", "pitch", "phase_ref", "z_ref")}
    seed = {k: np.repeat(v, 3, axis=0) for k, v in base.items()}
    seed["success"] = np.array([False, True, False])
    res = wire_residuals(base, mid, d, np.full(n, 10.), np.full(n, 100.), hlen)
    assoc = dict(evkey=keys, source_hit_ev=sh["ev"], source_hit_sid=sh["sid"], hit_group=np.ones(n, np.int64),
                 group_ev=np.arange(3), association_source_used=np.ones(n, bool), association_group_support=np.array([0, n, 0]))
    assoc.update({"associated_seed_"+k: v for k, v in seed.items()})
    assoc.update({"association_source_"+k: res[f][0] for k, f in (("doca", "doca"), ("wire_pull", "wire_pull"), ("poca_z", "z"))})
    success = np.array([False, True, False])
    fit = dict(evkey=keys, source_hit_ev=sh["ev"], source_hit_sid=sh["sid"], hit_group=assoc["hit_group"],
               group_ev=assoc["group_ev"], source_selected=np.ones(n, bool), group_support=np.array([0, n, 0]),
               fit_success=success, fit_converged=success.copy(), fit_physical_identifiable=success.copy(),
               fit_local_parameters=np.repeat(local, 3, axis=0), fit_z_ref=np.zeros(3),
               fit_p_abs=np.full(3, .299792458*1.0117*np.sqrt(1+.4**2+.2**2)/.0006),
               fit_status=np.array(["empty", "converged", "empty"]), source_poca_z=z,
               source_doca=doca, source_wire_u=np.full(n, 10.), source_wire_pull=np.zeros(n),
               source_poca_valid=np.ones(n, bool), source_poca_converged=np.ones(n, bool),
               source_poca_inside=np.ones(n, bool), source_poca_boundary=np.zeros(n, bool))
    cal, hv = mid+hlen[:, None]*d, mid-hlen[:, None]*d
    geom = {k: v.copy() for k, v in st.items()}
    geom.update(sid=np.arange(n), hpv2=np.full(n, 100.), kgrid=np.array([0., 1., 2., 3.]), hpv_s0=np.full(4, 100.))
    geom.update(zip(("wcalx", "wcaly", "wcalz"), cal.T))
    geom.update(zip(("whvx", "whvy", "whvz"), hv.T))
    response = dict(eBins=3, eBinWidth=1., eHalfPVScale=[1., 1., 1.], driftIgnorePhi=False,
                    dRdTScale=1., driftOffBins=[0., 2.5], driftOffset=[0., 0., 0., 0.],
                    driftRMSBins=[0., 2.5], signedDriftRMS=[.16]*4, unsignedDriftRMS=[.12]*4)
    drift = dict(wireVoltage=1400., kVcm=[.4, .8, 2., 6., 12., 24.], cmus=[4., 4.5, 5., 6., 6.5, 6.6],
                 phiBins=7, deltaDistance=.01, deltaTime=.1)
    model = build_model(drift, response, wire_radius=.0125, outer_radius=2.5235, bz=1.0117056584464001)
    return keys, sh, cond, assoc, fit, geom, response, model


def test_prepare_saved_full_census():
    args = fixture()
    data = prepare_saved(*args[:5])
    np.testing.assert_array_equal(data["eligible"], [False, True, False])
    assert len(data["selected"]) == 12
    assert len(data["keys"]) == 3


def test_measurements_use_early_end_and_resolved_speed():
    args = fixture()
    data = prepare_saved(*args[:5])
    values = measurement_arrays(data, args[5], args[6])
    np.testing.assert_array_equal(values["early_time"], np.minimum(args[1]["et0"], args[1]["et1"]))
    np.testing.assert_array_equal(values["signal_speed"], np.full(12, 200.))
    modified = copy.deepcopy(args)
    modified[1]["time"][:] = -1e8
    modified[1]["timevar"][:] = 1e12
    modified[1]["ptime"] = np.full(12, 1e7)
    modified[1]["dtime"] = np.full(12, -1e6)
    other = measurement_arrays(prepare_saved(*modified[:5]), modified[5], modified[6])
    for k in values:
        np.testing.assert_array_equal(values[k], other[k])


@pytest.mark.parametrize("field", ["evkey", "source_hit_ev", "source_hit_sid", "group_support", "source_selected"])
def test_wirefit_census_alteration_rejected(field):
    args = fixture()
    args[4][field] = args[4][field] + 1
    with pytest.raises(ValueError, match="census/support"):
        prepare_saved(*args[:5])


@pytest.mark.parametrize("field,value", [("fit_success", np.array([True, True, False])),
                                        ("fit_physical_identifiable", np.array([True, True, False])),
                                        ("fit_status", np.array(["empty", "empty", "empty"])),
                                        ("source_poca_boundary", np.ones(12, bool))])
def test_invalid_fit_flags_rejected(field, value):
    args = fixture()
    args[4][field] = value
    with pytest.raises(ValueError):
        prepare_saved(*args[:5])


@pytest.mark.parametrize("key", ["wcalx", "whvz", "midx", "hpv_s0", "sid"])
def test_endpoint_or_calibration_binding(key):
    args = fixture()
    data = prepare_saved(*args[:5])
    args[5][key] = args[5][key] + 1
    with pytest.raises(ValueError):
        measurement_arrays(data, args[5], args[6])


def test_energy_dependent_propagation_is_not_constant():
    args = fixture()
    args[6]["eHalfPVScale"] = [1., 1.1, 1.2]
    args[5]["hpv2"][:] = 120.
    args[5]["hpv_s0"] = np.array([100., 110., 120., 130.])
    args[1]["edep"] = np.linspace(0, .003, 12)
    result = measurement_arrays(prepare_saved(*args[:5]), args[5], args[6])
    np.testing.assert_allclose(result["signal_speed"], 200*(1+100*args[1]["edep"]), atol=1e-12)


def test_run_preserves_all_events_support_and_no_direction_assignment():
    report, out = run(*fixture(), bz=1.0117)
    assert report["events"] == report["groups"] == 3
    assert report["hits"] == report["fixed_selected_hits"] == 12
    assert report["zero_hit_events"] == 2
    assert report["both_directions_profiled"] == report["eligible_wirefits"] == 1
    assert report["assigned_directions"] == 0
    assert out["profile_t0"].shape == (3, 2)
    assert out["source_profile_delta_t"].shape == (12, 2)
    np.testing.assert_array_equal(out["event_timing_profiles"], [0, 1, 0])
    assert np.all(out["source_selected"])
    json.dumps(report, allow_nan=False)


def test_group_batch_and_query_cell_exact():
    args = fixture()
    _, whole = run(*args, bz=1.0117)
    _, chunk = run(*args, bz=1.0117, group_batch=1, max_query_cells=37)
    for key in whole:
        np.testing.assert_array_equal(whole[key], chunk[key], err_msg=key)


def test_failed_spatial_fit_retained_but_not_profiled():
    args = fixture()
    args[4]["fit_success"] = np.zeros(3, bool)
    args[4]["fit_converged"] = np.zeros(3, bool)
    args[4]["fit_physical_identifiable"] = np.zeros(3, bool)
    args[4]["fit_status"] = np.array(["empty", "max_iterations", "empty"])
    report, out = run(*args, bz=1.0117)
    assert report["both_directions_profiled"] == 0
    assert out["source_selected"].sum() == 12
    assert report["fixed_selected_hits"] == 12


@pytest.mark.parametrize("field", ["source_doca", "fit_p_abs", "source_poca_z"])
def test_changed_fit_diagnostics_rejected(field):
    args = fixture()
    args[4][field] = args[4][field] + 1
    with pytest.raises(ValueError):
        run(*args, bz=1.0117)


@pytest.mark.parametrize("fit_drift", [False, True])
def test_no_hits_keeps_all_events_and_groups(fit_drift):
    args = list(fixture())
    args[1] = {k: v[:0] for k, v in args[1].items()}
    assoc, fit = args[3], args[4]
    for key in tuple(assoc):
        if key.startswith("source_hit_") or key == "hit_group" or key.startswith("association_source_"):
            assoc[key] = assoc[key][:0]
    assoc["association_group_support"][:] = 0
    assoc["associated_seed_success"][:] = False
    for key in tuple(fit):
        if key.startswith("source_") or key == "hit_group":
            fit[key] = fit[key][:0]
    fit["group_support"][:] = 0
    for key in ("fit_success", "fit_converged", "fit_physical_identifiable"):
        fit[key] = np.zeros(3, bool)
    fit["fit_status"] = np.full(3, "empty")
    report, out = run(*args, bz=1.0117, fit_drift=fit_drift)
    assert report["events"] == report["zero_hit_events"] == report["groups"] == 3
    assert report["fixed_selected_hits"] == report["both_directions_profiled"] == 0
    assert out["source_profile_delta_t"].shape == (0, 2)


@pytest.mark.parametrize("fit_drift", [False, True])
def test_no_groups(fit_drift):
    args = list(fixture())
    args[1] = {k: v[:0] for k, v in args[1].items()}
    for table in (args[3], args[4]):
        for key in tuple(table):
            if key != "evkey":
                table[key] = table[key][:0]
    report, out = run(*args, bz=1.0117, fit_drift=fit_drift)
    assert report["events"] == report["zero_hit_events"] == 3
    assert report["groups"] == report["both_directions_profiled"] == 0
    assert out["profile_t0"].shape == (0, 2)


def test_conditions_service_match():
    from trkfastsim.validation.run1a_timing import match_conditions_config
    services = {k: {"configuration": 1} for k in ("ProditionsService", "GeometryService", "DbService", "GlobalConstantsService")}
    cfg = dict(services=services)
    match_conditions_config(cfg, copy.deepcopy(cfg))
    altered = copy.deepcopy(cfg)
    altered["services"]["GeometryService"]["configuration"] = 2
    with pytest.raises(ValueError, match="GeometryService"):
        match_conditions_config(cfg, altered)


def cli_fixture(tmp_path, monkeypatch):
    import awkward as ak
    import uproot
    from trkfastsim.validation.run1a_wirefit import _sha256
    args = fixture()
    keys, sh, cond, assoc, fit, geom, response, model = args
    hits = tmp_path / "hits.npz"
    conditions = tmp_path / "conditions.json"
    association = tmp_path / "association.npz"
    reference = tmp_path / "hits.root"
    np.savez_compressed(hits, evkey=keys, **{"s_"+k: v for k, v in sh.items()})
    saved_conditions = {"straw": {k: v.tolist() for k, v in cond["straw"].items()},
                        **{k: [] for k in ("halfvpscale", "centres", "resslope", "totdtime", "totderror")}}
    conditions.write_text(json.dumps(saved_conditions))
    np.savez_compressed(association, **assoc)
    wirefit = tmp_path / "wirefit"
    wirefit.mkdir()
    np.savez_compressed(wirefit / "diagnostics.npz", **fit)
    report = dict(bz_tesla=1.0117, inputs={k: dict(sha256=_sha256(v)) for k, v in
                  (("hits", hits), ("conditions", conditions), ("association", association))})
    (wirefit / "report.json").write_text(json.dumps(report))
    with uproot.recreate(reference) as root:
        root.mktree("trkdump/conditions", {k: ak.Array([v]) for k, v in geom.items()})
        root.mktree("trkdump/events", dict(run=keys[:, 0], subrun=keys[:, 1], event=keys[:, 2],
                                         s_sid=ak.unflatten(sh["sid"], [0, 12, 0])))
    services = {k: {} for k in ("ProditionsService", "GeometryService", "DbService", "GlobalConstantsService")}
    services["ProditionsService"] = dict(strawResponse=response, strawDrift={})
    config = dict(services=services)
    config_path = tmp_path / "expanded.fcl"
    config_path.write_text(json.dumps(config))
    oracle = tmp_path / "oracle.root"
    oracle.write_bytes(b"synthetic test double")
    oracle_data = dict(iov_start_run=1400, iov_start_subrun=0, iov_end_run=1499, iov_end_subrun=999999,
                       wire_radius=.0125, outer_radius=2.5235, origin_field=np.array([0., 0., model.bz]))
    for name in ("export.log", "export.done", "source-sha256.txt"):
        (tmp_path / name).write_text("synthetic sidecar")
    check = tmp_path / "drift-check.json"
    paths = dict(oracle=oracle, config=config_path, export_log=tmp_path/"export.log",
                 export_done=tmp_path/"export.done", source_manifest=tmp_path/"source-sha256.txt")
    check.write_text(json.dumps(dict(passed=True, model=dict(passed=True),
                                    inputs={k: dict(sha256=_sha256(v)) for k, v in paths.items()})))
    monkeypatch.setattr("trkfastsim.validation.run1a_timing.parse_fhicl_dump", json.loads)
    monkeypatch.setattr("trkfastsim.validation.run1a_timing.read_oracle", lambda path: oracle_data)
    monkeypatch.setattr("trkfastsim.validation.run1a_timing.audit", lambda *args: dict(queries=10))
    monkeypatch.setattr("trkfastsim.validation.run1a_timing.compare_model", lambda *args: dict(passed=True))
    monkeypatch.setattr("trkfastsim.validation.run1a_timing.build_model", lambda *args, **kwargs: model)
    named = dict(hits=hits, conditions=conditions, association=association, wirefit=wirefit,
                 reference=reference, oracle=oracle, config=config_path, outdir=tmp_path/"out", **{"drift-check": check})
    argv = [s for k, v in named.items() for s in ("--"+k, str(v))]
    return argv, named


def test_cli_hashes_and_no_overwrite(tmp_path, monkeypatch):
    from trkfastsim.validation.run1a_timing import main
    argv, paths = cli_fixture(tmp_path, monkeypatch)
    report = main(argv)
    assert report["wirefit_recorded_input_hashes_verified"]
    assert report["both_directions_profiled"] == 1
    assert len(report["inputs"]["reference_config"]["sha256"]) == 64
    with pytest.raises(FileExistsError):
        main(argv)


@pytest.mark.parametrize("key", ["hits", "conditions", "association"])
def test_stale_wirefit_input_hash_rejected(tmp_path, monkeypatch, key):
    from trkfastsim.validation.run1a_timing import main
    argv, paths = cli_fixture(tmp_path, monkeypatch)
    report_path = paths["wirefit"] / "report.json"
    report = json.loads(report_path.read_text())
    report["inputs"][key]["sha256"] = "0"*64
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="recorded input hash differs"):
        main(argv)


def test_stale_drift_report_rejected(tmp_path, monkeypatch):
    from trkfastsim.validation.run1a_timing import main
    argv, paths = cli_fixture(tmp_path, monkeypatch)
    report = json.loads(paths["drift-check"].read_text())
    report["inputs"]["oracle"]["sha256"] = "0"*64
    paths["drift-check"].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="drift audit recorded input"):
        main(argv)


@pytest.mark.parametrize("key", ["evkey", "source_hit_ev", "source_hit_sid", "hit_group", "group_ev", "group_support", "source_selected"])
def test_census_dtype_rejected(key):
    args = fixture()
    args[4][key] = args[4][key].astype(int if key == "source_selected" else float)
    with pytest.raises(ValueError, match="census/support"):
        prepare_saved(*args[:5])


def test_scaled_wire_axis_and_rebuilt_endpoints_rejected():
    args = fixture()
    for key in ("dirx", "diry", "dirz"):
        args[2]["straw"][key] *= 2
        args[5][key] *= 2
    mid = np.column_stack([args[5][k] for k in ("midx", "midy", "midz")])
    d = np.column_stack([args[5][k] for k in ("dirx", "diry", "dirz")])
    cal, hv = mid+args[5]["hlen"][:, None]*d, mid-args[5]["hlen"][:, None]*d
    args[5].update(zip(("wcalx", "wcaly", "wcalz"), cal.T))
    args[5].update(zip(("whvx", "whvy", "whvz"), hv.T))
    data = prepare_saved(*args[:5])
    with pytest.raises(ValueError, match="unit wire"):
        measurement_arrays(data, args[5], args[6])


def test_opt_in_joint_fit_keeps_both_hypotheses_and_support():
    report, out = run(*fixture(), bz=1.0117, fit_drift=True, drift_max_iterations=4)
    assert report["drift_fit"]["attempted_hypotheses"] == 2
    assert out["drift_success"].shape == (3, 2)
    assert out["source_drift_mask"].shape == (12, 2)
    assert np.all(out["source_drift_mask"])
    assert np.all(out["source_drift_initial_LR"] != 0)
    ok = out["drift_objective_valid"]
    assert np.all(out["drift_final_objective"][ok] <= out["drift_initial_objective"][ok])
    assert report["assigned_directions"] == 0
    json.dumps(report, allow_nan=False)


def test_joint_fit_batch_exact():
    args = fixture()
    _, whole = run(*args, bz=1.0117, fit_drift=True, drift_max_iterations=4)
    _, small = run(*args, bz=1.0117, fit_drift=True, drift_max_iterations=4, group_batch=1)
    for key in whole:
        np.testing.assert_array_equal(whole[key], small[key], err_msg=key)
