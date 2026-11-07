"""Independent stock-oracle census, provenance and no-overwrite validation tests."""
import copy
import hashlib
import json

import awkward as ak
import numpy as np
import pytest
import uproot

from trkfastsim.validation.run1a_drift import (FIELDS, SCALARS, SCOPE, VECTORS, audit,
                                             main, read_oracle)


def fixture():
    query = dict(module_type="FieldOnDriftOracle", TimeMin=-1., TimeMax=1., TimeStep=.5,
                 AngleBins=3, ExtraProbes=2, MaxQueries=100, TimeProbes=[.123], AngleProbes=[-7.])
    j = np.arange(2) + .5
    dt = np.r_[np.repeat(np.linspace(-1, 1, 5), 3), -1+2*np.fmod(j*.6180339887498948, 1), .123]
    phi = np.r_[np.tile(np.linspace(0, np.pi/2, 3), 5),
                (8*np.fmod(j*.4142135623730950, 1)-4)*np.pi/2, -7.]
    data = dict(schema=1, scope=SCOPE, run=1430, subrun=0, event=23, reference_straw=0,
                iov_start_run=1400, iov_start_subrun=0, iov_end_run=1499, iov_end_subrun=999999,
                angle_bins=3, extra_probes=2, cc=3.8, wire_radius=.0125, outer_radius=2.5235,
                origin_bz_float=float(np.float32(1.01170565)), time_min=-1., time_max=1., time_step=.5,
                dt=dt, phi=phi, used_phi=phi.copy(), c_drift=dt*.1, r_drift=dt*.09,
                drift_velocity=np.full(len(dt), .09), signed_error=np.full(len(dt), .16),
                unsigned_error=np.full(len(dt), .12), detector_origin_mu2e=np.array([-3904., 0., 10171.]),
                origin_field=np.array([-.0016, -.0003, 1.01170565]),
                response_cids=np.array([19, 41, 117]), tracker_cids=np.array([138, 139, 140, 171]))
    config = dict(physics=dict(trigger_paths=[], RecoPath=[], end_paths=["DumpPath"], DumpPath=["driftoracle"],
                               analyzers=dict(driftoracle=query)), outputs={},
                  services=dict(ProditionsService=dict(strawResponse=dict(driftIgnorePhi=False))))
    return data, config


def write_root(path, data):
    values = {k: ak.Array([data[k]]) for k in VECTORS}
    values.update({k: np.asarray([data[k]]) for k in SCALARS})
    values["scope"] = ak.Array([data["scope"]])
    with uproot.recreate(path) as root:
        root.mktree("driftoracle/oracle", values)


def test_full_query_census_and_metadata():
    data, config = fixture()
    report = audit(data, config)
    assert report["queries"] == 18
    assert report["rectangular_queries"] == 15
    assert report["off_grid_queries"] == 2
    assert report["explicit_queries"] == 1
    assert report["metadata"]["event"] == 23
    assert report["ranges"]["c_drift"][0] < 0
    json.dumps(report, allow_nan=False)


def test_read_root(tmp_path):
    data, _ = fixture()
    path = tmp_path / "oracle.root"
    write_root(path, data)
    read = read_oracle(path)
    assert set(read) == set(FIELDS)
    for key in FIELDS:
        np.testing.assert_array_equal(read[key], data[key], err_msg=key)


@pytest.mark.parametrize("key,value", [("schema", 2), ("scope", "interpolation"), ("cc", np.nan),
                                      ("origin_field", [1., 2.]), ("origin_bz_float", 1.),
                                      ("wire_radius", 3.), ("run", 1500), ("time_step", .25),
                                      ("r_drift", np.array([1.])), ("c_drift", np.full(18, np.inf))])
def test_bad_payload_rejected(key, value):
    data, config = fixture()
    data[key] = value
    with pytest.raises(ValueError):
        audit(data, config)


@pytest.mark.parametrize("field", ["dt", "phi", "used_phi"])
def test_changed_coordinates_or_used_angle_rejected(field):
    data, config = fixture()
    data[field][0] += 1e-5
    with pytest.raises(ValueError):
        audit(data, config)


@pytest.mark.parametrize("key,value", [("trigger_paths", ["RecoPath"]), ("RecoPath", ["makeSH"]),
                                      ("DumpPath", ["driftoracle", "trkdump"]), ("end_paths", [])])
def test_physics_processing_rejected(key, value):
    data, config = fixture()
    config["physics"][key] = value
    with pytest.raises(ValueError, match="schedule"):
        audit(data, config)


def test_outputs_rejected():
    data, config = fixture()
    config["outputs"] = dict(out=dict(module_type="RootOutput"))
    with pytest.raises(ValueError, match="schedule"):
        audit(data, config)


@pytest.mark.parametrize("key,value", [("MaxQueries", 2), ("MaxQueries", 1_000_001),
                                      ("TimeProbes", [0., 1.]), ("AngleProbes", [np.nan])])
def test_query_limits_and_paired_probes(key, value):
    data, config = fixture()
    config["physics"]["analyzers"]["driftoracle"][key] = value
    with pytest.raises(ValueError):
        audit(data, config)


def test_ignore_phi_is_applied_not_folded():
    data, config = fixture()
    config["services"]["ProditionsService"]["strawResponse"]["driftIgnorePhi"] = True
    with pytest.raises(ValueError, match="ignore-phi"):
        audit(data, config)
    data["used_phi"][:] = 0
    assert audit(data, config)["finite"]


def test_read_rejects_multiple_oracle_rows(tmp_path):
    data, _ = fixture()
    values = {k: ak.Array([data[k], data[k]]) for k in VECTORS}
    values.update({k: np.asarray([data[k], data[k]]) for k in SCALARS})
    values["scope"] = ak.Array([data["scope"], data["scope"]])
    path = tmp_path / "two.root"
    with uproot.recreate(path) as root:
        root.mktree("driftoracle/oracle", values)
    with pytest.raises(ValueError, match="one complete"):
        read_oracle(path)


def cli_fixture(tmp_path):
    data, config = fixture()
    oracle, fcl = tmp_path / "oracle.root", tmp_path / "expanded.fcl"
    write_root(oracle, data)
    fcl.write_text(json.dumps(config))
    (tmp_path / "export.done").write_text("finished\n")
    (tmp_path / "export.log").write_text("TrigReport Events total = 10 passed = 10 failed = 0\n"
                                        "Art has completed and will exit with status 0.\n")
    names = ("FieldOnDriftOracle_module.cc", "StrawDrift.cc", "StrawDriftMaker.cc", "StrawResponse.cc")
    manifest = hashlib.sha256(fcl.read_bytes()).hexdigest() + "  expanded.fcl\n"
    manifest += "\n".join("0"*64 + "  " + k for k in names) + "\n"
    (tmp_path / "source-sha256.txt").write_text(manifest)
    args = ["--oracle", str(oracle), "--config", str(fcl), "--outdir", str(tmp_path / "out")]
    return data, config, args


def test_cli_hash_repeat_and_no_overwrite(tmp_path, monkeypatch):
    data, config, args = cli_fixture(tmp_path)
    monkeypatch.setattr("trkfastsim.validation.run1a_drift.parse_fhicl_dump", lambda text: config)
    repeat = tmp_path / "repeat.root"
    write_root(repeat, data)
    report = main(args + ["--compare-oracle", str(repeat)])
    assert report["passed"] and report["repeat"]["exact"]
    assert len(report["inputs"]["oracle"]["sha256"]) == 64
    with pytest.raises(FileExistsError):
        main(args)


def test_cli_mismatch_preserves_failure_report(tmp_path, monkeypatch):
    data, config, args = cli_fixture(tmp_path)
    monkeypatch.setattr("trkfastsim.validation.run1a_drift.parse_fhicl_dump", lambda text: config)
    repeat = tmp_path / "repeat.root"
    modified = copy.deepcopy(data)
    modified["c_drift"][0] += .001
    write_root(repeat, modified)
    with pytest.raises(RuntimeError, match="comparison failed"):
        main(args + ["--compare-oracle", str(repeat)])
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert not report["passed"]
    assert report["repeat"]["different_fields"] == ["c_drift"]


@pytest.mark.parametrize("text", ["", "Art has completed and will exit with status 0.",
                                  "TrigReport Events total = 2 passed = 1 failed = 1\n"
                                  "Art has completed and will exit with status 0."])
def test_cli_failed_event_log_rejected(tmp_path, monkeypatch, text):
    _, config, args = cli_fixture(tmp_path)
    monkeypatch.setattr("trkfastsim.validation.run1a_drift.parse_fhicl_dump", lambda text: config)
    (tmp_path / "export.log").write_text(text)
    with pytest.raises(ValueError, match="census"):
        main(args)


def test_cli_changed_config_manifest_rejected(tmp_path, monkeypatch):
    _, config, args = cli_fixture(tmp_path)
    monkeypatch.setattr("trkfastsim.validation.run1a_drift.parse_fhicl_dump", lambda text: config)
    (tmp_path / "source-sha256.txt").write_text("\n".join("0"*64 + "  " + k for k in
                                                      ("expanded.fcl", "FieldOnDriftOracle_module.cc", "StrawDrift.cc", "StrawDriftMaker.cc", "StrawResponse.cc")))
    with pytest.raises(ValueError, match="hash differs"):
        main(args)


@pytest.mark.parametrize("key,value", [("reference_straw", -1.5), ("event", -4.5), ("run", 2**32),
                                      ("iov_start_run", -.5), ("reference_straw", 65536),
                                      ("tracker_cids", np.array([-.5])),
                                      ("response_cids", np.array([-1])),
                                      ("response_cids", np.arange(4097))])
def test_exported_integer_types_and_bounds(key, value):
    data, config = fixture()
    data[key] = value
    with pytest.raises(ValueError):
        audit(data, config)


@pytest.mark.parametrize("name", ["export.log", "source-sha256.txt", "export.done"])
def test_missing_sidecar_is_not_a_provenance_pass(tmp_path, monkeypatch, name):
    _, config, args = cli_fixture(tmp_path)
    monkeypatch.setattr("trkfastsim.validation.run1a_drift.parse_fhicl_dump", lambda text: config)
    (tmp_path / name).rename(tmp_path / (name + ".omitted"))
    with pytest.raises(ValueError, match="sidecars"):
        main(args)


@pytest.mark.parametrize("bad", ["duplicate", "nonhex", "wrongname", "missingdigest"])
def test_malformed_source_manifest(tmp_path, monkeypatch, bad):
    _, config, args = cli_fixture(tmp_path)
    monkeypatch.setattr("trkfastsim.validation.run1a_drift.parse_fhicl_dump", lambda text: config)
    manifest = tmp_path / "source-sha256.txt"
    rows = manifest.read_text().splitlines()
    if bad == "duplicate":
        rows[2] = rows[1]
    elif bad == "nonhex":
        rows[1] = "z"*64 + rows[1][64:]
    elif bad == "wrongname":
        rows[1] = "0"*64 + "  wrong.cc"
    else:
        rows[1] = "bad"
    manifest.write_text("\n".join(rows))
    with pytest.raises(ValueError, match="malformed"):
        main(args)


def test_file_size_bound_before_root_deserialization(tmp_path):
    path = tmp_path / "large.root"
    with path.open("wb") as stream:
        stream.truncate(128*1024**2 + 1)
    with pytest.raises(ValueError, match="bounded reader"):
        read_oracle(path)


def test_uncompressed_bound_before_array_read(tmp_path, monkeypatch):
    path = tmp_path / "stub.root"
    path.write_bytes(b"stub")

    class Tree:
        classname = "TTree"
        num_entries = 1
        uncompressed_bytes = 96*1024**2 + 1

        def keys(self):
            return FIELDS

        def arrays(self, *args, **kwargs):
            raise AssertionError("must reject before materializing payload")

    class Root:
        def __enter__(self):
            return {"driftoracle/oracle": Tree()}

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("trkfastsim.validation.run1a_drift.uproot.open", lambda p: Root())
    with pytest.raises(ValueError, match="uncompressed"):
        read_oracle(path)


@pytest.mark.parametrize("changed", [None, "cc", "c_drift", "r_drift", "drift_velocity",
                                     "signed_error", "unsigned_error", "used_phi"])
def test_model_comparison_reports_each_failure(monkeypatch, changed):
    from types import SimpleNamespace
    from trkfastsim.validation.run1a_drift import compare_model
    data, config = fixture()
    config["services"]["ProditionsService"]["strawDrift"] = {}
    model = SimpleNamespace(cc=data["cc"] + (.001 if changed == "cc" else 0))
    predicted = {k: data[k].copy() for k in
                 ("used_phi", "c_drift", "r_drift", "drift_velocity", "signed_error", "unsigned_error")}
    if changed in predicted:
        predicted[changed][0] += .001
    monkeypatch.setattr("trkfastsim.reco.fielddrift.build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr("trkfastsim.reco.fielddrift.drift_info", lambda *args: predicted)
    assert compare_model(data, config)["passed"] == (changed is None)
