"""Bounded capture-check publication tests; no GDML parsing or real transport."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from trkfastsim import transportcapture


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "transport_capture_check.py"
    spec = importlib.util.spec_from_file_location("_test_transport_capture_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture(particles):
    """Real, complete sidecar with deliberately stale legacy restart coordinates."""
    capture = transportcapture.TransportCapture({"fixture": True})
    capture.bind(particles)
    lanes = np.arange(len(particles))
    stopped = lanes == 0
    for name in ("interior", "exit"):
        phase = capture.phase(name, lanes, path_offset=np.full(len(lanes), 100.))
        phase.observe("entry", lanes, path_mm=np.full(len(lanes), 10.),
                      position_mm=np.zeros((len(lanes), 3)),
                      direction=np.tile([0., 0., 1.], (len(lanes), 1)),
                      model_ke_MeV=np.full(len(lanes), 20.),
                      momentum_MeV_c=np.full(len(lanes), 80.),
                      lost_support_MeV=np.zeros(len(lanes)),
                      lost_thin_MeV=np.ones(len(lanes)))
        phase.terminate(lanes[stopped], "unresolved_stop")
        phase.terminate(lanes[~stopped], "final_non_kink")
        phase.finish(path=np.full(len(lanes), 2.), stopped=stopped)
    capture.finish()
    return capture


@pytest.fixture
def fake_run(runner, monkeypatch, tmp_path):
    # Full keys distinguish repeated event numbers; nonadjacent primaries include a nonmuon.
    generated = pd.DataFrame([
        (2, 1, 7, 13), (1, 2, 7, -13), (2, 1, 7, 13),
        (1, 1, 7, 13), (0, 0, 4, 13), (1, 2, 7, 11), (1, 1, 7, -13),
    ], columns=["run", "subrun", "event", "pdg"])
    generated["input_row"] = np.arange(len(generated))
    gen_path, gdml_path = tmp_path / "input.root", tmp_path / "geometry.gdml"
    gen_path.write_bytes(b"mock input table\n")
    gdml_path.write_bytes(b"mock geometry, deliberately not GDML\n")
    source = tmp_path / "dependency.py"
    source.write_text("# tracked dependency\n")
    monkeypatch.setattr(transportcapture, "source_paths", lambda: {"fixture.transport": source})
    base_path = tmp_path / "base_steps.py"
    base_path.write_text("from _capture_test_base import load_world, TrackerInterior, make_steps\n")
    args = SimpleNamespace(gen=str(gen_path), gdml=str(gdml_path),
                           base_source=str(base_path), outdir=str(tmp_path / "output"),
                           max_events=3, seed=17, arm="legacy")
    state = SimpleNamespace(runner=runner, args=args, generated=generated, source=source,
                            calls=[], fault=None, tamper=None, capture=None)
    geom, world, interior = object(), object(), object()
    geometry_read = Mock(return_value=geom)
    table_read = Mock(return_value=generated)
    monkeypatch.setattr(runner, "geometry", SimpleNamespace(from_gdml=geometry_read))
    monkeypatch.setattr(runner, "tables", SimpleNamespace(KEY=["run", "subrun", "event"], read=table_read))

    def make_module(label):
        module = ModuleType(label)
        module.load_world = Mock(return_value=world)
        module.TrackerInterior = Mock(return_value=interior)

        def make_steps(gen, geometry, rng, **kwargs):
            arm = "capture" if kwargs.get("capture_transport", False) else label
            state.calls.append((arm, gen.copy(), kwargs.copy()))
            assert geometry is geom
            assert kwargs == dict(stage="C", world=world, interior=interior,
                                  sliced_msc=True, arms=args.arm, history=True,
                                  **({"capture_transport": True} if arm == "capture" else {}))
            rng.random(3)  # Equal stochastic work in every arm.
            particles = gen.loc[gen.pdg.abs() == 13].reset_index(drop=True).copy()
            particles["mukey"] = np.arange(1, len(particles) + 1)
            particles["value"] = 1.
            frames = {name: pd.DataFrame({"value": [1.]}) for name in
                      ("steps", "vertices", "segments", "muons")}
            frames["particles"] = particles
            stats = {"iterations": 1}
            if state.fault and state.fault[0] == arm:
                key = state.fault[1]
                if key in frames:
                    frames[key].loc[0, "value"] += 1e-12  # Not allclose equality.
                elif key == "rng":
                    rng.random()
                elif key == "seed_sequence":
                    rng.bit_generator.seed_seq.spawn(1)
                elif key == "stats":
                    stats["iterations"] += 1
            make_steps.path_vertices = frames["vertices"]
            make_steps.segment_ledger = frames["segments"]
            make_steps.muon_ledger = frames["muons"]
            make_steps.interior_stats = stats
            if arm == "capture":
                state.capture = _capture(particles)
                make_steps.transport_capture = state.capture
                if state.tamper:
                    with state.tamper.open("ab") as stream:
                        stream.write(b"# modified during transport\n")
            return frames["steps"], particles

        module.make_steps = make_steps
        return module

    base, candidate = make_module("baseline"), make_module("default")
    monkeypatch.setitem(sys.modules, "_capture_test_base", base)
    monkeypatch.setattr(runner, "steps", candidate)

    def surface_steps(sgs, vertices):
        arm = state.calls[-1][0]
        return pd.DataFrame({"value": [1. + (1e-12 if state.fault == (arm, "surfaces") else 0.)]})

    monkeypatch.setattr(runner, "surfaces", SimpleNamespace(surface_steps=surface_steps))
    state.modules = (base, candidate)
    state.geometry_read, state.table_read = geometry_read, table_read
    return state


@pytest.mark.parametrize("limit,event_count,row_count,muon_count", [
    (1, 1, 2, 2), (3, 3, 6, 5), (20, 4, 7, 6),
])
@pytest.mark.parametrize("arm", ["legacy", "AMT"])
def test_first_unique_full_event_keys_and_multi_primary_counts(
        fake_run, limit, event_count, row_count, muon_count, arm):
    state = fake_run
    state.args.max_events, state.args.arm = limit, arm
    state.runner.run(state.args)
    report = json.loads((Path(state.args.outdir) / "complete.json").read_text())
    assert report["generated_file_events"] == 4
    assert report["processed_events"] == event_count
    assert report["generated_rows"] == row_count
    assert report["primary_muons"] == muon_count
    assert report["arm"] == arm and report["seed"] == 17 and report["identical"] is True
    assert [call[0] for call in state.calls] == ["baseline", "default", "capture"]
    expected_keys = [(2, 1, 7), (1, 2, 7), (1, 1, 7), (0, 0, 4)][:limit]
    expected = state.generated.loc[
        [tuple(row) in expected_keys for row in state.generated[["run", "subrun", "event"]].to_numpy()]
    ].reset_index(drop=True)
    for _, selected, _ in state.calls:
        pd.testing.assert_frame_equal(selected.reset_index(drop=True), expected, check_exact=True)
    state.geometry_read.assert_called_once_with(state.args.gdml)
    state.table_read.assert_called_once_with(state.args.gen, "gen")
    for module, count in zip(state.modules, (1, 2)):
        assert module.load_world.call_count == module.TrackerInterior.call_count == count


@pytest.mark.parametrize("arm", ["default", "capture"])
@pytest.mark.parametrize("key", ["steps", "particles", "vertices", "surfaces", "segments",
                                 "muons", "rng", "seed_sequence", "stats"])
def test_any_exact_three_way_difference_prevents_completion(fake_run, arm, key):
    fake_run.fault = (arm, key)
    with pytest.raises(AssertionError):
        fake_run.runner.run(fake_run.args)
    out = Path(fake_run.args.outdir)
    assert not (out / "complete.json").exists()
    assert not (out / "capture.npz").exists()


@pytest.mark.parametrize("target", ["source", "base_source", "gen", "gdml"])
def test_before_after_content_tamper_rejects_publication(fake_run, target):
    fake_run.tamper = fake_run.source if target == "source" else Path(getattr(fake_run.args, target))
    with pytest.raises(RuntimeError, match="content changed during verification"):
        fake_run.runner.run(fake_run.args)
    assert not (Path(fake_run.args.outdir) / "complete.json").exists()


@pytest.mark.parametrize("existing_file", [False, True])
def test_existing_output_is_never_overwritten(fake_run, existing_file):
    out = Path(fake_run.args.outdir)
    if existing_file:
        out.write_bytes(b"keep me")
    else:
        out.mkdir()
        (out / "complete.json").write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        fake_run.runner.run(fake_run.args)
    assert (out if existing_file else out / "complete.json").read_bytes() == b"keep me"
    assert fake_run.calls == []
    fake_run.geometry_read.assert_not_called()
    fake_run.table_read.assert_not_called()


def test_manifest_provenance_and_offset_aware_inventory(fake_run):
    state = fake_run
    state.runner.run(state.args)
    out = Path(state.args.outdir)
    report = json.loads((out / "complete.json").read_text())
    paths = [state.args.gen, state.args.gdml, state.args.base_source,
             state.runner.__file__, str(state.source)]
    expected_hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    assert report["sha256"] == expected_hashes
    assert report["captured_source_sha256"] == state.capture.sources == {
        "fixture.transport": expected_hashes[str(state.source)]}
    assert report["environment"] == state.capture.environment
    assert report["environment"]["numpy"] == np.__version__
    assert {"python", "pandas", "system", "machine"} <= report["environment"].keys()
    assert report["capture_sha256"] == hashlib.sha256((out / "capture.npz").read_bytes()).hexdigest()
    assert len(report["seconds_base_off_on"]) == 3
    assert all(seconds >= 0 for seconds in report["seconds_base_off_on"])
    assert report["steps"] == report["vertices"] == report["surfaces"] == 1
    with np.load(out / "capture.npz", allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"]))
        assert manifest["environment"] == report["environment"]
        assert manifest["source_sha256"] == report["captured_source_sha256"]
        assert data["handoff_identity"].shape == (5, 4)
    for name in ("interior", "exit"):
        summary = pd.read_csv(out / f"{name}_summary.csv")
        assert len(summary) == 5
        np.testing.assert_array_equal(summary.last_path_mm, np.full(5, 110.))
        phase = report["phases"][name]
        assert phase["lanes"] == phase["last_observed_ahead_of_legacy_restart"] == 5
        assert phase["max_last_observed_minus_restart_mm"] == 8.
        assert phase["dispositions"] == {"unresolved_stop": 1, "final_non_kink": 4}
        assert phase["last_kind"] == {"entry": 5}
        assert phase["observations"] == 10
    interior = report["phases"]["interior"]
    assert interior["stopped_observed_lanes"] == interior["legacy_stopped_flags"] == 1
    assert interior["stopped_flag_identity_equal"] is True
