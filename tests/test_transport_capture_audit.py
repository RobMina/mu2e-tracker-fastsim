"""Saved algorithm-state audit tests; no real inputs or physical stopping claims."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trkfastsim import transportcapture
from test_transport_capture_check import _capture


@pytest.fixture
def auditor():
    path = Path(__file__).resolve().parents[1] / "scripts" / "transport_capture_audit.py"
    spec = importlib.util.spec_from_file_location("_test_transport_capture_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phases(n=3, *, stops=True, exit_knots=False):
    # Repeated event and particle numbers require the entire identity key.
    identity = np.array([[2, 1, 7, 1], [1, 2, 7, 1], [1, 2, 7, 2]], dtype=np.int64)[:n]
    lanes = np.arange(n)
    stop_lanes = lanes[:2] if stops else lanes[:0]
    offset = np.array([100., 200., 300.])[:n]
    phases = {}
    for name in ("interior", "exit"):
        phase = transportcapture.PhaseCapture(identity, path_offset=offset)
        phase.observe("entry", lanes, path_mm=np.array([10., 20., 30.])[:n],
                      position_mm=np.zeros((n, 3)), direction=np.tile([0., 0., 1.], (n, 1)),
                      model_ke_MeV=np.full(n, 20.), momentum_MeV_c=np.full(n, 80.),
                      lost_support_MeV=np.array([4., 6., 8.])[:n],
                      lost_thin_MeV=np.array([5., 7., 11.])[:n])
        if name == "interior":
            if len(stop_lanes):
                phase.record("stop_attempt", stop_lanes,
                             attempted_local_path_mm=np.array([12., 22.])[:len(stop_lanes)],
                             lower_local_path_mm=np.array([10., 20.])[:len(stop_lanes)],
                             model_ke_after_MeV=np.array([-1., -2.])[:len(stop_lanes)],
                             committed_candidate=np.array([False, True])[:len(stop_lanes)])
            phase.terminate(stop_lanes, "unresolved_stop")
            phase.terminate(lanes[len(stop_lanes):], "final_non_kink")
        else:
            phase.terminate(lanes, "extent_complete")
            if exit_knots:
                # Restart paths are global; attempt paths remain phase-local.
                phase.record("restart", np.array([0, 1]), path_mm=np.array([111., 222.]))
                phase.record("restart", np.array([0, 0]), path_mm=np.array([113., 112.]))
                phase.record("piece_attempt", np.array([0, 1]),
                             accepted=np.array([False, True]), model_ke_after_MeV=np.array([-2., -9.]))
                phase.record("piece_attempt", np.array([0, 1]),
                             accepted=np.array([False, False]), model_ke_after_MeV=np.array([0., 1.]))
        phase.finish(path=np.array([2., 20., 35.])[:n], ke=np.full(n, 25.),
                     lost_thin=np.array([2., 7., 13.])[:n],
                     lost_support=np.array([3., 6., 9.])[:n], stopped=np.isin(lanes, stop_lanes))
        records = {}
        for kind, arrays in phase.records():
            records.setdefault(kind, []).append(arrays)
        phases[name] = phase.summary(), records
    return phases


def test_stale_restart_gaps_use_full_identity_and_phase_offset(auditor):
    phases = _phases(stops=False)
    summary, records = phases["interior"]
    phases["interior"] = summary.iloc[[2, 0, 1]], records
    restart = records["legacy_restart"][0]
    records["legacy_restart"] = [{key: value[index] for key, value in restart.items()}
                                 for index in ([1], [2, 0])]
    result = auditor.audit(phases)
    assert result["interior_lanes"] == 3
    assert result["restart_behind_last_observed"] == 1
    assert result["max_path_gap_mm"] == 8.
    assert result["final_entry_missing_thin_MeV_sum"] == 1.
    assert result["max_final_entry_missing_thin_MeV"] == 3.
    assert result["final_entry_support_gap_MeV_sum"] == 0.
    assert result["transitions"] == [dict(disposition_interior="final_non_kink",
                                           disposition_exit="extent_complete", lanes=3)]
    assert "no physical stop" in result["scope"]
    assert "rate attribution" in result["scope"]


@pytest.mark.parametrize("join", ["restart", "exit"])
@pytest.mark.parametrize("mismatch", ["missing", "extra", "different"])
def test_identity_outer_join_refuses_unmatched_lanes(auditor, join, mismatch):
    phases = _phases()
    if join == "restart":
        records = phases["interior"][1]
        restart = records["legacy_restart"][0]
        if mismatch == "missing":
            records["legacy_restart"] = [{key: value[:-1] for key, value in restart.items()}]
        elif mismatch == "extra":
            extra = {key: value[:1].copy() for key, value in restart.items()}
            extra["identity"][0, 3] = 99
            records["legacy_restart"].append(extra)
        else:
            restart["identity"][0, 3] = 99
        message = "restart/summary identity coverage mismatch"
    else:
        summary, records = phases["exit"]
        if mismatch == "missing":
            summary = summary.iloc[:-1]
        elif mismatch == "extra":
            extra = summary.iloc[:1].copy()
            extra.loc[:, "sp"] = 99
            summary = pd.concat([summary, extra], ignore_index=True)
        else:
            summary.loc[0, "sp"] = 99
        phases["exit"] = summary, records
        message = "phase identity coverage mismatch"
    with pytest.raises(ValueError, match=message):
        auditor.audit(phases)


def test_unresolved_attempts_join_by_identity_without_exit_knots(auditor):
    phases = _phases()
    summary, records = phases["interior"]
    phases["interior"] = summary.iloc[::-1], records
    result = auditor.audit(phases)
    attempts = result["stop_attempts"]
    assert len(attempts) == 2
    for row, identity, local, offset, committed in zip(
            attempts, [(2, 1, 7, 1), (1, 2, 7, 1)], [12., 22.], [100., 200.], [False, True]):
        assert tuple(row[key] for key in auditor.KEY) == identity
        assert row["disposition"] == "unresolved_stop"
        assert row["attempt_local_path_mm"] == local
        assert row["lower_local_path_mm"] == local - 2.
        assert row["phase_path_offset_mm"] == offset
        assert row["last_path_mm"] == local - 2. + offset
        assert row["raw_model_ke_MeV"] < 0.
        assert row["committed_candidate"] is committed
        assert row["exit_knot_path_mm"] is None
        assert row["exit_knot_past_attempt"] is False
    assert sum(row["lanes"] for row in result["transitions"]) == 3
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", ["unknown_identity", "wrong_disposition", "duplicate"])
def test_stop_attempt_requires_unique_unresolved_identity(auditor, fault):
    phases = _phases()
    summary, records = phases["interior"]
    attempts = records["stop_attempt"][0]
    if fault == "unknown_identity":
        attempts["identity"][0, 3] = 99
    elif fault == "wrong_disposition":
        summary.loc[0, "disposition"] = "final_non_kink"
    else:
        records["stop_attempt"].append({key: value[:1].copy() for key, value in attempts.items()})
    message = "more than one first stop attempt" if fault == "duplicate" else "stop attempt without stop disposition"
    with pytest.raises(ValueError, match=message):
        auditor.audit(phases)


def test_exit_knots_past_attempt_are_offset_aware_and_use_latest_path(auditor):
    result = auditor.audit(_phases(exit_knots=True))
    first, second = result["stop_attempts"]
    assert first["exit_knot_path_mm"] == 113.
    assert first["exit_knot_past_attempt"] is True
    assert second["exit_knot_path_mm"] == 222.
    assert second["exit_knot_past_attempt"] is False
    assert result["exit_failed_attempts"] == 3
    assert result["exit_negative_raw_ke_attempts"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("n", [0, 3])
def test_no_stop_and_empty_capture_have_empty_attempt_inventory(auditor, n):
    result = auditor.audit(_phases(n=n, stops=False))
    assert result["interior_lanes"] == n
    assert result["stop_attempts"] == []
    assert result["exit_failed_attempts"] == 0
    assert result["exit_negative_raw_ke_attempts"] == 0
    if n == 0:
        assert result["transitions"] == []
        assert result["restart_behind_last_observed"] == 0
        assert result["max_path_gap_mm"] is None
        assert result["max_final_entry_missing_thin_MeV"] is None
        assert result["final_entry_missing_thin_MeV_sum"] == 0.
        assert result["final_entry_support_gap_MeV_sum"] == 0.
    json.dumps(result, allow_nan=False)


@pytest.fixture
def saved_capture(tmp_path, monkeypatch):
    source = tmp_path / "fixture_source.py"
    source.write_text("# Synthetic source identity only.\n")
    monkeypatch.setattr(transportcapture, "source_paths", lambda: {"fixture.transport": source})
    particles = pd.DataFrame([(2, 1, 7, 1), (1, 2, 7, 1), (1, 2, 7, 2)],
                             columns=["run", "subrun", "event", "mukey"])
    capture = _capture(particles)
    capture.save(tmp_path / "capture.npz")
    report = dict(identical=True,
                  capture_sha256=hashlib.sha256((tmp_path / "capture.npz").read_bytes()).hexdigest(),
                  captured_source_sha256=capture.sources)
    (tmp_path / "complete.json").write_text(json.dumps(report))
    return tmp_path, capture, report


def test_load_capture_round_trips_real_sidecar_without_pickle(auditor, saved_capture):
    directory, capture, expected_report = saved_capture
    before = {name: (directory / name).read_bytes() for name in ("capture.npz", "complete.json")}
    report, phases = auditor.load_capture(directory)
    assert report == expected_report
    assert set(phases) == {"interior", "exit"}
    for name, (summary, records) in phases.items():
        expected = capture.get_phase(name)
        pd.testing.assert_frame_equal(summary, expected.summary(), check_exact=True)
        expected_records = expected.records()
        assert sum(map(len, records.values())) == len(expected_records)
        for kind, arrays in expected_records:
            assert len(records[kind]) == 1
            assert records[kind][0].keys() == arrays.keys()
            for key, value in arrays.items():
                np.testing.assert_array_equal(records[kind][0][key], value)
    assert before == {name: (directory / name).read_bytes() for name in before}


@pytest.mark.parametrize("identical", [False, None, 1, "true"])
def test_load_capture_requires_true_completion_identity(auditor, saved_capture, identical):
    directory, _, report = saved_capture
    report["identical"] = identical
    (directory / "complete.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="capture identity/completion gate failed"):
        auditor.load_capture(directory)


def test_load_capture_refuses_hash_tampering(auditor, saved_capture):
    directory, _, _ = saved_capture
    with (directory / "capture.npz").open("ab") as stream:
        stream.write(b"tampered sidecar\n")
    with pytest.raises(ValueError, match="capture identity/completion gate failed"):
        auditor.load_capture(directory)


@pytest.mark.parametrize("target", ["completion", "manifest"])
def test_load_capture_refuses_source_identity_tampering(auditor, saved_capture, target):
    directory, _, report = saved_capture
    if target == "completion":
        report["captured_source_sha256"] = {"fixture.transport": "0" * 64}
    else:
        with np.load(directory / "capture.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        manifest = json.loads(str(arrays["manifest_json"]))
        manifest["source_sha256"] = {"fixture.transport": "0" * 64}
        arrays["manifest_json"] = np.array(json.dumps(manifest))
        np.savez_compressed(directory / "capture.npz", **arrays)
        report["capture_sha256"] = hashlib.sha256((directory / "capture.npz").read_bytes()).hexdigest()
    (directory / "complete.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="source identity mismatch"):
        auditor.load_capture(directory)
