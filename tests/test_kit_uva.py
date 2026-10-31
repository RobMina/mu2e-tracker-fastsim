"""Fake tracker streaming endpoint: all artifact/binding/checksum code is real."""
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mu2efastsim.endpoints import EndpointError, JobStatus
from mu2efastsim.endpoints.kit import FastsimKit
from trkfastsim import kit_task, kit_uva
from test_kit_task import metrics


class Source(kit_uva.FnalSource):
    revision = "original"
    def __init__(self):
        super().__init__("gpvm", "/path/to/data")
    def fingerprint(self, path):
        data = (path + self.revision).encode()
        return {"sha256": hashlib.sha256(data).hexdigest(), "md5": hashlib.md5(data).hexdigest(), "size": len(data)}


class Endpoint:
    name, host = "uva", "fake"
    def __init__(self):
        self.state = "working"
        self.inbox, self.cache, self.streams, self.submissions = "", False, [], []
        self.cache_ok = self.md5_ok = self.cleanup = True
        self.output_metrics, self.output_manifest = metrics(), None
    def admit_resources(self, resources):
        if resources.minutes > 90:
            raise EndpointError("budget")
    def submit(self, name, body, resources):
        self.submissions.append((name, body, resources))
        line = next(line for line in body.splitlines() if 'artifact-manifest.json' in line)
        self.manifest = json.loads(shlex.split(line)[2])
    def status(self, name):
        return JobStatus(self.state, site_state=self.state.upper(), job_id="7")
    def read_file(self, name, path):
        return self.inbox
    def run_on_node(self, name, host, cmd, timeout=120):
        if "VERIFIED" in cmd:
            return "VERIFIED" if self.cache_ok else "BAD"
        if "test -f" in cmd:
            return "have" if self.cache else ""
        if cmd.startswith("touch"):
            self.cache = True
        return ""
    def stream_in(self, name, host, path, argv):
        self.streams.append((path, argv))
        dest = Path(path).name
        entry = next(e for e in self.manifest["inputs"]["files"] if e["dest"] == dest)
        return entry["md5"] if self.md5_ok else "bad"
    def cancel(self, name):
        self.state = "cancelled"
        return self.state
    def fetch(self, name, target):
        out = Path(target) / name / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(self.output_metrics))
        (out / "provenance.json").write_text("{}")
        (out / "artifact-manifest.json").write_text(json.dumps(self.output_manifest or self.manifest))
        if self.cleanup:
            (out / "cleanup.txt").write_text("deleted tmpfs\n")
        return list(out.iterdir())


@pytest.fixture
def setup(tmp_path, monkeypatch, site_env):
    calls = []
    def prepare(ep, path, ref, **kw):
        calls.append(kw["modules"])
        return {"path": path + ".snapshots/" + "e" * 32, "requested_ref": ref,
                "umbrella_commit": "a" * 40, "submodules": {m: "b" * 40 for m in kw["modules"]},
                "policy": {"name": "retained-readonly-git-v1", "modules": list(kw["modules"]), "branch": ""}}
    monkeypatch.setattr(kit_uva, "prepare_code", prepare)
    monkeypatch.setattr(kit_uva, "resolve_image", lambda ep, path: {
        "path": path, "sha256": "d" * 64, "size": 100, "policy": "verify-before-run-not-snapshot"})
    ep, source = Endpoint(), Source()
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path, sif="/image.sif")
    handler = kit_uva.TrackerPointStream(kit, source=source, code="/snapshot/code", background=False)
    kit._delegates["tracker_point"] = handler
    return kit, handler, ep, source, calls


def start(setup):
    kit, handler, ep, source, calls = setup
    assert kit.submit("p", {"task": "tracker_point"}, [], []) == {"handle": "p"}
    ep.inbox = "node /dev/shm/trkfs_kit_j7_abcdef12"
    return kit, handler, ep


def test_submit_stream_and_results(setup):
    kit, handler, ep = start(setup)
    assert setup[4] == [("core", "tracker")]
    body = ep.submissions[0][1]
    assert "PYTHONPATH=$CODE/core:$CODE/tracker" in body and "beam" not in body
    assert "--retain-ntuple" not in body and "--nv" not in body
    assert "staged artifact digest mismatch" in body and "runtime image changed" in body
    assert "no input arrived within 900 s" in body and 'stat -f -c %T' in body
    assert subprocess.run(["bash", "-n"], input=body, text=True, capture_output=True).returncode == 0
    assert kit.status("p")["state"] == "working"
    assert len(ep.streams) == 5 and ep.cache
    assert {Path(p).name for p, a in ep.streams} == set(kit_task.DESTINATIONS.values())
    ep.state = "completed"
    result = kit.results("p")
    assert result["metrics"] == metrics()
    assert result["metadata"]["artifact_binding"] == "verified"
    assert result["metadata"]["artifact_manifest"]["coverage"]["full_immutability"] is False
    assert not any(f["name"].endswith(".root") for f in result["files"])


def test_retry_uses_binding_and_rejects_target_or_params(setup):
    kit, handler, ep = start(setup)
    original = ep.submissions[0][1]
    setup[3].revision = "changed upstream bytes"
    kit.submit("p", {"task": "tracker_point"}, [], [])
    assert len(setup[4]) == 1 and ep.submissions[-1][1] == original
    with pytest.raises(ValueError, match="different params"):
        kit.submit("p", {"task": "tracker_point", "seed": 2}, [], [])
    setup[3].root = "/different"
    with pytest.raises(ValueError, match="target/source"):
        kit.submit("p", {"task": "tracker_point"}, [], [])


@pytest.mark.parametrize("bad", ["md5", "cache", "reuse"])
def test_bad_checksums_cancel(setup, bad):
    kit, handler, ep = start(setup)
    if bad == "md5":
        ep.md5_ok = False
    else:
        ep.cache_ok = False
        ep.cache = bad == "reuse"
    status = kit.status("p")
    assert status["state"] == "cancelled" and "stream error" in status["message"]
    if bad == "reuse":
        assert not ep.streams
    with pytest.raises(ValueError, match="not completed"):
        kit.results("p")


def test_ready_cache_rehashed_and_reused(setup):
    kit, handler, ep = start(setup)
    ep.cache = True
    kit.status("p")
    assert not ep.streams and handler._has("p", "inputsreused")


@pytest.mark.parametrize("bad", ["cleanup", "manifest", "metrics", "failed"])
def test_result_gates(setup, bad):
    kit, handler, ep = start(setup)
    kit.status("p")
    ep.state = "completed"
    if bad == "cleanup":
        ep.cleanup = False
    elif bad == "manifest":
        ep.output_manifest = {"changed": True}
    elif bad == "metrics":
        ep.output_metrics["wall_s"] = float("nan")
    else:
        ep.state = "failed"
    with pytest.raises(ValueError):
        kit.results("p")


def test_cancel_waiting_and_invalid_files(setup):
    kit, handler, ep, _, _ = setup
    assert kit.check("p", {"task": "tracker_point"}, [], [])["ok"]
    assert not kit.check("p", {"task": "tracker_point"}, [{}], [])["ok"]
    kit.submit("p", {"task": "tracker_point"}, [], [])
    assert handler.stream_step("p") == "waiting"
    assert kit.cancel("p")["state"] == "cancelled"
    assert handler.stream_step("p") == "gone" and not ep.streams


def test_tampered_binding_rejected(setup):
    kit, handler, ep = start(setup)
    path = handler.dir / "p.json"
    rec = json.loads(path.read_text())
    rec["body"] += "\nchanged"
    path.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="binding"):
        kit.submit("p", {"task": "tracker_point"}, [], [])


@pytest.mark.parametrize("inbox", ["node /path/to/inputs", "node /dev/shm/../tmp/x", "node /dev/shm/trkfs_kit_j7_a extra"])
def test_inbox_must_be_tmpfs_job_directory(setup, inbox):
    kit, handler, ep = start(setup)
    ep.inbox = inbox
    assert kit.status("p")["state"] == "cancelled"
    assert not ep.streams


def test_resource_refusal_before_preparation(setup, monkeypatch):
    kit, handler, ep, source, calls = setup
    def refuse(resources):
        raise EndpointError("budget refused")
    monkeypatch.setattr(ep, "admit_resources", refuse)
    assert not kit.check("p", {"task": "tracker_point"}, [], [])["ok"]
    with pytest.raises(ValueError, match="budget"):
        kit.submit("p", {"task": "tracker_point"}, [], [])
    assert not calls and not ep.submissions


def test_direct_geometry_rejected(setup):
    kit, handler, ep, _, calls = setup
    with pytest.raises(ValueError, match="files/inputs"):
        handler.submit("p", {}, geometry={"any": "value"})
    assert not calls


def test_recovery_rechecks_marker_after_waiting_for_stream_lock(setup, monkeypatch):
    from contextlib import contextmanager
    kit, handler, ep = start(setup)
    @contextmanager
    def lock_after_other_controller(*args):
        handler._mark("p", "streamed", "other controller completed transfer")
        ep.state = "completed"
        yield
    monkeypatch.setattr(kit_uva, "submission_lock", lock_after_other_controller)
    assert handler.stream_step("p") == "streamed"
    assert not handler._has("p", "streamerr") and not ep.streams
    assert ep.state == "completed"


def test_success_marker_published_under_stream_lock(setup, monkeypatch):
    from contextlib import contextmanager
    kit, handler, ep = start(setup)
    @contextmanager
    def check_lock(*args):
        yield
        assert handler._has("p", "streamed")
    monkeypatch.setattr(kit_uva, "submission_lock", check_lock)
    assert handler.stream_step("p") == "streamed"
