"""Tracker admission and mounted rendering, with synthetic metrics only."""
import json
import math
import subprocess
from types import SimpleNamespace

import pytest

from mu2efastsim.endpoints.kit import FastsimKit, TASKS
from trkfastsim import kit_task, kit_uva


@pytest.fixture
def kit(tmp_path, site_env):
    ep = SimpleNamespace(name="eaf", fnal_data="/path/to/data/crv_fastsim")
    return FastsimKit(endpoint=ep, state_dir=tmp_path)


def test_registration_defaults_and_paths(kit):
    assert TASKS["tracker_point"].defaults == kit_task.DEFAULTS
    p = kit_uva.validate({"task": "tracker_point", "shard": 20.0, "seed": 0.0})
    assert p == {"sample": kit_task.SAMPLE, "shard": 20, "seed": 0}
    paths = kit_task.input_paths(p)
    assert paths["gen"] == "crv_fastsim/g4ref_v1_tables/gen.001400_00000020.root"
    assert paths["digi_cond"] == "trkfs_digi_20261026/v2/digicond.npz"
    assert paths["reco_cond"] == "trk_fastreco/cond_mc_au.json"
    assert paths["drift_source"] == "trk_fastreco/trk_sa2/tracks_ref_mc_saveall.001400_00000000.root"
    assert set(kit_uva.TrackerPointStream.inputs(p).values()) == set(kit_task.DESTINATIONS.values())


@pytest.mark.parametrize("key,value", [("shard", 0), ("shard", 21), ("shard", 1.5), ("shard", True),
    ("shard", "11"), ("seed", -1), ("seed", 2**31), ("seed", float("inf")), ("seed", None),
    ("sample", "other"), ("sample", "$(bad)"), ("fit", "f1"), ("library", "lib.root")])
def test_admission_refuses(kit, key, value):
    p = {"task": "tracker_point", key: value}
    assert not kit.check("p", p, [], [])["ok"]
    with pytest.raises(ValueError):
        kit.submit("p", p, [], [])
    assert not (kit.out_root.parent / "tasks").exists()


@pytest.mark.parametrize("field", ["files", "inputs"])
def test_no_file_handoff(kit, field):
    args = {"files": [], "inputs": [], field: [{"kind": "geom", "uri": "file:///path/to/x"}]}
    assert not kit.check("p", {"task": "tracker_point"}, **args)["ok"]
    with pytest.raises(ValueError, match="only beam_point"):
        kit.submit("p", {"task": "tracker_point"}, **args)


def test_eaf_body_and_capabilities(kit):
    body, res = kit.render({"task": "tracker_point"})
    assert "D=/path/to/data/crv_fastsim/.." in body
    assert "--seed 1" in body and "--workdir" in body
    assert "--retain-ntuple" not in body and "--library" not in body
    assert "cleanup.txt" in body and "trap 'exit 143'" in body
    assert res.gpus == 0
    assert subprocess.run(["bash", "-n"], input=body, text=True, capture_output=True).returncode == 0
    desc = kit.describe()["tasks"]["tracker_point"]
    assert desc["endpoints"] == ["eaf", "uva"] and desc["required"] == []
    assert desc["inputs_supported"] is False and desc["file_kinds"] == []
    assert set(desc["metrics"]) == set(kit_task.METRICS)
    assert desc["params"]["shard"]["maximum"] == 20
    check = kit.check("p", {"task": "tracker_point"}, [], [], ["fit_event_fraction_se"])
    assert check["ok"] and check["capabilities"]["route"]["name"] == "mounted"


def metrics():
    m = dict.fromkeys(kit_task.METRICS, 0)
    m.update(generated_events=10, written_events=10, events_with_fit=2, fit_successes=2, seeds=2, fit_attempts=2,
             fit_event_fraction=0.2, fit_event_fraction_se=math.sqrt(0.2 * 0.8 / 10))
    return m


def test_full_finite_denominator_contract():
    assert kit_task.validate_metrics(metrics()) == metrics()
    with pytest.raises(ValueError, match="nonempty"):
        kit_task.validate_metrics(dict.fromkeys(kit_task.METRICS, 0))
    for changed in ({"wall_s": float("nan")}, {"digis": True}, {"written_events": 2},
                    {"fit_event_fraction": 1}, {"events_with_fit": 11}, {"fit_event_fraction_se": 0}):
        with pytest.raises(ValueError):
            kit_task.validate_metrics({**metrics(), **changed})
    with pytest.raises(ValueError):
        kit_task.validate_metrics({"generated_events": 1})


@pytest.mark.parametrize("fail", [False, True])
def test_mounted_script_executes_metadata_only_and_cleans_scratch(tmp_path, fail, site_env):
    """Run the shell locally with a stub runner, no input data or endpoint access."""
    data, out, scratch = tmp_path / "data root", tmp_path / "out", tmp_path / "scratch"
    out.mkdir()
    scratch.mkdir()
    for path in kit_task.input_paths({}).values():
        target = data / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic")
    runner = tmp_path / "runner.py"
    runner.write_text('''import json, pathlib, sys
args = sys.argv[1:]
assert args[:3] == ["python", "-m", "trkfastsim.kitpoint"]
assert "--retain-ntuple" not in args
out = pathlib.Path(args[args.index("--outdir") + 1])
work = pathlib.Path(args[args.index("--workdir") + 1])
(work / "EventNtuple.root").write_text("synthetic scratch root")
(out / "metrics.json").write_text("{}")
(out / "provenance.json").write_text("{}")
''')
    if fail:
        runner.write_text(runner.read_text().split('(out / "metrics.json")')[0] + 'raise SystemExit(9)\n')
    import os
    import shlex
    import sys
    ep = SimpleNamespace(name="eaf", fnal_data=str(data / "crv_fastsim"), runtime=lambda: {
        "a": f'A() {{ {shlex.quote(sys.executable)} {shlex.quote(str(runner))} "$@"; }}\n', "t": ""})
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "state")
    body, _ = kit.render({"task": "tracker_point"})
    result = subprocess.run(["bash", "-c", body], capture_output=True, text=True,
                            env={**os.environ, "M2FS_OUT": str(out), "TMPDIR": str(scratch)})
    assert result.returncode == (6 if fail else 0), result.stderr
    assert not list(scratch.iterdir())
    assert (out / "cleanup.txt").read_text().strip()
    assert not list(out.rglob("*.root"))
    assert {p.name for p in out.iterdir()} == ({"cleanup.txt", "point.txt"} if fail else
                                              {"metrics.json", "provenance.json", "cleanup.txt", "point.txt"})


def test_mounted_results_gate(tmp_path, site_env):
    from mu2efastsim.endpoints import JobStatus
    out = tmp_path / "out" / "p" / "out"
    out.mkdir(parents=True)
    ep = SimpleNamespace(name="eaf", fnal_data="/path/to/crv_fastsim", status=lambda _: JobStatus("completed"),
                         fetch=lambda *args: list(out.iterdir()))
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path)
    kit._record_task("p", "tracker_point")
    (out / "metrics.json").write_text(json.dumps(metrics()))
    with pytest.raises(ValueError, match="cleanup"):
        kit.results("p")
    (out / "cleanup.txt").write_text("deleted")
    with pytest.raises(ValueError, match="provenance"):
        kit.results("p")
    (out / "provenance.json").write_text("{}")
    assert kit.results("p")["metrics"] == metrics()
    (out / "metrics.json").write_text(json.dumps({**metrics(), "writer_s": float("inf")}))
    with pytest.raises(ValueError, match="finite"):
        kit.results("p")
