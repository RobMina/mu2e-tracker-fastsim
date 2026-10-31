"""Task-private runtime adapters neither build nor mutate the umbrella runner."""
import os
from pathlib import Path
import subprocess

import pytest

from prepare_digitable_runtime import CALL, REPOSITORIES, prepare


def test_runtime_adapter_preserves_source_and_checks_version(tmp_path):
    source = tmp_path / "offline_uva.sh"
    original = f'#!/bin/bash\n# {REPOSITORIES}\n# {REPOSITORIES}\n# OFFLINE_OUTPUT_FILE OFFLINE_EVENTS OFFLINE_FCL_PATH OFFLINE_WARM_REPEAT;\n{CALL}\n'
    source.write_text(original)
    out = tmp_path / "runtime"
    prepare(source, out)
    assert source.read_text() == original
    for name, script in (("offline_sequence.sh", "run_digitable_sequence.sh"), ("offline_fast.sh", "run_fastdigis_reco.sh"),
                         ("offline_payload_diagnostic.sh", "run_payload_diagnostic.sh"),
                         ("offline_payload_control.sh", "run_payload_control.sh")):
        text = (out / name).read_text()
        assert "mu2e-development.opensciencegrid.org" in text and script in text
        assert "OFFLINE_WARM_REPEAT PAYLOAD;" in text
        assert CALL not in text and "muse build" not in text
        subprocess.run(["bash", "-n", str(out / name)], check=True)
    with pytest.raises(FileExistsError):
        prepare(source, out)
    source.write_text("unexpected runner\n")
    with pytest.raises(ValueError, match="unsupported"):
        prepare(source, tmp_path / "bad")


def test_sequence_wrapper_uses_explicit_verified_snapshot(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    area, runner = root / "built_area_v2", root / "private_runner.sh"
    runner.write_text('#!/bin/bash\nprintf "%s\\n" "$@" "$PAYLOAD" > args.txt\n')
    tools = tmp_path / "bin"
    tools.mkdir()
    module = tools / "module"
    module.write_text("#!/bin/bash\nexit 0\n")
    module.chmod(0o755)
    script = Path(__file__).parents[1] / "scripts/uva_digitable_art.sh"
    env = {**os.environ, "ROOT": str(root), "INPUT_ROOT": "/shared", "MODE": "mc", "TAG": "payload",
           "AREA": str(area), "RUNNER": str(runner), "PAYLOAD": "1", "CVMFSEXEC_DIST": "/unused", "PATH": str(tools) + ":" + os.environ["PATH"]}
    subprocess.run(["bash", str(script)], env=env, check=True)
    assert (root / "mc_payload/args.txt").read_text().splitlines() == [
        str(area), str(area / "TrkFastSim/fcl/dump_digis.fcl"),
        "/shared/trkfs_digi_20261026/ref/dig.v18L_1001.art", "1"]
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(["bash", str(script)], env=env, check=True)
