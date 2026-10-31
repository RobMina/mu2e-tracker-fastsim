"""Round-trip reports require finished jobs and the full original population."""
import importlib.util
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot

from trkfastsim import digitable

SPEC = importlib.util.spec_from_file_location("digitable_report", Path(__file__).parents[1] / "scripts/digitable_report.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def fixture(directory):
    cols = {key: np.array([1, 2], np.uint32) for key in digitable.KEYS}
    cols.update({key: ak.unflatten(np.array([], np.uint32), [0, 0]) for key in digitable.LIMITS})
    digitable.write(directory / "digis.root", cols)
    digitable.write(directory / "replay.root", cols)
    reco = {key: value for key, value in cols.items() if key != "d_flags"}
    reco["kl_status"] = ak.unflatten(np.array([], np.uint32), [0, 0])
    for name in ("original_reco.root", "replay_reco.root"):
        with uproot.recreate(directory / name) as root:
            root.mktree("trkdump/events", {key: (value.type.content if isinstance(value, ak.Array) else value.dtype)
                                         for key, value in reco.items()}).extend(reco)
            root.mktree("trkdump/conditions", {"c": "float64"}).extend({"c": np.array([1.5])})
    (directory / "sequence.done").write_text("done\n")


def test_complete_report_checks_every_empty_event(tmp_path):
    fixture(tmp_path)
    result = mod.report(tmp_path, 2)
    assert result["equal"] and result["digis"]["reference"]["empty_digi_events"] == 2
    assert result["exported_reco"]["trees"]["trkdump/conditions"]["equal"]
    assert not mod.report(tmp_path, 3)["equal"]


def test_partial_jobs_do_not_get_a_report(tmp_path):
    with pytest.raises(ValueError, match="did not finish"):
        mod.report(tmp_path, 2)


def test_reports_changed_condition_and_new_branch(tmp_path):
    fixture(tmp_path)
    with uproot.update(tmp_path / "replay_reco.root") as root:
        root.mktree("trkdump/conditions", {"c": "float64", "extra": "int32"}).extend(
            {"c": np.array([2.0]), "extra": np.array([1], np.int32)})
    result = mod.report(tmp_path, 2)
    assert not result["equal"]
    changed = result["exported_reco"]["trees"]["trkdump/conditions"]
    assert changed["different"] == ["c"] and not changed["same_branches"]
