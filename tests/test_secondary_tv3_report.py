"""Saved-result summaries preserve paired attribution and rescore differences."""
import importlib.util
import json
from pathlib import Path

import pytest


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / f"scripts/{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


REPORT = module("straw_occ_report")
RESCORE = module("secondary_tv3_rescore")


def test_saved_summary_preserves_paired_counts_covariance_and_flow(tmp_path):
    data = {"paired": True, "selected_events": [10, 12], "selected_in_ntuple": [10, 12], "tracks": [8, 10],
            "yield_per_s": {}, "yield_factorised": {}, "thresholds": {},
            "yield_paired": {"tracks": [8, 10], "selection_content_covariance": -.001,
                             "a_selection_only": {"tracks_fast_only": 2, "tracks_g4_only": 3},
                             "b_both_selected_content": {"tracks": [6, 7]}},
            "width_jackknife_groups": {"available": True, "groups": 20},
            "tdc0_event_clustered": {"available": True, "max_abs_dev": .1, "max_abs_dev_minus_2sigma": .02,
                                     "flow_fast": {"underflow": 12}, "flow_ref": {"underflow": 15}}}
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(data))
    out = REPORT.summarize(path)
    assert out["yield_paired"] == data["yield_paired"]
    assert out["yield_attribution_counts"]["total_track_difference"] == -2
    assert out["yield_attribution_counts"]["selection_only_track_difference"] == -1
    assert out["yield_attribution_counts"]["both_selected_track_difference"] == -1
    assert out["clustered_shapes"]["tdc0_event_clustered"]["flow_fast"]["underflow"] == 12
    assert out["width_jackknife_groups"]["groups"] == 20
    assert len(out["sha256"]) == 64


def test_rescore_reports_precise_field_changes_and_preserves_saved(tmp_path, monkeypatch):
    data = {"paired": True, "input_files": {k: [k] for k in
            ("ntuple_fast", "ntuple_g4", "steps_fast", "steps_g4", "hits_fast", "hits_g4")},
            "same": [float("nan"), 1.], "x": {"value": 1.}}
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps(data))
    original = saved.read_bytes()
    def run(inputs, outdir, **kwargs):
        assert inputs == data["input_files"]["ntuple_fast"] and kwargs["paired"]
        result = {**data, "width_jackknife_groups": {"available": True}, "x": {"value": 2.}}
        Path(outdir).mkdir()
        (Path(outdir) / "tv2compare.json").write_text(json.dumps(result))
    monkeypatch.setattr(RESCORE.TV, "run", run)
    audit = RESCORE.run(saved, tmp_path / "result")
    assert audit["changed_fields"] == ["width_jackknife_groups", "x.value"]
    assert saved.read_bytes() == original
    with pytest.raises(FileExistsError):
        RESCORE.run(saved, tmp_path / "result")
    assert RESCORE.differences([1], [1, 2], "list") == ["list"]
