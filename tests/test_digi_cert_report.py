"""Closure reporting refuses a partial, intersected or mistimed population."""
import json
import pytest

import digi_cert_report as report


def test_report_rejects_incomplete_ten_group_receipt(tmp_path):
    (tmp_path / "ten_comparison").mkdir()
    (tmp_path / "ten_comparison/digicmp.json").write_text(json.dumps({"digis": {"fast": 0}}))
    (tmp_path / "ten_jax_cpu").mkdir()
    (tmp_path / "ten_jax_cpu/timing.json").write_text(json.dumps({"runs": [{"events": 1, "file_index": 0}]}))
    with pytest.raises(ValueError, match="ten complete"):
        report.main([str(tmp_path), "--out", str(tmp_path / "report.json")])
    assert not (tmp_path / "report.json").exists()


def test_report_refuses_incomplete_f2_receipt(tmp_path):
    compare = dict(events=10, digis=dict(fast=10, ref=10, floor=10),
                   event_population={"identical_unique_keys": True}, digis_per_event={}, panelhits_per_event={},
                   chi2ndf={}, f2={"fast": {"paired_rel_diff": .001, "paired_rel_err": .002}})
    (tmp_path / "ten_comparison").mkdir()
    (tmp_path / "ten_comparison/digicmp.json").write_text(json.dumps(compare))
    progress = tmp_path / "ten_comparison/f2_progress.jsonl"
    progress.write_text(json.dumps(dict(arm="fast", stop=9, wall_s=1, shape_counts={"64x64": 1})) + "\n")
    for name in ("ten_jax_cpu", "ten_numpy", "ten_floor"):
        (tmp_path / name).mkdir()
        rows = [dict(events=1, digis=1, file_index=i) for i in range(10)]
        (tmp_path / name / "timing.json").write_text(json.dumps({"runs": rows}))
    with pytest.raises(ValueError, match="incomplete F2"):
        report.main([str(tmp_path), "--out", str(tmp_path / "report.json")])
    assert not (tmp_path / "report.json").exists()


def test_timing_distinguishes_cold_and_warm(tmp_path):
    rows = [dict(events=10, steps=30, digis=9, repeat=i, seconds=s,
                 stats={"digitize_s": s - .1}, maxrss_kib=200 + i) for i, s in enumerate((20., 2., 3.))]
    (tmp_path / "timing.json").write_text(json.dumps(dict(reproducible=True, device="cpu", runs=rows)))
    got = report.timing(tmp_path)
    assert got["cold_s"] == 20 and got["warm_s"] == [2., 3.]
    assert got["digitize_warm_s"] == [1.9, 2.9]
    assert got["peak_process_rss_kib"] == 202
    (tmp_path / "timing.json").write_text(json.dumps(dict(reproducible=True, device="cpu", runs=rows[:1])))
    with pytest.raises(ValueError, match="two reproducible warm"):
        report.timing(tmp_path)
