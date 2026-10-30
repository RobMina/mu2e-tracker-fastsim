"""Real-table benchmark admission and same-seed timing receipts."""
import json
import numpy as np
import pytest

import digi_real_bench as bench


def test_real_bench_retains_empty_events_and_checks_repeats(tmp_path, monkeypatch):
    keys = np.array([[1400, 22, 7], [1400, 22, 8]])
    monkeypatch.setattr(bench.digicond.Conditions, "load", lambda p: type("C", (), {"nADCSamples": 15})())
    monkeypatch.setattr(bench.fastdigi, "event_keys", lambda p: keys)
    calls = []
    def digitize(path, cond, rng, population, stats, backend):
        calls.append(population.copy())
        return population, {"sid": np.zeros(0, int)}, 0
    monkeypatch.setattr(bench.fastdigi, "digitize_table", digitize)
    monkeypatch.setattr(bench.fastdigi, "write_digis", lambda *a: None)
    out = tmp_path / "result"
    argv = ["--steps", "s.root", "--events", "e.root", "--cond", "c.npz", "--backend", "numpy", "--outdir", str(out)]
    bench.main(argv)
    report = json.loads((out / "timing.json").read_text())
    assert report["reproducible"] and len(report["runs"]) == 3
    assert all(row["events"] == 2 and row["digis"] == 0 for row in report["runs"])
    assert all(np.array_equal(k, keys) for k in calls)
    with pytest.raises(FileExistsError):
        bench.main(argv)


def test_real_bench_rejects_duplicate_events(tmp_path, monkeypatch):
    monkeypatch.setattr(bench.digicond.Conditions, "load", lambda p: None)
    monkeypatch.setattr(bench.fastdigi, "event_keys", lambda p: np.ones((2, 3), int))
    with pytest.raises(ValueError, match="duplicate"):
        bench.main(["--steps", "s", "--events", "e", "--cond", "c", "--backend", "numpy", "--outdir", str(tmp_path / "bad")])
