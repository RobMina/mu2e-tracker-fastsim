"""Paired per-shard statistic (shardstats) and per-shard histogram filling (shardhist) on synthetic shards."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import shardhist as SH
from trkfastsim.validation import shardstats as SS


def _write(tmp_path, arms, nshard=60, name="a.npz"):
    z = {}
    for arm, hists in arms.items():
        z[f"shards|{arm}"] = np.arange(nshard, dtype=np.int64) + 21
        for k, v in hists.items():
            z[f"h|{arm}|{k}"] = v.astype(np.int64)
            z[f"edges|{k}"] = np.arange(v.shape[1] - 1, dtype=float)
    p = tmp_path / name
    np.savez(p, **z)
    return p


def _pair(rng, nshard, nbin, mean, shift=None, clus=3):
    """Two arms sharing a clustered common part per shard (shared primaries) plus independent parts."""
    common = rng.poisson(mean * 0.7 / clus, size=(nshard, nbin)) * clus
    a = common + rng.poisson(mean * 0.3, size=(nshard, nbin))
    lam = mean * 0.3 * (np.ones(nbin) if shift is None else shift)
    b = common + rng.poisson(lam, size=(nshard, nbin))
    return a, b


def test_null_is_calibrated_and_shift_is_flagged(tmp_path):
    rng = np.random.default_rng(3)
    ref, cand = {}, {}
    for i in range(12):
        a, b = _pair(rng, 80, 25, 200.0)
        ref[f"h{i:02d}"], cand[f"h{i:02d}"] = a, b
    shift = np.ones(25)
    shift[10:15] = 1.4
    a, b = _pair(rng, 80, 25, 200.0, shift)
    ref["shifted"], cand["shifted"] = a, b
    sh = SS.Shards(_write(tmp_path, {"ref": ref, "cand": cand}, nshard=80))
    per, meta = SS.compare(sh, "ref", "cand", nflip=4000, nboot=100)
    assert per["shifted"]["flag"]
    assert per["shifted"]["rms_rel_excess"] > 0.02
    nulls = [per[f"h{i:02d}"]["chi2ndf"] for i in range(12)]
    assert 0.8 < np.mean(nulls) < 1.25
    assert sum(per[f"h{i:02d}"]["flag"] for i in range(12)) <= 1
    assert abs(np.mean([per[f"h{i:02d}"]["null_mean"] for i in range(12)]) - 1.0) < 0.05
    assert meta["N"] == 80 and meta["p_per_hist"] < 0.01


def test_identical_and_empty_status(tmp_path):
    x = np.random.default_rng(1).poisson(5, size=(20, 4))
    z = np.zeros((20, 4))
    sh = SS.Shards(_write(tmp_path, {"r": {"same": x, "empty": z}, "c": {"same": x, "empty": z}}, nshard=20))
    per, meta = SS.compare(sh, "r", "c", nflip=200, nboot=0)
    assert per["same"]["status"] == "identical" and not per["same"]["flag"]
    assert per["empty"]["status"] == "both_empty"
    assert meta["n_ok"] == 0


def test_total_pull_and_shape_mode(tmp_path):
    rng = np.random.default_rng(9)
    a = rng.poisson(100, size=(50, 10))
    b = rng.poisson(110, size=(50, 10))               # 10 % more in every bin: a rate difference only
    ea = a.sum(1, keepdims=True) // 10
    eb = (ea * 1.1).round()
    sh = SS.Shards(_write(tmp_path, {"r": {"x": a, "sel/events": ea}, "c": {"x": b, "sel/events": eb}}, nshard=50))
    per, _ = SS.compare(sh, "r", "c", keys=["x"], nflip=1000, nboot=0)
    assert per["x"]["total_rel"] == pytest.approx(0.1, abs=0.02)
    assert per["x"]["flag"]
    per_s, meta = SS.compare(sh, "r", "c", keys=["x"], nflip=1000, nboot=0, shape=True)
    assert meta["scale"] == pytest.approx(ea.sum() / eb.sum())
    assert per_s["x"]["chi2ndf"] < per["x"]["chi2ndf"]


def test_split_null_and_multi_file(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.poisson(50, size=(40, 8)) * 2
    p1 = _write(tmp_path, {"g4": {"k": x}}, nshard=40, name="g4.npz")
    p2 = _write(tmp_path, {"fast": {"k": x + 1}}, nshard=40, name="fast.npz")
    sh = SS.Shards([p1, p2])
    assert sh.arms() == ["fast", "g4"]
    out = SS.split_null(sh, "g4", nsplit=500)
    assert set(out["hists"]["k"]) >= {"first_vs_second", "even_vs_odd", "random_mean"}
    assert 0.6 < out["hists"]["k"]["random_mean"] < 1.4
    with pytest.raises(ValueError):
        SS.Shards([p1, p1])


def test_fill_keeps_flows():
    H = SH.Hists([21, 22])
    H.fill("sec/place", np.array([0, 0, 1, 1]), np.array([-3.0, 0.0, 3.0, 9.0]))
    h = H.h["sec/place"]
    assert h.shape == (2, 7)                              # 5 bins + underflow + overflow
    assert h[0, 0] == 1 and h[0, 1] == 1 and h[1, 4] == 1 and h[1, 6] == 1


def test_place_classes():
    s = pd.DataFrame({"run": 1, "subrun": 1, "event": [1] * 6 + [2],
                      "sid": [10, 11, 10, 99, 200, 300, 5], "plane": [3, 4, 3, 3, 6, 9, 2],
                      "mu": [True, True, False, False, False, False, False]})
    place, dmin = SH._place(s)
    # non-muon rows in order: (sid 10, plane 3) in muon straw; (99, 3) same plane; (200, 6) two planes from 4;
    # (300, 9) five planes away; event 2 has no muon step
    assert place.tolist() == [0, 1, 2, 3, 4]
    assert dmin.tolist() == [-1, 0, 2, 5, 99]


def test_batched_shard_ids():
    ids = SH._sid(np.array([11, 11, 12]), np.array([7, 10, 3]), 10)
    assert ids.tolist() == [117, 110, 123]
    full = (np.array([11, 12])[:, None] * 10 + np.arange(10)[None, :]).ravel()
    assert SH._sidx(full, ids).tolist() == [7, 0, 13]
    with pytest.raises(ValueError):
        SH._sidx(full, np.array([135]))


def test_bin_table(tmp_path):
    a = np.full((10, 3), 100)
    b = a.copy()
    b[:, 1] += np.arange(10) % 3          # +10 in bin 1 with a spread
    sh = SS.Shards(_write(tmp_path, {"r": {"k": a}, "c": {"k": b}}, nshard=10))
    t = SS.bin_table(sh, "r", "c", "k")
    assert t["ratio"][1] == pytest.approx(1.009)
    assert t["ratio"][0] == 1.0 and t["pull"][1] > 0
    assert "k" in SS.bin_text(t, "k")


def test_value_sum_is_the_multiplicity():
    e = np.arange(0, 6) - 0.5                      # values 0..4
    r = np.zeros((4, 7))
    c = np.zeros((4, 7))
    r[:, 1 + 2] = 10                               # 10 events with value 2 per shard
    c[:, 1 + 2] = 9
    c[:, 1 + 4] = 1                                # one event moves from 2 to 4
    v = SS.value_sum(e, r, c)
    assert v["mean_ref"] == pytest.approx(2.0) and v["mean_cand"] == pytest.approx(2.2)
    assert v["value_rel"] == pytest.approx(0.1) and v["value_rel_err"] == pytest.approx(0.0)
