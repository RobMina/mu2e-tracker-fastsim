"""Event-cluster uncertainty treats replicated steps as correlated, not new observations."""
import numpy as np
import pandas as pd
from trkfastsim.analyses.jaxcheck import event_clustered


def _input():
    rng = np.random.default_rng(27)
    gen = pd.DataFrame({"run": 1, "subrun": 1, "event": np.arange(100), "t": 0.0})
    frames = []
    for _ in range(2):
        df = gen.iloc[:80].copy()
        df["sid"] = (np.arange(80) % 4) * 128
        for col in ("t", "x0", "y0", "z0", "len", "eion"):
            df[col] = rng.normal(size=len(df))
        frames.append(df)
    return frames, gen


def test_duplicate_steps_do_not_shrink_event_error():
    (a, b), gen = _input()
    ref = event_clustered(a, b, gen)
    got = event_clustered(pd.concat([a] * 10), pd.concat([b] * 10), gen)
    for col, r in ref["means"].items():
        for k in ("difference", "paired_event_se", "diff_sigma"):
            np.testing.assert_allclose(got["means"][col][k], r[k], atol=1e-14)
    assert ref["generated_events"] == 100
    assert ref["events_with_steps_union"] == 80


def test_identical_events_have_zero_paired_error():
    (a, _), gen = _input()
    got = event_clustered(a, a.copy(), gen)
    assert all(q["paired_event_se"] == q["difference"] == 0 for q in got["means"].values())
    assert got["panel_occupancy"]["chi2"] == 0
    assert got["panel_occupancy"]["pvalue"] == 1


def test_panel_occupancy_uses_event_covariance():
    (a, b), gen = _input()
    b["sid"] = b.sid.to_numpy()[::-1]
    one = event_clustered(a, b, gen)["panel_occupancy"]
    many = event_clustered(pd.concat([a] * 7), pd.concat([b] * 7), gen)["panel_occupancy"]
    assert one["rank"] > 0
    np.testing.assert_allclose(one["chi2"], many["chi2"], atol=1e-14)


def test_unequal_multiplicity_and_arm_only_events():
    (source, _), gen = _input()
    a = source.iloc[[0, 0, 1, 3]].copy()
    b = source.iloc[[0, 2, 2, 2, 3, 3]].copy()
    a["x0"] = [1.0, 3.0, 7.0, 4.0]
    b["x0"] = [2.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    got = event_clustered(a, b, gen)
    na, nb = np.array([2, 1, 0, 1]), np.array([1, 0, 3, 2])
    sa, sb = np.array([4.0, 7.0, 0.0, 4.0]), np.array([2.0, 0.0, 18.0, 18.0])
    influence = (sb - sb.sum() / nb.sum() * nb) / nb.sum() - (sa - sa.sum() / na.sum() * na) / na.sum()
    np.testing.assert_allclose(got["means"]["x0"]["paired_event_se"], np.sqrt(100 / 99 * (influence @ influence)))
    delta = np.r_[nb - na, np.zeros(96)]
    np.testing.assert_allclose(got["steps_per_generated_event"]["paired_event_se"], delta.std(ddof=1) / 10)


def test_nonzero_panel_quadratic_form():
    (a, b), gen = _input()
    a["sid"] = (np.arange(80) % 2) * 128
    b["sid"] = (np.arange(80) % 3 == 0).astype(int) * 128
    got = event_clustered(a, b, gen)["panel_occupancy"]
    da = (a.sid.to_numpy() == 0).astype(float)
    db = (b.sid.to_numpy() == 0).astype(float)
    delta = db.mean() - da.mean()
    influence = ((db - db.mean()) - (da - da.mean())) / 80
    variance = 100 / 99 * (influence @ influence)
    assert got["rank"] == 1 and got["chi2"] > 0
    np.testing.assert_allclose(got["chi2"], delta ** 2 / variance)


def test_discarded_mode_disagreement_is_not_a_pass():
    (a, b), gen = _input()
    a["sid"], b["sid"] = 0, 128
    got = event_clustered(a, b, gen)["panel_occupancy"]
    assert got["rank"] == 0 and got["unresolved_discarded_modes"]
    assert got["pvalue"] is None


def test_invalid_cluster_inputs_rejected():
    import pytest
    (a, b), gen = _input()
    for bad in (a.iloc[:0], a.assign(x0=np.nan), a.assign(event=1000)):
        with pytest.raises(ValueError):
            event_clustered(bad, b, gen)
    with pytest.raises(ValueError):
        event_clustered(a.iloc[:1], b.iloc[:1], gen.iloc[:1])


def test_near_singular_covariance_reports_discarded_modes():
    (a, b), gen = _input()
    a["sid"] = np.where(np.arange(80) < 79, 0, 256)
    b["sid"] = np.where(np.arange(80) % 2, 0, 128)
    got = event_clustered(a, b, gen)["panel_occupancy"]
    assert "eigenvalues" in got and got["rank_tolerance"] >= 0
    assert got["rank"] <= 2
    assert not got["unresolved_discarded_modes"]
