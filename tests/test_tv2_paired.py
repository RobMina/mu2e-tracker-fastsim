"""Paired attribution of the T-V3 track yield (tv2compare._paired_yield)."""
import json

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from trkfastsim.analyses import tv2compare as TV


def test_paired_yield_splits_selection_and_content():
    # events 1-6 selected in G4 (one track each), fast selects 1-5 and 7; event 2 loses its track in fast; 7 has one
    kg = np.array([1, 2, 3, 4, 5, 6])
    kf = np.array([1, 2, 3, 4, 5, 7])
    nr = {"key": kg, "ntrk": np.array([1, 1, 1, 1, 1, 1])}
    nf = {"key": kf, "ntrk": np.array([1, 0, 1, 1, 1, 1])}
    r = TV._paired_yield(nf, nr, kf, kg, np.full(6, 20), np.full(6, 20))
    assert r["tracks"] == [5.0, 6.0]
    assert np.isclose(r["a_selection_only"]["value"], 0.0)          # +1 (event 7) - 1 (event 6)
    assert np.isclose(r["b_both_selected_content"]["value"], -1 / 6)
    assert np.isclose(r["total_ratio_minus_1"], 5 / 6 - 1)
    assert r["b_both_selected_content"]["events_lost_track"] == 1
    assert r["a_selection_only"]["events_fast_only"] == 1 and r["a_selection_only"]["events_g4_only"] == 1


def test_per_key_sums_duplicates_and_zero_fills_absent_keys():
    r = TV._per_key(np.array([5, 3, 5]), np.array([1, 2, 4]), np.array([3, 4, 5, 9]))
    assert r.tolist() == [2.0, 0.0, 5.0, 0.0]
    assert TV._per_key(np.zeros(0), np.zeros(0), np.array([1])).tolist() == [0.0]


def test_paired_proportional_counts_have_zero_ratio_error():
    k = np.arange(1, 5)
    nr = {"key": k, "ntrk": np.array([0, 1, 2, 3])}
    nf = {"key": k, "ntrk": 2 * nr["ntrk"]}
    r = TV._paired_yield(nf, nr, k, k, np.full(4, 20), np.full(4, 20))
    assert r["total_ratio_minus_1"] == 1
    assert r["total_err"] == 0
    assert r["b_both_selected_content"]["err"] == 0
    assert r["tracks_per_selected_event_err"] == 0


def test_paired_influence_errors_include_denominator_and_covariance():
    kf, kg = np.array([1, 2, 4]), np.array([1, 2, 3])
    nf = {"key": kf, "ntrk": np.array([3, 0, 2])}
    nr = {"key": kg, "ntrk": np.array([1, 2, 1])}
    r = TV._paired_yield(nf, nr, kf, kg, [20] * 3, [20] * 3)
    ge = np.array([1, 2, 1, 0])
    ae, be = np.array([0, 0, -1, 2]), np.array([2, -2, 0, 0])
    ia = (ae - ae.sum() / ge.sum() * ge) / ge.sum()
    ib = (be - be.sum() / ge.sum() * ge) / ge.sum()
    assert np.isclose(r["selection_content_covariance"], ia @ ib)
    assert not np.isclose(ia @ ib, 0)
    assert np.isclose(r["total_err"] ** 2, (ia + ib) @ (ia + ib))
    assert np.isclose(r["total_err"] ** 2, r["a_selection_only"]["err"] ** 2
                      + r["b_both_selected_content"]["err"] ** 2 + 2 * r["selection_content_covariance"])


@pytest.mark.parametrize("bad_keys,match", [([1], "missing"), ([1, 2, 2], "duplicate")])
def test_paired_rejects_missing_or_duplicate_event_rows(bad_keys, match):
    good = {"key": np.array([1, 2]), "ntrk": np.array([1, 1])}
    bad = {"key": np.array(bad_keys), "ntrk": np.ones(len(bad_keys))}
    with pytest.raises(ValueError, match=match):
        TV._paired_yield(bad, good, [1, 2], [1, 2], [20, 20], [20, 20])


def test_zero_reference_yield_is_explicit():
    nr = {"key": np.array([1]), "ntrk": np.array([0])}
    nf = {"key": np.array([1]), "ntrk": np.array([1])}
    with pytest.raises(ValueError, match="zero reference track yield"):
        TV._paired_yield(nf, nr, [1], [1], [20], [20])
    with pytest.raises(ValueError, match="zero reference track yield"):
        TV._yield_summary(nf, nr, [1], [1], 10, 10)


def test_yield_factorization_uses_selected_events_including_explicit_zeroes():
    nf = {"key": np.array([1, 2, 3, 99]), "ntrk": np.array([0, 1, 3, 100])}
    nr = {"key": np.array([1, 2]), "ntrk": np.array([0, 2])}
    r = TV._yield_summary(nf, nr, [1, 2, 3], [1, 2], 20, 10)
    assert r["tracks"] == [4, 2]  # unselected rows are not in the numerator either
    assert r["tracks_per_selected_event"] == [4 / 3, 1]
    f = r["yield_factorised"]
    assert f["factorization_verified"]
    product = (1 + f["selected_rate_ratio_minus_1"]) * (1 + f["tracks_per_selected_event_ratio_minus_1"])
    assert np.isclose(product, 1 + r["yield_per_s"]["ratio_minus_1"])
    with pytest.raises(ValueError, match="missing"):
        TV._yield_summary(nf, nr, [1, 2, 3, 4], [1, 2], 20, 10)


def test_zero_fast_yield_is_valid():
    nf = {"key": np.array([1]), "ntrk": np.array([0])}
    nr = {"key": np.array([1]), "ntrk": np.array([2])}
    r = TV._yield_summary(nf, nr, [1], [1], 10, 10)
    assert r["yield_per_s"]["ratio_minus_1"] == -1
    assert r["yield_factorised"]["factorization_verified"]


def test_selection_uses_any_primary_muon_not_mcprimary_or_secondary(monkeypatch):
    # Only the second primary muon (not ismcprim) passes in event 1. A secondary
    # muon passes in event 2; two subthreshold primary muons must not sum in event 3.
    p = pd.DataFrame({"run": [1] * 5, "subrun": [1] * 5, "event": [1, 1, 2, 3, 3],
                      "key": [1, 2, 1, 1, 2], "pdg": [13, -13, 13, 13, -13],
                      "isprim": [1, 1, 0, 1, 1], "ismcprim": [1, 0, 0, 1, 0]})
    s = pd.DataFrame([{"run": 1, "subrun": 1, "event": ev, "sp": sp, "eion": 1e-3,
                       "px": 100, "py": 0, "pz": 0}
                      for ev, sp, n in [(1, 1, 1), (1, 2, 12), (2, 1, 20), (3, 1, 6), (3, 2, 6)]
                      for _ in range(n)])
    def read(paths, name, columns=None):
        frame = {"particles": p, "sgsteps": s}[name]
        return frame if columns is None else frame[columns]
    monkeypatch.setattr(TV.T, "read_many", read)
    monkeypatch.setattr(TV.T, "livetime", lambda _: 10)
    keys, lt, nmax = TV._selected(["unused"], with_nmax=True)
    assert keys.tolist() == TV._keyid_rows([[1, 1, 1]]).tolist()
    assert lt == 10 and nmax.tolist() == [12]


def _fake_ntuple(monkeypatch, events):
    fields = {f"evtinfo/{k}": [v] * len(events) for k, v in (("run", 1), ("subrun", 1))}
    fields["evtinfo/event"] = events
    for k in ("status", "nactive", "chisq", "ndof", "fitcon", "nhits"):
        fields[f"trk/trk.{k}"] = ak.Array([[1], []] + [[1]] * (len(events) - 2))[:len(events)]
    class Segments:
        def array(self):
            return ak.Array([[] for _ in events])
    class Tree:
        def arrays(self, *args, **kwargs):
            return ak.Array(fields)
        def __getitem__(self, key):
            return Segments()
    monkeypatch.setattr(TV.uproot, "open", lambda _: {TV.TREE: Tree()})
    monkeypatch.setattr(TV, "_pairs_at", lambda *args: (np.zeros(0),) * 7)


@pytest.mark.parametrize("events,selected,match", [([1], [1, 2], "missing"), ([1, 1], [1], "duplicate"),
                                                  ([1, 2, 2], [1], "duplicate")])
def test_ntuple_rejects_bad_coverage(monkeypatch, events, selected, match):
    _fake_ntuple(monkeypatch, events)
    keys = TV._keyid_rows([[1, 1, e] for e in selected])
    with pytest.raises(ValueError, match=match):
        TV._ntuple(["unused"], keys)


def test_ntuple_explicit_zero_tracks_valid(monkeypatch):
    _fake_ntuple(monkeypatch, [1, 2])
    keys = TV._keyid_rows([[1, 1, 1], [1, 1, 2]])
    assert TV._ntuple(["unused"], keys)["ntrk"].tolist() == [1, 0]
    with pytest.raises(ValueError, match="duplicate"):
        TV._ntuple(["unused", "unused"], keys)


@pytest.mark.parametrize("events,selected,match", [([1], [1, 2], "missing"), ([1, 1], [1], "duplicate"),
                                                  ([1, 2, 2], [1], "duplicate"), ([1, 2], [1, 2], None)])
def test_hit_table_coverage(tmp_path, events, selected, match):
    path = tmp_path / "hits.root"
    data = {"run": np.ones(len(events), dtype=int), "subrun": np.ones(len(events), dtype=int),
            "event": np.array(events)}
    data.update({k: ak.Array([[1]] + [[]] * (len(events) - 1)) for k in ("d_sid", "d_tdc0", "d_tot0")})
    with uproot.recreate(path) as f:
        f["trkdump/events"] = data
    keys = TV._keyid_rows([[1, 1, e] for e in selected])
    if match:
        with pytest.raises(ValueError, match=match):
            TV._hits([str(path)], keys)
    else:
        assert TV._hits([str(path)], keys)["ndigi"].tolist() == [1, 0]
        with pytest.raises(ValueError, match="duplicate"):
            TV._hits([str(path), str(path)], keys)


def _subruns():
    return pd.DataFrame({"run": [1, 1], "subrun": [1, 2], "livetime": [2., 3.], "primaries": [20, 30]})


def test_paired_exposure_records_identity_and_event_count_limitations(monkeypatch):
    a = _subruns()
    monkeypatch.setattr(TV.T, "read_many", lambda *args: a.copy())
    r = TV._paired_exposure(["f"], ["g"], 5, 5)
    assert r["livetime_verified"]
    assert not r["generated_identities_verified"]
    assert not r["generated_event_counts_verified"]
    assert "nEvents is unavailable" in r["limitation"]
    a["nEvents"] = [10, 15]
    r = TV._paired_exposure(["f"], ["g"], 5, 5)
    assert r["generated_event_counts_verified"]
    assert not r["generated_identities_verified"]


@pytest.mark.parametrize("kind,match", [("subrun", "different generated"), ("count", "different generated"),
                                       ("livetime", "equal subrun"), ("duplicate", "duplicate subrun"),
                                       ("missing", "requires matching"), ("nEvents", "different generated")])
def test_paired_exposure_rejects_mismatch(monkeypatch, kind, match):
    a, b = _subruns(), _subruns()
    if kind == "subrun":
        b.loc[0, "subrun"] = 9
    elif kind == "count":
        b.loc[0, "primaries"] += 1
    elif kind == "livetime":
        b["livetime"] = [1., 4.]  # equal total is insufficient
    elif kind == "duplicate":
        b = pd.concat([b, b.iloc[:1]], ignore_index=True)
    elif kind == "missing":
        a, b = a.drop(columns="primaries"), b.drop(columns="primaries")
    elif kind == "nEvents":
        a["nEvents"], b["nEvents"] = [10, 15], [10, 16]
    monkeypatch.setattr(TV.T, "read_many", lambda paths, _: a if paths == ["f"] else b)
    with pytest.raises(ValueError, match=match):
        TV._paired_exposure(["f"], ["g"], 5, 5)


def test_occupancy_threshold_uses_full_covariance_and_labels_paired_approximation():
    res = {"yield_per_s": {"ratio_minus_1": 0, "err": 0}, "nactive_mean": [1, 1, 0, 0],
           "chi2ndf_band": 0, "digis_per_event": [1, 1, 0, 0], "tdc0_band": 0, "tot0_band": 0,
           "panel_occupancy": {"chi2ndf_clustered": 0.1, "chi2ndf_poisson": 0.2,
                               "chi2ndf_covariance": 2., "covariance_rank": 2, "panels_beyond_5sigma": 0}}
    res.update({f"resid_{k}": {"width": [1, 1, 0], "width_err": 0} for k in ("dx_mm", "dy_mm", "dang_mrad", "dt_ns")})
    r = TV._thresholds(res)
    key = "T-V2 panel occupancy full covariance"
    assert r[key]["certifying"] and not r[key]["pass"]
    assert r["T-V2 panel occupancy diagonal clustered score (diagnostic)"]["pass"] is None
    res["paired"] = True
    r = TV._thresholds(res)
    assert not r[key]["certifying"] and r[key]["pass"] is None
    assert "independent-arm approximation" in r[key]["interpretation"]
    res["panel_occupancy"]["cross_arm_covariance_included"] = True
    r = TV._thresholds(res)
    assert r[key]["certifying"] and not r[key]["pass"]
    res["panel_occupancy"]["chi2ndf_covariance"] = 1.0
    assert TV._thresholds(res)[key]["pass"]


def test_run_paired_uses_influence_error_for_headline_yield(monkeypatch, tmp_path):
    keys = np.arange(1, 5)
    def selected(paths, with_nmax=False):
        return keys, 5., np.full(4, 20)
    def ntuple(paths, sel):
        counts = np.array([0, 1, 2, 3]) * (2 if paths == ["f"] else 1)
        n = int(counts.sum())
        out = {"key": keys, "ntrk": counts, "nactive": np.full(n, 20.), "tfile": np.zeros(n), "tkey": np.repeat(keys, counts),
               "file_subruns": [np.array([[1, 1], [1, 2]])]}
        out.update({k: np.ones(n) for k in ("chi2ndf", "fitcon", "dx", "dy", "dt", "dang", "xrank")})
        out["xfile"] = np.zeros(n, dtype=int)
        return out
    monkeypatch.setattr(TV, "_selected", selected)
    monkeypatch.setattr(TV, "_ntuple", ntuple)
    monkeypatch.setattr(TV.T, "read_many", lambda *args: _subruns())
    monkeypatch.setattr(TV.stats, "ratio_band", lambda *args, **kw: {"max_abs_dev_minus_2sigma": None})
    TV.run(["f"], str(tmp_path), compare=["g"], steps=["sf"], steps_ref=["sg"], paired=True)
    r = json.loads((tmp_path / "tv2compare.json").read_text())
    assert r["yield_per_s"]["ratio_minus_1"] == 1
    assert r["yield_per_s"]["err"] == 0
    assert r["yield_factorised"]["tracks_per_selected_event_err"] == 0
    assert r["yield_factorised"]["factorization_verified"]
    assert not r["paired_exposure"]["generated_identities_verified"]
    assert not r["thresholds"]["T-V3 chi2/ndf binned ratio (minus 2 sigma)"]["pass"]


def test_single_reference_file_floor_is_unavailable(monkeypatch):
    def no_call(*args, **kwargs):
        raise AssertionError("empty comparison must not call ratio_band")
    monkeypatch.setattr(TV.stats, "ratio_band", no_call)
    assert TV._band_or_none(np.ones(10), np.array([])) is None
    assert TV._band_or_none(np.array([]), np.ones(10)) is None


def test_paired_selected_rate_error_includes_reference_fluctuations():
    kf, kg = np.array([1, 2, 3, 4]), np.array([1, 2])
    nf, nr = {"key": kf, "ntrk": np.ones(4)}, {"key": kg, "ntrk": np.ones(2)}
    r = TV._paired_yield(nf, nr, kf, kg, [20] * 4, [20] * 2)
    assert r["selected_rate_ratio_minus_1"] == 1
    # Influence vector [-1, -1, 1, 1] / Ng, not sqrt(fast-only + G4-only) / Ng.
    assert r["selected_rate_err"] == 1
    assert r["tracks_per_selected_event_err"] == 0


@pytest.mark.parametrize("bad", [[[1, 1, 2**24]], [[1, 2**24, 1]], [[2**15, 1, 1]],
                                  [[65537, 1, 1]], [[1, -1, 1]], [[1, 1, 1.5]], [[1, 1, np.nan]]])
def test_packed_event_identity_rejects_aliases_and_invalid_fields(bad):
    with pytest.raises(ValueError, match="packed bounds"):
        TV._keyid_rows(bad)
    with pytest.raises(ValueError, match="packed bounds"):
        TV._keyid(bad)


def test_packed_event_identity_maximum_and_empty():
    assert TV._keyid_rows([[2**15-1, 2**24-1, 2**24-1]]).item() == np.iinfo(np.int64).max
    assert len(TV._keyid_rows(np.empty((0, 3), dtype=int))) == 0
    with pytest.raises(ValueError, match="numeric"):
        TV._keyid_rows([1, 2, 3])


def test_width_error_excludes_empty_groups_and_refuses_invalid_deletions():
    x = np.linspace(-1, 1, 20)
    fn = lambda v: TV._width(v)[0]
    for ng in (1, 2, 20):
        r, e = TV._jack_ratio(fn, x, np.zeros(len(x)), x, np.zeros(len(x)), ng)
        assert r == 0 and np.isnan(e)
    x = np.r_[np.linspace(-1, 1, 20), np.linspace(-2, 2, 20)]
    g = np.repeat([0, 1], 20)
    assert TV._jack_ratio(fn, x, g, x, g, 2) == TV._jack_ratio(fn, x, g, x, g, 20) == (0, 0)
    g = np.r_[np.zeros(37), 1, 2, 2]
    assert np.isnan(TV._jack_ratio(fn, x, g, x, g, 3)[1])


def test_width_group_matching_invariant_to_file_order():
    x = np.r_[np.linspace(-1, 1, 20), np.linspace(-2, 2, 20)]
    g = np.repeat([0, 1], 20)
    nf = {"xfile": g, "file_subruns": [np.array([[1, 1], [1, 2]]), np.array([[1, 3], [2, 1]])]}
    nr = {"xfile": 1-g, "file_subruns": nf["file_subruns"][::-1]}
    gf, gr, ng, meta = TV._width_groups(nf, nr, True)
    assert meta["available"] and meta["reference_file_to_fast_group"] == [1, 0]
    assert np.array_equal(gf, gr) and ng == 2
    assert TV._jack_ratio(lambda v: TV._width(v)[0], x, gf, x, gr, ng) == (0, 0)


@pytest.mark.parametrize("blocks", [[[[1, 1]], [[1, 1]]], [[], [[1, 1]]], [[[1, 1], [1, 2]]]])
def test_width_unmatched_or_overlapping_blocks_are_unavailable(blocks):
    nf = {"xfile": np.array([0, 1]), "file_subruns": [np.array([[1, 1]]), np.array([[1, 2]])]}
    nr = {"xfile": np.array([0, 0]), "file_subruns": [np.array(b).reshape(-1, 2) for b in blocks]}
    _, _, ng, meta = TV._width_groups(nf, nr, True)
    assert ng == 0 and not meta["available"]
