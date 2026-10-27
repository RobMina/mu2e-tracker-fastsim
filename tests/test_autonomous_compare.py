"""Synthetic, all-event paired yields; no detector inputs or reconstruction."""
import json

import numpy as np
import pytest

from trkfastsim.reco import autonomous_compare as C


def _keys(n):
    return np.column_stack((np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64), np.arange(n)))


def test_all_events_multitrack_and_reordering():
    keys = np.array([[1, 0, 7], [1, 1, 7], [2, 0, 7], [2, 1, 7]])
    a, b = np.array([0, 1, 2, 0]), np.array([1, 0, 3, 0])
    order = np.array([3, 1, 0, 2])
    result = C.summarize_counts(keys, a, keys[order], b[order])
    assert result == C.summarize_counts(keys, a, keys, b)
    assert result["events"] == 4
    assert result["tracks"]["A_total"] == 3
    assert result["tracks"]["B_total"] == 4
    assert result["tracks"]["A_per_event"] == 0.75
    assert result["tracks"]["B_per_event"] == 1
    assert result["tracks"]["difference_B_minus_A"] == 0.25
    assert result["tracks"]["relative_yield_B_over_A_minus_1"] == pytest.approx(1 / 3)
    assert result["events_with_fit"]["A_per_event"] == 0.5
    assert result["events_with_fit"]["B_per_event"] == 0.5
    assert result["multiplicity"]["A"]["fractions_0_1_2plus"] == [0.5, 0.25, 0.25]
    assert result["multiplicity"]["B"]["counts_0_1_2plus"] == [2, 1, 1]
    assert result["contingency_A_rows_B_columns_0_1_2plus"] == [[1, 1, 0], [1, 0, 0], [0, 0, 1]]


@pytest.mark.parametrize("indicator", [False, True])
def test_covariance_delta_method_and_paired_difference(indicator):
    a, b = np.array([0, 1, 4, 0, 2]), np.array([1, 2, 3, 0, 0])
    result = C.summarize_counts(_keys(5), a, _keys(5), b)
    x, y = (a > 0, b > 0) if indicator else (a, b)
    x, y = x.astype(float), y.astype(float)
    row = result["events_with_fit" if indicator else "tracks"]
    cov = np.cov(x, y, ddof=1)
    r = y.mean() / x.mean()
    variance = (cov[1, 1] + r**2 * cov[0, 0] - 2 * r * cov[0, 1]) / (5 * x.mean()**2)
    assert row["sample_variance_A"] == pytest.approx(cov[0, 0])
    assert row["sample_variance_B"] == pytest.approx(cov[1, 1])
    assert row["sample_covariance_AB"] == pytest.approx(cov[0, 1])
    assert row["relative_yield_se"] == pytest.approx(np.sqrt(variance))
    assert row["difference_se"] == pytest.approx(np.std(y - x, ddof=1) / np.sqrt(5))


def test_covariance_cancels_for_identical_and_proportional_counts():
    a = np.array([0, 1, 4, 0, 2])
    identical = C.summarize_counts(_keys(5), a, _keys(5), a)
    assert identical["tracks"]["difference_se"] == 0
    assert identical["tracks"]["relative_yield_se"] == 0
    assert identical["events_with_fit"]["relative_yield_se"] == 0
    proportional = C.summarize_counts(_keys(5), a, _keys(5), 2 * a)
    assert proportional["tracks"]["relative_yield_B_over_A_minus_1"] == 1
    assert proportional["tracks"]["relative_yield_se"] == 0
    assert proportional["tracks"]["difference_se"] > 0


@pytest.mark.parametrize("arm", ["A", "B"])
def test_duplicate_identity_rejected_even_with_same_population(arm):
    k = _keys(3)
    duplicate = k.copy()
    duplicate[2] = duplicate[1]
    with pytest.raises(ValueError, match="duplicate"):
        C.summarize_counts(duplicate if arm == "A" else k, [0, 1, 2],
                           duplicate if arm == "B" else k, [0, 1, 2])
    with pytest.raises(ValueError, match="duplicate"):
        C.summarize_counts(duplicate, [0, 1, 2], duplicate, [0, 1, 2])


@pytest.mark.parametrize("component", [0, 1, 2])
def test_population_mismatch_in_any_identity_component_rejected(component):
    a, b = _keys(3), _keys(3)
    b[1, component] += 10
    with pytest.raises(ValueError, match="populations"):
        C.summarize_counts(a, [0, 1, 2], b, [0, 1, 2])


def test_missing_zero_track_event_is_not_silently_intersected():
    with pytest.raises(ValueError, match="populations"):
        C.summarize_counts(_keys(3), [1, 2, 0], _keys(2), [1, 2])


@pytest.mark.parametrize("keys,counts", [
    ([[1, 2]], [0]), ([[1.0, 2.0, 3.0]], [0]), ([[1, None, 3]], [0]),
    ([[1, 2, 3]], [-1]), ([[1, 2, 3]], [1.5]), ([[1, 2, 3]], [np.nan]),
    ([[1, 2, 3]], []), ([[1, 2, 3]], [[1]]),
    (np.ma.array([[1, 2, 3]], mask=[[False, True, False]]), [0]),
    ([[1, 2, 3]], np.ma.array([1], mask=[True])),
])
def test_invalid_keys_and_counts_rejected(keys, counts):
    with pytest.raises(ValueError):
        C.summarize_counts(keys, counts, _keys(1), [0])


@pytest.mark.parametrize("b", [[0, 0, 0], [0, 1, 3]])
def test_zero_reference_denominator_is_null(b):
    result = C.summarize_counts(_keys(3), [0, 0, 0], _keys(3), b)
    for name in ("tracks", "events_with_fit"):
        assert result[name]["relative_yield_B_over_A_minus_1"] is None
        assert result[name]["relative_yield_se"] is None
        assert result[name]["A_per_event"] == 0
        assert result[name]["difference_se"] is not None
    json.dumps(result, allow_nan=False)


def test_zero_candidate_is_defined_minus_one():
    result = C.summarize_counts(_keys(3), [0, 1, 3], _keys(3), [0, 0, 0])
    assert result["tracks"]["relative_yield_B_over_A_minus_1"] == -1
    assert result["tracks"]["relative_yield_se"] == 0


def test_empty_population_has_null_fractions_and_uncertainties():
    result = C.summarize_counts([], [], np.empty((0, 3), dtype=int), [])
    assert result["events"] == 0
    assert result["multiplicity"]["A"]["fractions_0_1_2plus"] == [None] * 3
    for name in ("tracks", "events_with_fit"):
        assert result[name]["A_total"] == result[name]["B_total"] == 0
        assert result[name]["A_per_event"] is None
        assert result[name]["difference_B_minus_A"] is None
        assert result[name]["difference_se"] is None
        assert result[name]["relative_yield_se"] is None
    json.dumps(result, allow_nan=False)


def test_one_event_has_point_estimates_but_no_standard_errors():
    result = C.summarize_counts(_keys(1), [2], _keys(1), [3])
    assert result["tracks"]["relative_yield_B_over_A_minus_1"] == 0.5
    for name in ("tracks", "events_with_fit"):
        assert result[name]["difference_se"] is None
        assert result[name]["relative_yield_se"] is None
        assert result[name]["sample_covariance_AB"] is None


def _root(path, keys, counts, *, hits=False, chi2=1.0, parameters=None, sids=None):
    ak = pytest.importorskip("awkward")
    uproot = pytest.importorskip("uproot")
    counts = np.asarray(counts, dtype=np.int64)
    nt = int(counts.sum())
    # Deliberately failed fits: the evaluator must count them without selection.
    goodfit = ak.unflatten(np.zeros(nt, dtype=np.int32), counts)
    trk = ak.zip({"trk.goodfit": goodfit,
                  "trk.nhits": ak.unflatten(np.full(nt, 5, dtype=np.int32), counts),
                  "trk.chisq": ak.unflatten(np.full(nt, chi2), counts),
                  "trk": ak.zip({"goodfit": goodfit}, depth_limit=1)}, depth_limit=1)
    branches = {"evtinfo": ak.zip(dict(zip(("run", "subrun", "event"), keys.T))), "trk": trk}
    if hits:
        zeros = np.zeros(nt * 5, dtype=np.int32)
        pos = ak.zip({"fCoordinates": ak.zip({"fX": zeros, "fY": zeros, "fZ": zeros})})
        ordinal = np.arange(nt) - np.repeat(np.cumsum(counts) - counts, counts)
        h = ak.zip({"plane": zeros, "panel": zeros, "layer": zeros,
                    "straw": np.repeat(5 * ordinal, 5) + np.tile(np.arange(5), nt), "state": np.ones(nt * 5, dtype=np.int32),
                    "poca": pos})
        branches["trkhits"] = ak.unflatten(ak.unflatten(h, np.full(nt, 5)), counts)
    if parameters is not None:
        z = np.zeros(nt)
        pos = ak.zip({"fCoordinates": ak.zip({"fX": z, "fY": z, "fZ": z})})
        mom = ak.zip({"fCoordinates": ak.zip({"fX": z, "fY": z, "fZ": np.ones(nt)})})
        seg = ak.zip({"sid": np.full(nt, 4) if sids is None else np.asarray(sids),
                      "sindex": np.zeros(nt, dtype=np.int32), "inbounds": np.ones(nt, dtype=bool),
                      "time": z, "pos": pos, "mom": mom, "momerr": z, "dmom": z})
        branches["trksegs"] = ak.unflatten(ak.unflatten(seg, np.ones(nt, dtype=int)), counts)
        pars = ak.zip({k: np.asarray(v, dtype=float) for k, v in parameters.items()})
        branches["trksegpars_kl"] = ak.unflatten(ak.unflatten(pars, np.ones(nt, dtype=int)), counts)
    with uproot.recreate(path) as root:
        root[C.TREE] = ak.zip(branches, depth_limit=1)


def test_cli_counts_only_reorders_and_refuses_overwrite(tmp_path, monkeypatch):
    stock, fast, out = tmp_path / "stock.root", tmp_path / "fast.root", tmp_path / "result.json"
    keys = _keys(4)
    order = np.array([3, 1, 0, 2])
    _root(stock, keys, [0, 1, 2, 0])
    _root(fast, keys[order], np.array([1, 0, 3, 0])[order])
    result = C.main([str(fast), str(stock), "--out", str(out)])
    assert json.loads(out.read_text()) == result
    assert result["events"] == 4
    assert result["tracks"]["B_total"] == 4  # all goodfit == 0; no hidden fit cut
    assert not result["shared_hit_diagnostics"]["enabled"]
    assert "comparison" not in result["shared_hit_diagnostics"]
    saved = out.read_bytes()
    monkeypatch.setattr(C, "evaluate", lambda *a, **k: pytest.fail("must refuse before loading"))
    with pytest.raises(SystemExit):
        C.main([str(fast), str(stock), "--out", str(out)])
    assert out.read_bytes() == saved


@pytest.mark.parametrize("n", [0, 3])
def test_root_empty_and_zero_track_populations(tmp_path, n):
    path = tmp_path / "empty.root"
    _root(path, _keys(n), np.zeros(n, dtype=int))
    result = C.evaluate(path, path)
    assert result["events"] == n
    assert result["tracks"]["A_total"] == 0
    assert result["tracks"]["relative_yield_se"] is None


def test_same_digis_enables_shared_hit_matching_and_residual_quantiles(tmp_path):
    stock, fast = tmp_path / "stock.root", tmp_path / "fast.root"
    _root(stock, _keys(3), [1, 0, 1], hits=True, chi2=1.0)
    _root(fast, _keys(3)[::-1], [1, 0, 1], hits=True, chi2=3.0)
    result = C.evaluate(fast, stock, same_digis=True)
    diagnostic = result["shared_hit_diagnostics"]
    assert diagnostic["same_digis_asserted_by_caller"]
    assert diagnostic["quantile_probabilities"] == [0.05, 0.16, 0.5, 0.84, 0.95]
    assert diagnostic["comparison"]["track_match"]["matched"] == 2
    assert diagnostic["comparison"]["matched_trk"]["chisq"]["B_minus_A"]["q"] == [2.0] * 5
    assert result["events"] == 3  # matched-track conditioning never changes yield denominator
    bootstrap = diagnostic["matched_parameter_bootstrap"]
    assert bootstrap["events"] == 3
    for row in bootstrap["variables"].values():
        assert row["missing_tracks"] == 2
        assert row["statistics"]["q50"]["status"] == "no_finite_tracks"


def test_same_digis_rejects_missing_hit_branches(tmp_path):
    path = tmp_path / "counts.root"
    _root(path, _keys(2), [0, 1])
    with pytest.raises(ValueError, match="require trkhits"):
        C.evaluate(path, path, same_digis=True)


def test_cli_population_mismatch_does_not_create_output(tmp_path):
    stock, fast, out = tmp_path / "stock.root", tmp_path / "fast.root", tmp_path / "bad.json"
    _root(stock, _keys(2), [0, 1])
    _root(fast, _keys(1), [0])
    with pytest.raises(SystemExit):
        C.main([str(fast), str(stock), "--out", str(out)])
    assert not out.exists()


@pytest.mark.parametrize("n", [0, 3])
def test_same_digis_with_no_tracks_keeps_population(tmp_path, n):
    path = tmp_path / "empty_hits.root"
    _root(path, _keys(n), np.zeros(n, dtype=int), hits=True)
    result = C.evaluate(path, path, same_digis=True)
    assert result["events"] == n
    assert result["tracks"]["A_total"] == 0
    assert result["shared_hit_diagnostics"]["comparison"]["events"]["common"] == n
    boot = result["shared_hit_diagnostics"]["matched_parameter_bootstrap"]
    assert boot["events"] == n
    for row in boot["variables"].values():
        assert row["matched_tracks"] == 0
        assert row["bootstrap_empty_replicates"] == boot["bootstrap_replicates"]
        assert row["statistics"]["q50"]["estimate"] is None


def test_identity_is_exact_above_float_integer_precision():
    keys = np.array([[1, 0, 2**53], [1, 0, 2**53 + 1]], dtype=np.int64)
    result = C.summarize_counts(keys, [0, 1], keys[::-1], [2, 0])
    assert result["tracks"]["difference_B_minus_A"] == 0.5
    assert result["tracks"]["relative_yield_se"] == 0


def test_cli_exclusive_creation_protects_against_race(tmp_path, monkeypatch):
    out = tmp_path / "race.json"

    def evaluate(*args, **kwargs):
        out.write_text("another writer")
        return C.summarize_counts([], [], [], [])

    monkeypatch.setattr(C, "evaluate", evaluate)
    with pytest.raises(SystemExit):
        C.main(["unused-fast.root", "unused-stock.root", "--out", str(out)])
    assert out.read_text() == "another writer"


def test_cli_rejects_dangling_output_symlink(tmp_path):
    out = tmp_path / "result.json"
    target = tmp_path / "missing.json"
    out.symlink_to(target)
    with pytest.raises(SystemExit):
        C.main(["unused-fast.root", "unused-stock.root", "--out", str(out)])
    assert out.is_symlink()
    assert not target.exists()


def test_same_digis_bootstrap_full_population_and_cli_controls(tmp_path):
    from trkfastsim.reco.autonomous_stats import event_cluster_bootstrap

    stock, fast, out = tmp_path / "s.root", tmp_path / "f.root", tmp_path / "result.json"
    keys = np.array([[1, 0, 7], [1, 1, 7], [2, 0, 7], [2, 1, 7]])
    # Four matches in three events; all tracks in event 0 must be clustered.
    counts = np.array([2, 1, 0, 1])
    x = np.array([-8., 2., 13., 5.])
    phi_a, phi_b = np.array([3., 0., 0., 0.]), np.array([-3., 1., -1., 2.])
    _root(stock, keys, counts, hits=True, parameters={"d0": np.zeros(4), "phi0": phi_a})
    event_order, track_order = np.array([3, 1, 0, 2]), np.array([3, 2, 0, 1])
    _root(fast, keys[event_order], counts[event_order], hits=True,
          parameters={"d0": x[track_order], "phi0": phi_b[track_order]})
    result = C.main([str(fast), str(stock), "--same-digis", "--out", str(out),
                     "--bootstrap-replicates", "71", "--bootstrap-seed", "42"])
    diagnostic = result["shared_hit_diagnostics"]
    boot = diagnostic["matched_parameter_bootstrap"]
    assert diagnostic["comparison"]["track_match"]["matched"] == 4
    assert boot["events"] == 4
    assert boot["events_with_matched_tracks"] == 3
    assert boot["bootstrap_replicates"] == 71
    assert boot["bootstrap_seed"] == 42
    expected = event_cluster_bootstrap(keys, [0, 0, 1, 3], {"d0": x}, replicates=71, seed=42)
    assert boot["variables"]["d0"] == expected["variables"]["d0"]
    delta_phi = (phi_b - phi_a + np.pi) % (2 * np.pi) - np.pi
    assert boot["variables"]["phi0"]["statistics"]["q50"]["estimate"] == pytest.approx(np.median(delta_phi))
    assert boot["variables"]["mom"]["missing_tracks"] == 4
    assert json.loads(out.read_text()) == result
    assert result == C.evaluate(fast, stock, same_digis=True, bootstrap_replicates=71, bootstrap_seed=42)
    counts_only = C.evaluate(fast, stock, bootstrap_replicates=71, bootstrap_seed=42)
    assert "matched_parameter_bootstrap" not in counts_only["shared_hit_diagnostics"]


def test_parameter_missing_crossing_and_nonfinite_are_counted(tmp_path):
    path = tmp_path / "pars.root"
    _root(path, _keys(4), [1, 1, 1, 1], hits=True,
          parameters={"d0": [1., np.nan, np.inf, 20.]}, sids=[4, 4, 4, 2])
    result = C.evaluate(path, path, same_digis=True, bootstrap_replicates=19)
    boot = result["shared_hit_diagnostics"]["matched_parameter_bootstrap"]
    row = boot["variables"]["d0"]
    assert [row[k] for k in ("finite_tracks", "missing_tracks", "nonfinite_tracks")] == [1, 1, 2]
    assert row["statistics"]["q50"]["bootstrap_se"] is None
    assert row["statistics"]["q50"]["status"] == "insufficient_effective_events"
    assert boot["variables"]["z0"]["missing_tracks"] == 4
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("option,value", [("--bootstrap-replicates", "0"), ("--bootstrap-replicates", "1"),
                                          ("--bootstrap-replicates", "2.5"), ("--bootstrap-seed", "-1"),
                                          ("--bootstrap-seed", str(2**64))])
def test_cli_rejects_bad_bootstrap_controls_before_io(tmp_path, monkeypatch, option, value):
    monkeypatch.setattr(C, "evaluate", lambda *a, **k: pytest.fail("invalid options must fail before IO"))
    out = tmp_path / "bad.json"
    with pytest.raises(SystemExit):
        C.main(["unused-fast", "unused-stock", "--out", str(out), option, value])
    assert not out.exists()


def test_earliest_outer_crossing_matches_legacy_and_preserves_missing_options():
    import awkward as ak

    data = {"n": 3, "trk": ak.Array({"trk.goodfit": [[0, 0], [], [0]]}),
            "trksegs": ak.Array([[[{"sid": 4, "time": 30.}, {"sid": 4, "time": 10.},
                                     {"sid": 2, "time": 0.}], [{"sid": 4, "time": 4.}]], [],
                                   [[{"sid": 2, "time": 1.}]]]),
            "trksegpars_kl": ak.Array([[[{"d0": 30.}, {"d0": 10.}, {"d0": 0.}],
                                         [{"d0": None}]], [], [[{"d0": 1.}]]])}
    p = C._matched_parameters(data, np.arange(3), np.arange(3))
    assert p["d0"][0] == 10.
    assert p["d0"].mask.tolist() == [False, True, True]
    assert p["mom"].mask.all()
    # Missing an entire branch must not prevent explicit per-variable counts.
    assert C._matched_parameters({**data, "trksegs": None}, np.arange(3), np.arange(3))["d0"].mask.all()


@pytest.mark.parametrize("n", [0, 3])
def test_empty_parameter_branches_are_supported(tmp_path, n):
    path = tmp_path / "empty_pars.root"
    _root(path, _keys(n), np.zeros(n, dtype=int), hits=True, parameters={"d0": []})
    result = C.evaluate(path, path, same_digis=True, bootstrap_replicates=2)
    boot = result["shared_hit_diagnostics"]["matched_parameter_bootstrap"]
    assert boot["events"] == n
    assert boot["variables"]["d0"]["statistics"]["q50"]["status"] == "no_finite_tracks"


def test_writer_sentinels_masked_before_subtraction_and_phi_wrapping(tmp_path):
    stock, fast = tmp_path / "sentinel_s.root", tmp_path / "sentinel_f.root"
    # Both sentinel, A-only sentinel, B-only sentinel, valid, present nonfinite.
    a = [-9999., -9999., 1., 3., np.inf]
    b = [-9999., 2., -9999., -3., 0.]
    _root(stock, _keys(5), [1] * 5, hits=True,
          parameters={"d0": a, "phi0": a, "d0err": [1.] * 5})
    _root(fast, _keys(5), [1] * 5, hits=True, parameters={"d0": b, "phi0": b})
    result = C.evaluate(fast, stock, same_digis=True, bootstrap_replicates=31)
    diagnostic = result["shared_hit_diagnostics"]
    for name, delta in (("d0", -6.), ("phi0", 2 * np.pi - 6.)):
        row = diagnostic["matched_parameter_bootstrap"]["variables"][name]
        assert [row[k] for k in ("finite_tracks", "missing_tracks", "nonfinite_tracks")] == [1, 3, 1]
        assert row["statistics"]["q50"]["estimate"] == pytest.approx(delta)
        descriptive = diagnostic["comparison"]["matched_pars_at_TT_Outer_entry"][name]
        assert descriptive["A"]["n"] == 2
        assert descriptive["B"]["n"] == 3
        assert descriptive["B_minus_A"]["n"] == 1
        assert descriptive["B_minus_A"]["q"] == pytest.approx([delta] * 5)
    assert diagnostic["comparison"]["matched_pars_at_TT_Outer_entry"]["d0"]["pull_by_A_err"]["n"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("missing", ["segments", "parameters", "both"])
def test_missing_per_track_lists_preserve_flat_track_ownership(missing):
    import awkward as ak

    seg = [{"sid": 4, "time": 1.}]
    par = [{"d0": 1.}]
    data = {"n": 2, "trk": ak.Array({"trk.goodfit": [[0, 0], [0]]}),
            "trksegs": ak.Array([[None if missing != "parameters" else seg, seg], [seg]]),
            "trksegpars_kl": ak.Array([[None if missing != "segments" else par, [{"d0": 11.}]],
                                      [[{"d0": 22.}]]])}
    values = C._matched_parameters(data, np.array([1, 0]), np.arange(3))["d0"]
    assert values.mask.tolist() == [False, True, False]
    assert values.compressed().tolist() == [22., 11.]


@pytest.mark.parametrize("segments,pars", [
    ([[[{"sid": 4, "time": 1.}]]], [[[]]]),
    ([[[]]], [[[ {"d0": 1.} ]]]),
    ([[None]], [[[], []]]),
])
def test_known_inconsistent_parameter_counts_still_fail(segments, pars):
    import awkward as ak

    data = {"n": 1, "trk": ak.Array({"trk.goodfit": [[0]]}),
            "trksegs": ak.Array(segments), "trksegpars_kl": ak.Array(pars)}
    with pytest.raises(ValueError, match="parallel"):
        C._matched_parameters(data, np.array([0]), np.array([0]))


@pytest.mark.parametrize("missing", ["trksegs", "trksegpars_kl", "both"])
def test_evaluate_option_lists_normalizes_legacy_path(tmp_path, monkeypatch, missing):
    import awkward as ak
    from trkfastsim.reco import ntcompare

    path = tmp_path / "optional.root"
    _root(path, _keys(2), [1, 1], hits=True, parameters={"d0": [2., 7.]})
    data = ntcompare.load(str(path))
    for branch in ("trksegs", "trksegpars_kl"):
        if missing in (branch, "both"):
            data[branch] = ak.mask(data[branch], ak.Array([[False], [True]]))
    monkeypatch.setattr(ntcompare, "load", lambda _: data)
    result = C.evaluate(path, path, same_digis=True, bootstrap_replicates=17)
    diagnostic = result["shared_hit_diagnostics"]
    row = diagnostic["matched_parameter_bootstrap"]["variables"]["d0"]
    assert row["finite_tracks"] == row["missing_tracks"] == 1
    assert row["effective_events"] == 1
    descriptive = diagnostic["comparison"]["matched_pars_at_TT_Outer_entry"]["d0"]
    assert descriptive["A"]["q"] == [7.] * 5
    assert descriptive["B_minus_A"]["q"] == [0.] * 5
    json.dumps(result, allow_nan=False)


def test_evaluate_extreme_parameters_descriptive_and_bootstrap_json_safe(tmp_path):
    stock, fast = tmp_path / "extreme_s.root", tmp_path / "extreme_f.root"
    limit = np.finfo(float).max
    _root(stock, _keys(2), [1, 1], hits=True, parameters={"d0": [0., 0.]})
    _root(fast, _keys(2), [1, 1], hits=True, parameters={"d0": [-limit, limit]})
    result = C.evaluate(fast, stock, same_digis=True, bootstrap_replicates=200)
    diagnostic = result["shared_hit_diagnostics"]
    descriptive = diagnostic["comparison"]["matched_pars_at_TT_Outer_entry"]["d0"]["B_minus_A"]
    assert np.asarray(descriptive["q"]) / limit == pytest.approx([-.9, -.68, 0., .68, .9])
    assert descriptive["mean"] == 0.
    row = diagnostic["matched_parameter_bootstrap"]["variables"]["d0"]
    assert row["bootstrap_empty_replicates"] == 0
    assert row["statistics"]["q50"]["status"] == "ok"
    json.dumps(result, allow_nan=False)


def test_all_per_track_lists_absent_are_missing_not_malformed():
    import awkward as ak

    data = {"n": 2, "trk": ak.Array({"trk.goodfit": [[0], [0]]}),
            "trksegs": ak.Array([[None], [None]]),
            "trksegpars_kl": ak.Array([[None], [None]])}
    values = C._matched_parameters(data, np.arange(2), np.arange(2))
    assert all(v.mask.all() for v in values.values())


def test_evaluate_rejects_known_misalignment_before_legacy_segment_diagnostics(tmp_path, monkeypatch):
    from trkfastsim.reco import ntcompare

    path = tmp_path / "misaligned.root"
    _root(path, _keys(2), [1, 1], hits=True, parameters={"d0": [2., 7.]})
    data = ntcompare.load(str(path))
    data["trksegpars_kl"] = data["trksegpars_kl"][:, :, :0]
    monkeypatch.setattr(ntcompare, "load", lambda _: data)
    monkeypatch.setattr(ntcompare, "compare", lambda *a, **k: pytest.fail("validate before legacy comparison"))
    with pytest.raises(ValueError, match="parallel"):
        C.evaluate(path, path, same_digis=True, bootstrap_replicates=2)


def test_error_sentinel_does_not_contribute_descriptive_pulls(tmp_path):
    stock, fast = tmp_path / "error_s.root", tmp_path / "error_f.root"
    _root(stock, _keys(2), [1, 1], hits=True,
          parameters={"d0": [0., 0.], "d0err": [-9999., 2.]})
    _root(fast, _keys(2), [1, 1], hits=True, parameters={"d0": [10., 4.]})
    result = C.evaluate(fast, stock, same_digis=True, bootstrap_replicates=17)
    descriptive = result["shared_hit_diagnostics"]["comparison"]["matched_pars_at_TT_Outer_entry"]["d0"]
    assert descriptive["B_minus_A"]["n"] == 2
    assert descriptive["pull_by_A_err"]["n"] == 1
    assert descriptive["pull_by_A_err"]["q"] == [2.] * 5
    json.dumps(result, allow_nan=False)
