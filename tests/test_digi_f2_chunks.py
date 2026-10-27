"""Event-batched F2 yields: exact pooled statistics, event/ADC mappings and empty batches."""
from types import SimpleNamespace

import numpy as np
import pytest

from trkfastsim.analyses import digicmp


def _digis(events=(70, 10, 90, 20, 80, 30, 60), ev=(0, 0, 1, 4, 6, 6, 6)):
    events, ev = np.asarray(events, np.int64), np.asarray(ev, np.int64)
    n = len(ev)
    d = {k: np.arange(n, dtype=np.int64) for k in ("sid", "tdc0", "tdc1", "tot0", "tot1")}
    d.update(evkey=np.column_stack((np.full(len(events), 1400), np.full(len(events), 11), events)),
             ev=ev, pbt=events.astype(float), nadc=np.arange(n, dtype=np.int64) % 3 + 1)
    d["adc"] = np.arange(d["nadc"].sum(), dtype=np.int64)
    return d


ACTIVE = np.array([5, 6, 7, 4, 50, 60, 70])
CHI2NDF = np.array([0., 1., 2., 999., 100., 500., 1000.])


def _fake_batch(d, cond, drift_ref, settings="mc", niter=4):
    good = ACTIVE[d["sid"]] >= 5
    return {"seeds": len(d["sid"]),
            "fits_per_event": np.bincount(d["ev"][good], minlength=len(d["evkey"])),
            "nactive": ACTIVE[d["sid"]][good], "chi2ndf": CHI2NDF[d["sid"]][good]}


def _assert_yield_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_restrict_event_and_flat_adc_mapping():
    d = _digis()
    out = digicmp._restrict(d, np.array([6, 2, 0]))
    np.testing.assert_array_equal(out["evkey"], d["evkey"][[6, 2, 0]])
    np.testing.assert_array_equal(out["pbt"], [60., 90., 70.])
    np.testing.assert_array_equal(out["ev"], [0, 0, 0, 2, 2])
    rows = np.array([4, 5, 6, 0, 1])
    for k in ("sid", "tdc0", "tdc1", "tot0", "tot1", "nadc"):
        np.testing.assert_array_equal(out[k], d[k][rows])
    adc_rows = np.split(d["adc"], np.cumsum(d["nadc"])[:-1])
    np.testing.assert_array_equal(out["adc"], np.concatenate([adc_rows[i] for i in rows]))
    empty = digicmp._restrict(d, np.array([], np.int64))
    assert all(len(v) == 0 for v in empty.values())


@pytest.mark.parametrize("chunk", [0, 1, 2, 3, 7, 20])
def test_exact_chunked_yield_and_pooled_quantiles(chunk, monkeypatch):
    d, cond, calls = _digis(), object(), []

    def batch(sub, c, drift_ref, settings, niter):
        assert c is cond and drift_ref == "drift.root" and settings == "data" and niter == 9
        calls.append(sub)
        return _fake_batch(sub, c, drift_ref)

    monkeypatch.setattr(digicmp, "_f2_batch", batch)
    got = digicmp.f2_yield(d, cond, "drift.root", settings="data", niter=9, chunk=chunk)
    good = ACTIVE >= 5
    expected = {"seeds": 7, "good_fits": 6, "events_with_fit": 3,
                "fits_per_event": np.array([2, 1, 0, 0, 0, 0, 3]),
                "nactive_q": digicmp._q(ACTIVE[good]), "chi2ndf_q": digicmp._q(CHI2NDF[good])}
    _assert_yield_equal(got, expected)
    if not chunk:
        assert len(calls) == 1 and calls[0] is d
    else:
        assert len(calls) == (len(d["evkey"]) + chunk - 1) // chunk
        for start, sub in zip(range(0, len(d["evkey"]), chunk), calls):
            stop = min(start + chunk, len(d["evkey"]))
            mask = (d["ev"] >= start) & (d["ev"] < stop)
            np.testing.assert_array_equal(sub["evkey"], d["evkey"][start:stop])
            np.testing.assert_array_equal(sub["pbt"], d["pbt"][start:stop])
            np.testing.assert_array_equal(sub["ev"], d["ev"][mask] - start)
            np.testing.assert_array_equal(sub["adc"], d["adc"][np.repeat(mask, d["nadc"])])
        if chunk == 2:
            assert len(calls[1]["evkey"]) == 2 and len(calls[1]["ev"]) == 0
            # Unequal batches require pooled rather than averaged quantiles.
            assert not np.allclose(got["chi2ndf_q"],
                                   np.mean([digicmp._q(CHI2NDF[:3]), digicmp._q(CHI2NDF[4:])], axis=0))


@pytest.mark.parametrize("chunk", [0, 1, 2, 10])
@pytest.mark.parametrize("events", [(), (10, 20, 30)])
def test_empty_events_and_all_zero_hit_chunks(events, chunk, monkeypatch):
    monkeypatch.setattr(digicmp.hitreco, "straw_hits", lambda *a: pytest.fail("empty batch must skip reco"))
    got = digicmp.f2_yield(_digis(events, ()), None, "unused", chunk=chunk)
    _assert_yield_equal(got, {"seeds": 0, "good_fits": 0, "events_with_fit": 0,
                             "fits_per_event": np.zeros(len(events), np.int64),
                             "nactive_q": [], "chi2ndf_q": []})


def _mock_reco(monkeypatch, seeds, fits=None):
    from trkfastsim import reco
    from trkfastsim.reco import cli as reco_cli, linefind
    monkeypatch.setattr(digicmp.hitreco, "straw_hits", lambda *a: {"ev": np.array([0])})
    monkeypatch.setattr(linefind, "time_clusters", lambda sh: {"n": np.array([6])})
    monkeypatch.setattr(reco, "linefind_jax", SimpleNamespace(line_seeds=lambda *a: seeds), raising=False)
    calls = []

    def fit(*a, **kw):
        calls.append(kw)
        assert fits is not None, "zero-seed batch must not execute fitter"
        return fits

    monkeypatch.setattr(reco, "linefit_jax", SimpleNamespace(fit_own_states=fit), raising=False)
    monkeypatch.setattr(reco, "linefit", SimpleNamespace(build_hits=lambda *a: None, seed_params=lambda *a: None), raising=False)
    monkeypatch.setattr(reco_cli, "_drift_table", lambda path: calls.append(path))
    return calls


def test_genuine_zero_seeds_does_not_execute_fitter(monkeypatch):
    calls = _mock_reco(monkeypatch, {"n": np.empty(0, np.int64)})
    got = digicmp.f2_yield(_digis((10, 20), (0,)), None, "unused", chunk=1)
    assert calls == []
    _assert_yield_equal(got, {"seeds": 0, "good_fits": 0, "events_with_fit": 0,
                             "fits_per_event": np.array([0, 0]), "nactive_q": [], "chi2ndf_q": []})


def test_no_time_clusters_skips_seeding_and_fitter(monkeypatch):
    from trkfastsim import reco
    from trkfastsim.reco import linefind
    calls = _mock_reco(monkeypatch, None)
    monkeypatch.setattr(linefind, "time_clusters", lambda sh: {"n": np.empty(0, np.int64)})
    monkeypatch.setattr(reco, "linefind_jax", SimpleNamespace(line_seeds=lambda *a: pytest.fail("no clusters")))
    got = digicmp.f2_yield(_digis((10,), (0,)), None, "unused")
    assert got["seeds"] == got["good_fits"] == 0 and calls == []


def test_single_batch_keeps_original_good_fit_cuts(monkeypatch):
    seeds = {k: np.zeros(6) for k in ("a0", "b0", "a1", "b1", "t0")}
    seeds.update(n=np.full(6, 6), ev=np.array([0, 1, 1, 1, 1, 0]), member=np.arange(36))
    nactive = np.array([5, 4, 6, 7, 8, 6])
    fits = {"active": np.arange(8)[None, :] < nactive[:, None],
            "chi2": np.array([10., 1., np.inf, np.nan, 4., 9.]), "ndof": np.array([2, 1, 1, 1, 0, 3])}
    calls = _mock_reco(monkeypatch, seeds, fits)
    got = digicmp.f2_yield(_digis((10, 20, 30), (0,)), None, "drift.root", niter=7)
    _assert_yield_equal(got, {"seeds": 6, "good_fits": 2, "events_with_fit": 1,
                             "fits_per_event": np.array([2, 0, 0]),
                             "nactive_q": digicmp._q([5, 6]), "chi2ndf_q": digicmp._q([5., 3.])})
    assert calls == ["drift.root", {"niter": 7}]


def test_chunk_cli_and_early_validation(tmp_path, monkeypatch):
    from trkfastsim import cli
    parser = cli.build_parser()
    args = ["digicmp", "fast.root", "--ref", "ref.root"]
    assert parser.parse_args(args).f2_chunk == 0
    assert parser.parse_args(args + ["--f2-chunk", "0"]).f2_chunk == 0
    assert parser.parse_args(args + ["--f2-chunk", "17"]).f2_chunk == 17
    for value in ("-1", "1.5"):
        with pytest.raises(SystemExit):
            parser.parse_args(args + ["--f2-chunk", value])
    with pytest.raises(ValueError, match="chunk"):
        digicmp.run([], tmp_path / "not-created", f2_chunk=-1)
    assert not (tmp_path / "not-created").exists()
    with pytest.raises(ValueError, match="chunk"):
        digicmp.f2_yield(None, None, None, chunk=-1)
    seen = []
    monkeypatch.setitem(cli.ANALYSES, "digicmp", (lambda *a, **kw: seen.append(kw), "test"))
    cli.main(args + ["--f2", "--f2-chunk", "17"])
    assert seen[0]["f2_chunk"] == 17 and seen[0]["f2"]


@pytest.mark.parametrize("chunk", [0, 2])
@pytest.mark.parametrize("source", [None, "separate.root"])
def test_run_chunks_every_arm_with_common_event_mapping(chunk, source, tmp_path, monkeypatch):
    arms = {"fast": _digis(), "ref": _digis(), "floor": _digis(ev=())}
    # Different event ordering in ref: key matching must happen before chunking.
    arms["ref"]["evkey"] = arms["ref"]["evkey"][::-1].copy()
    arms["ref"]["pbt"] = arms["ref"]["pbt"][::-1].copy()
    monkeypatch.setattr(digicmp.hitreco, "read_digis", lambda paths: arms[paths[0]])
    monkeypatch.setattr(digicmp.hitreco, "load_conditions", lambda path: {})
    monkeypatch.setattr(digicmp, "HIST", {})
    monkeypatch.setattr(digicmp, "paired_single", lambda *a: {})
    monkeypatch.setattr(digicmp, "observables", lambda d, c: {k: np.ones(len(d["evkey"]), int) for k in
                        ("sid", "ndigi", "nsh", "nph", "tot0", "tot1", "pmp", "dt", "wdist")})
    seen = []

    def batch(d, cond, drift, settings, niter):
        seen.append((d, drift))
        return _fake_batch(d, cond, drift)

    monkeypatch.setattr(digicmp, "_f2_batch", batch)
    got = digicmp.run(["fast"], tmp_path, ref=["ref"], floor=["floor"], hit_cond="unused",
                      f2=True, f2_chunk=chunk, drift_source=source)
    assert got["events"] == 7 and set(got["f2"]) == {"fast", "ref", "floor"}
    per_arm = 4 if chunk else 1
    assert len(seen) == 3 * per_arm
    for arm_index, name in enumerate(arms):
        batches = seen[arm_index * per_arm:(arm_index + 1) * per_arm]
        np.testing.assert_array_equal(np.concatenate([d["evkey"][:, 2] for d, _ in batches]),
                                      [10, 20, 30, 60, 70, 80, 90])
        assert all(drift == (source or "ref") for _, drift in batches)
        ordered = digicmp._restrict(arms[name], np.argsort(arms[name]["evkey"][:, 2]))
        expected = digicmp.f2_yield(ordered, {}, source or "ref")
        for k in ("seeds", "good_fits", "events_with_fit", "nactive_q", "chi2ndf_q"):
            np.testing.assert_array_equal(got["f2"][name][k], expected[k])
    assert got["f2"]["floor"]["good_fits"] == 0
    assert got["f2"]["floor"]["paired_rel_diff"] == -1.


def test_fixed_shape_yield_loads_drift_once_and_reports_chunks(monkeypatch):
    from trkfastsim.reco import cli as reco_cli
    d, seen, reports = _digis(), [], []
    drift = {"dt": np.array([0., 1.])}
    monkeypatch.setattr(reco_cli, "_drift_table", lambda p: seen.append(p) or drift)
    def fixed(sub, cond, cal, settings, niter):
        assert cal is drift
        return {**_fake_batch(sub, cond, cal), "profile": {"shape_counts": {"64x64": 1}}}
    monkeypatch.setattr(digicmp, "_f2_fixed_batch", fixed)
    got = digicmp.f2_yield(d, {}, "drift.root", chunk=2, fixed_shapes=True, progress=reports.append)
    assert seen == ["drift.root"] and got["good_fits"] == 6
    assert [(r["start"], r["stop"]) for r in reports] == [(0, 2), (2, 4), (4, 6), (6, 7)]
    assert all(r["shape_counts"] == {"64x64": 1} for r in reports)


@pytest.mark.parametrize("bad", ["missing", "duplicate"])
def test_strict_population_gate_before_reco(tmp_path, monkeypatch, bad):
    arms = {"fast": _digis(), "ref": _digis()}
    if bad == "missing":
        arms["fast"]["evkey"] = arms["fast"]["evkey"][:-1]
    else:
        arms["fast"]["evkey"][1] = arms["fast"]["evkey"][0]
    monkeypatch.setattr(digicmp.hitreco, "load_conditions", lambda p: {})
    monkeypatch.setattr(digicmp.hitreco, "read_digis", lambda p: arms[p[0]])
    monkeypatch.setattr(digicmp, "observables", lambda *a: pytest.fail("population must fail first"))
    with pytest.raises(ValueError, match="population"):
        digicmp.run(["fast"], tmp_path, ref=["ref"], hit_cond="unused", require_same_events=True)


def test_fixed_shape_cli_flags():
    from trkfastsim.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["digicmp", "fast.root", "--ref", "ref.root", "--f2-fixed-shapes",
                              "--f2-progress", "--require-same-events"])
    assert args.f2_fixed_shapes and args.f2_progress and args.require_same_events


def test_full_identity_keys_do_not_alias_overlapping_bitfields():
    keys = np.array([[1400, 2, 1048577], [1400, 3, 1]], np.int64)
    assert len(np.unique(digicmp._keyid(keys))) == 2


@pytest.mark.parametrize("arm", ["fast", "ref", "floor"])
def test_strict_gate_rejects_collision_or_duplicate_in_every_arm(tmp_path, monkeypatch, arm):
    arms = {n: _digis(events=(1, 2), ev=()) for n in ("fast", "ref", "floor")}
    monkeypatch.setattr(digicmp.hitreco, "load_conditions", lambda p: {})
    monkeypatch.setattr(digicmp.hitreco, "read_digis", lambda p: arms[p[0]])
    for value in (np.array([[1400, 2, 1048577], [1400, 11, 2]]),
                  np.array([[1400, 11, 1], [1400, 11, 1]])):
        arms[arm]["evkey"] = value
        with pytest.raises(ValueError, match="population"):
            digicmp.run(["fast"], tmp_path, ref=["ref"], floor=["floor"], hit_cond="unused", require_same_events=True)


def test_restrict_round_trip_preserves_variable_adc_and_empty_events():
    d = _digis()
    perm = np.array([6, 2, 4, 0, 5, 1, 3])
    reordered = digicmp._restrict(d, perm)
    assert np.all(np.diff(reordered["ev"]) >= 0)
    restored = digicmp._restrict(reordered, np.argsort(perm))
    for key in d:
        np.testing.assert_array_equal(restored[key], d[key])


def test_batch_check_requires_exercised_reconstruction():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts/f2_batch_check.py"
    spec = importlib.util.spec_from_file_location("f2_batch_check_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for args in ((0, 100, 32), (30, 100, 32), (30, 30, 10), (100, 100, 0)):
        with pytest.raises(ValueError):
            module.require_population(*args)
    module.require_population(100, 100, 32)
    with pytest.raises(ValueError, match="coverage"):
        module.require_exercised({"seeds": 0, "good_fits": 0}, {"seeds": 0})
    module.require_exercised({"seeds": 30, "good_fits": 29}, {"seeds": 96})
