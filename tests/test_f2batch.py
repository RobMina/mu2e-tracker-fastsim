"""Fixed-shape caller closure and packing tests for F2."""
import builtins

import numpy as np
import pytest

from trkfastsim.reco import f2batch, linefit


PARAMS = ("a0", "b0", "a1", "b1", "t0")


def _empty():
    return {**{k: np.empty(0) for k in PARAMS},
            **{k: np.empty(0, np.int64) for k in ("n", "ev", "member")}}


def _synthetic(ntrk, nhit):
    from test_reco_jax import _toy_tracks

    _, x0, H, drift = _toy_tracks(ntrk, nhit, np.random.default_rng(721))
    size = ntrk * nhit
    us = np.arange(size)
    sid = ((us // 576) << 10) | (((us // 96) % 6) << 7) | (us % 96)
    ev = 2 * (np.arange(ntrk) // 2)
    st = {k: H["mid"][..., i].ravel() for i, k in enumerate(("midx", "midy", "midz"))}
    st.update({k: H["wdir"][..., i].ravel() for i, k in enumerate(("dirx", "diry", "dirz"))})
    st["hlen"] = H["hlen"].ravel()
    sh = {"ev": np.repeat(ev, nhit), "sid": sid, "eend": (H["esign"].ravel() < 0).astype(int),
          "et0": H["tmeas"].ravel(), "et1": H["tmeas"].ravel(), "halfpv": H["vsig"].ravel() / 2,
          "dtime": H["totdt"].ravel(), "timevar": H["tvar"].ravel(), "uvar": H["uvar"].ravel(),
          "wdist": H["wdist"].ravel(), "edep": np.full(size, 0.001)}
    sd = {k: x0[:, i] for i, k in enumerate(PARAMS)}
    sd.update(n=np.full(ntrk, nhit, np.int64), ev=ev, member=us)
    return sh, sd, {"straw": st}, drift


def _raw(sh, sd, cond, drift):
    from trkfastsim.reco import linefit_jax

    trk = np.repeat(np.arange(len(sd["n"])), sd["n"])
    H = linefit.build_hits(trk, sd["member"], np.zeros(len(trk)), np.ones(len(trk), bool), sh, cond, len(sd["n"]))
    x0 = linefit.seed_params(*(sd[k] for k in PARAMS))
    return linefit_jax.fit_own_states(x0, H, drift, niter=2)


def _assert_closure(monkeypatch, ntrk, nhit):
    pytest.importorskip("jax")
    from trkfastsim.reco import linefit_jax

    sh, sd, cond, drift = _synthetic(ntrk, nhit)
    ref = _raw(sh, sd, cond, drift)
    fit = linefit_jax.fit_own_states
    calls = []

    def capture(x0, H, drift, **kw):
        assert x0.shape == (64, 5)
        capacity = max(64, 1 << (nhit - 1).bit_length())
        assert kw == {"niter": 2, "n_pad": capacity, "chunk": 64}
        assert all(v.shape[:2] == (64, capacity) for v in H.values())
        assert np.all(H["lr"] == 0)
        assert np.all(H["active"][:, :nhit])
        assert not np.any(H["active"][:, nhit:])
        nreal = min(64, ntrk - len(calls) * 64)
        if nreal < 64:
            np.testing.assert_array_equal(x0[nreal:], np.repeat(x0[nreal - 1:nreal], 64 - nreal, axis=0))
            for v in H.values():
                np.testing.assert_array_equal(v[nreal:], np.repeat(v[nreal - 1:nreal], 64 - nreal, axis=0))
        result = fit(x0, H, drift, **kw)
        calls.append({k: v[:nreal] for k, v in result.items()})
        return result

    monkeypatch.setattr(linefit_jax, "fit_own_states", capture)
    nevents = ntrk + 3
    out = f2batch.fit_seeds(sh, sd, cond, drift, nevents, niter=2)
    padded = {k: np.concatenate([f[k] for f in calls]) for k in ref}
    for k in ("active", "lr"):
        np.testing.assert_array_equal(padded[k][:, :nhit], ref[k])
    for k in ("ndof", "failed"):
        np.testing.assert_array_equal(padded[k], ref[k])
    for k in ("x", "cov", "chi2"):
        np.testing.assert_allclose(padded[k], ref[k], rtol=1e-10, atol=1e-9)
    nactive = ref["active"].sum(1)
    good = (nactive >= 5) & np.isfinite(ref["chi2"]) & (ref["ndof"] >= 1)
    assert good.any()
    assert out["seeds"] == ntrk
    np.testing.assert_array_equal(out["fits_per_event"], np.bincount(sd["ev"][good], minlength=nevents))
    np.testing.assert_array_equal(out["nactive"], nactive[good])
    np.testing.assert_allclose(out["chi2ndf"], ref["chi2"][good] / np.maximum(ref["ndof"][good], 1),
                               rtol=1e-10, atol=1e-9)
    nblocks = (ntrk + 63) // 64
    capacity = max(64, 1 << (nhit - 1).bit_length())
    assert out["profile"]["shape_counts"] == {f"64x{capacity}": nblocks}
    assert out["profile"]["tracks_padded"] == 64 * nblocks - ntrk
    assert out["profile"]["fit_s"] > 0


@pytest.mark.parametrize("ntrk", [1, 63, 64, 65])
def test_real_track_boundaries(monkeypatch, ntrk):
    _assert_closure(monkeypatch, ntrk, 20)


@pytest.mark.parametrize("nhit", [63, 64, 65])
def test_real_hit_boundaries(monkeypatch, nhit):
    _assert_closure(monkeypatch, 2, nhit)


@pytest.mark.parametrize("nevents", [0, 7])
def test_empty_never_imports_fitter(monkeypatch, nevents):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert "jax" not in name
        assert "linefit_jax" not in kwargs.get("fromlist", ())
        if len(args) >= 3 and args[2]:
            assert "linefit_jax" not in args[2]
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    out = f2batch.fit_seeds({"ev": np.empty(0, np.int64)}, _empty(), None, None, nevents)
    assert out["seeds"] == 0
    for k in ("nactive", "fits_per_event"):
        assert out[k].dtype == np.int64
    assert out["chi2ndf"].dtype == np.float64
    assert out["nactive"].shape == out["chi2ndf"].shape == (0,)
    np.testing.assert_array_equal(out["fits_per_event"], np.zeros(nevents, np.int64))
    assert out["profile"]["blocks"] == 0
    assert out["profile"]["shape_counts"] == {}
    assert out["profile"]["fit_s"] == 0


def test_bucket_mapping_predicate_and_order(monkeypatch):
    pytest.importorskip("jax")
    from trkfastsim.reco import linefit_jax

    ns = np.array([65, 63, 64, 5, 65, 64, 0, 63])
    sd = {k: np.arange(len(ns), dtype=float) + i * 10 for i, k in enumerate(PARAMS)}
    sd.update(n=ns, ev=np.array([4, 0, 4, 2, 0, 2, 4, 4]), member=np.arange(ns.sum())[::-1])
    sh = {"ev": np.zeros(ns.sum(), np.int64)}
    starts = np.cumsum(ns) - ns
    expected_groups = [np.array([1, 2, 3, 5, 6, 7]), np.array([0, 4])]
    built = []
    fitted = []
    active_counts = np.array([6, 5, 6, 4, 7, 7, 0, 8])
    chi2 = np.array([12., 10., np.nan, 8., 21., 14., 0., 32.])
    ndof = np.array([2, 1, 1, 1, 3, 0, -5, 4])

    def build(trk, member, lr, active, sh_arg, cond, ntrk, nmax=None):
        ids = expected_groups[len(built)]
        assert sh_arg is sh
        assert ntrk == len(ids)
        assert np.all(active)
        assert np.all(lr == 0)
        np.testing.assert_array_equal(trk, np.repeat(np.arange(ntrk), ns[ids]))
        expected = np.concatenate([sd["member"][starts[i]:starts[i] + ns[i]] for i in ids])
        np.testing.assert_array_equal(member, expected)
        capacity = 64 if len(built) == 0 else 128
        assert nmax == capacity
        H = {"active": np.arange(capacity)[None, :] < ns[ids, None],
             "lr": np.zeros((ntrk, capacity)), "sh": np.full((ntrk, capacity), -1),
             "extra_vector": np.broadcast_to(ids[:, None, None], (ntrk, capacity, 3)).copy()}
        positions = np.arange(len(trk)) - np.repeat(np.cumsum(ns[ids]) - ns[ids], ns[ids])
        H["sh"][trk, positions] = member
        built.append(H)
        return H

    def fit(x0, H, drift, **kw):
        ids = expected_groups[len(fitted)]
        rows = np.minimum(np.arange(64), len(ids) - 1)
        np.testing.assert_array_equal(x0, linefit.seed_params(*(sd[k][ids] for k in PARAMS))[rows])
        assert kw == {"niter": 4, "n_pad": H["active"].shape[1], "chunk": 64}
        for k, v in built[-1].items():
            np.testing.assert_array_equal(H[k], v[rows])
        active = np.ones_like(H["active"])
        active[:len(ids)] = np.arange(active.shape[1])[None, :] < active_counts[ids, None]
        result = {"active": active, "chi2": np.full(64, 1e12), "ndof": np.ones(64, int),
                  "failed": np.ones(64, bool), "unused": np.ones((64, 5, 5))}
        result["chi2"][:len(ids)] = chi2[ids]
        result["ndof"][:len(ids)] = ndof[ids]
        fitted.append(ids)
        return result

    monkeypatch.setattr(linefit, "build_hits", build)
    monkeypatch.setattr(linefit_jax, "fit_own_states", fit)
    out = f2batch.fit_seeds(sh, sd, None, None, 6)
    assert out["seeds"] == 8
    np.testing.assert_array_equal(out["nactive"], [6, 5, 7, 8])
    np.testing.assert_array_equal(out["chi2ndf"], [6., 10., 7., 8.])
    np.testing.assert_array_equal(out["fits_per_event"], [2, 0, 0, 0, 2, 0])
    assert out["profile"]["shape_counts"] == {"64x64": 1, "64x128": 1}
    assert out["profile"]["tracks_padded"] == 120


@pytest.mark.parametrize("key,value", [
    ("n", np.array([1])), ("n", np.array([-1])), ("n", np.array([2.])),
    ("member", np.array([0])), ("member", np.array([0, 3])), ("member", np.array([-1, 0])),
    ("member", np.array([[0, 1]])), ("ev", np.array([-1])), ("ev", np.array([2])),
    ("ev", np.array([0, 1])), ("ev", np.array([0.])), ("t0", np.empty(0)),
])
def test_malformed_seed_rejected_before_jax(monkeypatch, key, value):
    sd = {k: np.zeros(1) for k in PARAMS}
    sd.update(n=np.array([2]), ev=np.array([0]), member=np.array([0, 1]))
    sd[key] = value
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert "jax" not in name
        if len(args) >= 3 and args[2]:
            assert "linefit_jax" not in args[2]
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError):
        f2batch.fit_seeds({"ev": np.zeros(2, np.int64)}, sd, None, None, 2)


@pytest.mark.parametrize("kwargs", [
    {"track_batch": 0}, {"track_batch": 1.5}, {"min_hits": 0}, {"min_hits": -1},
    {"nevents": -1}, {"nevents": 0.5}, {"niter": -1},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        f2batch.fit_seeds({"ev": np.empty(0, np.int64)}, _empty(), None, None, **{"nevents": 0, **kwargs})
