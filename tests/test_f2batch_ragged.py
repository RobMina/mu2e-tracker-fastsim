"""Real-JAX closure for ragged F2 batches and bounded line-finder chunks."""
import numpy as np
import pytest

from test_f2batch import PARAMS, _raw, _synthetic
from trkfastsim.reco import f2batch, linefit


# Keep optional-JAX installations able to collect the rest of the test suite.
pytest.importorskip("jax")


def test_real_ragged_reordered_seed_closure(monkeypatch):
    from trkfastsim.reco import linefit_jax

    # 67 seeds in capacity 64 force a full block plus a partial block; the
    # interspersed 65-hit seeds require a different capacity. One short seed
    # exercises rejection without introducing fabricated fitter results.
    counts = np.r_[np.tile([63, 64], 33), [65, 65, 65, 4]]
    nseed, width = len(counts), 65
    sh, sd, cond, drift = _synthetic(nseed, width)
    members = sd["member"].reshape(nseed, width)
    valid = np.arange(width)[None, :] < counts[:, None]
    sd.update(n=counts, member=members[valid])  # vectorized ragged truncation
    ref = _raw(sh, sd, cond, drift)

    rng = np.random.default_rng(1921)
    seed_order = rng.permutation(nseed)
    # Independently permute each seed's retained members; absent slots remain
    # at the end so flattening preserves the new ragged segment boundaries.
    hit_order = np.argsort(np.where(valid, rng.random(valid.shape), np.inf), axis=1)
    shuffled = np.take_along_axis(members, hit_order, axis=1)[seed_order]
    reordered = {k: sd[k][seed_order] for k in (*PARAMS, "n", "ev")}
    reordered["member"] = shuffled[np.arange(width)[None, :] < reordered["n"][:, None]]
    assert not np.array_equal(seed_order, np.arange(nseed))
    assert np.any(np.diff(reordered["ev"]) < 0)
    assert np.any(np.diff(reordered["member"]) < 0)

    capacities = np.where(reordered["n"] <= 64, 64, 128)
    blocks = []
    for capacity in (64, 128):
        ids = np.flatnonzero(capacities == capacity)
        blocks.extend((capacity, ids[i:i + 64]) for i in range(0, len(ids), 64))
    restored = {k: np.zeros_like(ref[k]) for k in ref}
    seen = []
    fit = linefit_jax.fit_own_states
    original_x0 = linefit.seed_params(*(sd[k] for k in PARAMS))

    def capture(x0, H, drift_arg, **kwargs):
        capacity, ids = blocks[len(seen)]
        nreal = len(ids)
        original_ids = seed_order[ids]
        rows = np.minimum(np.arange(64), nreal - 1)
        assert x0.shape == (64, 5)
        assert kwargs == {"niter": 2, "n_pad": capacity, "chunk": 64}
        assert all(v.shape[:2] == (64, capacity) for v in H.values())
        np.testing.assert_array_equal(x0, original_x0[original_ids][rows])
        np.testing.assert_array_equal(H["sh"][:, 0] // width, original_ids[rows])
        np.testing.assert_array_equal(H["active"].sum(1), counts[original_ids][rows])
        assert not np.any(H["lr"])
        for v in H.values():
            np.testing.assert_array_equal(v, v[:nreal][rows])

        # This wrapper only records the real fitter's output, unchanged.
        result = fit(x0, H, drift_arg, **kwargs)
        occupied = H["sh"][:nreal] >= 0
        rr, cc = np.nonzero(occupied)
        hit_ids = H["sh"][:nreal][occupied]
        np.testing.assert_array_equal(hit_ids // width, original_ids[rr])
        for k in ref:
            if k in ("active", "lr", "doca", "dT"):
                restored[k][original_ids[rr], hit_ids % width] = result[k][rr, cc]
            else:
                restored[k][original_ids] = result[k][:nreal]
        assert not np.any(result["active"][:nreal][~occupied])
        seen.append(original_ids)
        return result

    monkeypatch.setattr(linefit_jax, "fit_own_states", capture)
    nevents = nseed + 3  # retain holes and a trailing zero-fit event
    out = f2batch.fit_seeds(sh, reordered, cond, drift, nevents, niter=2)
    assert len(seen) == 3
    np.testing.assert_array_equal(np.sort(np.concatenate(seen)), np.arange(nseed))
    for k in ("active", "lr"):
        np.testing.assert_array_equal(restored[k][valid], ref[k][valid], err_msg=k)
    for k in ("ndof", "failed"):
        np.testing.assert_array_equal(restored[k], ref[k], err_msg=k)
    for k in ("x", "cov", "chi2"):
        np.testing.assert_allclose(restored[k], ref[k], rtol=1e-10, atol=1e-9, err_msg=k)
    for k in ("doca", "dT"):
        np.testing.assert_allclose(restored[k][valid], ref[k][valid], rtol=1e-10, atol=1e-9, err_msg=k)
    nactive = ref["active"].sum(1)
    np.testing.assert_array_equal(restored["active"].sum(1), nactive)
    good = (nactive >= 5) & np.isfinite(ref["chi2"]) & (ref["ndof"] >= 1)
    assert good.sum() > 64
    assert not good[-1]
    assert out["seeds"] == nseed
    np.testing.assert_array_equal(out["fits_per_event"], np.bincount(sd["ev"][good], minlength=nevents))
    good_ids = seed_order[good[seed_order]]
    expected_chi2ndf = ref["chi2"][good_ids] / np.maximum(ref["ndof"][good_ids], 1)
    # Both seed-order preservation and sorted, unaggregated good-fit summaries.
    np.testing.assert_array_equal(out["nactive"], nactive[good_ids])
    np.testing.assert_allclose(out["chi2ndf"], expected_chi2ndf, rtol=1e-10, atol=1e-9)
    np.testing.assert_array_equal(np.sort(out["nactive"]), np.sort(nactive[good]))
    np.testing.assert_allclose(np.sort(out["chi2ndf"]), np.sort(expected_chi2ndf), rtol=1e-10, atol=1e-9)
    assert out["profile"]["shape_counts"] == {"64x64": 2, "64x128": 1}
    assert out["profile"]["blocks"] == 3
    assert out["profile"]["tracks_padded"] == 3 * 64 - nseed
    assert out["profile"]["fit_s"] > 0


def _line_clusters():
    """Small variant of test_reco_jax.test_line_seeds_jax_matches_numpy.

    All 67 ordinary clusters share bucket 16, keeping pair tensors small;
    two interspersed oversize clusters check fallback index restoration.
    """
    rng = np.random.default_rng(9)
    ncl = 69
    nh = rng.integers(12, 17, ncl)
    nh[[1, 35]] = [257, 300]
    n = int(nh.sum())
    ev = np.repeat(np.arange(ncl), nh)
    nstraw = 36 * 6 * 96
    st = {"midx": np.zeros(nstraw), "midy": np.zeros(nstraw), "midz": np.zeros(nstraw),
          "dirx": np.ones(nstraw), "diry": np.zeros(nstraw), "dirz": np.zeros(nstraw),
          "hlen": np.full(nstraw, 600.0)}
    us = rng.choice(nstraw, n, replace=False)
    a0, b0 = rng.uniform(-300, 300, ncl), rng.uniform(-800, 800, ncl)
    a1, b1 = rng.uniform(-0.6, 0.6, ncl), rng.uniform(-0.6, 0.6, ncl)
    y = rng.uniform(-500, 500, n)
    x = a0[ev] - a1[ev] * y + rng.normal(0, 3, n)
    z = b0[ev] - b1[ev] * y + rng.normal(0, 2, n)
    phi = rng.uniform(0, np.pi, n)
    st["dirx"][us], st["dirz"][us] = np.cos(phi), np.sin(phi)
    st["midx"][us] = x - 50 * np.cos(phi)
    st["midy"][us] = y
    st["midz"][us] = z - 50 * np.sin(phi)
    noise = rng.uniform(size=n) < 0.15
    st["midz"][us[noise]] += rng.uniform(50, 300, noise.sum())
    plane, rem = us // 576, us % 576
    sid = ((plane << 10) | ((rem // 96) << 7) | (rem % 96)).astype(np.uint16)
    f32 = np.float32
    sh = {"ev": ev, "sid": sid, "x": x.astype(f32), "y": y.astype(f32), "z": z.astype(f32),
          "wdist": np.full(n, 50.0, f32), "uvar": np.full(n, 900.0, f32), "eend": np.zeros(n, np.int64),
          "et0": rng.uniform(1000, 1100, n).astype(f32), "et1": np.zeros(n, f32),
          "dtime": np.full(n, 10.0, f32), "ptime": np.full(n, 2.0, f32)}
    tc = {"ev": np.arange(ncl), "n": nh, "member": np.arange(n), "member_cl": ev}
    return sh, tc, {"straw": st}


def test_linefind_bounded_budget_and_numpy_fallback(monkeypatch):
    from trkfastsim.reco import linefind, linefind_jax

    sh, tc, cond = _line_clusters()
    assert np.count_nonzero(tc["n"] <= 16) == 67
    oversized = np.flatnonzero(tc["n"] > 256)
    ref = linefind.line_seeds(sh, tc, cond)
    assert len(ref["n"]) > 64
    assert np.all(np.isin(oversized, ref["tc"]))
    real_bucket = linefind_jax._seed_bucket
    real_numpy = linefind.line_seeds
    bucket_calls, fallback_calls = [], []

    def record_bucket(*args, **kwargs):
        bucket_calls.append((args[0].shape[:2], np.asarray(args[7]).copy()))
        return real_bucket(*args, **kwargs)

    def record_numpy(sh_arg, tc_arg, cond_arg, *args, **kwargs):
        assert sh_arg is sh and cond_arg is cond
        np.testing.assert_array_equal(tc_arg["n"], tc["n"][oversized])
        np.testing.assert_array_equal(tc_arg["ev"], tc["ev"][oversized])
        expected_member = tc["member"][np.isin(tc["member_cl"], oversized)]
        np.testing.assert_array_equal(tc_arg["member"], expected_member)
        np.testing.assert_array_equal(tc_arg["member_cl"], np.repeat(np.arange(len(oversized)), tc_arg["n"]))
        fallback_calls.append(tc_arg["n"].copy())
        return real_numpy(sh_arg, tc_arg, cond_arg, *args, **kwargs)

    monkeypatch.setattr(linefind_jax, "_seed_bucket", record_bucket)
    monkeypatch.setattr(linefind, "line_seeds", record_numpy)
    default = linefind_jax.line_seeds(sh, tc, cond)
    assert [shape for shape, _ in bucket_calls] == [(128, 16)]
    assert [np.count_nonzero(n) for _, n in bucket_calls] == [67]
    bucket_calls.clear()
    bounded = linefind_jax.line_seeds(sh, tc, cond, budget=100000)
    assert [shape for shape, _ in bucket_calls] == [(64, 16), (64, 16)]
    assert [np.count_nonzero(n) for _, n in bucket_calls] == [64, 3]
    assert len(fallback_calls) == 2
    for out in (default, bounded):
        assert set(out) == set(ref)
        for k in ("tc", "ev", "n", "member", "ind"):
            np.testing.assert_array_equal(out[k], ref[k], err_msg=k)
        for k in PARAMS:
            np.testing.assert_allclose(out[k], ref[k], rtol=1e-9, atol=1e-9, err_msg=k)
    for k in ("tc", "ev", "n", "member", "ind"):
        np.testing.assert_array_equal(bounded[k], default[k], err_msg=k)
    # Batch shape can change floating-point reduction order (CPU t0 differs
    # by ~2e-13); membership and discrete counts above must still be exact.
    for k in PARAMS:
        np.testing.assert_allclose(bounded[k], default[k], rtol=1e-12, atol=1e-10, err_msg=k)
