import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from kalman_toy import make_track  # noqa: E402
from trkfastsim.reco import kalman_jax as K, linefit_jax as LJ  # noqa: E402


def _linear_toy(rng, E=30, noise_every=0):
    H = rng.normal(size=(E, 3, 5))
    w = np.abs(rng.normal(size=(E, 3))) + 0.5
    w[::4, 2] = 0
    ws = np.where(w > 0, w, 1.0)
    r = -(H @ rng.normal(size=5)) + rng.normal(size=(E, 3)) / np.sqrt(ws)
    Q = np.zeros((E, 5, 5))
    D = np.zeros((E, 5, 2))
    sv = np.zeros(E)
    if noise_every:
        for j in range(0, E, noise_every):
            D[j] = rng.normal(size=(5, 2))
            sv[j] = 0.05 ** 2
            Q[j] = sv[j] * D[j] @ D[j].T
    eff = dict(xref=jnp.zeros((E, 5)), H=jnp.asarray(H), r=jnp.asarray(r), w=jnp.asarray(w), mvar=jnp.asarray(1 / ws),
               D=jnp.asarray(D), sv=jnp.asarray(sv))
    P = jnp.eye(5) * 1e-12
    return H, w, r, Q, K.filter_smooth(eff, P, jnp.zeros(5), P, jnp.zeros(5)), eff


def test_filter_without_noise_is_least_squares_and_unbiased_is_refit():
    H, w, r, _, fs, eff = _linear_toy(np.random.default_rng(1))
    x, _, xu, _ = K.smoothed(fs, eff)
    A = np.einsum("eki,ek,ekj->ij", H, w, H)
    g = np.einsum("eki,ek,ek->i", H, w, -r)
    np.testing.assert_allclose(np.asarray(x), np.broadcast_to(np.linalg.solve(A, g), x.shape), atol=1e-4)
    e = 7
    xe = np.linalg.solve(A - H[e].T @ np.diag(w[e]) @ H[e], g - H[e].T @ (w[e] * -r[e]))
    np.testing.assert_allclose(np.asarray(xu)[e], xe, atol=1e-4)


def test_filter_with_noise_is_correlated_gls():
    H, w, r, Q, fs, eff = _linear_toy(np.random.default_rng(1), noise_every=3)
    x, C, _, _ = K.smoothed(fs, eff)
    E = H.shape[0]
    idx = [(a, k) for a in range(E) for k in range(3) if w[a, k] > 0]
    Hm = np.array([H[a, k] for a, k in idx])
    rm = np.array([r[a, k] for a, k in idx])
    Sc = [Q[a:].sum(0) for a in range(E)]
    Vm = np.diag([1 / w[a, k] for a, k in idx])
    for i, (a, k) in enumerate(idx):
        for j, (f, l) in enumerate(idx):
            Vm[i, j] += H[a, k] @ Sc[max(a, f)] @ H[f, l]
    Vi = np.linalg.inv(Vm)
    F = Hm.T @ Vi @ Hm
    np.testing.assert_allclose(np.asarray(x)[-1], np.linalg.solve(F, Hm.T @ Vi @ -rm), atol=1e-4)
    np.testing.assert_allclose(np.asarray(C)[-1], np.linalg.inv(F), rtol=1e-5, atol=1e-10)


def test_kink_jacobian_closed_form_equals_autodiff():
    for x in ([120.0, -300.0, 0.3, -0.2, 15.0], [-50.0, 40.0, -1.2, 0.8, 3.0]):
        for s in (-300.0, 0.0, 450.0):
            np.testing.assert_allclose(np.asarray(K.kink_jacobian(jnp.array(x), s)),
                                       np.asarray(K.kink_jacobian_ad(jnp.array(x), s)), atol=1e-10)


def test_kink_moves_line_by_lever_arm():
    x = jnp.array([120.0, -300.0, 0.3, -0.2, 15.0])
    p, u = K.point_dir(x)
    s, L, th = 400.0, 500.0, 1e-4
    pn, un = K.point_dir(K.kink(x, s, jnp.array([th, 0.0])))
    q = p + u * (s + L)
    d = q - pn
    assert float(jnp.linalg.norm(d - un * (d @ un))) == pytest.approx(L * th, rel=1e-6)
    np.testing.assert_allclose(np.asarray(K.params_from(p, u, x[4])), np.asarray(x), atol=1e-12)


def _toy_track(seed):
    rng = np.random.default_rng(seed)
    make_track(rng, smear=False)
    xt, h, crs, d = make_track(rng)
    N = len(h["lr"])
    hj = {k: jnp.asarray(v) for k, v in dict(h, valid=np.ones(N, bool)).items()}
    cj = {k: jnp.asarray(v) for k, v in dict(crs, hit=np.arange(N)).items()}
    dj = {k: jnp.asarray(d[k]) for k in LJ.DRIFT_KEYS}
    return xt, h, hj, cj, d, dj


def test_iteration_without_material_equals_f2():
    xt, h, hj, cj, d, dj = _toy_track(3)
    N = len(h["lr"])
    x0 = jnp.asarray(xt + np.array([1.0, -1.0, 0.003, 0.002, 0.5]))
    C0 = jnp.diag(jnp.array([25.0, 25.0, 1e-4, 1e-4, 4.0]))
    xe, Ce = jnp.tile(x0, (2 * N, 1)), jnp.tile(C0, (2 * N, 1, 1))
    hs = {"active": hj["active"], "lr": hj["lr"], "dvar": jnp.full(N, 0.15 ** 2)}
    pr = {"xf": x0, "Cf": C0, "xb": x0, "Cb": C0}
    for _ in range(6):
        o = K.iterate_once(xe, Ce, hj, cj, dj, hs, jnp.ones(N, bool), 1.0, 1.0, 0.0, pr, 1e6)
        xe, Ce = o["x"], o["C"]
        pr = {"xf": o["xfront"], "Cf": o["Cfront"], "xb": o["xback"], "Cb": o["Cback"]}
    f = LJ.fit(np.asarray(x0)[None], {k: np.asarray(v)[None] for k, v in h.items()},
               {k: np.asarray(d[k]) for k in LJ.DRIFT_KEYS}, niter=12)
    sig = np.sqrt(np.diag(f["cov"][0]))
    assert np.all(np.abs(np.asarray(xe[0]) - f["x"][0]) / sig < 1e-4)
    np.testing.assert_allclose(np.sqrt(np.diag(np.asarray(Ce[0]))) / sig, 1.0, rtol=1e-4)


def _anns_available():
    try:
        K.load_anns()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _anns_available(), reason="KinKal ANN weights not found (set MU2E_SEARCH_PATH)")
def test_full_schedule_converges_on_toy_tracks():
    xt, h, hj, cj, d, dj = _toy_track(3)
    x0 = jnp.asarray(xt + np.array([3.0, -2.0, 0.01, 0.005, 2.0]))
    cf, hist, good = jax.jit(lambda x: K.fit_track(x, hj, cj, dj, K.stock_stages(), K.load_anns()))(x0)
    assert bool(good)
    assert np.all(np.asarray(hist[0]) == K.CONVERGED)
    sig = np.sqrt(np.diag(np.asarray(cf["Ce"][0])))
    assert np.all(np.abs(np.asarray(cf["xe"][0]) - xt) / sig < 4)


def test_covariance_filter_equals_information_filter():
    xt, h, hj, cj, d, dj = _toy_track(3)
    N = len(h["lr"])
    x0 = jnp.asarray(xt + np.array([1.0, -1.0, 0.003, 0.002, 0.5]))
    C0 = K.seed_covariance(x0)
    xe, Ce = jnp.tile(x0, (2 * N, 1)), jnp.tile(C0, (2 * N, 1, 1))
    hs = {"active": hj["active"], "lr": hj["lr"], "dvar": jnp.full(N, 0.15 ** 2)}
    pr = {"xf": x0, "Cf": C0, "xb": x0, "Cb": C0}
    out, keep = {}, K.FILTER
    try:
        for mode in ("info", "cov"):
            K.FILTER = mode
            a = K.iterate_once(xe, Ce, hj, cj, dj, hs, jnp.ones(N, bool), 4.0, 1.0, 50.0, pr, 1e6)
            out[mode] = K.iterate_once(a["x"], a["C"], hj, cj, dj, hs, jnp.ones(N, bool), 4.0, 1.0, 50.0,
                                       {"xf": a["xfront"], "Cf": a["Cfront"], "xb": a["xback"], "Cb": a["Cback"]}, 1e6)
    finally:
        K.FILTER = keep
    A, B = out["info"], out["cov"]
    # hits (the cov path gives crossings their KinKal reference piece, the one before their boundary; the info path
    # keeps the piece that follows them, so only hit pieces are compared)
    for k, c, sl in (("x", "C", slice(0, N)), ("xu", "Cu", slice(None))):
        sig = np.sqrt(np.diagonal(np.asarray(A[c])[sl], axis1=1, axis2=2))
        # after the 2nd iteration the crossing references differ (piece before vs after), so 1e-3 sigma, not 1e-9
        assert np.max(np.abs(np.asarray(A[k])[sl] - np.asarray(B[k])[sl]) / sig) < 1e-3
        dC = np.abs(np.asarray(B[c])[sl] - np.asarray(A[c])[sl])
        assert np.all(dC <= 1e-3 * np.sqrt(np.einsum("nii,njj->nij", np.asarray(A[c])[sl], np.asarray(A[c])[sl])))
    assert float(B["chi2"]) == pytest.approx(float(A["chi2"]), rel=1e-4)
    assert float(B["avggap"]) == pytest.approx(float(A["avggap"]), rel=1e-2, abs=1e-6)
