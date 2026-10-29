"""Independent NumPy Lynch-Dahl and saved-noise isolation/guard tests; no fit."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import f1_material_audit as A
from trkfastsim.reco import nativematerial as N
from trkfastsim.reco import kkmaterial
from test_f1_trace_compare import fixture, analytic


def independent_constants(name):
    # Explicit atom/mass mixtures, not production effective_za/material_constants.
    Z = np.array([1., 6., 8., 13., 18., 79.])
    atomic = np.array([1.007940, 12.011000, 15.999400, 26.981539, 39.948000, 196.966540])
    mylar = atomic*np.array([8, 10, 4, 0, 0, 0]); mylar /= mylar.sum()
    co2 = atomic*np.array([0, 1, 2, 0, 0, 0]); co2 /= co2.sum()
    if name == "straw-wall":
        mass = .9695*mylar + np.array([0, 0, 0, .0125, 0, .018])
        density = 1.4325e-3
    else:
        mass = .21696*co2 + np.array([0, 0, 0, 0, .78304, 0])
        density = .0016980e-3
    number = mass/atomic
    return number@Z/number.sum(), number@atomic/number.sum(), density


def oracle(name, p, path, fraction):
    z, a, density = independent_constants(name)
    beta2 = p*p/(p*p+105.6583755**2)
    chi_c = 15.7*z*(z+1)*np.abs(path)*density/(a*beta2*p*p)
    chi_a = 2.007e-5*z**(2/3)*(1+3.34*(z/137.036)**2/beta2)/(p*p)
    nu = chi_c/(2*(1-fraction)*chi_a)
    nonzero = np.where(nu != 0, nu, 1.)
    integral = (1+nonzero)/nonzero*np.log1p(nonzero)-1
    return np.where(nu == 0, 0., np.maximum(0., chi_c*integral/(1+fraction**2)))


@pytest.mark.parametrize("name", ["straw-wall", "straw-gas"])
@pytest.mark.parametrize("fraction", [.5, .999999, .9999999])
def test_independent_lynch_dahl(name, fraction):
    p = np.array([35., 105., 450., 2000.])[:, None]
    path = np.array([0., 1e-7, .015, 4.5, -8.])[None, :]
    actual = N.scatter_angle_var(name, p, path, fraction=fraction)
    np.testing.assert_allclose(actual, oracle(name, p, path, fraction), rtol=3e-12, atol=0.)
    assert np.all(actual[:, 0] == 0)


def test_explicit_variants_match_default_jax_and_do_not_change_environment(monkeypatch):
    p, wall, gas = np.array([380., 450.]), np.array([.04, .07]), np.array([4., 6.])
    old_default = kkmaterial.GAS_SCATTERFRAC
    monkeypatch.setenv("F1_GASFRAC", "anything-is-ignored-by-this-diagnostic")
    variants = N.scattering_variants(p, wall, gas, 81.)
    default = np.asarray(kkmaterial.scatter_angle_var("straw-wall", p, wall, scatterfrac=.999999)
                         + kkmaterial.scatter_angle_var("straw-gas", p, gas, scatterfrac=.999999))*81
    np.testing.assert_allclose(variants["default"], default, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(variants["p109"], 81*(oracle("straw-wall", p, wall, .999999)
        + oracle("straw-gas", p, gas, .9999999)), rtol=1e-13, atol=0.)
    assert np.all(variants["p109"] > variants["default"])
    assert kkmaterial.GAS_SCATTERFRAC == old_default
    assert __import__("os").environ["F1_GASFRAC"] == "anything-is-ignored-by-this-diagnostic"



def independent_kink_D(x, distance):
    """Complex-step differentiation of a geometric kink, independent of production D."""
    p0 = np.array([x[0], 0., x[1]])
    u = np.array([x[2], -1., x[3]]); u /= np.linalg.norm(u)
    e1 = np.array([u[1], -u[0], 0.]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    q = p0+distance*u
    velocity = lambda p: 299.792458*p/np.sqrt(p*p+105.6583755**2)
    tq = x[4]+distance/velocity(x[5])
    def kink(delta):
        direction = u+delta[0]*e1+delta[1]*e2
        direction /= np.sqrt(direction@direction)
        back = -q[1]/direction[1]
        origin = q+back*direction
        p = x[5]+delta[2]
        return np.array([origin[0], origin[2], -direction[0]/direction[1],
                         -direction[2]/direction[1], tq+back/velocity(p), p])
    return np.column_stack([kink(1j*1e-25*axis).imag/1e-25 for axis in np.eye(3)])


def noise_fixture():
    args = fixture()
    t = args[0]
    x = t["owner_parameters"][0]
    _, jac = analytic(x)
    # Independent geometric CA-position D, with natural-slot time distinct from sorting.
    distance = 350.
    D = independent_kink_D(x, distance)
    B = jac@D
    velocity = 299.792458*x[5]/np.hypot(x[5], 105.6583755)
    t["crossing_offset"][:] = .0125/velocity
    t["physical_time"][3] = x[4]+distance/velocity+t["crossing_offset"][0]
    p, wall, gas, scale = x[5], .04, 4., 81.
    sv = scale*(oracle("straw-wall", p, wall, .999999)+oracle("straw-gas", p, gas, .999999))
    new = scale*(oracle("straw-wall", p, wall, .999999)+oracle("straw-gas", p, gas, .9999999))
    dm, dmv = -.12, .008
    slot = np.flatnonzero(t["order"] == 3)[0]
    t["eff_D"][slot] = D
    t["eff_sv"][slot], t["eff_dm"][slot], t["eff_dmv"][slot] = sv, dm, dmv
    t["crossing_wall"][:] = wall
    t["crossing_gas"][:] = gas
    t["crossing_active"][:] = [True, False]
    t["crossing_inbounds"][:] = [True, False]
    effects = ak.to_list(args[3])
    material_row = np.flatnonzero(np.asarray(args[3].kind) == 2)[0]
    effects[material_row]["change_covariance"] = ((B*np.array([new, new, dmv]))@B.T).ravel().tolist()
    effects[material_row]["change_parameters"] = (B[:, 2]*dm).tolist()
    args[3] = ak.Array(effects)
    return args, (x[None], D[None], np.array([sv]), np.array([new]), np.array([dm]), np.array([dmv]),
                  (B[:, 2]*dm)[None], ((B*np.array([new, new, dmv]))@B.T)[None])


def test_saved_noise_substitution_independent_covariance_and_unchanged_operands():
    _, inputs = noise_fixture()
    before = tuple(a.copy() for a in inputs)
    result = A.substitutions(*inputs)
    assert result["saved"]["change_covariance"]["mismatches"] > 0
    new = result["explicit_p109_gas"]
    assert new["change_covariance"]["mismatches"] == 0
    assert new["change_parameters"]["passed"]
    assert result["saved"]["change_parameters"] == new["change_parameters"]
    assert result["saved"]["momentum_variance_scalar"] == new["momentum_variance_scalar"]
    recovered = N.native_scalars(inputs[-1], 81.)
    np.testing.assert_allclose(recovered["scaled_sv"], inputs[3], rtol=1e-14, atol=0.)
    np.testing.assert_allclose(recovered["scaled_dmv"], inputs[5], rtol=1e-14, atol=0.)
    for a, b in zip(inputs, before):
        np.testing.assert_array_equal(a, b)
    json.dumps(result, allow_nan=False)


def test_metric_retains_all_failures_and_tiny_relative_variance():
    actual, native = np.ones((4, 6, 6)), np.zeros((4, 6, 6))
    r = A.metric(actual, native)
    assert r["mismatches"] == len(r["mismatch_indices"]) == 144
    assert r["n_failing_effects"] == 4
    assert len(r["actual_mismatch_values"]) == 144
    assert r["relative_zero_reference_nonzero_residual"] == 144
    tiny = A.metric(np.array([1.1e-7]), np.array([1e-7]))
    assert tiny["passed"]  # unchanged old gate; report does not conceal the 10% difference
    assert tiny["max_relative_residual_unfloored"] == pytest.approx(.1)
    json.dumps(r, allow_nan=False)


def saved_fixture(tmp_path):
    args, _ = noise_fixture()
    t, it, old, effects, hits, crossings, membership, partners = args
    trace = tmp_path / "trace"; trace.mkdir()
    np.savez(trace / "trace.npz", **{name: value[None] for name, value in t.items()},
             stage_index=np.array([0]), key=np.array([1, 2, 3, 0]), hit_ids=hits, crossing_ids=crossings,
             seed_hit_mask=membership, crossing_partner=partners)
    condition = tmp_path / "conditions.json"; condition.write_text("{}")
    source = {"path": str(condition), "sha256": A.digest(condition)}
    root = Path(A.__file__).resolve().parents[1]
    manifest = dict(schema_version=1, completed=True, trace_passivity_exact=True, environment={},
        check_sha256="gate", inputs={}, trace_sha256=A.digest(trace / "trace.npz"),
        key=[1, 2, 3, 0], n_iterations=1, native_seed=True, native_time=True,
        time_model="stock", reference_model="previous", straw_geometry="approximate", crossing_time="current",
        schedule={"T": [8., 4.]}, conditions=source, alignment=source, ann={"test": source},
        source={name: A.digest(root / name) for name in A.REQUIRED_SOURCES})
    (trace / "manifest.json").write_text(json.dumps(manifest))
    return trace, {"check_sha256": "gate", "inputs": {}}, it, old, effects


def test_full_saved_audit_and_source_provenance(tmp_path):
    args = saved_fixture(tmp_path)
    path = args[0] / "manifest.json"
    manifest = json.loads(path.read_text())
    for name in ("vr_f1", "nativemeasurement", "nativeca"):
        manifest["source"][f"trkfastsim/reco/{name}.py"] = "deliberately-old-unused-source"
    path.write_text(json.dumps(manifest))
    report = A.audit_one(*args)
    assert report["varscale"] == 81
    assert report["default_reconstruction"]["tighter_scalar_diagnostic"]["passed"]
    assert report["n_compared_effects"] == 1
    assert report["compared_strawids"] == [900]
    assert report["compared_native_effect_indices"] == [1]
    assert not report["source_provenance"]["trkfastsim/reco/vr_f1.py"]["source_equal"]
    assert report["variants"]["saved"]["change_covariance"]["mismatches"] > 0
    assert report["variants"]["explicit_p109_gas"]["change_covariance"]["mismatches"] == 0
    for name in ("vr_f1", "nativemeasurement", "nativeca"):
        provenance = report["source_provenance"][f"trkfastsim/reco/{name}.py"]
        assert not provenance["source_equal"] and not provenance["equality_required"]
    assert report["kink_time_diagnostic"]["saved_D_reconstruction"]["strict_D_diagnostic"]["passed"]
    assert "explicit_p109_gas_physical_time_D" in report["variants"]
    assert not report["full_closure"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("field,value,error", [
    ("trace_passivity_exact", False, "passive"), ("completed", False, "passive"),
    ("environment", {"F1_GASFRAC": "p109"}, "environment"),
    ("check_sha256", "different", "gate"), ("trace_sha256", "different", "trace hash"),
    ("inputs", {"capture": "different"}, "evidence"), ("key", [1, 2, 4, 0], "key"),
    ("native_time", False, "constraint time"), ("n_iterations", 2, "count"),
    ("schedule", {"T": [-1.]}, "temperature"), ("time_model", "legacy", "policy"),
])
def test_manifest_guards(tmp_path, field, value, error):
    args = saved_fixture(tmp_path)
    path = args[0] / "manifest.json"
    m = json.loads(path.read_text()); m[field] = value; path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match=error):
        A.audit_one(*args)


def test_reconstruction_guard_and_native_identity(tmp_path):
    args = list(saved_fixture(tmp_path))
    path = args[0] / "trace.npz"
    with np.load(path) as f:
        saved = {name: f[name] for name in f.files}
    saved["eff_sv"] *= 1.001  # still passes absolute old scalar gate, fails tight reconstruction
    np.savez(path, **saved)
    mp = args[0] / "manifest.json"
    m = json.loads(mp.read_text()); m["trace_sha256"] = A.digest(path); mp.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="default scalar"):
        A.audit_one(*args)
    changed = ak.to_list(args[-1]); changed[0]["miter"] += 1; args[-1] = ak.Array(changed)
    with pytest.raises(ValueError, match="identity"):
        A.audit_one(*args)


def test_source_hash_and_config_guards(tmp_path):
    args = saved_fixture(tmp_path)
    mp = args[0] / "manifest.json"
    m = json.loads(mp.read_text()); original = copy.deepcopy(m)
    m["source"][A.REQUIRED_SOURCES[0]] = "changed"; mp.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="source changed"):
        A.verify_manifest(*args[:2])
    mp.write_text(json.dumps(original))
    Path(m["conditions"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="configuration hash"):
        A.verify_manifest(*args[:2])


@pytest.mark.parametrize("fraction", [-.1, 1., np.nan, [.5]])
def test_fraction_guards(fraction):
    with pytest.raises(ValueError, match="fraction"):
        N.scatter_angle_var("straw-wall", 450., 1., fraction=fraction)


@pytest.mark.parametrize("momentum,path", [(0., 1.), (-1., 1.), (np.nan, 1.), (450., np.inf)])
def test_scalar_guards(momentum, path):
    with pytest.raises(ValueError):
        N.scatter_angle_var("straw-gas", momentum, path, fraction=.9999999)


@pytest.mark.parametrize("scale", [0., -1., np.nan, [1.]])
def test_scale_guards(scale):
    with pytest.raises(ValueError, match="scale"):
        N.scattering_variants(450., 1., 1., scale)
    with pytest.raises(ValueError, match="scale"):
        N.native_scalars(np.eye(6), scale)


def test_shapes_empty_variance_and_path_guards():
    assert N.scattering_variants(np.empty((2, 0)), 0., 0., 1.)["p109"].shape == (2, 0)
    assert N.native_scalars(np.empty((0, 6, 6)), 1.)["sv"].shape == (0,)
    with pytest.raises(ValueError, match="nonnegative"):
        N.scattering_variants(450., -1., 0., 1.)
    with pytest.raises(ValueError, match="shape"):
        N.native_scalars(np.zeros((6, 5)), 1.)
    with pytest.raises(ValueError, match="negative"):
        N.native_scalars(-np.eye(6), 1.)
    with pytest.raises(ValueError, match="nonfinite"):
        A.metric(np.array([np.nan]), np.zeros(1))
    assert A.metric(np.empty((0, 6, 6)), np.empty((0, 6, 6)))["n_effects"] == 0


def test_exclusive_output_and_environment_guard(tmp_path, monkeypatch):
    out = tmp_path / "exists.json"; out.write_text("immutable")
    args = SimpleNamespace(out=out)
    with pytest.raises(ValueError, match="existing"):
        A.run(args)
    assert out.read_text() == "immutable"
    out.unlink()
    monkeypatch.setenv("F1_GASFRAC", "p109")
    with pytest.raises(ValueError, match="environment"):
        A.run(args)
    assert not out.exists()


def test_participation_intersection_and_every_failure_retained(tmp_path):
    args = list(saved_fixture(tmp_path))
    path = args[0] / "trace.npz"
    with np.load(path) as f:
        saved = {name: f[name] for name in f.files}
    # Add a constructed second crossing, F1-inactive but native participating.
    saved["crossing_partner"][1] = 53
    np.savez(path, **saved)
    mp = args[0] / "manifest.json"
    m = json.loads(mp.read_text()); m["trace_sha256"] = A.digest(path); mp.write_text(json.dumps(m))
    effects = ak.to_list(args[-1])
    extra = copy.deepcopy(effects[np.flatnonzero(np.asarray(args[-1].kind) == 2)[0]])
    extra.update(effect_index=4, strawid=1200, partner_shindex=53)
    args[-1] = ak.Array(effects + [extra])
    it = ak.to_list(args[2]); it[0]["neffects"] = 5; args[2] = ak.Array(it)
    report = A.audit_one(*args)
    assert report["n_compared_effects"] == 1
    assert report["participation"]["mismatches"] == 1
    assert report["participation"]["n_native_only"] == 1
    assert report["participation"]["n_native_participating"] == 2
    assert report["first_iteration_comparison"]["phases"]["material"]["status"] == "different"
    assert len(report["participation"]["mismatch_indices"]) == 1
    assert report["variants"]["explicit_p109_gas"]["change_covariance"]["passed"]
    # A passing restricted substitution never erases the participation failure.
    assert not report["participation"]["passed"]


def test_relative_only_is_diagnostic_not_gate():
    report = A.metric(np.array([1.01e-7]), np.array([1e-7]))
    assert report["passed"]
    assert not report["relative_only_diagnostic"]["passed"]
    assert report["relative_only_diagnostic"]["mismatch_indices"] == [[0]]


def test_displaced_D_independent_covariance_scalars_and_explicit_mean_time_shift():
    _, inputs = noise_fixture()
    x, D, saved_sv, sv, dm, dmv, _, _ = inputs
    time = x[:, 4]+np.array([350.0125])/(299.792458*x[:, 5]/np.hypot(x[:, 5], 105.6583755))
    s, shifted = N.kink_at_time(x, time)
    oracle_D = independent_kink_D(x[0], 350.0125)[None]
    np.testing.assert_allclose(s, [350.0125], rtol=1e-14)
    np.testing.assert_allclose(shifted, oracle_D, rtol=1e-12, atol=1e-12)
    _, jac = analytic(x[0])
    B = jac@oracle_D[0]
    mean = (B[:, 2]*dm[0])[None]
    covariance = ((B*np.array([sv[0], sv[0], dmv[0]]))@B.T)[None]
    before = [v.copy() for v in (x, D, saved_sv, sv, dm, dmv, shifted)]
    result = A.substitutions(x, D, saved_sv, sv, dm, dmv, mean, covariance, shifted_D=shifted)
    v1 = A.substitutions(x, D, saved_sv, sv, dm, dmv, mean, covariance)
    assert {name: result[name] for name in v1} == v1
    new = result["explicit_p109_gas_physical_time_D"]
    assert new["change_covariance"]["passed"] and new["change_parameters"]["passed"]
    delta = new["mean_shift_from_saved_D"]
    velocity = 299.792458*x[0, 5]/np.hypot(x[0, 5], 105.6583755)
    dvdp = 299.792458*105.6583755**2/np.hypot(x[0, 5], 105.6583755)**3
    expected = .0125*dvdp/velocity**2*dm[0]
    assert expected != 0
    np.testing.assert_allclose(delta["local_t0_delta"], [expected], rtol=1e-9)
    np.testing.assert_allclose(delta["native_t0_delta"], [expected], rtol=1e-9)
    np.testing.assert_array_equal(np.asarray(delta["native_parameter_delta"])[0, [0, 1, 2, 3, 5]], 0.)
    from trkfastsim.reco.nativecompare import transform_material
    for local_D in (D, shifted):
        _, cov = transform_material(x, local_D, sv, dm, dmv)
        recovered = N.native_scalars(cov, 81.)
        np.testing.assert_allclose(recovered["scaled_sv"], sv, rtol=1e-13)
        np.testing.assert_allclose(recovered["scaled_dmv"], dmv, rtol=1e-13)
    bad = covariance.copy(); bad[0, 0, 0] += .1
    failed = A.substitutions(x, D, saved_sv, sv, dm, dmv, mean, bad, shifted_D=shifted)
    assert not failed["explicit_p109_gas_physical_time_D"]["change_covariance"]["passed"]
    for value, original in zip((x, D, saved_sv, sv, dm, dmv, shifted), before):
        np.testing.assert_array_equal(value, original)
    json.dumps(result, allow_nan=False)


def test_vectorized_kink_and_strict_saved_D_guard():
    x = np.array([[125., -730., .35, .8, 42., 380.], [-10., 250., -.6, -.2, 18., 105.]])
    distance = np.array([350., -250.])
    velocity = 299.792458*x[:, 5]/np.hypot(x[:, 5], 105.6583755)
    offset = .0125/velocity
    time = x[:, 4]+distance/velocity+offset
    D = np.stack([independent_kink_D(xx, ss) for xx, ss in zip(x, distance)])
    shifted, report = A.kink_time_diagnostic(x, D, time, offset)
    np.testing.assert_allclose(shifted, np.stack([independent_kink_D(xx, ss+.0125)
        for xx, ss in zip(x, distance)]), rtol=1e-12, atol=1e-12)
    assert report["saved_D_reconstruction"]["strict_D_diagnostic"]["passed"]
    damaged = D.copy(); damaged[0, 4, 2] += 1e-8
    assert A.metric(D, damaged)["passed"]
    _, diagnostic = A.kink_time_diagnostic(x, damaged, time, offset)
    assert not diagnostic["saved_D_reconstruction"]["strict_D_diagnostic"]["passed"]
    damaged[0, 4, 2] += 1e-3
    with pytest.raises(ValueError, match="reconstruct saved D"):
        A.kink_time_diagnostic(x, damaged, time, offset)
    with pytest.raises(ValueError, match="reconstruct saved D"):
        A.kink_time_diagnostic(x, D, time, np.zeros(2))


@pytest.mark.parametrize("x,time,error", [
    (np.zeros(6), np.zeros(1), "shape"), (np.ones((2, 6)), np.zeros(1), "shape"),
    (np.zeros((1, 6)), np.zeros(1), "positive"),
    (np.ones((1, 6)), np.array([np.nan]), "nonfinite")])
def test_kink_time_input_guards(x, time, error):
    with pytest.raises(ValueError, match=error):
        N.kink_at_time(x, time)


def test_offset_increment_uses_saved_D_without_absolute_time_reconstruction():
    x = np.array([[125., -730., .35, .8, 73000., 2000.], [-10., 250., -.6, -.2, 4400., 1800.]])
    distances = np.array([350., -250.])
    velocity = 299.792458*x[:, 5]/np.hypot(x[:, 5], 105.6583755)
    offset = .0125/velocity
    D = np.stack([independent_kink_D(xx, ss) for xx, ss in zip(x, distances)])
    shifted, ds = N.offset_kink_D(x, D, offset)
    expected = np.stack([independent_kink_D(xx, ss+.0125) for xx, ss in zip(x, distances)])
    np.testing.assert_allclose(shifted, expected, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(ds, .0125, rtol=1e-14)
    zero, _ = N.offset_kink_D(x, D, np.zeros(2))
    np.testing.assert_array_equal(zero, D)
    with pytest.raises(ValueError, match="offset"):
        N.offset_kink_D(x, D, -offset)


def test_saved_audit_retains_strict_D_diagnostic_and_rejects_gate_failure(tmp_path):
    args = saved_fixture(tmp_path)
    path = args[0] / "trace.npz"
    with np.load(path) as f:
        saved = {name: f[name] for name in f.files}
    slot = np.flatnonzero(saved["order"][0] == 3)[0]
    original = saved["eff_D"][0, slot].copy()
    saved["eff_D"][0, slot, 4, 2] += 1e-8
    assert A.metric(saved["eff_D"][0, slot][None], original[None])["passed"]
    np.savez(path, **saved)
    mp = args[0] / "manifest.json"
    manifest = json.loads(mp.read_text())
    manifest["trace_sha256"] = A.digest(path)
    mp.write_text(json.dumps(manifest))
    report = A.audit_one(*args)
    assert not report["kink_time_diagnostic"]["saved_D_reconstruction"]["strict_D_diagnostic"]["passed"]
    saved["eff_D"][0, slot, 4, 2] += 1e-3
    np.savez(path, **saved)
    manifest["trace_sha256"] = A.digest(path)
    mp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="CA-time D does not reconstruct saved D"):
        A.audit_one(*args)
