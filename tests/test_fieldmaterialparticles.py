"""Frozen frame, identity, inertness, geometry partitions and closed replay gates."""
import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import tables
from trkfastsim import geometry
from trkfastsim.sim import fieldsteps as fs
from trkfastsim.validation import fieldcrossing as fc, fieldmaterial as fm
from trkfastsim.validation import fieldmaterialparticles as fp


def geom():
    return geometry.StrawGeometry(
        plane=np.array([0]), panel=np.array([0]), straw=np.array([0]),
        mid=np.zeros((1, 3)), dir=np.array([[0., 0., 1.]]), halflen=np.array([2.]), radius=1.,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)), plane_R=np.eye(3)[None],
        plane_t=np.zeros((1, 3)), wall_rmax=1.2, metal_t=.05)


def start(n=4):
    return pd.DataFrame(dict(run=np.full(n, 2**63 + 1, dtype=np.uint64), subrun=0,
                             event=np.arange(n, dtype=np.uint64) + np.uint64(2**63 + 7),
                             sp=np.arange(n, dtype=np.uint64) + np.uint64(2**63 + 11),
                             pdg=13, x=-3., y=np.linspace(-.5, .5, n), z=0., px=150., py=1., pz=1., t=0.))


def rays(a=(-2., 0., 0.), b=(2., 0., 0.)):
    return fc.endpoint_secants(np.array([a]), np.array([b]), np.zeros(3))[:3]


def test_direct_helper_central_graze_boundary_zero_denominator_and_pure_means():
    den, g = start(), geom()
    before = den.copy(deep=True)
    material_before = [m.__dict__.copy() for m in (fs.materials.MYLAR, fs.materials.ALUMINIUM_6061)]
    obs = fp.ParticleWallObserver(0)
    obs.begin(den)
    for row, args in enumerate((rays(), rays((-2., 1.1, 0.), (2., 1.1, 0.)), rays((-1.2, 0., 0.), (-1.1, 0., 0.)))):
        pieces = fs._straw_pieces(*args, g, 1.2, 0., 0., fs.ChordIndex(g))
        obs.observe(np.array([row]), np.array([150.]), np.array([fs.eloss.MMU]), *args, g, 1.2, pieces)
    d, report = obs.ledger, obs.report()
    assert report["consistent"] and report["entries"] == 4 and report["zero_chord_entries"] == 1
    assert d.sp.dtype == np.uint64 and d.sp.tolist() == den.sp.tolist()
    assert d.event.tolist() == den.event.tolist() and d.run.tolist() == den.run.tolist()
    np.testing.assert_array_equal(d.nactualchords, [1, 1, 1, 0])
    assert d.loc[0, "consumed_wall_mm"] == pytest.approx(.4)
    assert d.loc[1, "graze_wall_mm"] == pytest.approx(2 * np.sqrt(1.2**2 - 1.1**2))
    assert d.loc[2, "boundary_wall_mm"] == pytest.approx(.1)
    assert d.loc[3, list(fp.COUNTS + fp.LENGTHS + fp.MODELS)].eq(0).all()
    assert [m.__dict__ for m in (fs.materials.MYLAR, fs.materials.ALUMINIUM_6061)] == material_before
    split = .05 / .2
    rate = sum(w * fs.eloss.dedx_restricted(replace(m), np.array([150.]), 1e9, np.array([fs.eloss.MMU]))[0]
               for m, w in ((fs.materials.MYLAR, 1-split), (fs.materials.ALUMINIUM_6061, split)))
    xrate = sum(w / fs.eloss.radiation_length_mm(m)
                for m, w in ((fs.materials.MYLAR, 1-split), (fs.materials.ALUMINIUM_6061, split)))
    np.testing.assert_allclose(d.mean_missing_collision_loss_MeV, d.missing_wall_mm * rate)
    np.testing.assert_allclose(d.mean_consumed_collision_loss_MeV, d.consumed_wall_mm * rate)
    np.testing.assert_allclose(d.missing_wall_x0, d.missing_wall_mm * xrate)
    pd.testing.assert_frame_equal(den, before, check_exact=True)
    assert report["label"] == "MEANMODELGEOMETRYDIAGNOSTIC"
    assert "NOT realized" in report["scope"]


@pytest.mark.parametrize("act,p,m", [([0, 0], [1., 1.], [1., 1.]), ([99], [1.], [1.]),
                                     ([-1], [1.], [1.]), ([.5], [1.], [1.]),
                                     ([0], [np.nan], [1.]), ([0], [1.], [np.inf]),
                                     ([0], [1., 2.], [1.]), ([0], [0.], [1.])])
def test_invalid_act_and_kinematics_fail_closed(act, p, m):
    obs = fp.ParticleWallObserver(0)
    obs.begin(start())
    args = rays()
    if len(act) == 2:
        args = tuple(np.repeat(a, 2, axis=0) for a in args)
    with pytest.raises(ValueError):
        obs.observe(np.asarray(act), p, m, *args, geom(), 1.2, None)
    assert obs.ledger.nactualchords.sum() == 0


@pytest.mark.parametrize("kind", ["duplicate", "float", "spzero", "allzero"])
def test_original_identity_validation(kind):
    den = start()
    if kind == "duplicate":
        for key in fp.TRACK:
            den.loc[1, key] = den.loc[0, key]
    elif kind == "float":
        den["event"] = den.event.astype(float)
    elif kind == "spzero":
        den["sp"] = np.uint64(0)
    else:
        den[tables.KEY] = np.uint64(0)
    with pytest.raises(ValueError):
        fp.ParticleWallObserver().begin(den)


def test_seeded_real_transport_payload_rng_and_start_exact_equality():
    den, g = start(), geom()
    den.loc[3, "x"] = 99.
    before = den.copy(deep=True)
    opts = dict(chord_mm=.5, envelope=fs.Envelope(g, pad_r=2., pad_z=2., axis_xy=(0., 0.)), origin=np.zeros(3))
    rng0, rng1 = np.random.default_rng(19), np.random.default_rng(19)
    baseline = fs.transport(rng0, den, g, np.array([0., 0., 1.]), **opts)
    original_transport, original_straw, index_class = fs.transport, fs._straw_pieces, fs.ChordIndex
    obs = fp.ParticleWallObserver(16)
    with fp.observing_particles(obs):
        assert fs.ChordIndex is index_class
        observed = fs.transport(rng1, den, g, np.array([0., 0., 1.]), **opts)
    assert fs.transport is original_transport and fs._straw_pieces is original_straw
    for a, b in zip(baseline, observed):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(den, before, check_exact=True)
    assert rng0.bit_generator.state == rng1.bit_generator.state
    assert obs.report()["consistent"] and len(observed[0]) > 0
    np.testing.assert_array_equal(obs.ledger.nactualchords, observed[1].chords)
    assert obs.ledger.loc[3, "nactualchords"] == 0


def synthetic_transport(rng, start, geom, field, *, gas=None, loss=None, chord_mm=20., sagitta_mm=.02, radiative=False):
    xa, c, L = rays()
    act, pa, ma = np.array([0]), np.array([150.]), np.array([fs.eloss.MMU])
    sp = fs._straw_pieces(xa, c, L, geom, 1.2, 0., 0., fs.ChordIndex(geom))
    rng.random(7)
    synthetic_transport.last_pieces = sp
    synthetic_transport.last = (start.copy(deep=True), pd.DataFrame(dict(value=[rng.random()])))
    return synthetic_transport.last


def test_synthetic_original_once_same_tuple_and_straw_object(monkeypatch):
    monkeypatch.setattr(fs, "transport", synthetic_transport)
    original = fs._straw_pieces
    calls = []
    pieces = original(*rays(), geom(), 1.2, 0., 0.)

    def straw(*args):
        calls.append(1)
        return pieces

    monkeypatch.setattr(fs, "_straw_pieces", straw)
    r0, r1 = np.random.default_rng(1), np.random.default_rng(1)
    den = start()
    baseline = synthetic_transport(r0, den, geom(), None)
    calls.clear()
    obs = fp.ParticleWallObserver(0)
    with fp.observing_particles(obs):
        result = fs.transport(r1, den, geom(), None)
        assert result is synthetic_transport.last
        assert synthetic_transport.last_pieces is pieces
    assert calls == [1] and obs.transport_calls == 1
    assert r0.bit_generator.state == r1.bit_generator.state
    for a, b in zip(baseline, result):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert fs.transport is synthetic_transport and fs._straw_pieces is straw


def test_direct_context_failclosed_and_restores_on_errors(monkeypatch):
    t, s, cls = fs.transport, fs._straw_pieces, fs.ChordIndex
    with pytest.raises(ValueError, match="caller context"):
        with fp.observing_particles(fp.ParticleWallObserver()):
            fs._straw_pieces(*rays(), geom(), 1.2, 0., 0.)
    assert (fs.transport, fs._straw_pieces, fs.ChordIndex) == (t, s, cls)
    with pytest.raises(RuntimeError, match="body"):
        with fp.observing_particles(fp.ParticleWallObserver()):
            raise RuntimeError("body")
    assert fs.transport is t and fs._straw_pieces is s

    def broken(*args):
        raise RuntimeError("original")

    monkeypatch.setattr(fs, "_straw_pieces", broken)
    with pytest.raises(RuntimeError, match="original"):
        with fp.observing_particles(fp.ParticleWallObserver()):
            fs._straw_pieces(*rays(), geom(), 1.2, 0., 0.)
    assert fs.transport is t and fs._straw_pieces is broken


def test_frozen_options_rejected_and_observer_exception_restores(monkeypatch):
    monkeypatch.setattr(fs, "transport", synthetic_transport)
    settings = dict(chord_mm=20., sagitta_mm=.02, gas_state="gas", mu_radiative=False)
    with pytest.raises(ValueError, match="options/gas"):
        with fp.observing_particles(fp.ParticleWallObserver(), settings):
            fs.transport(np.random.default_rng(0), start(), geom(), None)
    assert fs.transport is synthetic_transport

    class Broken(fp.ParticleWallObserver):
        def observe(self, *args):
            raise RuntimeError("observer")

    s = fs._straw_pieces
    with pytest.raises(RuntimeError, match="observer"):
        with fp.observing_particles(Broken()):
            fs.transport(np.random.default_rng(0), start(), geom(), None)
    assert fs.transport is synthetic_transport and fs._straw_pieces is s


def test_global_report_gate_strict_lengths_and_counts():
    obs = fp.ParticleWallObserver(0)
    obs.begin(start())
    obs.observe(np.array([0]), [150.], [fs.eloss.MMU], *rays((-2., 1.1, 0.), (2., 1.1, 0.)), geom(), 1.2, None)
    current = obs.global_observer.report()
    assert fp.compare_material_totals(current, current)["equal"]
    for key, delta in (("missing_wall_mm", 2e-7), ("calls", 1), ("lanes", 1), ("outer_rows", 1)):
        prior = dict(current)
        prior[key] += delta
        assert not fp.compare_material_totals(current, prior)["equal"]
    prior = dict(current, missing_wall_mm=current["missing_wall_mm"] + .5e-7)
    assert fp.compare_material_totals(current, prior)["equal"]


def test_cli_requires_entry_and_material_report():
    for missing in ("--entry", "--material-report"):
        argv = []
        for flag in ("--field-dir", "--input-table", "--gdml", "--map", "--material-report", "--outdir", "--entry"):
            if flag != missing:
                argv.extend([flag, "first" if flag == "--entry" else "fixture"])
        with pytest.raises(SystemExit) as exc:
            fp.main(argv)
        assert exc.value.code == 2


@pytest.mark.parametrize("mutation", ["alias", "start", "duplicate", "foreign", "nonfinite", "helper"])
def test_context_rejects_unexpected_locals_and_indirect_call(monkeypatch, mutation):
    namespace = dict(fs=fs, rays=rays, np=np, mutation=mutation)
    exec('''def frozen(rng, start, geom, field, *, gas=None, loss=None, chord_mm=20., sagitta_mm=.02, radiative=False):
    xa, c, L = rays()
    act, pa, ma = np.array([0]), np.array([150.]), np.array([fs.eloss.MMU])
    if mutation == "start":
        start = start.copy()
    if mutation == "foreign":
        act = np.array([999])
    if mutation == "nonfinite":
        pa = np.array([np.nan])
    if mutation == "duplicate":
        xa, c, L = (np.repeat(v, 2, axis=0) for v in (xa, c, L))
        act, pa, ma = np.array([0, 0]), np.array([150., 150.]), np.array([fs.eloss.MMU, fs.eloss.MMU])
    if mutation == "helper":
        def helper():
            return fs._straw_pieces(xa, c, L, geom, 1.2, 0., 0.)
        return helper()
    xx = xa.copy() if mutation == "alias" else xa
    return fs._straw_pieces(xx, c, L, geom, 1.2, 0., 0.)
''', namespace)
    frozen, s, cls = namespace["frozen"], fs._straw_pieces, fs.ChordIndex
    monkeypatch.setattr(fs, "transport", frozen)
    with pytest.raises(ValueError):
        with fp.observing_particles(fp.ParticleWallObserver(0)):
            fs.transport(np.random.default_rng(1), start(), geom(), None)
    assert fs.transport is frozen and fs._straw_pieces is s and fs.ChordIndex is cls


def payload():
    ids = np.array([2**63 + 1, 2**63 + 3], dtype=np.uint64)
    return dict(particles=pd.DataFrame(dict(run=1, subrun=0, event=ids, key=ids, pdg=13)),
                sgsteps=pd.DataFrame(dict(run=1, subrun=0, event=ids, sp=ids, len=[1., 2.])))


def write_arms(directory):
    directory.mkdir()
    for filename in fc.FILES.values():
        tables.write(str(directory / filename), payload(), as_ttree=True)


def settings():
    return dict(species=13, inputs=1, entries=4, seed=19, chord_mm=20., sagitta_mm=.02,
                gas_state="gas", mu_radiative=True, origin_mm=[0., 0., 0.],
                entry_vds=[11, 12, 13], field="map.header")


def prior_fixture(tmp_path):
    old = tmp_path / "original"
    write_arms(old)
    (old / "fieldcompare.json").write_text(json.dumps(settings()))
    gdml, map_, input_ = [tmp_path / n for n in ("world.gdml", "map.header", "input.root")]
    for path in (gdml, map_, input_, map_.with_suffix(".bin")):
        path.write_bytes(b"fixture")
    wall = fm.WallObserver(0)
    args, g = rays(), geom()
    wall.observe(*args, g, 1.2, fs._straw_pieces(*args, g, 1.2, 0., 0., fs.ChordIndex(g)))
    paths = dict(input_table=input_, gdml=gdml, map_header=map_, map_adjacent_binary=map_.with_suffix(".bin"))
    for path in old.rglob("*"):
        if path.is_file():
            paths[f"original/{path.relative_to(old)}"] = path
    modules = dict(observer=fm, fieldsteps=fs, fieldcompare=fm.fieldcompare, fieldcrossing=fc,
                   fieldpathaudit=fm.pa, crossings=fm.crossings, geometry=geometry, core_tables=tables,
                   core_ehelix=fs.ehelix, core_eloss=fs.eloss, core_transport=fs.tr,
                   core_materials=fs.materials, steps=fs.st)
    paths.update({f"code/{name}": Path(module.__file__) for name, module in modules.items()})
    receipts = fm.pa.receipts(paths)
    checks = {f"{arm}/{tree}": dict(equal=True, rows=2)
              for arm in fc.FILES for tree in ("particles", "sgsteps")}
    prior = dict(passed=True, inputs_unchanged=True, reproduction=dict(reproduced=True, tables=checks),
                 frozen_settings=fm.pa.frozen_settings(settings(), "first"),
                 comparisons_diagnostic_only=False, original_field_dir=str(old),
                 material_stats=wall.report(), hashes_before=receipts, hashes_after=receipts)
    report = tmp_path / "fieldmaterial.json"
    report.write_text(json.dumps(prior))
    return old, input_, gdml, map_, report, prior


@pytest.mark.parametrize("mode", ["equal", "payload", "integer", "order", "mutate", "error", "global", "evidence", "priorfailed"])
def test_cli_closed_evidence_exact_all_six_payload_gate_and_failed_retention(tmp_path, monkeypatch, mode):
    old, input_, gdml, map_, report, prior = prior_fixture(tmp_path)
    if mode == "global":
        prior["material_stats"]["missing_wall_mm"] += 2e-7
    elif mode == "evidence":
        prior["hashes_before"]["input_table"]["sha256"] = "0" * 64
    elif mode == "priorfailed":
        prior["passed"] = False
    report.write_text(json.dumps(prior))
    monkeypatch.setattr(fs, "transport", synthetic_transport)
    original_straw, cls = fs._straw_pieces, fs.ChordIndex
    called = []

    def replay(inputs, output, **kw):
        called.append((inputs, kw))
        fs.transport(np.random.default_rng(19), start(), geom(), None,
                     gas=fs.STRAW_GAS_G4STATE, radiative=True)
        write_arms(Path(output))
        if mode in ("payload", "integer", "order"):
            changed = payload()
            if mode == "payload":
                changed["sgsteps"].loc[0, "len"] = np.nextafter(1., 2.)
            elif mode == "integer":
                changed["particles"].loc[0, "key"] = np.uint64(2**63 + 2)
            else:
                changed["sgsteps"] = changed["sgsteps"].iloc[::-1].reset_index(drop=True)
            tables.write(str(Path(output) / fc.FILES["fast"]), changed, as_ttree=True)
        elif mode == "mutate":
            input_.write_bytes(b"changed")
        elif mode == "error":
            raise RuntimeError("replay failure")

    monkeypatch.setattr(fp.fieldcompare, "run", replay)
    out = tmp_path / "audit"
    result = fp.run(old, input_, gdml, map_, report, out, entry="first")
    assert fs.transport is synthetic_transport and fs._straw_pieces is original_straw and fs.ChordIndex is cls
    assert result["passed"] == (mode == "equal")
    assert result["interpretation_allowed"] == (mode == "equal")
    assert (out / "FAILED").exists() == (mode != "equal")
    assert (out / "particles.csv").exists()
    saved = json.loads((out / "fieldmaterialparticles.json").read_text())
    assert saved["passed"] == result["passed"]
    assert "map_adjacent_binary" in result["hashes_before"]
    assert "material_report" in result["hashes_before"]
    assert "code/fieldmaterial" in result["hashes_before"]
    assert "code/cecontract" in result["hashes_before"]
    if mode in ("evidence", "priorfailed"):
        assert called == [] and "error" in result
    else:
        assert len(called) == 1
        assert len(result["reproduction"]["tables"]) == (0 if mode == "error" else 6)
        d = pd.read_csv(out / "particles.csv", dtype={key: "uint64" for key in fp.TRACK})
        assert d.sp.tolist() == start().sp.tolist() and len(d) == 4
        assert d.nactualchords.tolist() == [1, 0, 0, 0]
    if mode == "global":
        assert not result["global_material_comparison"]["equal"]
    with pytest.raises(FileExistsError):
        fp.run(old, input_, gdml, map_, report, out, entry="first")


def test_complete_43156_entry_denominator_no_event_loops_or_zero_filter():
    den = start(43156)
    obs = fp.ParticleWallObserver(0)
    obs.begin(den)
    obs.observe(np.array([43155]), [150.], [fs.eloss.MMU], *rays((5., 5., 5.), (6., 5., 5.)), geom(), 1.2, None)
    ledger, report = obs.ledger, obs.report()
    assert report["consistent"] and report["entries"] == 43156 and report["zero_chord_entries"] == 43155
    assert report["zero_consumed_wall_entries"] == 43156
    pd.testing.assert_frame_equal(ledger[fp.TRACK], fp.fieldmultiplicity.track_keys(den, "test")[fp.TRACK])


def test_transport_original_error_restores_both_and_original_called_once(monkeypatch):
    calls = []

    def broken(rng, start, geom, field, *, gas=None, loss=None, chord_mm=20., sagitta_mm=.02, radiative=False):
        calls.append(1)
        raise RuntimeError("transport original")

    s, cls = fs._straw_pieces, fs.ChordIndex
    monkeypatch.setattr(fs, "transport", broken)
    with pytest.raises(RuntimeError, match="transport original"):
        with fp.observing_particles(fp.ParticleWallObserver()):
            fs.transport(np.random.default_rng(0), start(), geom(), None)
    assert calls == [1] and fs.transport is broken and fs._straw_pieces is s and fs.ChordIndex is cls


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "43156", None, True])
def test_nonfinite_or_non_native_global_totals_failclosed_and_json_serializable(bad):
    current = fm.WallObserver(0).report()
    prior = dict(current, geometric_wall_mm=bad)
    comparison = fp.compare_material_totals(current, prior)
    assert not comparison["equal"]
    json.dumps(comparison, allow_nan=False)


def test_global_category_missing_or_mismatched_counts_failclosed():
    current = fm.WallObserver(0).report()
    for field, value in (("rows", 1), ("wall_mm", 2e-7), ("lanes", .0)):
        prior = dict(current, finite_chord_boundary=dict(current["finite_chord_boundary"]))
        prior["finite_chord_boundary"][field] = value
        assert not fp.compare_material_totals(current, prior)["equal"]
    prior = dict(current, finite_chord_boundary={})
    assert not fp.compare_material_totals(current, prior)["equal"]


def test_map_binary_required_and_output_must_be_exclusive(tmp_path):
    old, input_, gdml, map_, report, _ = prior_fixture(tmp_path)
    map_.with_suffix(".bin").unlink()
    with pytest.raises(ValueError, match="adjacent"):
        fp.run(old, input_, gdml, map_, report, tmp_path / "audit", entry="first")
    with pytest.raises(ValueError, match="exclusive"):
        fp.run(old, input_, gdml, map_, report, old / "audit", entry="first")


def test_permuted_active_lanes_attribute_original_sp_and_use_each_current_pa():
    obs, g = fp.ParticleWallObserver(0), geom()
    den = start()
    obs.begin(den)
    args = tuple(np.repeat(v, 2, axis=0) for v in rays((-2., 1.1, 0.), (2., 1.1, 0.)))
    current_p = np.array([30., 900.])
    masses = np.full(2, fs.eloss.MMU)
    obs.observe(np.array([2, 0]), current_p, masses, *args, g, 1.2, None)
    d = obs.ledger
    rate = (.75 * fs.eloss.dedx_restricted(replace(fs.materials.MYLAR), current_p, 1e9, masses)
            + .25 * fs.eloss.dedx_restricted(replace(fs.materials.ALUMINIUM_6061), current_p, 1e9, masses))
    np.testing.assert_allclose(d.loc[[2, 0], "mean_missing_collision_loss_MeV"],
                               d.loc[[2, 0], "missing_wall_mm"].to_numpy() * rate)
    assert d.loc[2, "pentry_MeV_c"] == pytest.approx(np.linalg.norm([150., 1., 1.]))
    assert d.loc[2, "sp"] == den.loc[2, "sp"]
    np.testing.assert_array_equal(d.nactualchords, [1, 0, 1, 0])
    assert obs.report()["consistent"]


def test_zero_chord_start_momentum_zero_retained_without_division():
    den = start()
    den[["px", "py", "pz"]] = 0.
    obs = fp.ParticleWallObserver(0)
    obs.begin(den)
    assert obs.report()["consistent"] and obs.report()["zero_chord_entries"] == len(den)
    np.testing.assert_array_equal(obs.ledger.pentry_MeV_c, np.zeros(len(den)))


def test_bad_entry_count_retains_failed_evidence_and_ledger(tmp_path, monkeypatch):
    old, input_, gdml, map_, report, prior = prior_fixture(tmp_path)
    original = json.loads((old / "fieldcompare.json").read_text())
    original["entries"] = 43156
    (old / "fieldcompare.json").write_text(json.dumps(original))
    receipts = prior["hashes_before"]
    receipts["original/fieldcompare.json"]["sha256"] = fm.pa.sha256(old / "fieldcompare.json")
    report.write_text(json.dumps(prior))
    monkeypatch.setattr(fs, "transport", synthetic_transport)

    def replay(inputs, output, **kw):
        fs.transport(np.random.default_rng(19), start(), geom(), None, gas=fs.STRAW_GAS_G4STATE, radiative=True)
        write_arms(Path(output))

    monkeypatch.setattr(fp.fieldcompare, "run", replay)
    out = tmp_path / "audit"
    result = fp.run(old, input_, gdml, map_, report, out, entry="first")
    assert not result["passed"] and "denominator count" in result["error"]
    assert (out / "FAILED").exists() and (out / "particles.csv").exists()


def test_prior_code_snapshot_relocation_requires_identical_bytes_but_data_paths_exact(tmp_path):
    old, input_, gdml, map_, report, prior = prior_fixture(tmp_path)
    paths = {name: Path(receipt["path"]) for name, receipt in prior["hashes_before"].items()}
    paths["code/observer"] = Path(fp.__file__)
    paths["material_report"] = report
    original = Path(prior["hashes_before"]["code/fieldsteps"]["path"])
    relocated = tmp_path / "next_snapshot" / original.name
    relocated.parent.mkdir()
    relocated.write_bytes(original.read_bytes())
    paths["code/fieldsteps"] = relocated
    fp._prior_evidence(prior, paths.copy(), old.resolve())
    relocated.write_bytes(original.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="path/hash mismatch"):
        fp._prior_evidence(prior, paths.copy(), old.resolve())
    paths["code/fieldsteps"] = original
    other_data = tmp_path / "same_bytes_different_input.root"
    other_data.write_bytes(input_.read_bytes())
    paths["input_table"] = other_data
    with pytest.raises(ValueError, match="path/hash mismatch"):
        fp._prior_evidence(prior, paths.copy(), old.resolve())
