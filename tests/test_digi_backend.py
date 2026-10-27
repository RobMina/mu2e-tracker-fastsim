"""Opt-in digi routing, explicit keys, unchanged NumPy streams and local point provenance."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trkfastsim import digi as DG
from trkfastsim.analyses import digicmp, fastdigi
from test_digi import cond, _step


def _frame(cond, events=(3, 99, 1, 2)):
    rows = []
    for i, event in enumerate(events):
        row = _step(cond, 5 * 96 + 40 + i)
        row.update(run=1400, subrun=11, event=event, plane=0, panel=5, straw=40 + i,
                   px=0., py=0., pz=-2000., sp=1)
        rows.append(row)
    return pd.DataFrame(rows, columns=fastdigi.STEP_COLS)


def _keys():
    return np.array([[1400, 11, e] for e in (3, 1, 2, 4, 5)], np.int64)


@pytest.mark.parametrize("backend", ["gpu", "JAX", "", None])
def test_invalid_backend_fails_before_io(backend, tmp_path):
    from trkfastsim import kitpoint
    with pytest.raises(ValueError, match="backend"):
        fastdigi.digitize_table("missing.root", None, None, backend=backend)
    with pytest.raises(ValueError, match="backend"):
        fastdigi.run_digi([], str(tmp_path / "digi"), backend=backend)
    with pytest.raises(ValueError, match="backend"):
        kitpoint.run(None, None, None, None, None, 1, tmp_path / "point", backend=backend)
    assert not list(tmp_path.iterdir())


def test_numpy_default_preserves_reference_rng_and_arrays(cond, monkeypatch):
    frame, keys = _frame(cond), _keys()
    monkeypatch.setattr(fastdigi, "load_steps", lambda p: frame)
    rng = np.random.default_rng(19)
    got_keys, got, nstep = fastdigi.digitize_table("unused", cond, rng, keys, chunk=2)
    ref_rng = np.random.default_rng(19)
    parts = []
    for e0, rows, ev in ((0, [0, 2], [0, 1]), (2, [3], [0]), (4, [], [])):
        sub = frame.iloc[rows]
        st = {k: sub[k].to_numpy() for k in ("shape", "ion", "eion", "len", "width", "t",
                                             "x0", "y0", "z0", "x1", "y1", "z1")}
        st.update(sid=fastdigi.geometry_sid(sub), ev=np.asarray(ev, np.int64))
        d = DG.digitize(st, min(2, len(keys) - e0), cond, ref_rng)
        d["ev"] += e0
        d["step"] = np.asarray(rows, np.int64)[d["step"]]
        parts.append(d)
    assert nstep == 3 and np.array_equal(got_keys, keys)
    assert rng.bit_generator.state == ref_rng.bit_generator.state
    for k in got:
        assert np.array_equal(got[k], np.concatenate([d[k] for d in parts]))
    _, explicit, _ = fastdigi.digitize_table("unused", cond, np.random.default_rng(19), keys, chunk=2, backend="numpy")
    assert all(np.array_equal(got[k], explicit[k]) for k in got)


def test_jax_chunk_keys_and_truth_routing(cond, monkeypatch):
    jax = pytest.importorskip("jax")
    calls, stats = [], {}
    monkeypatch.setattr(fastdigi, "load_steps", lambda p: _frame(cond))
    def fake(st, nev, c, key, counters):
        assert c is cond and counters is stats
        calls.append((nev, jax.random.key_data(key), st["ev"].copy()))
        counters["calls"] = counters.get("calls", 0) + 1
        n = len(st["ev"])
        return {"ev": st["ev"].copy(), "sid": st["sid"].copy(), "step": np.repeat(np.arange(n)[:, None], 2, axis=1)}
    monkeypatch.setattr(fastdigi, "digitize_jax", fake)
    root = jax.random.key(17)
    keys, d, nstep = fastdigi.digitize_table("unused", cond, root, _keys(), chunk=2, stats=stats, backend="jax")
    assert nstep == 3 and np.array_equal(keys, _keys()) and stats["calls"] == 3
    assert np.array_equal(d["ev"], [0, 1, 2])
    assert np.array_equal(d["step"], [[0, 0], [2, 2], [3, 3]])
    for (nev, key, ev), offset in zip(calls, (0, 2, 4)):
        assert nev == min(2, 5 - offset)
        assert np.array_equal(key, jax.random.key_data(jax.random.fold_in(root, offset)))
        assert np.all((ev >= 0) & (ev < nev))
    assert not np.array_equal(calls[0][1], calls[1][1])


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_run_digi_file_streams(backend, tmp_path, monkeypatch):
    jax = pytest.importorskip("jax") if backend == "jax" else None
    calls = []
    monkeypatch.setattr(fastdigi.digicond.Conditions, "load", lambda p: SimpleNamespace(nADCSamples=15))
    monkeypatch.setattr(fastdigi.tables, "expand", lambda p: ["first.root", "second.root"])
    monkeypatch.setattr(fastdigi, "write_digis", lambda *a: None)
    def fake(path, c, rng, keys, chunk, stats, backend):
        calls.append((path, rng, backend, chunk))
        return _keys(), {"sid": np.zeros(0)}, 0
    monkeypatch.setattr(fastdigi, "digitize_table", fake)
    fastdigi.run_digi(["unused"], str(tmp_path), cond="unused", seed=13, chunk=7, backend=backend)
    assert [c[0] for c in calls] == ["first.root", "second.root"]
    for i, (_, rng, selected, chunk) in enumerate(calls):
        assert selected == backend and chunk == 7
        if backend == "numpy":
            assert rng.bit_generator.state == np.random.default_rng([13, i]).bit_generator.state
        else:
            assert np.array_equal(jax.random.key_data(rng), jax.random.key_data(jax.random.fold_in(jax.random.key(13), i)))


def test_numpy_import_and_execution_without_jax(cond, tmp_path):
    from trkfastsim import kitpoint
    cpath, spath = tmp_path / "cond.npz", tmp_path / "steps.root"
    cond.save(cpath)
    kitpoint.write_step_table(spath, _frame(cond, (1,)))
    code = '''
import importlib.abc
import sys
class NoJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'trkfastsim.digi_jax':
            raise AssertionError('unexpected digitizer import: ' + fullname)
        if fullname.split('.')[0] in ('jax', 'jaxlib'):
            raise ModuleNotFoundError('JAX intentionally unavailable', name=fullname)
sys.meta_path.insert(0, NoJax())
import numpy as np
from trkfastsim import cli, digicond, kitpoint
from trkfastsim.analyses import fastdigi
assert cli.build_parser().parse_args(['digi', 'steps.root']).backend == 'numpy'
c = digicond.Conditions.load(sys.argv[1])
keys, d, n = fastdigi.digitize_table(sys.argv[2], c, np.random.default_rng(3), keys=np.array([[1400, 11, 1]]))
assert n == 1 and len(d['sid']) == 1
assert not any(k == 'jax' or k.startswith('jax.') or k == 'trkfastsim.digi_jax' for k in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", code, str(cpath), str(spath)], text=True, capture_output=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=60)
    assert result.returncode == 0, result.stderr


def test_cli_backend_and_drift_source(monkeypatch):
    from trkfastsim import cli, kitpoint
    parser = cli.build_parser()
    assert parser.parse_args(["digi", "steps.root"]).backend == "numpy"
    assert parser.parse_args(["digi", "steps.root", "--backend", "jax"]).backend == "jax"
    with pytest.raises(SystemExit):
        parser.parse_args(["digi", "steps.root", "--backend", "gpu"])
    args = parser.parse_args(["digicmp", "jax.root", "--ref", "numpy.root", "--f2", "--drift-source", "drift.root"])
    assert args.drift_source == "drift.root" and args.f2
    assert parser.parse_args(["digicmp", "fast.root", "--ref", "ref.root"]).drift_source is None
    calls = []
    monkeypatch.setattr(cli, "ANALYSES", {**cli.ANALYSES, "digi": (lambda *a, **k: calls.append(k), "digi")})
    cli.main(["digi", "steps.root", "--backend", "jax"])
    assert calls[-1]["backend"] == "jax"
    monkeypatch.setattr(kitpoint, "run", lambda **kw: kw)
    flags = [a for name in ("gen", "gdml", "digi-cond", "reco-cond", "drift-source", "outdir") for a in ("--" + name, name)]
    assert kitpoint.main(flags)["backend"] == "numpy"
    assert kitpoint.main(flags + ["--backend", "jax"])["backend"] == "jax"
    with pytest.raises(SystemExit):
        kitpoint.main(flags + ["--backend", "gpu"])


@pytest.mark.parametrize("source", [None, "separate-drift.root"])
def test_digicmp_shares_separate_drift_for_every_arm(source, tmp_path, monkeypatch):
    keys = _keys()[:2]
    monkeypatch.setattr(digicmp.hitreco, "load_conditions", lambda p: {})
    monkeypatch.setattr(digicmp.hitreco, "read_digis", lambda p: {"evkey": keys})
    monkeypatch.setattr(digicmp, "_restrict", lambda d, keep: d)
    monkeypatch.setattr(digicmp, "HIST", {})
    monkeypatch.setattr(digicmp, "observables", lambda *a: {k: np.ones(2, int) for k in
                        ("sid", "ndigi", "nsh", "nph", "tot0", "tot1", "pmp", "dt", "wdist")})
    monkeypatch.setattr(digicmp, "paired_single", lambda *a: {})
    seen = []
    def fake(d, cond, drift_ref):
        seen.append(drift_ref)
        return {"seeds": 2, "good_fits": 2, "events_with_fit": 2, "fits_per_event": np.ones(2, int)}
    monkeypatch.setattr(digicmp, "f2_yield", fake)
    digicmp.run(["jax.root"], str(tmp_path), ref=["numpy.root"], floor=["floor.root"], hit_cond="rc.json",
                f2=True, drift_source=source)
    assert seen == [source or "numpy.root"] * 3


@pytest.fixture
def point_inputs(tmp_path, cond):
    pytest.importorskip("jax")
    from test_kitpoint import inputs
    return inputs.__wrapped__(tmp_path, cond)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_point_backend_key_provenance_and_numpy_transport(point_inputs, cond, tmp_path, monkeypatch, backend):
    from trkfastsim import kitpoint
    from test_kitpoint import _sgs
    jax = pytest.importorskip("jax")
    monkeypatch.setattr(kitpoint, "simulate", lambda g, gdml, seed: (_sgs(cond, 0), g[g.pdg.abs() == 13]))
    real_table, calls = fastdigi.digitize_table, []
    def fake(path, c, rng, **kw):
        calls.append((rng, kw.pop("backend")))
        return real_table(path, c, np.random.default_rng(1), **kw)
    monkeypatch.setattr(fastdigi, "digitize_table", fake)
    for i in range(2):
        out = tmp_path / f"out{i}"
        metrics = kitpoint.run(**point_inputs, outdir=out, backend=backend)
        prov = json.loads((out / "provenance.json").read_text())
        assert prov["configuration"]["digitizer"] == backend
        assert prov["configuration"]["step_backend"] == "numpy"
        assert metrics["generated_events"] == metrics["written_events"] == 3 and metrics["digis"] == 0
    for rng, selected in calls:
        assert selected == backend
        if backend == "numpy":
            assert rng.bit_generator.state == np.random.default_rng([1, 0, 3]).bit_generator.state
        else:
            expected = jax.random.fold_in(jax.random.fold_in(jax.random.key(1), 0), 3)
            assert np.array_equal(jax.random.key_data(rng), jax.random.key_data(expected))
            assert not np.array_equal(jax.random.key_data(rng), jax.random.key_data(jax.random.fold_in(jax.random.key(1), 0)))


@pytest.mark.parametrize("nstep", [0, 1])
def test_real_jax_empty_point(point_inputs, cond, tmp_path, monkeypatch, nstep):
    pytest.importorskip("trkfastsim.digi_jax")
    from trkfastsim import kitpoint
    from test_kitpoint import _sgs
    monkeypatch.setattr(kitpoint, "simulate", lambda g, gdml, seed: (_sgs(cond, nstep), g[g.pdg.abs() == 13]))
    out = tmp_path / "jax-point"
    m = kitpoint.run(**point_inputs, outdir=out, backend="jax", retain_ntuple=True)
    assert m["generated_events"] == m["written_events"] == 3
    assert m["digis"] == m["hits"] == nstep and m["fit_successes"] == 0
    p = json.loads((out / "provenance.json").read_text())
    assert p["configuration"]["digitizer"] == "jax" and p["configuration"]["step_backend"] == "numpy"


def _bench():
    path = Path(__file__).resolve().parents[1] / "scripts" / "digi_bench.py"
    spec = importlib.util.spec_from_file_location("digi_bench_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bench_defaults_same_seed_and_json(tmp_path, monkeypatch):
    bench, seeds = _bench(), []
    monkeypatch.setattr(bench, "conditions", lambda: None)
    monkeypatch.setattr(bench, "events", lambda c, nev, rng: {"sid": np.zeros(nev, int)})
    def fake(st, nev, c, rng):
        seeds.append(rng.bit_generator.state)
        return {"sid": rng.integers(0, 100, nev)}
    monkeypatch.setattr(bench.DG, "digitize", fake)
    report = bench.main([])
    assert report["events"] == 1000 and report["repeats"] == 3 and report["backend"] == "numpy"
    assert report["reproducible"] and seeds[0] == seeds[1] == seeds[2]
    assert len(report["warm_s"]) == 2 and report["runs"][0]["phase"] == "cold"
    output = tmp_path / "bench.json"
    got = bench.main(["2", "1", "--seed", "9", "--json", str(output)])
    assert json.loads(output.read_text()) == got
    assert got["warm_mean_s"] is None and got["seed"] == 9 and got["device"] == "cpu (numpy)"
    with pytest.raises(SystemExit):
        bench.main(["0"])
