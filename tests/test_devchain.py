"""In-memory chain: the step and digi handoffs equal their table round trips; devchain with NumPy backends is the
tracker_point chain; backend validation and digests."""
import json

import numpy as np
import pandas as pd
import pytest

from trkfastsim.digi import fastdigi
from trkfastsim.reco import hitreco, tracktable
from test_digi import cond, _step  # noqa: F401


def _frame(cond, events=(3, 1, 2, 2), straws=(0, 1, 2, 30)):
    rows = []
    for i, (event, s) in enumerate(zip(events, straws)):
        row = _step(cond, 5 * 96 + 40 + s, t=900.0 + 25 * i)
        row.update(run=1400, subrun=11, event=event, plane=0, panel=5, straw=40 + s, px=0., py=0., pz=-2000., sp=1)
        rows.append(row)
    return pd.DataFrame(rows, columns=fastdigi.STEP_COLS)


def _keys():
    return np.array([[1400, 11, e] for e in (1, 2, 3, 4)], np.int64)


@pytest.mark.parametrize("events", [(3, 1, 2, 2), ()])
def test_digitize_steps_equals_table_round_trip(cond, tmp_path, events):
    from trkfastsim.kit import kitpoint
    frame = _frame(cond) if events else _frame(cond).iloc[:0]
    path = tmp_path / "steps.root"
    kitpoint.write_step_table(path, frame)
    k1, d1, n1 = fastdigi.digitize_table(str(path), cond, np.random.default_rng(5), _keys(), chunk=2)
    k2, d2, n2 = fastdigi.digitize_steps(frame, cond, np.random.default_rng(5), _keys(), chunk=2)
    assert n1 == n2 and np.array_equal(k1, k2) and d1.keys() == d2.keys()
    for k in d1:
        assert d1[k].dtype == d2[k].dtype and np.array_equal(d1[k], d2[k]), k


@pytest.mark.parametrize("events", [(3, 1, 2, 2), ()])
def test_digis_and_events_as_read_equal_the_file(cond, tmp_path, events):
    frame = _frame(cond) if events else _frame(cond).iloc[:0]
    keys, d, _ = fastdigi.digitize_steps(frame, cond, np.random.default_rng(7), _keys(), chunk=1000)
    path = tmp_path / "digis.root"
    fastdigi.write_digis(str(path), keys, d, cond.nADCSamples)
    got, ref = fastdigi.digis_as_read(keys, d, cond.nADCSamples), hitreco.read_digis([str(path)])
    assert got.keys() == ref.keys()
    for k in ref:
        assert got[k].dtype == ref[k].dtype and np.array_equal(got[k], ref[k]), k
    ge, re = fastdigi.events_as_read(keys), tracktable._events(str(path))
    assert ge.keys() == re.keys()
    for k in re:
        assert ge[k].dtype == re[k].dtype and np.array_equal(ge[k], re[k]), k


def test_backend_choices_and_digest():
    from trkfastsim.kit import devchain
    assert devchain.check_backends(None) == devchain.NUMPY
    assert devchain.check_backends({"digi": "jax"})["digi"] == "jax"
    for bad in ({"digi": "gpu"}, {"fit": "numpy"}):
        with pytest.raises(ValueError, match="backends"):
            devchain.check_backends(bad)
    a = {"x": np.arange(3), "y": np.zeros(2, np.float32)}
    assert devchain.digest(a) == devchain.digest(dict(reversed(list(a.items()))))
    assert devchain.digest(a) != devchain.digest({"x": np.arange(3).astype(np.int32), "y": a["y"]})


@pytest.mark.parametrize("nstep", [0, 1])
def test_numpy_chain_is_the_tracker_point_chain(cond, tmp_path, monkeypatch, nstep):
    pytest.importorskip("jax")
    from trkfastsim.kit import devchain, kitpoint
    from trkfastsim.reco.cli import _drift_table
    from mu2efastsim import tables
    from test_kitpoint import _sgs, inputs
    inp = inputs.__wrapped__(tmp_path, cond)
    sgs = _sgs(cond, nstep)
    monkeypatch.setattr(kitpoint, "simulate", lambda g, gdml, seed: (sgs, g[g.pdg.abs() == 13]))
    monkeypatch.setattr(devchain.steps_mod, "make_steps", lambda g, *a, **k: (sgs, g[g.pdg.abs() == 13]))
    m = kitpoint.run(**inp, outdir=tmp_path / "point")
    ctx = object.__new__(devchain.Context)
    ctx.dc = fastdigi.digicond.Conditions.load(str(inp["digi_cond"]))
    ctx.rc = hitreco.load_conditions(str(inp["reco_cond"]))
    ctx.drift = kitpoint.validate_drift(_drift_table(str(inp["drift_source"])))
    ctx.geom = ctx.world = ctx.interior = None
    r = devchain.run(tables.read(str(inp["gen"]), "gen"), ctx, 1)
    c = r["counts"]
    assert (c["generated_events"], c["primary_muons"], c["steps"], c["digis"], c["hits"], c["seeds"]) == \
        (m["generated_events"], m["primary_muons"], m["straw_steps"], m["digis"], m["hits"], m["seeds"])
    assert c["events_with_fit"] == m["events_with_fit"] and c["good_fits"] == m["fit_successes"]
    assert set(r["hashes"]) == {"steps", "digis", "hits", "combos", "clusters", "seeds", "table"}
    assert devchain.run(tables.read(str(inp["gen"]), "gen"), ctx, 1)["hashes"] == r["hashes"]
    json.dumps(r["counts"]), json.dumps(r["timing"])


def _report(path, counts, hists, seed=1):
    shards = [{"gen": f"g{i}", "counts": {"generated_events": 10, **c}, "histograms": h,
               "hashes": {"steps": str(c["digis"])}} for i, (c, h) in enumerate(zip(counts, hists))]
    path.write_text(json.dumps({"backends": {}, "seed": seed, "shards": shards}))
    return str(path)


def test_chain_pair_paired_errors_and_null(tmp_path):
    from trkfastsim.validation import chain_pair
    rng = np.random.default_rng(3)
    base = rng.poisson(100, (8, 5)).astype(int)
    a = _report(tmp_path / "a.json", [{"digis": int(x.sum())} for x in base], [{"h": x.tolist()} for x in base])
    shift = base + np.array([2, -2, 2, -2, 2, -2, 2, -2])[:, None]
    b = _report(tmp_path / "b.json", [{"digis": int(x.sum())} for x in shift], [{"h": x.tolist()} for x in shift])
    r = chain_pair.compare(a, b, null=a)
    d = shift.sum(1) - base.sum(1)
    se = np.sqrt(8 / 7 * ((d - d.mean()) ** 2).sum()) / base.sum()
    assert np.isclose(r["counts"]["digis"]["rel"], 0.0) and np.isclose(r["counts"]["digis"]["rel_se"], se)
    assert r["null"]["hists"]["h"]["identical"] and r["null"]["counts"]["digis"]["identical_shards"] == 8
    assert r["stage_hash_identical"]["steps"] == 0 and r["hists"]["h"]["ndf"] == 5
    assert np.isclose(r["hists"]["h"]["expected_chi2_ndf"], 7 / 5)
    with pytest.raises(ValueError, match="shards"):
        chain_pair.compare(a, _report(tmp_path / "c.json", [{"digis": 1}] * 2, [{"h": [1]}] * 2))


def test_compact_digitisation_keeps_the_event_identities(cond):
    keys = np.array([[1400, 11, e] for e in range(1, 2001)], np.int64)
    frame = _frame(cond, events=(3, 1500, 2, 2), straws=(0, 1, 2, 30))
    _, d, n = fastdigi.digitize_steps(frame, cond, np.random.default_rng(5), keys, chunk=1000, compact=True)
    _, p, _ = fastdigi.digitize_steps(frame, cond, np.random.default_rng(5), keys, chunk=1000)
    assert n == 4 and set(d["ev"]) <= {0, 1, 2, 1499} and set(d["ev"]) == set(p["ev"])
    assert np.all(np.diff(d["ev"]) >= 0) and d["ev"].dtype == p["ev"].dtype
    assert {k: v.dtype for k, v in d.items()} == {k: v.dtype for k, v in p.items()}
    _, e, _ = fastdigi.digitize_steps(frame.iloc[:0], cond, np.random.default_rng(5), keys, compact=True)
    assert len(e["ev"]) == 0


def test_shape_growth_buckets(monkeypatch):
    sj = pytest.importorskip("trkfastsim.sim.steps_jax")
    assert [sj._bucket(n, 64) for n in (0, 1, 64, 65, 300)] == [64, 64, 64, 128, 512]
    monkeypatch.setattr(sj, "SHAPE_GROWTH", 8)
    assert [sj._bucket(n, 64) for n in (1, 64, 65, 513, 5000)] == [64, 64, 512, 4096, 32768]
    assert sj._bucket(600, 32, 4097) == 2048 and sj._bucket(5000, 32, 4097) == 8192
    for n in range(1, 3000, 37):
        assert sj._bucket(n, 32, 3000) >= n
