"""Tracker point orchestration, real empty digi/reco/writer and fail-closed populations."""
import json

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from mu2efastsim import tables
from trkfastsim import kitpoint
from trkfastsim.analyses import fastdigi
from trkfastsim.reco import hitreco, ntuple, tracktable
from test_digi import cond, _step
from test_reco_jax import _cond
from test_ntuple_writer import _table, VERT


def _json_arrays(value):
    if isinstance(value, dict):
        return {k: _json_arrays(v) for k, v in value.items()}
    return value.tolist() if isinstance(value, np.ndarray) else value


@pytest.fixture
def inputs(tmp_path, cond):
    gen = pd.DataFrame({"run": [1400] * 4, "subrun": [11] * 4, "event": [1, 1, 2, 3],
                        "pdg": [13, -13, 11, 13]})
    gp = tmp_path / "gen.root"
    tables.write(gp, {"gen": gen}, as_ttree=True)
    gdml = tmp_path / "geometry.gdml"
    gdml.write_text("synthetic geometry replaced by fixture transport")
    dc = tmp_path / "dc.npz"
    cond.save(dc)
    rc = tmp_path / "rc.json"
    rc.write_text(json.dumps(_json_arrays(_cond(20736, np.random.default_rng(2)))))
    dp = tmp_path / "drift.root"
    with uproot.recreate(dp) as f:
        f.mktree("trkdump/conditions", {k: np.array([v]) for k, v in
            {"dtgrid": [0., 50.], "rdrift": [0., 2.5], "dvel": [.05, .05], "sderr": [.1, .1], "uderr": [.2, .2]}.items()})
    return dict(gen=gp, gdml=gdml, digi_cond=dc, reco_cond=rc, drift_source=dp, seed=1)


def _sgs(cond, n):
    row = _step(cond, 5 * 96 + 40)
    row.update(run=1400, subrun=11, event=1, plane=0, panel=5, straw=40, px=0., py=0., pz=-2000., sp=1)
    return pd.DataFrame({k: np.repeat(row[k], n) for k in fastdigi.STEP_COLS})


@pytest.mark.parametrize("nstep", [0, 1])
def test_point_keeps_zero_hit_events_and_writes_real_empty_f2(inputs, cond, tmp_path, monkeypatch, nstep):
    monkeypatch.setattr(kitpoint, "simulate", lambda g, gdml, seed: (_sgs(cond, nstep), g[g.pdg.abs() == 13]))
    out = tmp_path / "out"
    m = kitpoint.run(**inputs, outdir=out, retain_ntuple=True)
    assert m["generated_events"] == m["written_events"] == 3
    assert m["primary_muons"] == 3
    assert m["events_with_steps"] == nstep
    assert m["hits"] == m["digis"] == nstep
    assert m["fit_attempts"] == m["fit_successes"] == 0
    with uproot.open(out / "EventNtuple.root") as f:
        t = f[ntuple.TREE]
        assert np.array_equal(t["evtinfo/event"].array(library="np"), [1, 2, 3])
        assert int(ak.sum(ak.num(t["trk"].array()["trk.status"]))) == 0
    p = json.loads((out / "provenance.json").read_text())
    assert not p["writer"]["stock_conditioned"]
    assert p["product"]["retained"] and p["writer"]["fit_type"] == "F2"
    with pytest.raises(ValueError, match="overwrite"):
        kitpoint.run(**inputs, outdir=out)


def test_writer_is_executed_but_not_retained_by_default(inputs, cond, tmp_path, monkeypatch):
    monkeypatch.setattr(kitpoint, "simulate", lambda *a: (_sgs(cond, 0), pd.DataFrame()))
    out, work = tmp_path / "out", tmp_path / "scratch"
    kitpoint.run(**inputs, outdir=out, workdir=work)
    assert {p.name for p in out.iterdir()} == {"metrics.json", "provenance.json"}
    assert list(work.iterdir()) == []
    p = json.loads((out / "provenance.json").read_text())
    assert not p["product"]["retained"] and p["product"]["bytes"] > 0


def test_stage_failure_never_publishes_metrics(inputs, tmp_path, monkeypatch):
    def fail(*a):
        raise RuntimeError("transport failure")
    monkeypatch.setattr(kitpoint, "simulate", fail)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="transport failure"):
        kitpoint.run(**inputs, outdir=out, workdir=tmp_path / "scratch")
    assert not (out / "metrics.json").exists()
    assert list((tmp_path / "scratch").iterdir()) == []


def test_unknown_step_event_refused(inputs, cond, tmp_path, monkeypatch):
    s = _sgs(cond, 1)
    s["event"] = 99
    monkeypatch.setattr(kitpoint, "simulate", lambda *a: (s, pd.DataFrame()))
    with pytest.raises(ValueError, match="absent"):
        kitpoint.run(**inputs, outdir=tmp_path / "out")


def test_generated_population_and_identity_domains():
    g = pd.DataFrame({"run": [1400, 1400], "subrun": [11, 11], "event": [4, 4], "pdg": [13, -13]})
    assert kitpoint.generated_keys(g).shape == (1, 3)
    with pytest.raises(ValueError, match="empty"):
        kitpoint.generated_keys(g.iloc[:0])
    for bad in (-1, 2 ** 20, 1.5, np.nan):
        with pytest.raises(ValueError, match="keys"):
            kitpoint.generated_keys(g.assign(event=bad))


def test_peak_minus_ped_empty_contract():
    assert hitreco.peak_minus_ped(np.zeros(0, np.int64), np.zeros(0, np.int64), 5).shape == (0,)


def test_persisted_writer_census_multitrack_and_failed(tmp_path):
    tb = _table([VERT] * 3, ev=[0, 0, 2], good=[1, 1, 0])
    p = tmp_path / "tracks.root"
    ntuple.write(tb, p)
    keys = np.column_stack([tb["events"][k] for k in tables.KEY])
    m = kitpoint.inspect_written(p, keys, 2)
    assert m == dict(written_events=3, events_with_fit=1, fit_successes=2, active_hits=18)
    with pytest.raises(ValueError, match="track count"):
        kitpoint.inspect_written(p, keys, 3)
    with pytest.raises(ValueError, match="identities"):
        kitpoint.inspect_written(p, keys[::-1], 2)


@pytest.mark.parametrize("key,value", [("dt", [1., 0.]), ("sderr", [np.nan, 1.]), ("uderr", [-1., 1.])])
def test_invalid_drift_refused(key, value):
    d = {"dt": [0., 1.], "rdrift": [0., 1.], "dvel": [1., 1.], "sderr": [1., 1.], "uderr": [1., 1.]}
    d[key] = value
    with pytest.raises(ValueError):
        kitpoint.validate_drift(d)


def test_input_replacement_before_read_fails_publication(inputs, cond, tmp_path, monkeypatch):
    original = kitpoint.file_identity
    replaced = False
    def replace_after_hash(path):
        nonlocal replaced
        identity = original(path)
        if path == inputs["gen"] and not replaced:
            replaced = True
            tables.write(path, {"gen": pd.DataFrame({"run": [1400], "subrun": [11], "event": [1], "pdg": [13]})},
                         as_ttree=True)
        return identity
    monkeypatch.setattr(kitpoint, "file_identity", replace_after_hash)
    monkeypatch.setattr(kitpoint, "simulate", lambda *a: (_sgs(cond, 0), pd.DataFrame()))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="changed during consumption"):
        kitpoint.run(**inputs, outdir=out)
    assert not (out / "metrics.json").exists() and not (out / "provenance.json").exists()


def test_stock_negative_time_radius_extrapolation_preserved():
    d = {"dt": [-20., 0., 50.], "rdrift": [-.0223167, 0., 2.5], "dvel": [.00137, .05, .05],
         "sderr": [.16] * 3, "uderr": [.12] * 3}
    got = kitpoint.validate_drift(d)
    assert np.array_equal(got["rdrift"], d["rdrift"])
