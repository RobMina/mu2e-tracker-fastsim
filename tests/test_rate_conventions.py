"""Synthetic identity and event-count regressions; no ROOT files or transport runs."""
import numpy as np
import pandas as pd
import pytest

from mu2efastsim import eloss
from mu2efastsim.tables import KEY, all_primary_muons
from trkfastsim import geometry, tables as T
from trkfastsim.analyses import fatecheck as fc, seedcheck as sc


def _steps(counts):
    rows = []
    for event, sp, n in counts:
        for i in range(n):
            rows.append({"run": 1, "subrun": 1, "event": event, "sp": sp, "eion": 0.01,
                         "px": 0.0, "py": -100.0, "pz": 0.0, "t": float(i),
                         "x0": float(sp), "y0": 0.0, "z0": 0.0})
    return pd.DataFrame(rows, columns=KEY + ["sp", "eion", "px", "py", "pz", "t", "x0", "y0", "z0"])


def _read_tables(monkeypatch, tables):
    def read_many(paths, tree, columns=None):
        df = tables[tree]
        return df.copy() if columns is None else df[columns].copy()
    monkeypatch.setattr(T, "read_many", read_many)
    monkeypatch.setattr(T, "expand", lambda paths: paths)


@pytest.fixture
def fast_inputs(monkeypatch):
    rows = []
    for event in range(1, 8):
        for idx, pdg in ((0, 2212), (2, 13), (5, -13)):
            rows.append({"run": 1, "subrun": 1, "event": event, "idx": idx, "pdg": pdg,
                         "px": 0.0, "py": -(100.0 * event + idx), "pz": 0.0})
    gen = pd.DataFrame(rows[::-1])
    mu = all_primary_muons(gen)
    alive = np.array([False, True, True, True, True, False, False, True, False, False, True, True, True, True])
    lp = {"key": mu[KEY].to_numpy(), "alive": alive, "p_plane": np.arange(len(mu)) + 100.0,
          "o": np.zeros((len(mu), 3)), "d": np.tile([0.0, -1.0, 0.0], (len(mu), 1))}
    s = _steps([(1, 2, 12), (2, 1, 6), (2, 2, 6), (3, 1, 12), (5, 1000, 20),
                (6, 1, 11), (7, 1, 12), (7, 2, 13)])
    bad = _steps([(6, 1, 1), (6, 2, 1), (6, 2, 1)])
    bad.loc[0, "eion"] = 0.0
    bad.loc[1, "py"] = -T.FILTER_MIN_MOM
    bad.loc[2, "py"] = -T.FILTER_MAX_MOM
    s = pd.concat([s, bad], ignore_index=True)
    monkeypatch.setattr(fc, "_miss_distance", lambda o, d, geom, y: np.arange(len(o), dtype=float))
    return gen, s, lp


def test_fatecheck_any_muon_reduces_once_after_identity_match(fast_inputs):
    gen, s, lp = fast_inputs
    ev = fc._fast_events(gen, s, lp, None, 0.0).set_index("event")
    assert len(ev) == 7
    assert ev.fast_pass.tolist() == [True, False, True, False, False, False, True]
    assert ev.fast_first_pass.tolist() == [False, False, True, False, False, False, True]
    assert ev.ngood.tolist() == [12, 6, 12, 0, 0, 11, 13]
    assert ev.fate.tolist() == ["pass", "few", "pass", "missed", "stopped_above", "few", "pass"]
    assert ev.fast_sp.tolist() == [2, 1, 1, 2, 1, 1, 2]
    assert ev.loc[1, "fate_first"] == "stopped_above"
    assert ev.loc[1, "ngood_first"] == 0
    assert ev.loc[1, "p_plane"] == 101.0
    assert ev.loc[3, "p_plane"] == 104.0
    assert ev.loc[1, "miss_plane_mm"] == 1.0
    assert ev.loc[1, "p_last"] == 100.0
    assert np.isnan(ev.loc[5, "p_last"])


def test_fatecheck_full_event_identity(fast_inputs):
    gen, s, lp = fast_inputs
    gen2 = gen.assign(subrun=2)
    mu = all_primary_muons(pd.concat([gen, gen2], ignore_index=True))
    lp2 = {k: np.concatenate([v, v]) for k, v in lp.items()}
    lp2["key"] = mu[KEY].to_numpy()
    ev = fc._fast_events(pd.concat([gen, gen2]), s, lp2, None, 0.0)
    assert len(ev) == 14 and not ev.duplicated(KEY).any()
    assert ev.loc[ev.subrun == 1, "fast_pass"].sum() == 3
    assert not ev.loc[ev.subrun == 2, "fast_pass"].any()


def test_fatecheck_no_steps_and_empty_muons(fast_inputs):
    gen, s, lp = fast_inputs
    ev = fc._fast_events(gen, s.iloc[:0], lp, None, 0.0)
    assert len(ev) == 7 and not ev.fast_pass.any()
    assert (ev.ngood == 0).all()
    empty_lp = {k: v[:0] for k, v in lp.items()}
    assert fc._fast_events(gen.iloc[:0], s.iloc[:0], empty_lp, None, 0.0).empty


def test_fatecheck_rejects_unmatched_plane_order(fast_inputs):
    gen, s, lp = fast_inputs
    with pytest.raises(ValueError, match="last_plane rows"):
        fc._fast_events(gen, s, {**lp, "key": lp["key"][::-1]}, None, 0.0)


@pytest.fixture
def g4_tables(monkeypatch):
    pa = pd.DataFrame({"run": 1, "subrun": 1, "event": np.repeat([1, 2, 3, 4], 4),
                       "key": [1, 7, 3, 99] * 4, "pdg": [2212, 13, -13, 13] * 4,
                       "ismcprim": [1, 1, 1, 0] * 4, "genidx": 0,
                       "ey": 100.0, "ee": eloss.MMU, "ex": 0.0, "ez": 0.0,
                       "epx": 0.0, "epy": 0.0, "epz": 0.0})
    s = _steps([(1, 7, 2), (1, 3, 12), (2, 3, 12), (3, 7, 6), (3, 3, 6),
                (4, 7, 12), (3, 99, 20), (3, 1, 20)])
    s.loc[s.sp == 3, "px"] = 100.0
    s.loc[s.sp == 3, "py"] = 0.0
    gen = pa[KEY].drop_duplicates()
    tables = {"particles": pa, "sgsteps": s, "gen": gen}
    _read_tables(monkeypatch, tables)
    return tables


def test_g4_any_primary_filter_excludes_secondaries_and_does_not_sum_muons(g4_tables):
    ev = fc._g4(["synthetic"], 0.0)
    assert len(ev) == 4 and not ev.duplicated(KEY).any()
    assert ev.g4_pass.tolist() == [True, True, False, True]


def test_seedcheck_counts_same_simparticle_and_keeps_any_primary(g4_tables):
    pm = sc._primaries(["synthetic"])
    assert pm.sp.tolist() == [7, 7, 7, 7]
    counts = sc._g4_counts(["synthetic"], pm)
    assert not counts.duplicated(KEY).any()
    assert counts.seeded_sp.tolist() == [7, 7, 7, 7]
    assert counts.nmax.tolist() == [2, 0, 6, 12]
    assert counts.pass_trk.tolist() == [False, False, False, True]
    assert counts.ngood_g4_any.tolist() == [12, 12, 6, 12]
    assert counts.g4_any_pass.tolist() == [True, True, False, True]
    other = pm.assign(sp=3)
    assert sc._g4_counts(["synthetic"], other).nmax.tolist() == [12, 12, 6, 0]


def test_seedcheck_chord_envelope_and_first_step_select_identical_primary(g4_tables):
    traj, surf = [], []
    for event in (1, 2):
        for sp in (3, 7):
            if event == 2 and sp == 7:
                continue
            for ipt, y in enumerate((20.0, -20.0)):
                traj.append({"run": 1, "subrun": 1, "event": event, "sp": sp, "ipt": ipt,
                             "x": float(sp), "y": y, "z": 0.0, "ke": float(sp * 100)})
            surf.append({"run": 1, "subrun": 1, "event": event, "sp": sp, "surf": 4, "t": sp,
                         "x0": float(sp), "y0": 0.0, "z0": 0.0, "x1": float(sp), "y1": 0.0,
                         "z1": 0.0, "px": 0.0, "py": -sp * 100.0, "pz": 0.0})
    g4_tables["trajpts"] = pd.DataFrame(traj)
    ev = g4_tables["gen"].assign(iev=np.arange(4))
    pm = sc._primaries(["synthetic"])
    e, o, u, p = sc.chord_seeds(["synthetic"], ev, 0.0)
    assert e.tolist() == [0]
    np.testing.assert_allclose(o, [[7.0, 0.0, 0.0]])
    np.testing.assert_allclose(u, [[0.0, -1.0, 0.0]])
    np.testing.assert_allclose(p, sc._p_of_ke(np.array([700.0])))
    e, o, u, p, sid = sc.envelope_from(pd.DataFrame(surf), pm, ev)
    assert e.tolist() == [0] and sid.tolist() == [4]
    np.testing.assert_allclose(o, geometry.to_mu2e([[7.0, 0.0, 0.0]]))
    np.testing.assert_allclose(p, [700.0])
    v, x = sc._g4_first_dir(["synthetic"], pm, ev)
    np.testing.assert_allclose(v[0], [0.0, -1.0, 0.0])
    np.testing.assert_allclose(x[0], geometry.to_mu2e([7.0, 0.0, 0.0]))
    assert np.isnan(v[1]).all() and np.isnan(x[1]).all()


def _capture_output(monkeypatch, module):
    saved = {}
    monkeypatch.setattr(module.os, "makedirs", lambda *a, **kw: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda df, *a, **kw: saved.update(events=df.copy()))
    monkeypatch.setattr(module.plots, "write_json", lambda outdir, name, out: saved.update(report=out))
    return saved


def test_fatecheck_output_is_event_counted_with_first_muon_diagnostic(monkeypatch, fast_inputs):
    gen, s, lp = fast_inputs
    fa = fc._fast_events(gen, s, lp, None, 0.0)
    g4 = fa[KEY].assign(g4_pass=[True, True, True, False, False, False, True],
                         ey_mu2e=100.0, g4_stopped=True)
    monkeypatch.setattr(fc, "_fast", lambda *args: (fa, None, 0.0))
    monkeypatch.setattr(fc, "_g4", lambda *args: g4)
    _read_tables(monkeypatch, {"gen": gen})
    saved = _capture_output(monkeypatch, fc)
    fc.run(["synthetic"], "unused", compare=["synthetic"], gdml="unused")
    ev, out = saved["events"], saved["report"]
    assert len(ev) == out["generated"] == out["generated_muon_events"] == 7 and not ev.duplicated(KEY).any()
    assert out["generated_events"] == len(gen[KEY].drop_duplicates())
    assert out["fast_pass"] == 3 and out["fast_first_pass"] == 2 and out["g4_pass"] == 4
    assert ev.p_gen.tolist() == [102.0, 202.0, 302.0, 402.0, 502.0, 602.0, 702.0]
    assert ev.p_gen.equals(ev.p_gen_first)
    assert "first" in out["p_gen_attribution"]
    assert sum(r["g4_only"] for r in out["by_p_gen"]) == 1


def test_seedcheck_count_and_fate_tables_use_seeded_not_any(g4_tables):
    pm = sc._primaries(["synthetic"])
    df = sc._g4_counts(["synthetic"], pm).rename(columns={"pass_trk": "g4_pass", "nmax": "ngood_g4"})
    df = df.assign(seed_pass=False, straight_pass=False, n_seed=0, n_straight=0,
                   p_seed=500.0, fast_pmin=np.nan, g4_stop=False)
    tot = sc.count_table(df)["total"]
    assert tot["g4_pass"] == tot["g4_only"] == 1 and tot["g4_any_pass"] == 3
    assert sc.fate_table(df)["n"] == 1


def test_seedcheck_counts_preserve_full_event_key_and_zero_steps(g4_tables):
    g4_tables["particles"] = pd.concat([g4_tables["particles"], g4_tables["particles"].assign(subrun=2)])
    pm = sc._primaries(["synthetic"])
    counts = sc._g4_counts(["synthetic"], pm)
    assert len(counts) == 8 and not counts.duplicated(KEY).any()
    other_subrun = counts[counts.subrun == 2]
    assert not other_subrun[["pass_trk", "g4_any_pass"]].any().any()
    assert (other_subrun[["nmax", "ngood_g4_any"]] == 0).all().all()
    g4_tables["sgsteps"] = g4_tables["sgsteps"].iloc[:0]
    counts = sc._g4_counts(["synthetic"], pm)
    assert not counts[["pass_trk", "g4_any_pass"]].any().any()
    assert (counts[["nmax", "ngood_g4_any"]] == 0).all().all()
    assert sc._g4_counts(["synthetic"], pm.iloc[:0]).empty


def test_seedcheck_output_uses_seeded_identity_for_every_variant(monkeypatch, g4_tables):
    traj, surf = [], []
    for event in range(1, 5):
        for sp in (3, 7):
            for ipt, y in enumerate((20.0, -20.0)):
                traj.append({"run": 1, "subrun": 1, "event": event, "sp": sp, "ipt": ipt,
                             "x": float(sp), "y": y, "z": 0.0, "ke": 500.0})
            surf.append({"run": 1, "subrun": 1, "event": event, "sp": sp, "surf": 4, "t": sp,
                         "x0": float(sp), "y0": 0.0, "z0": 0.0, "x1": float(sp), "y1": 0.0,
                         "z1": 0.0, "px": 0.0, "py": -500.0, "pz": 0.0})
    g4_tables["trajpts"] = pd.DataFrame(traj)
    g4_tables["surfsteps"] = pd.DataFrame(surf)
    g4_tables["steps"] = g4_tables["gen"].assign(edep=0.2, px0=0.0, py0=-100.0, pz0=0.0)
    monkeypatch.setattr(sc.geometry, "from_gdml", lambda gdml: None)
    monkeypatch.setattr(sc.st, "plane_y", lambda geom: 0.0)
    monkeypatch.setattr(sc.st, "TrackerInterior", lambda *args: None)
    monkeypatch.setattr(sc.crossings, "panel_boxes", lambda geom: (np.full((1, 3), -200.0), np.full((1, 3), 200.0)))
    calls = []

    def seeded(rng, interior, geom, o, u, p, start, **kw):
        calls.append((start, rng.random()))
        n = np.array([0, 12, 0, 12])
        return n, n.copy(), np.ones(4), u.copy(), o.copy(), p.copy()

    monkeypatch.setattr(sc, "seeded", seeded)
    saved = _capture_output(monkeypatch, sc)
    sc.run(["synthetic"], "unused", crv=["synthetic"], gdml="unused", seed=42)
    ev, out = saved["events"], saved["report"]
    assert len(ev) == 12 and not ev.duplicated(KEY + ["variant"]).any()
    for name in ("top", "plane", "envelope"):
        df = ev[ev.variant == name]
        assert df.seeded_sp.tolist() == [7, 7, 7, 7]
        assert df.g4_pass.tolist() == [False, False, False, True]
        assert df.ngood_g4.tolist() == [2, 0, 6, 12]
        assert df.g4_any_pass.tolist() == [True, True, False, True]
        assert df.ngood_g4_any.tolist() == [12, 12, 6, 12]
        for subset in ("all", "crv"):
            totals = out["variants"][name][subset]["counts"]["total"]
            assert totals["g4_pass"] == 1 and totals["g4_any_pass"] == 3
            assert totals["seed_only"] == 1 and totals["g4_only"] == 0
    assert out["filter_terms_g4ref"]["trk"] == 1
    assert out["filter_terms_g4ref"]["trk_any"] == 3
    assert [start for start, draw in calls] == ["top", "as_is", "as_is"]
    assert [draw for start, draw in calls] == [np.random.default_rng([42, 77, i]).random() for i in range(3)]


def test_fatecheck_preserves_transport_call_and_file_seed(monkeypatch, fast_inputs):
    gen, s, lp = fast_inputs
    geom, world, interior = object(), object(), object()
    monkeypatch.setattr(fc.geometry, "from_gdml", lambda gdml: geom)
    monkeypatch.setattr(fc.st, "load_world", lambda *args: world)
    monkeypatch.setattr(fc.st, "TrackerInterior", lambda *args: interior)
    monkeypatch.setattr(fc.st, "plane_y", lambda geom: 0.0)
    monkeypatch.setattr(T, "expand", lambda paths: paths)
    monkeypatch.setattr(T, "read", lambda path, tree: gen)
    draws = []

    def make_steps(generated, geometry, rng, **kw):
        assert generated is gen and geometry is geom
        assert kw == {"stage": "C", "world": world, "sliced_msc": True, "interior": interior}
        draws.append(rng.random())
        make_steps.last_plane = lp
        return s, None

    monkeypatch.setattr(fc.st, "make_steps", make_steps)
    ev, returned_geom, y = fc._fast(["synthetic0", "synthetic1"], "unused", 42)
    assert returned_geom is geom and y == 0.0 and len(ev) == 14
    assert draws == [np.random.default_rng([42, i]).random() for i in range(2)]
