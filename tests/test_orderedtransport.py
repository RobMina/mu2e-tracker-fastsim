"""Opt-in support-only transport closure on known physical intervals; no production claims."""
import copy

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import eloss, materials
from trkfastsim import orderedtransport as ot, transportledger as ledger


@pytest.fixture
def metadata():
    return ledger.provenance(geometry_sha256="0" * 64, tracker_commit="1" * 40, core_commit="2" * 40,
        inputs=[{"name": "synthetic", "sha256": "3" * 64}], frame="synthetic Cartesian",
        settings={"test": True}, media=[ledger.medium(),
        ledger.medium(materials.ALUMINIUM, cut_MeV=1e9, loss_basis="unrestricted"),
        ledger.medium(materials.COPPER, cut_MeV=1e9, loss_basis="unrestricted")])


def intervals(rows):
    return pd.DataFrame(rows, columns=["lane", "s0_mm", "s1_mm", "material"]).astype(
        {"lane": "int64", "s0_mm": "float64", "s1_mm": "float64", "material": str})


def run(metadata, spans, ke=(20.,), identity=None, rng=None, **kwargs):
    n = len(ke)
    identity = np.column_stack((np.ones((n, 3), dtype=int), np.arange(1, n+1))) if identity is None else identity
    return ot.transport_intervals(np.random.default_rng(5) if rng is None else rng,
        identity, np.zeros((n, 3)), np.tile([1., 0., 0.], (n, 1)), np.array(ke), spans, metadata,
        mode=kwargs.pop("mode", "ordered-support-mean-replace-v1"), **kwargs)


def deterministic(monkeypatch):
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.ones_like(p))
    monkeypatch.setattr(eloss, "sample_loss", lambda rng, mat, p, length, cut: length)


def test_sequential_materials_energy_closes_and_times_match(metadata, monkeypatch):
    deterministic(monkeypatch)
    spans = intervals([(0, 0, 2, "vacuum"), (0, 2, 5, "G4_Al"), (0, 5, 6, "G4_Cu"), (0, 6, 7, "vacuum")])
    result = run(metadata, spans, timing="linear-in-path")
    history = result["history"]
    np.testing.assert_allclose(result["final_ke_MeV"], [16.], rtol=1e-14)
    assert result["traveled_mm"][0] == 7.
    np.testing.assert_allclose(history.accounting()["primary_residual_MeV"], 0., atol=1e-13)
    assert np.all(history["dt_ns"] > 0)
    np.testing.assert_array_equal(history["ke_in_MeV"][1:], history["ke_out_MeV"][:-1])
    np.testing.assert_array_equal(history["entry_mm"][1:], history["exit_mm"][:-1])
    assert not history.accounting()["partition_complete"][history["medium"] > 0].any()
    assert history.metadata["settings"]["ordered_transport"]["timing"] == "linear-in-path"


def test_order_really_uses_updated_energy(metadata, monkeypatch):
    calls = []
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.ones_like(p))

    def loss(rng, mat, p, length, cut):
        calls.append((mat.name, p.copy()))
        return length

    monkeypatch.setattr(eloss, "sample_loss", loss)
    run(metadata, intervals([(0, 0, 1, "G4_Al"), (0, 1, 2, "G4_Cu")]), fractional_loss=1.)
    assert [name for name, _ in calls] == ["G4_Al", "G4_Cu"]
    expected = np.sqrt(19 * (19 + 2 * eloss.MMU))
    np.testing.assert_allclose(calls[1][1], [expected])


def test_sampled_stop_is_absorbing_and_time_is_finite(metadata, monkeypatch):
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.ones_like(p))
    monkeypatch.setattr(eloss, "sample_loss", lambda rng, mat, p, length, cut: np.full_like(p, 40.))
    spans = intervals([(0, 0, 1, "vacuum"), (0, 1, 2, "G4_Al"), (0, 2, 3, "G4_Cu")])
    result = run(metadata, spans, mode="ordered-support-sampled-stop-v1", fractional_loss=1., timing="linear-in-path")
    h = result["history"]
    assert result["status"].tolist() == ["exhausted"]
    np.testing.assert_allclose(result["traveled_mm"], [1.5])
    assert h["terminal"][-1] == "exhausted" and h["ke_out_MeV"][-1] == 0
    assert set(h["medium"]) == {0, 1}
    assert np.isfinite(h["dt_ns"]).all()
    assert h["sampled_loss_MeV"][-1] == 40 and h["applied_loss_MeV"][-1] == 20


def test_model_cutoff_keeps_residual_and_no_extra_deposition(metadata, monkeypatch):
    deterministic(monkeypatch)
    spans = intervals([(0, 0, 3, "G4_Al"), (0, 3, 4, "vacuum")])
    result = run(metadata, spans, ke=(1.5,), fractional_loss=1., max_step_mm=1.)
    np.testing.assert_allclose(result["final_ke_MeV"], [.5])
    assert result["status"].tolist() == ["model_stop"]
    assert result["traveled_mm"].tolist() == [1.]
    h = result["history"]
    assert h["terminal"][-1] == "model_stop" and np.isnan(h["deposited_MeV"][-1])


def test_multimuon_empty_zero_extent_and_stopped(metadata):
    spans = intervals([(0, 0, 1, "vacuum"), (1, 0, 10, "G4_Al"), (3, 0, 2, "vacuum")])
    identity = np.array([[2, 1, 1, 1], [1, 1, 1, 2], [1, 1, 1, 1], [1, 2, 1, 1]])
    rng = np.random.default_rng(2)
    state = copy.deepcopy(rng.bit_generator.state)
    result = run(metadata, spans, ke=(5., .1, 0., 8.), identity=identity, rng=rng)
    assert rng.bit_generator.state == state
    np.testing.assert_array_equal(result["traveled_mm"], [1, 0, 0, 2])
    assert result["status"].tolist() == ["transmitted", "model_stop", "exhausted", "transmitted"]
    h = result["history"]
    np.testing.assert_array_equal(h["run"], [1, 1, 1, 2])
    assert len(run(metadata, intervals([]), ke=())["history"]) == 0


def test_timing_choice_never_changes_energy_or_rng(metadata):
    spans = intervals([(0, 0, .2, "G4_Al"), (0, .2, .4, "G4_Cu")])
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    unknown, timed = run(metadata, spans, rng=a), run(metadata, spans, rng=b, timing="linear-in-path")
    for name in ("ke_in_MeV", "ke_out_MeV", "sampled_loss_MeV", "path_mm"):
        np.testing.assert_array_equal(unknown["history"][name], timed["history"][name])
    np.testing.assert_array_equal(a.random(20), b.random(20))
    assert np.isnan(unknown["history"]["dt_ns"]).all()


@pytest.mark.parametrize("rows", [[(0, 1, 2, "vacuum")], [(0, 0, 2, "G4_Al"), (0, 1, 3, "G4_Cu")],
                                  [(0, 0, 1, "unknown")], [(1, 0, 1, "vacuum")], [(0, 0, 0, "vacuum")]])
def test_bad_navigation_rejected_before_rng(metadata, rows):
    rng = np.random.default_rng(1)
    state = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError):
        run(metadata, intervals(rows), rng=rng)
    assert rng.bit_generator.state == state


def test_manifest_and_duplicate_identity_refusal(metadata):
    with pytest.raises(ValueError, match="matching"):
        run({**metadata, "media": metadata["media"][:1], "materials_cuts_sha256": ledger._digest(metadata["media"][:1])},
            intervals([(0, 0, 1, "G4_Al")]))
    with pytest.raises(ValueError, match="unique"):
        run(metadata, intervals([]), ke=(5., 5.), identity=np.ones((2, 4), dtype=int))


def test_real_sampler_many_muons_repeatability_and_no_resurrection(metadata):
    n = 64
    lane = np.repeat(np.arange(n), 3)
    spans = pd.DataFrame(dict(lane=lane, s0_mm=np.tile([0., .1, .2], n), s1_mm=np.tile([.1, .2, .3], n),
                             material=np.tile(["G4_Al", "G4_Cu", "vacuum"], n)))
    ke = np.linspace(.25, 100., n)
    keys = np.column_stack((np.full(n, 5), np.repeat([1, 2], n//2), np.ones(n, int), np.arange(n)[::-1]))
    a = run(metadata, spans, ke=ke, identity=keys, timing="linear-in-path")
    b = run(metadata, spans, ke=ke, identity=keys, timing="linear-in-path")
    for name in ledger.FIELDS:
        np.testing.assert_equal(a["history"][name], b["history"][name])
    np.testing.assert_allclose(a["history"].accounting()["primary_residual_MeV"], 0., atol=1e-13)
    assert (a["final_ke_MeV"] <= ke).all()
    assert (a["traveled_mm"][a["status"] != "transmitted"] < .3).all()


@pytest.mark.parametrize("mode", list(ot.MODES))
def test_mean_and_sampled_arms_equal_without_overshoot(metadata, monkeypatch, mode):
    deterministic(monkeypatch)
    spans = intervals([(0, 0, 1, "G4_Al"), (0, 1, 2, "G4_Cu")])
    expected = run(metadata, spans, fractional_loss=1.)
    actual = run(metadata, spans, fractional_loss=1., mode=mode)
    for name in ledger.FIELDS:
        np.testing.assert_equal(actual["history"][name], expected["history"][name])
