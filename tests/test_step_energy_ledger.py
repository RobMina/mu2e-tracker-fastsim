"""Synthetic first-entry ledgers and constant-momentum flight times, not rate tests."""

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import eloss, materials
from mu2efastsim.world import CoaxialShells
from trkfastsim import geometry, steps


RADIUS = 2.5
P_START = 80.0
START_Y = 1002.5
T_START = 7.0


def _geometry(ys=(0.0,)):
    """Parallel x-axis straws in one synthetic panel, with no external geometry."""
    n = len(ys)
    zeros = np.zeros(n, dtype=int)
    return geometry.StrawGeometry(
        plane=zeros, panel=zeros, straw=np.arange(n),
        mid=np.column_stack((np.zeros(n), ys, np.zeros(n))),
        dir=np.tile([1.0, 0.0, 0.0], (n, 1)), halflen=np.full(n, 500.0),
        radius=RADIUS, panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)),
        plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)), wall_rmax=RADIUS,
    )


@pytest.fixture
def generated():
    return pd.DataFrame({
        "run": [1], "subrun": [1], "event": [1], "pdg": [13],
        "x": [0.0], "y": [START_Y], "z": [0.0],
        "px": [0.0], "py": [-P_START], "pz": [0.0], "t": [T_START],
    })


@pytest.fixture
def empty_interior():
    """Exercise real interior methods with no support and their default vacuum gap."""
    interior = steps.TrackerInterior.__new__(steps.TrackerInterior)
    interior.shells = CoaxialShells([], axis_point=[0.0, 0.0, 0.0])
    interior.nsub = 8
    return interior


@pytest.fixture
def plane_state(monkeypatch):
    """Inject only the stage-C boundary state; upstream transport is outside this test."""
    def install(momentum, origin_y=START_Y, upstream_path=500.0):
        def to_plane(gen, geom, rng, world, **kwargs):
            mu = steps._muons(gen, kwargs["all_muons"])
            return (mu, np.array([[0.0, origin_y, 0.0]]),
                    np.array([[0.0, -1.0, 0.0]]), np.array([momentum]),
                    np.array([upstream_path]), np.array([True]))
        monkeypatch.setattr(steps, "to_plane", to_plane)
    return install


def _run(generated, geom, **kwargs):
    rows, particles = steps.make_steps(
        generated, geom, np.random.default_rng(19), msc_tail=False,
        exit_scatter=False, **kwargs,
    )
    assert len(rows) == geom.nstraws
    assert len(particles) == 1
    np.testing.assert_allclose(rows["len"], 2 * RADIUS, rtol=0, atol=1e-10)
    return rows


def _momentum(rows):
    return np.linalg.norm(rows[["px", "py", "pz"]].to_numpy(), axis=1)


@pytest.mark.parametrize("interior_msc", [False, True])
def test_no_loss_preserves_first_entry_momentum_and_time(generated, empty_interior, interior_msc):
    rows = _run(generated, _geometry(), air_loss=False,
                interior=empty_interior, interior_msc=interior_msc)
    beta = P_START / np.hypot(P_START, eloss.MMU)
    expected_time = T_START + (START_Y - RADIUS) / (beta * steps.C_MM_PER_NS)
    np.testing.assert_allclose(_momentum(rows), P_START, rtol=1e-12)
    np.testing.assert_allclose(rows.t, expected_time, rtol=1e-12)


@pytest.mark.parametrize("interior_msc", [False, True])
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="make_steps subtracts air dE from momentum instead of total energy")
def test_air_loss_debits_energy_not_momentum(generated, empty_interior, interior_msc):
    # Stage A explicitly requests air; use its frozen entrance-momentum stopping power.
    distance = START_Y - RADIUS
    energy_in = np.hypot(P_START, eloss.MMU)
    loss = float(eloss.dedx_restricted(materials.AIR, P_START, 1e9)) * distance
    assert 0 < loss < energy_in - eloss.MMU - 0.5
    rows = _run(generated, _geometry(), air_loss=True,
                interior=empty_interior, interior_msc=interior_msc)
    np.testing.assert_allclose(np.hypot(_momentum(rows), eloss.MMU),
                               energy_in - loss, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("interior_msc", [False, True])
@pytest.mark.parametrize("air_loss", [
    False,
    pytest.param(True, marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason="make_steps adds air loss even though TrackerInterior gaps are vacuum",
    )),
])
def test_vacuum_before_first_straw_has_no_energy_debit(
        generated, empty_interior, plane_state, interior_msc, air_loss):
    geom = _geometry()
    plane_state(P_START, origin_y=steps.plane_y(geom))
    # The first gas entry has no preceding straw chord or support material.
    rows = _run(generated, geom, stage="C", air_loss=air_loss,
                interior=empty_interior, interior_msc=interior_msc)
    np.testing.assert_allclose(_momentum(rows), P_START, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("generated_p", [
    P_START,
    pytest.param(300.0, marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason="make_steps flight time uses generated beta, not local post-world beta",
    )),
])
def test_loss_disabled_segment_time_uses_local_momentum(generated, plane_state, generated_p):
    generated = generated.assign(py=-generated_p)
    plane_state(P_START)
    # Disable interior and air loss so the modeled segment momentum is exactly constant.
    rows = _run(generated, _geometry((0.0, -1000.0)), stage="C", air_loss=False)
    np.testing.assert_allclose(_momentum(rows), P_START, rtol=1e-12)
    separation = np.linalg.norm(rows[["x0", "y0", "z0"]].to_numpy()[1]
                                - rows[["x0", "y0", "z0"]].to_numpy()[0])
    beta_local = P_START / np.hypot(P_START, eloss.MMU)
    # Subtract arrival times to cancel the deliberately unspecified upstream flight time.
    np.testing.assert_allclose(rows.t.iloc[1] - rows.t.iloc[0],
                               separation / (beta_local * steps.C_MM_PER_NS), rtol=1e-12)
