"""Surface projection of supplied histories; no production transport qualification."""
import numpy as np
import pandas as pd
import pytest

from mu2efastsim import materials
from trkfastsim import historysurfaces, surfaces, transportledger
from trkfastsim.flighttime import linear_kinetic_dt_ns


def history(z=(-2000., 2000.), kinetic=(20., 5.), terminal="none", kind=None):
    z, kinetic = np.asarray(z, float), np.asarray(kinetic, float)
    n = len(z) - 1
    kind = np.full(n, "physical") if kind is None else np.asarray(kind)
    length = np.where(kind == "physical", np.diff(z), 0.)
    loss = kinetic[:-1] - kinetic[1:]
    meta = transportledger.provenance(
        geometry_sha256="0" * 64, tracker_commit="1" * 40, core_commit="2" * 40,
        inputs=[{"name": "synthetic-surface", "sha256": "3" * 64}], frame="detector",
        media=[transportledger.medium(), transportledger.medium(materials.ALUMINIUM, cut_MeV=1e9,
                                                               loss_basis="unrestricted")],
        settings={"energy_evolution": "linear-in-path", "scope": "synthetic"})
    known = np.full(n, "known")
    return transportledger.SegmentHistory(dict(
        run=np.ones(n, int), subrun=np.ones(n, int), event=np.ones(n, int), sp=np.ones(n, int),
        segment=np.arange(n), kind=kind, medium=np.where(kind == "physical", 1, -1),
        terminal=np.concatenate((np.full(n-1, "none"), [terminal])), path_mm=length,
        entry_mm=np.column_stack((np.full(n, 100.), np.zeros(n), z[:-1])),
        exit_mm=np.column_stack((np.full(n, 100.), np.zeros(n), z[1:])),
        ke_in_MeV=kinetic[:-1], ke_out_MeV=kinetic[1:], sampled_loss_MeV=loss, applied_loss_MeV=loss,
        deposited_MeV=loss, escaping_MeV=np.zeros(n), dt_ns=linear_kinetic_dt_ns(length, kinetic[:-1], kinetic[1:]),
        sample_status=known, deposit_status=known, escape_status=known, dt_status=known), meta)


def starts(identities=((1, 1, 1, 1),), times=(7.,)):
    return pd.DataFrame(identities, columns=list(transportledger.KEY[:-1])).assign(t_start_ns=times)


def project(h, anchor=None):
    return historysurfaces.surface_steps(h, starts() if anchor is None else anchor,
                                        energy_evolution="linear-in-path")


def test_partial_clock_and_momentum_at_all_plane_crossings():
    h = history()
    actual = project(h)
    assert actual.surf.tolist() == [0, 1, 2]
    distance = np.array([surfaces.Z_FRONT, surfaces.Z_MID, surfaces.Z_BACK]) + 2000.
    kinetic = 20. - 15. * distance / 4000.
    expected = 7. + linear_kinetic_dt_ns(distance, 20., kinetic)
    np.testing.assert_allclose(actual.t, expected, rtol=0, atol=1e-13)
    np.testing.assert_allclose(actual.pz, np.sqrt(kinetic * (kinetic + 2 * surfaces.MMU)), rtol=1e-7)
    naive = 7. + distance / 4000. * h["dt_ns"][0]
    assert np.max(np.abs(actual.t - naive)) > 1.


@pytest.mark.parametrize("terminal,energy", [("exhausted", 0.), ("model_stop", .5), ("none", 5.)])
def test_no_crossings_beyond_final_endpoint(terminal, energy):
    actual = project(history(z=(-2000., 0.), kinetic=(20., energy), terminal=terminal))
    assert actual.surf.tolist() == [0]


def test_terminal_on_surface_is_not_transmission():
    actual = project(history(z=(-2000., surfaces.Z_MID), kinetic=(20., 0.), terminal="exhausted"))
    assert actual.surf.tolist() == [0]


def test_boundary_at_surface_uses_outgoing_clock_once():
    z = (-2000., surfaces.Z_MID, 2000.)
    actual = project(history(z=z, kinetic=(20., 10., 5.)))
    assert actual.surf.tolist() == [0, 1, 2]
    mid = actual[actual.surf == 1].iloc[0]
    assert mid.t == 7. + linear_kinetic_dt_ns(surfaces.Z_MID + 2000., 20., 10.)


def test_displacement_does_not_manufacture_physical_crossing():
    h = history(z=(-2000., -100., 100., 2000.), kinetic=(20., 10., 10., 5.),
                kind=("physical", "displacement", "physical"))
    actual = project(h)
    assert actual.surf.tolist() == [0, 2]
    expected = (7. + linear_kinetic_dt_ns(1900., 20., 10.)
                + linear_kinetic_dt_ns(surfaces.Z_BACK - 100., 10., 10. - 5. * (surfaces.Z_BACK - 100.) / 1900.))
    np.testing.assert_allclose(actual.t.iloc[-1], expected, atol=1e-13, rtol=0)


def test_multi_primary_anchor_order_and_history_batch_invariance():
    h = history()
    columns = {k: np.concatenate((h[k], h[k])) for k in transportledger.FIELDS}
    columns["sp"] = np.array([1, 2])
    many = transportledger.SegmentHistory(columns, h.metadata)
    anchor = starts(((1, 1, 1, 2), (1, 1, 1, 1)), (700., 7.))
    actual = project(many, anchor)
    one = project(h)
    pd.testing.assert_frame_equal(actual[actual.sp == 1].reset_index(drop=True), one)
    shifted = actual[actual.sp == 2].reset_index(drop=True)
    np.testing.assert_allclose(shifted.t - one.t, 693., atol=1e-13, rtol=0)
    np.testing.assert_array_equal(shifted.pz, one.pz)


@pytest.mark.parametrize("modify,message", [
    (lambda a: a.assign(t_start_ns=np.nan), "finite"),
    (lambda a: pd.concat([a, a]), "unique"),
    (lambda a: a.assign(sp=2), "exactly cover"),
    (lambda a: a.assign(sp=1.), "integers"),
    (lambda a: a.assign(sp=-1), "integers"),
    (lambda a: a.assign(sp=np.uint64(2**63)), "integers"),
    (lambda a: a.assign(extra=1), "only"),
    (lambda a: a.drop(columns="run"), "only"),
])
def test_invalid_start_contract(modify, message):
    with pytest.raises(ValueError, match=message):
        project(history(), modify(starts()))


def test_empty_and_stopped_zero_length():
    h = history(z=(0., 0.), kinetic=(0., 0.), terminal="exhausted")
    assert project(h).empty
    assert project(transportledger.SegmentHistory.empty(h.metadata), starts().iloc[:0]).empty
    with pytest.raises(ValueError, match="exactly cover"):
        project(transportledger.SegmentHistory.empty(h.metadata))


def test_frame_unknown_time_and_evolution_refused():
    h = history()
    meta = h.metadata
    meta["frame"] = "Mu2e"
    columns = {k: h[k].copy() for k in transportledger.FIELDS}
    with pytest.raises(ValueError, match="detector"):
        project(transportledger.SegmentHistory(columns, meta))
    columns.update(dt_status=np.array(["unknown"]), dt_ns=np.array([np.nan]))
    with pytest.raises(ValueError, match="known"):
        project(transportledger.SegmentHistory(columns, h.metadata))
    with pytest.raises(ValueError, match="linear-in-path"):
        historysurfaces.surface_steps(h, starts(), energy_evolution="constant")


def test_surface_output_refuses_identity_truncation():
    h = history()
    columns = {k: h[k].copy() for k in transportledger.FIELDS}
    columns["sp"][:] = 2**32 + 1
    with pytest.raises(ValueError, match="int32"):
        project(transportledger.SegmentHistory(columns, h.metadata), starts(((1, 1, 1, 2**32+1),)))
    columns["sp"][:] = np.iinfo(np.int32).max
    actual = project(transportledger.SegmentHistory(columns, h.metadata),
                     starts(((1, 1, 1, np.iinfo(np.int32).max),)))
    assert (actual.sp == np.iinfo(np.int32).max).all()


def test_exact_cylinder_tangency_is_not_a_crossing():
    h = history(z=(0., 20.), kinetic=(20., 20.))
    columns = {k: h[k].copy() for k in transportledger.FIELDS}
    columns["entry_mm"][:] = [surfaces.R_IN, -10., 0.]
    columns["exit_mm"][:] = [surfaces.R_IN, 10., 0.]
    assert project(transportledger.SegmentHistory(columns, h.metadata)).empty
    columns["entry_mm"][:, 0] -= 1e-3
    columns["exit_mm"][:, 0] -= 1e-3
    actual = project(transportledger.SegmentHistory(columns, h.metadata))
    assert actual.surf.tolist() == [3, 3]
    assert actual.t.iloc[1] > actual.t.iloc[0]


@pytest.mark.parametrize("lengths", [[.01, .02, .3], [.01, .2, .6]])
def test_surface_clock_preserves_prefix_through_displacement(lengths):
    from test_historystate import history as make_history, row
    from trkfastsim import historystate
    meta = history().metadata
    supplied, z = [], -2000.
    for length in lengths:
        supplied.append(row([100., 0., z], [100., 0., z + length], path=length))
        z += length
    supplied.append(row([100., 0., z], [100., 0., -100.], kind="displacement"))
    supplied.append(row([100., 0., -100.], [100., 0., 100.], 20., 5., path=200.))
    h = make_history(meta, *supplied)
    result = project(h)
    assert result.surf.tolist() == [1]
    _, _, t0, _ = historystate.path_coordinates(h)
    fraction = (surfaces.Z_MID + 100.) / 200.
    expected = 7. + t0[-1] + linear_kinetic_dt_ns(200. * fraction, 20., 20. - 15. * fraction)
    np.testing.assert_allclose(result.t, expected, rtol=0, atol=1e-13)


def test_overflowing_absolute_crossing_time_is_refused():
    h = history(z=(-2e134, 2e134), kinetic=(1e-320, 1e-320))
    with pytest.raises(ValueError, match="absolute crossing time"):
        project(h, starts(times=(np.finfo(float).max,)))
