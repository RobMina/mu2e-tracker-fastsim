"""Synthetic read-only query tests, independent of experimental or production transport."""
from math import fsum

import numpy as np
import pytest

from mu2efastsim import eloss, materials
from trkfastsim import historystate, transportledger as ledger
from trkfastsim.flighttime import C_MM_PER_NS, linear_kinetic_dt_ns


@pytest.fixture
def meta():
    return ledger.provenance(
        geometry_sha256="0" * 64, tracker_commit="1" * 40, core_commit="2" * 40,
        inputs=[{"name": "synthetic-query-fixture", "sha256": "3" * 64}],
        frame="synthetic Cartesian", settings={"mode": "supplied states only"},
        media=[ledger.medium(), ledger.medium(materials.ALUMINIUM, cut_MeV=1e9,
                                               loss_basis="unrestricted")])


def row(entry, exit_, kin=20., kout=None, *, identity=(1, 1, 1, 1), kind="physical",
        terminal="none", path=None):
    """Supply a row, including analytic time, without invoking any transport producer."""
    kout = kin if kout is None else kout
    entry, exit_ = np.asarray(entry, float), np.asarray(exit_, float)
    length = (0. if kind == "displacement" else np.linalg.norm(exit_ - entry)) if path is None else path
    debit = kin - kout
    return dict(zip(ledger.KEY[:-1], identity), segment=0, medium=-1 if kind == "displacement" else int(debit > 0),
                kind=kind, terminal=terminal, path_mm=length, entry_mm=entry, exit_mm=exit_,
                ke_in_MeV=kin, ke_out_MeV=kout, applied_loss_MeV=debit,
                sampled_loss_MeV=debit, deposited_MeV=debit, escaping_MeV=0.,
                sample_status="known", deposit_status="known", escape_status="known",
                dt_status="known", dt_ns=float(linear_kinetic_dt_ns(length, kin, kout)))


def history(meta, *rows):
    if not rows:
        return ledger.SegmentHistory.empty(meta)
    counts, supplied = {}, []
    for item in rows:
        key = tuple(item[name] for name in ledger.KEY[:-1])
        supplied.append(dict(item, segment=counts.get(key, 0)))
        counts[key] = counts.get(key, 0) + 1
    return ledger.SegmentHistory({name: [item[name] for item in supplied] for name in ledger.FIELDS}, meta)


def query(h, paths, identities=None):
    if identities is None:
        identities = np.tile([1, 1, 1, 1], (len(paths), 1))
    return historystate.at_path(h, identities, paths, energy_evolution="linear-in-path")


def assert_unavailable(state, indices):
    assert not state["available"][indices].any()
    assert not state["alive"][indices].any()
    assert (state["segment"][indices] == -1).all()
    for name in ("position_mm", "ke_MeV", "p_MeV_c", "dt_from_start_ns"):
        assert np.isnan(state[name][indices]).all()


def momentum(kinetic):
    return np.sqrt(kinetic) * np.sqrt(kinetic + 2 * eloss.MMU)


def test_shared_boundaries_partial_ke_time_and_momentum(meta):
    h = history(meta, row([0, 0, 0], [2, 0, 0]), row([2, 0, 0], [4, 0, 0], 20., 2.),
                row([4, 0, 0], [5, 0, 0], 2.))
    paths = np.array([0., 1., 2., 2.5, 3., 4., 5., 5.1, -1.])
    state = query(h, paths)
    np.testing.assert_array_equal(state["available"], [True] * 7 + [False] * 2)
    np.testing.assert_array_equal(state["alive"], state["available"])
    np.testing.assert_array_equal(state["segment"], [0, 0, 1, 1, 1, 2, 2, -1, -1])
    np.testing.assert_allclose(state["ke_MeV"][:7], [20, 20, 20, 15.5, 11, 2, 2])
    np.testing.assert_allclose(state["p_MeV_c"][:7], momentum(state["ke_MeV"][:7]))
    np.testing.assert_allclose(state["position_mm"][:7], np.column_stack((paths[:7], np.zeros((7, 2)))))
    initial = 2 * (20 + eloss.MMU) / (C_MM_PER_NS * momentum(20))
    partial = (momentum(20) - momentum(11)) / (9 * C_MM_PER_NS)
    np.testing.assert_allclose(state["dt_from_start_ns"][4], initial + partial, rtol=3e-15)
    assert not np.isclose(partial, h["dt_ns"][1] / 2, rtol=1e-3)
    assert not np.isclose(state["p_MeV_c"][4], (momentum(20) + momentum(2)) / 2, rtol=1e-3)
    np.testing.assert_allclose(state["dt_from_start_ns"][6], h["dt_ns"].sum(), rtol=3e-15)
    assert (np.diff(state["dt_from_start_ns"][:7]) > 0).all()
    assert_unavailable(state, [7, 8])


@pytest.mark.parametrize("terminal,kout", [("exhausted", 0.), ("model_stop", 3.)])
def test_terminal_endpoint_available_not_alive(meta, terminal, kout):
    h = history(meta, row([0, 0, 0], [0, 3, 4], 20., kout, terminal=terminal))
    paths = [0., 2.5, np.nextafter(5., 0.), 5., np.nextafter(5., np.inf)]
    state = query(h, paths)
    np.testing.assert_array_equal(state["available"], [True] * 4 + [False])
    np.testing.assert_array_equal(state["alive"], [True] * 3 + [False] * 2)
    assert state["ke_MeV"][3] == kout
    assert state["p_MeV_c"][3] == momentum(kout)
    np.testing.assert_array_equal(state["position_mm"][3], [0, 3, 4])
    assert state["dt_from_start_ns"][3] == h["dt_ns"][0]
    assert_unavailable(state, [4])


@pytest.mark.parametrize("terminal,kin", [("exhausted", 0.), ("model_stop", 7.)])
def test_zero_length_terminal(meta, terminal, kin):
    h = history(meta, row([2, -3, 4], [2, -3, 4], kin, terminal=terminal))
    with np.errstate(all="raise"):
        state = query(h, [-1., 0., 1.])
    np.testing.assert_array_equal(state["available"], [False, True, False])
    assert not state["alive"].any()
    assert state["ke_MeV"][1] == kin
    assert state["dt_from_start_ns"][1] == 0.
    np.testing.assert_array_equal(state["position_mm"][1], [2, -3, 4])
    assert_unavailable(state, [0, 2])


def test_multiple_displacements_then_physical_outgoing_state(meta):
    h = history(meta, row([0, 0, 0], [2, 0, 0]),
                row([2, 0, 0], [100, 20, 0], kind="displacement"),
                row([100, 20, 0], [-8, 20, 2], kind="displacement"),
                row([-8, 20, 2], [-8, 23, 6], 20., 5.))
    state = query(h, [np.nextafter(2., 0.), 2., 4.5, 7.])
    np.testing.assert_array_equal(state["segment"], [0, 3, 3, 3])
    np.testing.assert_allclose(state["position_mm"], [[2, 0, 0], [-8, 20, 2], [-8, 21.5, 4], [-8, 23, 6]])
    assert state["dt_from_start_ns"][1] == h["dt_ns"][0]
    np.testing.assert_allclose(state["dt_from_start_ns"][2], h["dt_ns"][0] + linear_kinetic_dt_ns(2.5, 20., 12.5))
    np.testing.assert_allclose(state["dt_from_start_ns"][3], h["dt_ns"].sum())
    assert state["alive"].all()


@pytest.mark.parametrize("terminal", ["none", "model_stop"])
def test_trailing_displacements_expose_last_exit_without_flight(meta, terminal):
    h = history(meta, row([0, 0, 0], [2, 0, 0]),
                row([2, 0, 0], [6, 7, 8], kind="displacement"),
                row([6, 7, 8], [-2, -3, 4], kind="displacement", terminal=terminal))
    state = query(h, [np.nextafter(2., 0.), 2., np.nextafter(2., np.inf)])
    np.testing.assert_array_equal(state["segment"], [0, 2, -1])
    np.testing.assert_array_equal(state["position_mm"][1], [-2, -3, 4])
    assert state["dt_from_start_ns"][1] == h["dt_ns"][0]
    assert state["alive"][1] == (terminal == "none")
    assert_unavailable(state, [2])


@pytest.mark.parametrize("following_physical", [False, True])
def test_initial_nonzero_displacement_and_single_coordinate_domain(meta, following_physical):
    rows = [row([0, 0, 0], [8, -2, 1], kind="displacement")]
    if following_physical:
        rows.append(row([8, -2, 1], [8, -2, 4]))
    h = history(meta, *rows)
    state = query(h, [-1., 0., 1.5])
    assert_unavailable(state, [0])
    assert state["segment"][1] == int(following_physical)
    np.testing.assert_array_equal(state["position_mm"][1], [8, -2, 1])
    assert state["dt_from_start_ns"][1] == 0.
    if following_physical:
        np.testing.assert_array_equal(state["position_mm"][2], [8, -2, 2.5])
        assert state["dt_from_start_ns"][2] > 0.
    else:
        assert_unavailable(state, [2])


def test_zero_length_model_stop_exactly_at_displacement_coordinate(meta):
    h = history(meta, row([0, 0, 0], [2, 0, 0]),
                row([2, 0, 0], [10, 10, 10], kind="displacement"),
                row([10, 10, 10], [10, 10, 10], terminal="model_stop"))
    state = query(h, [np.nextafter(2., 0.), 2., np.nextafter(2., np.inf)])
    np.testing.assert_array_equal(state["alive"], [True, False, False])
    assert state["available"][1] and state["segment"][1] == 2
    np.testing.assert_array_equal(state["position_mm"][1], [10, 10, 10])
    assert state["dt_from_start_ns"][1] == h["dt_ns"][0]
    assert_unavailable(state, [2])


def test_exhausted_displacement_at_zero(meta):
    h = history(meta, row([0, 0, 0], [6, 7, 8], 0., kind="displacement", terminal="exhausted"))
    with np.errstate(all="raise"):
        state = query(h, [0.])
    assert state["available"][0] and not state["alive"][0]
    np.testing.assert_array_equal(state["position_mm"][0], [6, 7, 8])
    assert state["ke_MeV"][0] == state["p_MeV_c"][0] == state["dt_from_start_ns"][0] == 0.


def test_full_identity_multiple_muons_missing_and_all_stopped(meta):
    identities = [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)]
    h = history(meta, *(row([0, 0, i], [1, 0, i], i + 1., identity=key) for i, key in enumerate(identities)))
    keys = identities[::-1] + [(1, 1, 1, 2), (0, 0, 0, 0), (9, 9, 9, 9)]
    state = query(h, [0.5] * len(keys), keys)
    np.testing.assert_array_equal(state["ke_MeV"][:5], [5, 4, 3, 2, 1])
    np.testing.assert_array_equal(state["position_mm"][:5, 2], [4, 3, 2, 1, 0])
    assert_unavailable(state, [5, 6, 7])
    stopped = history(meta, *(row([0, 0, i], [0, 0, i], 0., identity=key, terminal="exhausted")
                              for i, key in enumerate(identities)))
    with np.errstate(all="raise"):
        state = query(stopped, [0.] * 5, identities[::-1])
    assert state["available"].all() and not state["alive"].any()
    np.testing.assert_array_equal(state["dt_from_start_ns"], 0.)
    np.testing.assert_array_equal(state["ke_MeV"], 0.)


def test_query_permutation_and_batch_invariance(meta):
    a = [row([0, 0, 0], [2, 0, 0]), row([2, 0, 0], [3, 0, 0], 20., 0., terminal="exhausted")]
    b = [row([5, 0, 0], [5, 4, 0], 7., identity=(1, 1, 1, 2))]
    h = history(meta, *a, *b)
    keys = np.array([[1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 1, 1], [1, 1, 1, 2],
                     [1, 1, 1, 9], [1, 1, 1, 1], [1, 1, 1, 2]])
    paths = np.array([2.5, 1., 3., 4., 0., -1., 0.])
    baseline = query(h, paths, keys)
    permutation = np.random.default_rng(51).permutation(len(keys))
    permuted = query(h, paths[permutation], keys[permutation])
    chunks = [query(h, paths[:3], keys[:3]), query(h, paths[3:], keys[3:])]
    for name in baseline:
        np.testing.assert_equal(permuted[name][np.argsort(permutation)], baseline[name])
        np.testing.assert_equal(np.concatenate([chunk[name] for chunk in chunks]), baseline[name])
    for rows, sp in ((a, 1), (b, 2)):
        selected = keys[:, -1] == sp
        alone = query(history(meta, *rows), paths[selected], keys[selected])
        for name in baseline:
            np.testing.assert_equal(alone[name], baseline[name][selected])


def test_per_muon_clock_isolation_from_huge_prior_time(meta):
    first = row([0, 0, 0], [1, 0, 0], 1e-200)
    second = row([0, 0, 0], [1, 0, 0], 20., identity=(1, 1, 1, 2))
    h = history(meta, first, second)
    state = query(h, [0., 0.5, 1.], [[1, 1, 1, 2]] * 3)
    alone = query(history(meta, second), [0., 0.5, 1.], [[1, 1, 1, 2]] * 3)
    assert h["dt_ns"][0] > 1e90
    np.testing.assert_array_equal(state["dt_from_start_ns"], alone["dt_from_start_ns"])
    np.testing.assert_allclose(state["dt_from_start_ns"], [0., second["dt_ns"] / 2, second["dt_ns"]])


def test_empty_history_and_empty_query(meta):
    h = history(meta)
    state = query(h, [0., -1., 2.])
    assert_unavailable(state, [0, 1, 2])
    for h in (h, history(meta, row([0, 0, 0], [1, 0, 0]))):
        state = query(h, [], np.empty((0, 4), dtype=np.int64))
        for name, value in state.items():
            assert value.shape == ((0, 3) if name == "position_mm" else (0,))
        assert state["available"].dtype == state["alive"].dtype == np.dtype(bool)
        assert state["segment"].dtype == np.dtype(np.int64)


@pytest.mark.parametrize("status", ["unknown", "unmodeled", "inconsistent"])
def test_reject_unknown_or_inconsistent_time_even_with_no_matching_query(meta, status):
    item = row([0, 0, 0], [1, 0, 0])
    item.update(dt_status="known" if status == "inconsistent" else status,
                dt_ns=item["dt_ns"] * 2 if status == "inconsistent" else np.nan)
    h = history(meta, item)
    for keys, paths in (([[9, 9, 9, 9]], [0.]), (np.empty((0, 4), int), [])):
        with pytest.raises(ValueError, match="disagree" if status == "inconsistent" else "known"):
            query(h, paths, keys)


def test_reject_curved_physical_segment(meta):
    h = history(meta, row([0, 0, 0], [1, 0, 0], path=2.))
    with pytest.raises(ValueError, match="straight"):
        query(h, [0.5])


@pytest.mark.parametrize("declaration", [None, False, "unknown", "constant", "linear-in-time"])
def test_explicit_evolution_required(meta, declaration):
    h = history(meta, row([0, 0, 0], [1, 0, 0]))
    with pytest.raises(ValueError, match="explicit"):
        historystate.at_path(h, [[1, 1, 1, 1]], [0.], energy_evolution=declaration)


def test_history_type_and_missing_declaration(meta):
    with pytest.raises(ValueError, match="SegmentHistory"):
        historystate.at_path({}, [[1, 1, 1, 1]], [0.], energy_evolution="linear-in-path")
    with pytest.raises(TypeError):
        historystate.at_path(history(meta), [[1, 1, 1, 1]], [0.])


@pytest.mark.parametrize("keys", [
    [[1., 1., 1., 1.]], [[1., 1., 1., np.nan]], [[1., 1., 1., np.inf]],
    [[1., 1., 1., 1.5]], [[1, 1, 1, -1]], [[True] * 4], [["1"] * 4],
    [[1, 1, 1]], [1, 1, 1, 1], np.ones((1, 4, 1), int),
    np.array([[0, 0, 0, 2 ** 63]], dtype=np.uint64), np.array([[1] * 4], dtype=object),
])
def test_invalid_identity_keys(meta, keys):
    with pytest.raises(ValueError, match="identities"):
        query(history(meta), [0.], keys)


@pytest.mark.parametrize("paths", [[np.nan], [np.inf], [-np.inf], [[0.]], [], [0., 1.], 0.])
def test_invalid_path_shape_or_nonfinite(meta, paths):
    with pytest.raises(ValueError, match="paths"):
        query(history(meta), paths, [[1, 1, 1, 1]])


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_maximum_int64_full_identity(meta, dtype):
    key = (np.iinfo(np.int64).max,) * 4
    h = history(meta, row([0, 0, 0], [1, 0, 0], identity=key))
    state = query(h, [0.5], np.array([key], dtype=dtype))
    assert state["available"][0] and state["alive"][0]
    assert state["position_mm"][0, 0] == 0.5


def test_adjacent_float_boundaries_use_exact_shared_prefixes(meta):
    h = history(meta, row([0, 0, 0], [0.1, 0, 0], path=0.1),
                row([0.1, 0, 0], [0.3, 0, 0], 20., 10., path=0.2),
                row([0.3, 0, 0], [0.6, 0, 0], 10., 0., terminal="exhausted", path=0.3))
    boundaries = np.array([fsum(h["path_mm"][:i]) for i in range(1, len(h) + 1)])
    paths = np.array([np.nextafter(0., -np.inf), 0., np.nextafter(boundaries[0], 0.), boundaries[0],
                      np.nextafter(boundaries[0], np.inf), np.nextafter(boundaries[1], 0.), boundaries[1],
                      np.nextafter(boundaries[1], np.inf), np.nextafter(boundaries[2], 0.), boundaries[2],
                      np.nextafter(boundaries[2], np.inf)])
    state = query(h, paths)
    np.testing.assert_array_equal(state["segment"], [-1, 0, 0, 1, 1, 1, 2, 2, 2, 2, -1])
    assert_unavailable(state, [0, 10])
    assert state["alive"][1:9].all() and not state["alive"][9]
    assert state["ke_MeV"][9] == state["p_MeV_c"][9] == 0.
    np.testing.assert_array_equal(state["position_mm"][[3, 6, 9], 0], [0.1, 0.3, 0.6])
    np.testing.assert_allclose(state["dt_from_start_ns"][[3, 6, 9]], np.cumsum(h["dt_ns"]), rtol=3e-15)
    assert (np.diff(state["dt_from_start_ns"][1:10]) >= 0.).all()


def test_query_does_not_mutate_history_metadata_or_return_aliases(meta):
    h = history(meta, row([0, 0, 0], [1, 0, 0]))
    before = {name: h[name].copy() for name in ledger.FIELDS}
    provenance = h.metadata
    state = query(h, [0., 0.5, 1.])
    for value in state.values():
        value[:] = 0
    for name in ledger.FIELDS:
        np.testing.assert_equal(h[name], before[name])
    assert h.metadata == provenance
    assert query(h, [0.5])["ke_MeV"][0] == 20.


@pytest.mark.parametrize("lengths,end", [([.01, .02, .3], .32999999999999996),
                                         ([.01, .2, .6], .81)])
@pytest.mark.parametrize("following", [False, True])
def test_compensated_zero_path_preserves_exact_coordinate(meta, lengths, end, following):
    rows, x = [], 0.
    for length in lengths:
        rows.append(row([x, 0, 0], [x + length, 0, 0], path=length))
        x += length
    rows.extend([row([x, 0, 0], [10, 0, 0], kind="displacement"),
                 row([10, 0, 0], [11, 0, 0], kind="displacement")])
    if following:
        rows.append(row([11, 0, 0], [12, 0, 0], path=1.))
    else:
        rows.append(row([11, 0, 0], [11, 0, 0], terminal="model_stop"))
    h = history(meta, *rows)
    s0, s1, t0, t1 = historystate.path_coordinates(h)
    assert s1[2] == end
    np.testing.assert_array_equal(s0[3:6], end)
    np.testing.assert_array_equal(s1[3:5], end)
    np.testing.assert_array_equal(t0[3:6], t1[2])
    np.testing.assert_array_equal(t1[3:5], t1[2])
    state = query(h, [np.nextafter(end, 0.), end, np.nextafter(end, np.inf)])
    np.testing.assert_array_equal(state["segment"], [2, 5, 5 if following else -1])
    assert state["alive"][1] == following
    assert state["dt_from_start_ns"][1] == t1[2]
    np.testing.assert_array_equal(state["position_mm"][1], [11, 0, 0])
    if not following:
        assert_unavailable(state, [2])


@pytest.mark.parametrize("times,end", [([.01, .02, .3], .32999999999999996),
                                       ([.01, .2, .6], .81)])
def test_compensated_zero_time_and_per_muon_reset(meta, times, end):
    rows = []
    time_per_mm = float(linear_kinetic_dt_ns(1., 20., 20.))
    for sp in (1, 2):
        key, x = (1, 1, 1, sp), 0.
        rows.append(row([x, 0, 0], [x, 0, 0], identity=key, kind="displacement"))
        for dt in times:
            length = dt / time_per_mm
            item = row([x, 0, 0], [x + length, 0, 0], identity=key, path=length)
            item["dt_ns"] = dt
            rows.append(item)
            x += length
        rows.append(row([x, 0, 0], [x, 0, 0], identity=key, terminal="model_stop"))
    h = history(meta, *rows)
    coordinates = historystate.path_coordinates(h)
    s0, s1, t0, t1 = coordinates
    for start in (0, 5):
        assert s0[start] == s1[start] == t0[start] == t1[start] == 0.
        assert t1[start + 3] == t0[start + 4] == t1[start + 4] == end
        for begin, finish in ((s0, s1), (t0, t1)):
            assert (finish[start:start + 5] >= begin[start:start + 5]).all()
    state = query(h, [s1[4], s1[9]], [[1, 1, 1, 1], [1, 1, 1, 2]])
    np.testing.assert_array_equal(state["dt_from_start_ns"], [end, end])
    assert state["available"].all() and not state["alive"].any()
    for values in coordinates:
        np.testing.assert_array_equal(values[:5], values[5:])


def test_reject_positive_path_that_cannot_advance_prefix(meta):
    h = history(meta, row([0, 0, 0], [1, 0, 0]),
                row([1, 0, 0], [1, 0, 0], path=1e-20))
    with pytest.raises(ValueError, match="unrepresentable.*path"):
        historystate.path_coordinates(h)
    for paths, keys in (([1.], [[1, 1, 1, 1]]), ([], np.empty((0, 4), int))):
        with pytest.raises(ValueError, match="unrepresentable.*path"):
            query(h, paths, keys)


def test_nearest_endpoint_interpolation_keeps_exhausted_interior_alive_and_positive(meta):
    h = history(meta, row([0, 0, 0], [.01, 0, 0], path=.01),
                row([.01, 0, 0], [.03, 0, 0], path=.02),
                row([.03, 0, 0], [.43, 0, 0], 20., 0., path=.4, terminal="exhausted"))
    s0, s1, t0, t1 = historystate.path_coordinates(h)
    start, end = s0[-1], s1[-1]
    assert end == .43000000000000005 and np.nextafter(end, 0.) == .43
    paths = [np.nextafter(start, 0.), start, np.nextafter(start, np.inf),
             np.nextafter(end, 0.), end, np.nextafter(end, np.inf)]
    state = query(h, paths)
    np.testing.assert_array_equal(state["segment"], [1, 2, 2, 2, 2, -1])
    assert state["alive"][:4].all() and not state["alive"][4:].any()
    assert (state["ke_MeV"][:4] > 0).all() and (state["p_MeV_c"][:4] > 0).all()
    near_exit_ke = 20. * ((end - paths[3]) / (end - start))
    assert state["ke_MeV"][3] == near_exit_ke
    assert state["ke_MeV"][4] == state["p_MeV_c"][4] == 0.
    assert state["dt_from_start_ns"][1] == t0[-1]
    assert state["dt_from_start_ns"][4] == t1[-1]
    assert (np.diff(state["dt_from_start_ns"][:5]) >= 0).all()
    assert_unavailable(state, [5])


def test_reject_unrepresentable_positive_interior_energy(meta):
    tiny = np.nextafter(0., 1.)
    h = history(meta, row([0, 0, 0], [1, 0, 0], tiny, 0., terminal="exhausted"))
    with pytest.raises(ValueError, match="unrepresentable positive interior kinetic energy"):
        query(h, [np.nextafter(1., 0.)])
    state = query(h, [0., 1.])
    np.testing.assert_array_equal(state["ke_MeV"], [tiny, 0.])
    np.testing.assert_array_equal(state["alive"], [True, False])


def test_path_coordinates_empty_and_no_aliases(meta):
    h = history(meta)
    assert all(values.shape == (0,) for values in historystate.path_coordinates(h))
    h = history(meta, row([0, 0, 0], [1, 0, 0]))
    before = {name: h[name].copy() for name in ledger.FIELDS}
    for values in historystate.path_coordinates(h):
        values[:] = -1.
    for name in ledger.FIELDS:
        np.testing.assert_array_equal(h[name], before[name])


@pytest.mark.parametrize("field", ["path_mm", "dt_ns"])
def test_path_coordinates_reject_overflow_before_filling_zeros(meta, field):
    rows = [row([0, 0, 0], [0, 0, 0], path=1.),
            row([0, 0, 0], [0, 0, 0], path=1.),
            row([0, 0, 0], [0, 0, 0], kind="displacement")]
    for item in rows[:2]:
        item[field] = 1e308
    # Accumulation itself imposes no straightness or energy/time law on supplied rows.
    h = history(meta, *rows)
    with pytest.raises(ValueError, match="must be finite"):
        historystate.path_coordinates(h)
