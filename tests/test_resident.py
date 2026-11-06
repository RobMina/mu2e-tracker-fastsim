"""Synthetic measured-hit resident contracts, not stock or physical validation.

The grouping oracle is deliberately plain Python and does not call production
selection/grouping code. Geometry is constructed analytically from measured
positions and wires; no truth labels, track memberships, or stock fits enter.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
from trkfastsim.reco import resident
from trkfastsim.reco import helixfit_jax, helixseed_jax


# Independent selection constants keep production-mask changes visible to the oracle.
_TDIV = 1 << 10
_BAD = (1 << 6) | (1 << 23) | (1 << 24)


def _keys(n):
    # Exercise lossless keys beyond float64's consecutive-integer range.
    return np.array([[2**53 + 7, 31, i + 1] for i in range(n)], dtype=np.int64).reshape(n, 3)


def _helix(ev, n=12, start=0., sid_start=0):
    phase = np.linspace(-.8, .8, n)
    point = np.column_stack((35. + 220.*np.cos(phase),
                             -18. + 220.*np.sin(phase), 80. + 300.*phase))
    angle = np.arange(n)*1.23
    direction = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(n)))
    return dict(ev=np.full(n, ev, np.int64), sid=np.arange(sid_start, sid_start+n, dtype=np.int64),
                time=start+np.arange(n, dtype=np.float64), flag=np.full(n, _TDIV, np.int64),
                wdist=np.zeros(n), uvar=np.full(n, 900.), mid=point, direction=direction,
                hlen=np.full(n, 1000.), valid=np.ones(n, bool))


def _join(*parts):
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def _run(keys, hits, **kwargs):
    options = dict(bz=1., max_hits=12, group_batch=4, max_iterations=8)
    options.update(kwargs)
    return jax.device_get(resident.reconstruct(keys, hits, **options))


def _oracle(keys, h, max_hits=12, bz=1.):
    """Return every group, including rejected ones, and original row identities."""
    ne, n = len(keys), len(h["ev"])
    rows = [[] for _ in range(ne)]
    invalid = np.zeros(ne, bool)
    counts = np.zeros(ne, np.int64)
    hit_status = np.zeros(n, np.int32)
    global_bad = (not np.isfinite(bz) or bz == 0 or np.any(keys < 0)
                  or len({tuple(k) for k in keys}) != ne)
    for i in range(n):
        if not h["valid"][i]:
            continue
        ev, sid, flag = (int(h[k][i]) for k in ("ev", "sid", "flag"))
        ev_ok = 0 <= ev < ne
        sid_ok = 0 <= sid < 36 * 1024 and sid % 128 < 96 and (sid // 128) % 8 < 6
        identity_ok = ev_ok and sid_ok and 0 <= flag <= 2**32-1
        selected = not (flag & _BAD) and bool(flag & _TDIV)
        bad = not identity_ok or (selected and not np.isfinite(h["time"][i]))
        if ev_ok:
            counts[ev] += 1
            invalid[ev] |= bad
        else:
            global_bad = True
        label = "invalid_input" if bad else "grouped" if selected else "flag_rejected"
        hit_status[i] = resident.HIT_STATUS[label]
        if not bad and selected:
            rows[ev].append(i)
    invalid |= global_bad
    groups, labels, group_ev, spans = [], [], [], []
    for ev, members in enumerate(rows):
        members.sort(key=lambda i: (h["time"][i], h["sid"][i], i))
        chunks = []
        for i in members:
            if not chunks or h["time"][i] - h["time"][chunks[-1][-1]] > 20.:
                chunks.append([])
            chunks[-1].append(i)
        for chunk in chunks:
            span = h["time"][chunk[-1]] - h["time"][chunk[0]]
            geometry_bad = any(
                not (np.all(np.isfinite(h["mid"][i]))
                     and np.all(np.isfinite(h["direction"][i]))
                     and np.any(h["direction"][i] != 0)
                     and np.isfinite(h["wdist"][i])
                     and np.isfinite(h["uvar"][i]) and h["uvar"][i] > 0
                     and np.isfinite(h["hlen"][i]) and h["hlen"][i] > 0)
                for i in chunk)
            if invalid[ev] or geometry_bad:
                label = "invalid_input"
            elif len(chunk) > max_hits:
                label = "hit_capacity"
            elif span > 100.:
                label = "wide_time"
            elif len({int(h["sid"][i]) for i in chunk}) < len(chunk):
                label = "duplicate_straw"
            elif len(chunk) < 4:
                label = "insufficient_hits"
            else:
                label = None  # The grouping oracle does not implement a fitter.
            groups.append(chunk)
            labels.append(label)
            group_ev.append(ev)
            spans.append(span)
    hit_group = np.full(n, -1, np.int64)
    for g, chunk in enumerate(groups):
        hit_group[chunk] = g
    return dict(groups=groups, labels=labels, group_ev=group_ev, spans=spans,
                hit_group=hit_group, hit_status=hit_status, event_n_hits=counts,
                event_invalid=invalid, batch_valid=not global_bad and not np.any(invalid))


def _assert_grouping(out, expected, hits):
    ng, n = len(expected["groups"]), len(hits["ev"])
    assert int(out["n_groups"]) == ng
    assert bool(out["batch_valid"]) == expected["batch_valid"]
    for key in ("hit_group", "hit_status", "event_n_hits", "event_invalid"):
        np.testing.assert_array_equal(out[key], expected[key], err_msg=key)
    np.testing.assert_array_equal(out["group_valid"], np.arange(n) < ng)
    np.testing.assert_array_equal(out["group_ev"][:ng], expected["group_ev"])
    np.testing.assert_array_equal(out["group_n_hits"][:ng], list(map(len, expected["groups"])))
    np.testing.assert_array_equal(out["group_time_span"][:ng], expected["spans"])
    np.testing.assert_array_equal(out["group_ev"][ng:], -1)
    np.testing.assert_array_equal(out["group_status"][ng:], resident.GROUP_STATUS["padding"])
    np.testing.assert_array_equal(out["group_n_hits"][ng:], 0)
    ne = len(out["event_keys"])
    np.testing.assert_array_equal(out["event_n_groups"],
                                  np.bincount(expected["group_ev"], minlength=ne))
    converged = np.zeros(ne, np.int64)
    for g, ev in enumerate(expected["group_ev"]):
        converged[ev] += int(out["group_success"][g])
        assert bool(out["group_success"][g]) == (
            out["group_status"][g] == resident.GROUP_STATUS["converged"])
        label = expected["labels"][g]
        if label is not None:
            assert out["group_status"][g] == resident.GROUP_STATUS[label], (g, label)
            assert not out["group_success"][g]
            assert not out["group_objective_valid"][g]
            assert out["group_iterations"][g] == out["group_accepted_steps"][g] == 0
            assert not np.any(out["hit_poca_valid"][expected["groups"][g]])
    np.testing.assert_array_equal(out["event_n_converged"], converged)


@pytest.fixture(scope="module")
def mixed():
    parts = [_helix(1)]  # event zero intentionally has no rows
    flags = _helix(2, 5)
    flags["flag"] = np.array([_TDIV | (1 << b) for b in (6, 23, 24)] + [0, 1 << 5], np.int64)
    parts += [flags, _helix(3, 3)]
    wide = _helix(4, 7)
    wide["time"] = np.arange(7)*20.
    parts += [wide, _helix(5, 13)]
    duplicate = _helix(6)
    duplicate["sid"][-1] = duplicate["sid"][0]
    parts.append(duplicate)
    bad_geometry = _helix(7)
    bad_geometry["mid"][3, 0] = np.nan
    parts.append(bad_geometry)
    bad_time = _helix(8, 1, start=50., sid_start=30)
    bad_time["time"][0] = np.nan
    parts += [_helix(8), bad_time, _helix(8, 4, start=200., sid_start=40),
              _helix(9), _helix(9, start=200.)]
    line = _helix(10, 4)
    line["mid"] = np.column_stack((np.arange(4), 2*np.arange(4), 3*np.arange(4)))
    parts.append(line)
    boundary = _helix(11, 6)
    boundary["time"] = np.arange(6)*20.
    parts += [boundary, _helix(11, 4, start=120.+1e-9)]
    rejected_nan = _helix(12, 1, sid_start=30)
    rejected_nan["flag"][:] = _TDIV | (1 << 6)
    rejected_nan["time"][:] = np.nan
    rejected_nan["mid"][:] = np.nan
    parts += [_helix(12), rejected_nan]
    masked = _helix(999, 2)
    masked["valid"][:] = False
    masked["sid"][:] = -1
    masked["flag"][:] = -1
    for name in ("time", "wdist", "uvar", "mid", "direction", "hlen"):
        masked[name][:] = np.nan
    parts.append(masked)
    hits = _join(*parts)
    # Deliberately destroy input time and event ordering.
    perm = np.random.default_rng(234).permutation(len(hits["ev"]))
    hits = {k: v[perm].copy() for k, v in hits.items()}
    keys = _keys(14)  # also retain an empty event at the end
    return keys, hits, _run(keys, hits)


def test_independent_grouping_all_events_and_failure_memberships(mixed):
    keys, hits, out = mixed
    expected = _oracle(keys, hits)
    _assert_grouping(out, expected, hits)
    np.testing.assert_array_equal(out["event_n_hits"][[0, 13]], 0)
    np.testing.assert_array_equal(out["event_n_groups"][[0, 2, 13]], 0)
    np.testing.assert_array_equal(out["event_n_groups"][[8, 9, 11]], [2, 2, 2])
    # Explicitly verify that overflow did not silently fit the first max_hits.
    g = np.flatnonzero(out["group_valid"] & (out["group_ev"] == 5))[0]
    assert out["group_n_hits"][g] == 13
    assert np.count_nonzero(out["hit_group"] == g) == 13
    assert out["group_status"][g] == resident.GROUP_STATUS["hit_capacity"]
    assert not out["group_objective_valid"][g]
    assert out["group_seed_status"][g] == out["group_fit_status"][g] == 0
    assert not np.any(out["hit_poca_valid"][out["hit_group"] == g])
    # Chained <=20 gaps do not split a group to get around the 100-span gate.
    g = np.flatnonzero(out["group_valid"] & (out["group_ev"] == 4))[0]
    assert out["group_time_span"][g] == 120.
    assert out["group_status"][g] == resident.GROUP_STATUS["wide_time"]


def test_exact_measured_helix_and_multiple_group_event_counters(mixed):
    _, hits, out = mixed
    for ev in (1, 9, 11, 12):
        groups = np.flatnonzero(out["group_valid"] & (out["group_ev"] == ev))
        assert len(groups) == (2 if ev in (9, 11) else 1)
        np.testing.assert_array_equal(out["group_status"][groups], resident.GROUP_STATUS["converged"])
        np.testing.assert_array_equal(out["group_seed_status"][groups], helixseed_jax.SUCCESS)
        np.testing.assert_array_equal(out["group_fit_status"][groups], helixfit_jax.CONVERGED)
        assert np.all(out["group_objective_valid"][groups])
        assert np.all(out["group_final_objective"][groups] < 1e-18)
        assert out["event_n_converged"][ev] == len(groups)
        for g in groups:
            rows = np.flatnonzero(out["hit_group"] == g)
            assert np.all(out["hit_poca_valid"][rows])
            np.testing.assert_allclose(out["hit_doca"][rows], 0., atol=1e-8)
            np.testing.assert_allclose(out["hit_wire_pull"][rows], 0., atol=1e-8)
            zref = out["group_z_ref"][g]
            phase = (zref-80.)/300.
            parameters = [35.+220.*np.cos(phase), -18.+220.*np.sin(phase),
                          -220./300.*np.sin(phase), 220./300.*np.cos(phase), 1./300.]
            np.testing.assert_allclose(out["group_local_parameters"][g], parameters, rtol=2e-9, atol=2e-8)
            assert len(np.unique(hits["mid"][rows, 2])) == len(rows)
    g = np.flatnonzero(out["group_valid"] & (out["group_ev"] == 10))[0]
    assert out["group_status"][g] == resident.GROUP_STATUS["seed_failed"]
    assert out["group_seed_status"][g] != helixseed_jax.SUCCESS
    assert not out["group_success"][g]


def test_original_rows_preserved_and_not_modified(mixed):
    keys, hits, out = mixed
    before = {k: v.copy() for k, v in hits.items()}
    key_before = keys.copy()
    repeated = _run(keys, hits)
    np.testing.assert_array_equal(keys, key_before)
    np.testing.assert_array_equal(out["event_keys"], keys)
    for k in hits:
        np.testing.assert_array_equal(hits[k], before[k], err_msg=k)
        np.testing.assert_array_equal(repeated["input_"+k], before[k], err_msg=k)
        assert repeated["input_"+k].dtype == before[k].dtype
    assert set(hits) == set(resident.HIT_FIELDS)
    masked = ~hits["valid"]
    assert not np.any(repeated["hit_poca_valid"][masked])
    np.testing.assert_array_equal(repeated["hit_group"][masked], -1)
    np.testing.assert_array_equal(repeated["hit_status"][masked], resident.HIT_STATUS["padding"])


def test_permutation_and_group_batch_invariance(mixed):
    keys, hits, baseline = mixed
    permutation = np.random.default_rng(567).permutation(len(hits["ev"]))
    inverse = np.argsort(permutation)
    shuffled = {k: v[permutation] for k, v in hits.items()}
    for batch in (1, 7):
        out = _run(keys, shuffled, group_batch=batch)
        for name, wanted in baseline.items():
            actual = out[name]
            if name.startswith(("hit_", "input_")):
                actual = actual[inverse]
            if np.asarray(wanted).dtype.kind == "f":
                np.testing.assert_allclose(actual, wanted, rtol=2e-9, atol=2e-8, equal_nan=True, err_msg=name)
            else:
                np.testing.assert_array_equal(actual, wanted, err_msg=name)



def test_equal_times_and_exact_gap_boundaries_are_stable_under_permutation():
    hits = _helix(0)
    # Equal-time rows survive a 100 ns group followed by a gap just above 20 ns.
    hits["time"] = np.array([100., 0., 20., 40., 60., 80., 100., 0.,
                             120.+1e-9, 121., 121., 122.])
    expected = _oracle(_keys(2), hits)
    assert list(map(len, expected["groups"])) == [8, 4]
    assert expected["spans"][0] == 100.
    baseline = _run(_keys(2), hits)
    _assert_grouping(baseline, expected, hits)
    permutation = np.random.default_rng(987).permutation(12)
    shuffled = {k: v[permutation] for k, v in hits.items()}
    out = _run(_keys(2), shuffled)
    _assert_grouping(out, _oracle(_keys(2), shuffled), shuffled)
    np.testing.assert_array_equal(out["hit_group"][np.argsort(permutation)], baseline["hit_group"])
    np.testing.assert_array_equal(out["group_status"], baseline["group_status"])
    np.testing.assert_allclose(out["group_local_parameters"], baseline["group_local_parameters"],
                               rtol=2e-9, atol=2e-8, equal_nan=True)


def test_event_partition_and_event_order_invariance(mixed):
    keys, hits, baseline = mixed
    # Full event keys identify reordered batches whose hit event indices are local.
    for event_ids in ([12, 0, 9], [11, 8, 1, 13]):
        row_indices = np.flatnonzero(np.isin(hits["ev"], event_ids))
        subset = {k: v[row_indices].copy() for k, v in hits.items()}
        event_map = {ev: i for i, ev in enumerate(event_ids)}
        subset["ev"] = np.array([event_map[int(ev)] for ev in subset["ev"]], np.int64)
        out = _run(keys[event_ids], subset)
        _assert_grouping(out, _oracle(keys[event_ids], subset), subset)
        for key in ("event_n_hits", "event_n_groups", "event_n_converged", "event_invalid"):
            np.testing.assert_array_equal(out[key], baseline[key][event_ids], err_msg=key)
        for g in np.flatnonzero(out["group_valid"]):
            local_rows = np.flatnonzero(out["hit_group"] == g)
            original_rows = row_indices[local_rows]
            old_groups = np.unique(baseline["hit_group"][original_rows])
            assert len(old_groups) == 1 and old_groups[0] >= 0
            old_group = old_groups[0]
            np.testing.assert_array_equal(original_rows, np.flatnonzero(baseline["hit_group"] == old_group))
            for key in ("group_status", "group_seed_status", "group_fit_status", "group_success",
                        "group_n_hits", "group_time_span", "group_local_parameters", "group_z_ref",
                        "group_objective_valid", "group_initial_objective", "group_final_objective"):
                np.testing.assert_allclose(out[key][g], baseline[key][old_group],
                                           rtol=2e-9, atol=2e-8, equal_nan=True, err_msg=key)
        for key in ("hit_status", "hit_doca", "hit_wire_pull", "hit_poca_valid"):
            np.testing.assert_allclose(out[key], baseline[key][row_indices], rtol=2e-9, atol=2e-8, err_msg=key)

def test_masked_nan_payload_cannot_change_valid_peer():
    clean = _helix(0)
    masked = _helix(999, 2)
    masked["valid"][:] = False
    for k in ("mid", "direction", "wdist", "uvar", "hlen", "time"):
        masked[k][:] = np.nan
    out = _run(_keys(2), _join(clean, masked))
    reference = _run(_keys(2), clean)
    assert bool(out["batch_valid"])
    for name in ("group_status", "group_success", "group_objective_valid", "group_local_parameters",
                 "group_final_objective", "group_z_ref"):
        np.testing.assert_allclose(out[name][0], reference[name][0], rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(out["event_n_hits"], [12, 0])


def test_each_invalid_geometry_rejects_whole_group_not_just_bad_row():
    cases = [("mid", np.inf), ("direction", np.nan), ("direction", 0.),
             ("wdist", np.inf), ("uvar", 0.), ("uvar", np.nan),
             ("hlen", 0.), ("hlen", np.inf)]
    parts = [_helix(0)]
    for ev, (field, value) in enumerate(cases, 1):
        h = _helix(ev)
        h[field][4] = value
        parts.append(h)
    hits, keys = _join(*parts), _keys(len(parts))
    out = _run(keys, hits)
    _assert_grouping(out, _oracle(keys, hits), hits)
    assert out["batch_valid"]  # geometry is a group failure, not an identity failure
    np.testing.assert_array_equal(out["group_status"][1:len(parts)], resident.GROUP_STATUS["invalid_input"])
    np.testing.assert_array_equal(out["group_n_hits"][:len(parts)], 12)
    assert out["group_success"][0]


def test_invalid_selected_time_invalidates_all_groups_of_only_its_event(mixed):
    _, hits, out = mixed
    np.testing.assert_array_equal(np.flatnonzero(out["event_invalid"]), [8])
    groups = out["group_valid"] & (out["group_ev"] == 8)
    assert np.count_nonzero(groups) == 2
    np.testing.assert_array_equal(out["group_status"][groups], resident.GROUP_STATUS["invalid_input"])
    row = (hits["ev"] == 8) & ~np.isfinite(hits["time"])
    np.testing.assert_array_equal(out["hit_status"][row], resident.HIT_STATUS["invalid_input"])
    np.testing.assert_array_equal(out["hit_group"][row], -1)
    assert not out["event_invalid"][12]  # bad-flag NaN timestamps are not selected



@pytest.mark.parametrize("timestamp", [np.inf, -np.inf])
def test_infinite_selected_timestamp_invalidates_event_not_a_partial_fit(timestamp):
    hits = _join(_helix(0), _helix(0, start=200.), _helix(1))
    hits["time"][0] = timestamp
    out = _run(_keys(3), hits)
    _assert_grouping(out, _oracle(_keys(3), hits), hits)
    np.testing.assert_array_equal(out["event_invalid"], [True, False, False])
    np.testing.assert_array_equal(out["event_n_converged"], [0, 1, 0])
    np.testing.assert_array_equal(out["group_status"][:2], resident.GROUP_STATUS["invalid_input"])
    assert out["hit_group"][0] == -1
    assert out["hit_status"][0] == resident.HIT_STATUS["invalid_input"]

def test_invalid_straw_ids_and_flags_fail_event_without_poisoning_peer():
    cases = [("sid", -1), ("sid", 96), ("sid", 6 << 7), ("sid", 65536), ("sid", 36 << 10),
             ("flag", -1), ("flag", 2**32)]
    parts = [_helix(0)]
    for ev, (field, value) in enumerate(cases, 1):
        h = _helix(ev)
        h[field][0] = value
        parts.append(h)
    hits, keys = _join(*parts), _keys(len(parts))
    out = _run(keys, hits)
    _assert_grouping(out, _oracle(keys, hits), hits)
    np.testing.assert_array_equal(out["event_invalid"], [False]+[True]*len(cases))
    assert out["group_success"][0]
    assert not out["batch_valid"]


@pytest.mark.parametrize("kind", ["nan_field", "inf_field", "zero_field", "negative_key",
                                  "duplicate_key", "negative_ev", "out_of_range_ev"])
def test_invalid_global_identity_or_field_fails_closed(kind):
    keys, hits, bz = _keys(2), _helix(0), 1.
    if kind.endswith("field"):
        bz = {"nan_field": np.nan, "inf_field": np.inf, "zero_field": 0.}[kind]
    elif kind == "negative_key":
        keys[1, 1] = -1
    elif kind == "duplicate_key":
        keys[1] = keys[0]
    else:
        hits["ev"][0] = -1 if kind == "negative_ev" else len(keys)
    out = _run(keys, hits, bz=bz)
    _assert_grouping(out, _oracle(keys, hits, bz=bz), hits)
    assert not out["batch_valid"]
    assert np.all(out["event_invalid"])
    assert not np.any(out["group_success"])


def test_fit_failure_is_not_mislabeled_as_seed_failure():
    hits = _helix(0)
    hits["mid"][4, 0] += 2.
    out = _run(_keys(2), hits, max_iterations=0)
    assert out["group_seed_status"][0] == helixseed_jax.SUCCESS
    assert out["group_status"][0] == resident.GROUP_STATUS["fit_failed"]
    assert out["group_fit_status"][0] == helixfit_jax.MAX_ITERATIONS
    assert out["group_objective_valid"][0]
    assert not out["group_success"][0]
    assert out["group_iterations"][0] == 0
    assert out["group_n_hits"][0] == len(hits["ev"])
    np.testing.assert_array_equal(out["event_n_converged"], [0, 0])


@pytest.mark.parametrize("ne,n,present", [(0, 0, False), (3, 0, False), (0, 2, False),
                                         (0, 2, True), (3, 2, False)])
def test_empty_shapes_and_all_padding(ne, n, present):
    hits = _helix(0, n)
    hits["valid"][:] = present
    if not present:
        hits["mid"][:] = np.nan
    out = _run(_keys(ne), hits)
    assert out["event_keys"].shape == (ne, 3)
    assert out["group_local_parameters"].shape == (n, 5)
    assert out["n_groups"] == 0
    assert bool(out["batch_valid"]) == (not present or ne > 0)
    for name in ("event_n_hits", "event_n_groups", "event_n_converged"):
        assert out[name].shape == (ne,)
        np.testing.assert_array_equal(out[name], 0)
    np.testing.assert_array_equal(out["hit_group"], -1)
    np.testing.assert_array_equal(out["hit_status"], resident.HIT_STATUS["invalid_input"] if present else 0)
    assert not np.any(out["group_valid"])
    assert not np.any(out["group_success"])
    for k in hits:
        np.testing.assert_array_equal(out["input_"+k], hits[k])


def _primitive_names(value):
    """Descend through jit/while/cond sub-jaxprs, not only the outer wrapper."""
    if hasattr(value, "eqns"):
        for eqn in value.eqns:
            yield eqn.primitive.name
            yield from _primitive_names(eqn.params)
    elif hasattr(value, "jaxpr"):
        yield from _primitive_names(value.jaxpr)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _primitive_names(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _primitive_names(child)


def test_outer_jit_lower_compile_and_execute_without_host_boundaries():
    hits = _join(_helix(0), _helix(0, start=200.), _helix(1, 3))
    keys, device_hits, field = jax.device_put((_keys(3), hits, np.float64(1.)))
    outer = jax.jit(lambda k, h, b: resident.reconstruct(
        k, h, bz=b, max_hits=12, group_batch=2, max_iterations=8))
    with jax.transfer_guard("disallow_explicit"):
        traced = jax.make_jaxpr(outer)(keys, device_hits, field)
        lowered = outer.lower(keys, device_hits, field)
        executable = lowered.compile()
        out = executable(keys, device_hits, field)
        jax.block_until_ready(out)
        again = outer(keys, device_hits, field)
        jax.block_until_ready(again)
    primitives = set(_primitive_names(traced))
    assert "while" in primitives  # seed/fit work was actually traced
    assert not any("callback" in p or "host" in p for p in primitives), primitives
    assert all(isinstance(v, jax.Array) for v in out.values())
    host = jax.device_get(out)  # the one explicit result boundary is outside the guard
    np.testing.assert_array_equal(host["event_n_converged"], [2, 0, 0])
    np.testing.assert_array_equal(host["event_n_groups"], [2, 1, 0])
    _assert_grouping(host, _oracle(np.asarray(keys), hits), hits)


@pytest.mark.parametrize("options,match", [
    ({"max_hits": 3}, "max_hits"), ({"max_hits": 513}, "max_hits"),
    ({"max_hits": True}, "max_hits"), ({"max_hits": 12.5}, "max_hits"),
    ({"group_batch": 0}, "group_batch"), ({"group_batch": 2049}, "group_batch"),
    ({"group_batch": False}, "group_batch"),
    ({"max_iterations": -1}, "max_iterations"), ({"max_iterations": 121}, "max_iterations"),
    ({"max_iterations": True}, "max_iterations"),
    ({"max_hits": 512, "group_batch": 513}, "workspace"),
])
def test_static_workspace_bounds_rejected_before_allocation(options, match):
    with pytest.raises(ValueError, match=match):
        _run(_keys(0), _helix(0, 0), **options)


@pytest.mark.parametrize("options", [dict(max_hits=4, group_batch=1, max_iterations=0),
                                     dict(max_hits=512, group_batch=512, max_iterations=120),
                                     dict(max_hits=128, group_batch=2048)])
def test_static_boundaries_are_inclusive_on_empty_input(options):
    out = _run(_keys(1), _helix(0, 0), **options)
    assert out["batch_valid"] and out["n_groups"] == 0


@pytest.mark.parametrize("case,match", [
    ("key_shape", "event_keys"), ("key_float", "event_keys"), ("key_unsigned64", "event_keys"),
    ("missing", "exactly"), ("extra", "exactly"), ("mid_shape", "mid"),
    ("time_shape", "time"), ("valid_int", "valid"), ("sid_float", "sid"),
    ("sid_unsigned64", "sid"), ("complex_time", "time"), ("field_shape", "scalar"),
])
def test_static_schema_errors(case, match):
    keys, hits, options = _keys(1), _helix(0, 0), {}
    if case == "key_shape":
        keys = keys[:, :2]
    elif case == "key_float":
        keys = keys.astype(float)
    elif case == "key_unsigned64":
        keys = keys.astype(np.uint64)
    elif case == "missing":
        del hits["hlen"]
    elif case == "extra":
        hits["extra"] = np.empty(0)
    elif case == "mid_shape":
        hits["mid"] = np.empty((0, 2))
    elif case == "time_shape":
        hits["time"] = np.empty((0, 1))
    elif case == "valid_int":
        hits["valid"] = hits["valid"].astype(np.int64)
    elif case == "sid_float":
        hits["sid"] = hits["sid"].astype(float)
    elif case == "sid_unsigned64":
        hits["sid"] = hits["sid"].astype(np.uint64)
    elif case == "complex_time":
        hits["time"] = hits["time"].astype(complex)
    elif case == "field_shape":
        options["bz"] = np.array([1.])
    with pytest.raises(ValueError, match=match):
        _run(keys, hits, **options)

@pytest.mark.parametrize("bz", [True, 1+2j])
def test_field_dtype_must_be_real_numeric(bz):
    with pytest.raises(ValueError, match="real numeric"):
        _run(_keys(1), _helix(0), bz=bz)


def test_stage_attempt_flags_distinguish_padding_rejection_and_fit():
    hits = _join(_helix(0), _helix(1, 3), _helix(2))
    hits["sid"][-1] = hits["sid"][-2]
    out = _run(_keys(4), hits)
    np.testing.assert_array_equal(out["group_seed_attempted"][:3], [True, False, False])
    np.testing.assert_array_equal(out["group_fit_attempted"][:3], [True, False, False])
    assert not out["group_seed_attempted"][3:].any()
    assert out["group_rank"][0] == 5 and np.isfinite(out["group_condition"][0])
