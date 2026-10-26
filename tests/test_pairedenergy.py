"""Pinned paired mean transport: exact geometry/identities and primary-KE accounting only."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import eloss
from trkfastsim import pairedenergy as pe, transportledger as ledger


REFS = dict(tracker_ref="1" * 40, core_ref="2" * 40)


def spans(rows):
    return pd.DataFrame(rows, columns=pe.COLS).astype(
        {"lane": "int64", "s0_mm": "float64", "s1_mm": "float64", "material": "object"})


@pytest.fixture
def saved(tmp_path):
    """Produce real hashed files without invoking geometry navigation or production runners."""
    count = 0

    def make(rows=None, *, outer=None, lengths=(4.,), kinetic=(20.,), keys=None, mutate=None):
        nonlocal count
        directory = tmp_path / str(count)
        count += 1
        directory.mkdir()
        n = len(lengths)
        if keys is None:
            keys = np.column_stack((np.full(n, 7), np.full(n, 2), np.full(n, 3), np.arange(n)[::-1] + 1))
        seeds = pd.DataFrame(np.asarray(keys, dtype=np.int64), columns=pe.KEY)
        seeds[["x", "y", "z"]] = np.arange(n * 3, dtype=float).reshape(n, 3)
        kinetic = np.asarray(kinetic, float)
        seeds["px"] = np.sqrt(kinetic * (kinetic + 2 * eloss.MMU))
        seeds["py"], seeds["pz"], seeds["length_mm"] = 0., 0., lengths
        if rows is None:
            rows = [(i, 0., length, "G4_Al") for i, length in enumerate(lengths) if length > 0]
        tables = {}
        for arm, records in (("inner", rows), ("outer", rows if outer is None else outer)):
            table = spans(records)
            for key in pe.KEY:
                table[key] = seeds[key].to_numpy()[table.lane.to_numpy()]
            tables[arm] = table
        decisions = seeds[pe.KEY].assign(lane=np.arange(n, dtype=np.int64),
            requested_length_mm=lengths, bounded_pair_admitted=True, admitted_for_transport=False)
        refusals = pd.DataFrame(columns=pe.KEY + ["lane", "reason"])
        if mutate is not None:
            mutate(seeds, tables, decisions, refusals)
        seeds.to_csv(directory / "seeds.csv", index=False)
        decisions.to_csv(directory / "envelope_0mm_decisions.csv", index=False)
        refusals.to_csv(directory / "envelope_0mm_refusals.csv", index=False)
        for arm, table in tables.items():
            table.to_csv(directory / f"envelope_0mm_{arm}_intervals.csv", index=False)
        gdml = directory / "synthetic.gdml"
        gdml.write_text("<gdml/>\n")
        control = directory / "control.json"
        control.write_text('{"synthetic": true}\n')
        summary = dict(schema="trkfastsim.support-envelope-replay.v1", complete=True,
            mode="support-navigation-torus-envelope-v1", admitted_for_transport=False, seed_rows=n,
            candidates=[dict(endpoint_budget_mm=0., all_pairs_admitted=True, refused_rays=0,
                bounded_pair_rays=n, inner_interval_rows=len(tables["inner"]), outer_interval_rows=len(tables["outer"]))],
            inputs={"seeds.csv": dict(path=str(directory / "seeds.csv"), sha256=pe.sha256(directory / "seeds.csv"))},
            control_inputs={"control": dict(path=str(control), sha256=pe.sha256(control))},
            gdml=dict(path=str(gdml), sha256=pe.sha256(gdml)),
            settings=dict(max_component_mm=1e-6, synthetic=True),
            sources_sha256={"synthetic_geometry": "4" * 64},
            seed_domain=dict(seed_rows=n, reference_generated_primaries=50, synthetic=True))
        (directory / "summary.json").write_text(json.dumps(summary))
        pin = pe.pin_inputs(directory, directory / "pin.json", endpoint_budget_mm=0.)
        return directory, pin

    return make


def load(saved, *args, **kwargs):
    directory, pin = saved(*args, **kwargs)
    return pe.load_pinned(pin["pin"], pin["sha256"]), directory, pin


@pytest.fixture
def no_rng(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("deterministic paired mean transport must neither sample loss nor create an RNG")

    monkeypatch.setattr(eloss, "sample_loss", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)


@pytest.mark.parametrize("rows,lengths,match", [
    ([(0, .1, 1, "vacuum")], [1], "gap or overlap"),
    ([(0, 0, .4, "vacuum"), (0, .5, 1, "vacuum")], [1], "gap or overlap"),
    ([(0, 0, .6, "vacuum"), (0, .5, 1, "vacuum")], [1], "gap or overlap"),
    ([(0, 0, .5, "vacuum"), (0, np.nextafter(.5, 1), 1, "vacuum")], [1], "gap or overlap"),
    ([(0, 0, .5, "vacuum"), (0, np.nextafter(.5, 0), 1, "vacuum")], [1], "gap or overlap"),
    ([(0, 0, np.nextafter(1., 0), "vacuum")], [1], "truncated"),
    ([(0, 0, 1, "vacuum")], [1, 1], "missing positive-length lane"),
    ([], [1], "missing positive-length lane"),
    ([(0, 0, 1, "aluminium")], [1], "core registry"),
    ([(0, 0, 1, "")], [1], "nonempty material"),
    ([(0, 0, 1, None)], [1], "nonempty material"),
    ([(0, 0, 0, "vacuum")], [0], "positive in-domain"),
    ([(0, 0, 1, "vacuum")], [0], "positive in-domain"),
    ([(0, -.1, 1, "vacuum")], [1], "positive in-domain"),
    ([(0, 0, np.inf, "vacuum")], [1], "positive in-domain"),
    ([(0, np.nan, 1, "vacuum")], [1], "positive in-domain"),
    ([(0, 1, .5, "vacuum")], [1], "positive in-domain"),
    ([(1, 0, 1, "vacuum"), (0, 0, 1, "vacuum")], [1, 1], "physically ordered"),
    ([(-1, 0, 1, "vacuum")], [1], "invalid interval lanes"),
    ([(1, 0, 1, "vacuum")], [1], "invalid interval lanes"),
])
def test_canonical_rejects_inexact_or_invalid_coverage(rows, lengths, match):
    with pytest.raises(ValueError, match=match):
        pe.canonical_intervals(spans(rows), lengths)


@pytest.mark.parametrize("lengths", [[-1], [np.nan], [np.inf], [[0]], 0.])
def test_canonical_rejects_invalid_lengths(lengths):
    with pytest.raises(ValueError, match="invalid requested lengths"):
        pe.canonical_intervals(spans([]), lengths)


@pytest.mark.parametrize("lane", [0., True, "0"])
def test_canonical_requires_integer_lane_dtype(lane):
    table = spans([(0, 0, 1, "vacuum")])
    table["lane"] = [lane]
    with pytest.raises(ValueError, match="invalid interval lanes"):
        pe.canonical_intervals(table, [1])


@pytest.mark.parametrize("lengths", [[], [0.], [0., 0.]])
def test_canonical_empty_and_zero_length_are_valid(lengths):
    table, mapping = pe.canonical_intervals(spans([]), lengths)
    assert table.empty and mapping.empty
    assert list(table.columns) == pe.COLS
    assert table.lane.dtype == np.dtype("int64")
    assert list(mapping.columns) == ["source_row", "canonical_row", "lane"]


def test_canonical_coalesces_exactly_is_idempotent_and_does_not_mutate():
    table = spans([(0, 0, .25, "vacuum"), (0, .25, 1, "vacuum"),
        (0, 1, 1.5, "G4_Al"), (0, 1.5, 2, "G4_Al"), (0, 2, 3, "vacuum"),
        (2, 0, .75, "vacuum"), (2, .75, 1, "vacuum")])
    table.index = [19, 18, 17, 16, 15, 14, 13]
    table["evidence"] = "untouched"
    before = table.copy(deep=True)
    lengths = np.array([3., 0., 1.])
    canon, mapping = pe.canonical_intervals(table, lengths)
    expected = spans([(0, 0, 1, "vacuum"), (0, 1, 2, "G4_Al"), (0, 2, 3, "vacuum"), (2, 0, 1, "vacuum")])
    pd.testing.assert_frame_equal(canon, expected)
    pd.testing.assert_frame_equal(table, before)
    np.testing.assert_array_equal(lengths, [3, 0, 1])
    np.testing.assert_array_equal(mapping.source_row, np.arange(7))
    np.testing.assert_array_equal(mapping.canonical_row, [0, 0, 1, 1, 2, 3, 3])
    np.testing.assert_array_equal(mapping.lane, table.lane)
    again, again_map = pe.canonical_intervals(canon, lengths)
    pd.testing.assert_frame_equal(again, canon)
    np.testing.assert_array_equal(again_map.canonical_row, np.arange(4))
    canon.loc[0, "s1_mm"] = .125
    pd.testing.assert_frame_equal(table, before)


def test_canonical_schema_required():
    with pytest.raises(ValueError, match="schema"):
        pe.canonical_intervals(spans([]).drop(columns="material"), [])


@pytest.mark.parametrize("column", pe.KEY + ["lane"])
def test_csv_exact_int64_identity_without_float_roundtrip(column):
    values = [0, 2**53 + 1, 2**53 + 2, np.iinfo(np.int64).max]
    raw = (column + ",value\n" + "".join(f"{value},1.25\n" for value in values)).encode()
    table = pe._read_csv(raw, len(values), [column])
    assert table[column].dtype == np.dtype("int64")
    assert table[column].tolist() == values


@pytest.mark.parametrize("bad", ["1.0", "1e2", "-1", str(2**63), "18446744073709551616", "", "01", "+1", "True", " 1"])
@pytest.mark.parametrize("column", pe.KEY + ["lane"])
def test_csv_rejects_float_negative_overflow_and_mixed_identity(column, bad):
    raw = f"{column},value\n9007199254740993,0\n{bad},1\n".encode()
    with pytest.raises(ValueError, match="exact nonnegative decimal integers|exceeds int64"):
        pe._read_csv(raw, 2, [column])


def test_csv_row_bound_and_required_identity():
    with pytest.raises(ValueError, match="row bound"):
        pe._read_csv(b"run\n1\n2\n", 1, ["run"])
    with pytest.raises(ValueError, match="missing integer identity"):
        pe._read_csv(b"event\n1\n", 1, ["run"])


def test_pin_load_preserves_large_full_identities_and_source_mapping(saved):
    keys = [[2**53 + 1, 2**53 + 2, 2**63 - 1, 2**63 - 2], [2**53 + 1, 2**53 + 2, 2**63 - 1, 2**53 + 3]]
    rows = [(0, 0, 1, "vacuum"), (0, 1, 2, "vacuum"), (1, 0, 1, "G4_Al")]
    bundle, directory, pin = load(saved, rows, lengths=(2., 1.), kinetic=(10., 12.), keys=keys)
    np.testing.assert_array_equal(bundle["seeds"][pe.KEY], keys)
    np.testing.assert_array_equal(bundle["direction"], [[1., 0., 0.], [1., 0., 0.]])
    np.testing.assert_allclose(bundle["kinetic_MeV"], [10., 12.], rtol=1e-15)
    np.testing.assert_array_equal(bundle["mapping"]["inner"].canonical_row, [0, 0, 1])
    assert pin["files"] == 6
    assert bundle["pin_sha256"] == pe.sha256(pin["pin"])
    with pytest.raises(FileExistsError):
        pe.pin_inputs(directory, pin["pin"], endpoint_budget_mm=0.)


@pytest.mark.parametrize("target", ["pin", "seeds.csv", "envelope_0mm_inner_intervals.csv", "control.json", "synthetic.gdml"])
def test_load_rejects_tampering_before_transport(saved, target):
    directory, pin = saved()
    path = directory / ("pin.json" if target == "pin" else target)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch|SHA256 mismatch"):
        pe.load_pinned(pin["pin"], pin["sha256"])


@pytest.mark.parametrize("column,value", [("length_mm", -1.), ("length_mm", np.nan),
    ("length_mm", np.inf), ("px", 0.), ("px", np.inf)])
def test_load_rejects_malicious_seed_lengths_and_momenta(saved, column, value):
    def mutate(seeds, tables, decisions, refusals):
        seeds.loc[0, column] = value

    directory, pin = saved(mutate=mutate)
    with pytest.raises(ValueError, match="finite|nonnegative"):
        pe.load_pinned(pin["pin"], pin["sha256"])


@pytest.mark.parametrize("column,value", [("bounded_pair_admitted", False), ("admitted_for_transport", True),
    ("bounded_pair_admitted", "unreviewed"), ("admitted_for_transport", "not-false"),
    ("requested_length_mm", -1.), ("requested_length_mm", np.nan), ("requested_length_mm", 3.)])
def test_load_rejects_malicious_decision_lengths_and_flags(saved, column, value):
    def mutate(seeds, tables, decisions, refusals):
        decisions[column] = value

    directory, pin = saved(mutate=mutate)
    with pytest.raises(ValueError, match="all original lanes|decision length"):
        pe.load_pinned(pin["pin"], pin["sha256"])


def test_load_rejects_identity_mismatch_without_large_integer_rounding(saved):
    def mutate(seeds, tables, decisions, refusals):
        tables["outer"].loc[0, "sp"] = 2**53 + 2

    directory, pin = saved(keys=[[1, 2, 3, 2**53 + 1]], mutate=mutate)
    with pytest.raises(ValueError, match="identity disagrees"):
        pe.load_pinned(pin["pin"], pin["sha256"])


def assert_same_numeric_result(left, right):
    for key in ("final_ke_MeV", "position_mm", "traveled_mm", "status"):
        np.testing.assert_equal(left[key], right[key])
    for key in ledger.FIELDS:
        np.testing.assert_equal(left["history"][key], right["history"][key])


def test_real_mean_same_arms_repeatable_zero_rng_and_provenance(saved, no_rng):
    rows = [(0, 0, 1, "vacuum"), (0, 1, 2, "G4_Al"), (0, 2, 3, "G4_Cu"),
            (1, 0, 3, "G4_Al")]
    bundle, _, _ = load(saved, rows, lengths=(3., 3.), kinetic=(25., 40.))
    before = copy.deepcopy(bundle)
    a = pe.transport_pair(bundle, **REFS)
    b = pe.transport_pair(bundle, **REFS)
    assert_same_numeric_result(a["results"]["inner"], a["results"]["outer"])
    assert_same_numeric_result(a["results"]["inner"], b["results"]["inner"])
    np.testing.assert_array_equal(a["paired"].outer_minus_inner_ke_MeV, 0.)
    np.testing.assert_array_equal(a["paired"].outer_minus_inner_path_mm, 0.)
    assert not a["paired"].status_changed.any()
    pd.testing.assert_frame_equal(bundle["seeds"], before["seeds"])
    for arm in ("inner", "outer"):
        pd.testing.assert_frame_equal(bundle["intervals"][arm], before["intervals"][arm])
        np.testing.assert_array_equal(a["finals"][arm][pe.KEY], bundle["seeds"][pe.KEY])
        h = a["results"][arm]["history"]
        np.testing.assert_allclose(h.accounting()["primary_residual_MeV"], 0., atol=1e-12)
        assert np.isnan(h["dt_ns"][h["path_mm"] > 0]).all()
        assert h.metadata["settings"]["ordered_transport"]["loss_model"] == "mean"
        assert h.metadata["settings"]["geometry_envelope"] == arm
    metadata = a["metadata"]
    assert metadata["geometry_sha256"] == bundle["summary"]["gdml"]["sha256"]
    assert metadata["tracker_commit"] == REFS["tracker_ref"]
    assert metadata["core_commit"] == REFS["core_ref"]
    assert {item["name"]: item["sha256"] for item in metadata["inputs"]} == bundle["pin"]["files"]
    settings = metadata["settings"]
    assert settings["mode"] == pe.MODE and settings["rng"].startswith("none;")
    assert settings["geometry_settings"] == bundle["summary"]["settings"]
    assert settings["geometry_source_hashes"] == bundle["summary"]["sources_sha256"]
    assert settings["reference_counts_inherited"] == bundle["summary"]["seed_domain"]
    assert settings["geometry_admitted_for_transport"] is False
    assert "no upstream history" in settings["seed_domain"]
    assert settings["input_pin_sha256"] == bundle["pin_sha256"]
    assert (a["finals"]["inner"].primary_debit_MeV > 0).all()


def test_multiple_primary_order_stops_residual_and_energy_closure(saved, monkeypatch, no_rng):
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.ones_like(p))
    rows = [(0, 0, 3, "G4_Al"), (0, 3, 4, "vacuum"), (1, 0, 4, "G4_Al"),
            (2, 0, 4, "vacuum"), (3, 0, 4, "vacuum")]
    bundle, _, _ = load(saved, rows, lengths=(4., 4., 4., 4., 0.), kinetic=(1.5, 10., .5, 1., 8.))
    # Exercise exact zero and exact cutoff independently of the momentum-to-KE round trip.
    bundle["kinetic_MeV"] = np.array([1.5, 10., .5, 0., 8.])
    result = pe.transport_pair(bundle, fractional_loss=1., **REFS)
    for arm in ("inner", "outer"):
        final, h = result["finals"][arm], result["results"][arm]["history"]
        np.testing.assert_array_equal(final[pe.KEY], bundle["seeds"][pe.KEY])
        np.testing.assert_array_equal(final.final_ke_MeV, [.5, 6., .5, 0., 8.])
        np.testing.assert_array_equal(final.traveled_mm, [1., 4., 0., 0., 0.])
        assert final.status.tolist() == ["model_stop", "transmitted", "model_stop", "exhausted", "transmitted"]
        np.testing.assert_allclose(final.primary_debit_MeV + final.final_ke_MeV, final.initial_ke_MeV, atol=1e-14)
        np.testing.assert_allclose(h.accounting()["primary_residual_MeV"], 0., atol=1e-14)
        for lane, key in enumerate(bundle["seeds"]["sp"]):
            mask = h["sp"] == key
            assert mask.any()
            np.testing.assert_allclose(h["applied_loss_MeV"][mask].sum(), final.primary_debit_MeV.iloc[lane])
            np.testing.assert_allclose(h["path_mm"][mask].sum(), final.traveled_mm.iloc[lane])
        stop = h["sp"] == bundle["seeds"]["sp"].iloc[0]
        assert h["ke_out_MeV"][stop][-1] == .5
        assert h["terminal"][stop][-1] == "model_stop"
        assert np.isnan(h["deposited_MeV"][stop]).all()
        np.testing.assert_allclose(final[["final_x_mm", "final_y_mm", "final_z_mm"]],
            bundle["seeds"][["x", "y", "z"]].to_numpy() + bundle["direction"] * final.traveled_mm.to_numpy()[:, None])


def test_different_arms_are_paired_without_dropping_stops(saved, monkeypatch, no_rng):
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.ones_like(p))
    bundle, _, _ = load(saved, [(0, 0, 4, "vacuum")], outer=[(0, 0, 4, "G4_Al")], kinetic=(1.5,))
    bundle["kinetic_MeV"][:] = 1.5
    result = pe.transport_pair(bundle, fractional_loss=1., **REFS)
    pair = result["paired"].iloc[0]
    assert pair.inner_status == "transmitted" and pair.outer_status == "model_stop"
    assert pair.outer_minus_inner_ke_MeV == -1.
    assert pair.outer_minus_inner_path_mm == -3.
    assert pair.status_changed


@pytest.mark.parametrize("fractional_loss", [.05, 1.])
def test_arbitrary_subdivision_has_identical_final_and_all_history_columns(saved, no_rng, fractional_loss):
    coarse = [(0, 0, .5, "vacuum"), (0, .5, 4, "G4_Al"), (0, 4, 6, "G4_Cu"),
              (1, 0, 7, "G4_Al")]
    fine = []
    for lane, start, end, material in coarse:
        boundaries = start + (end - start) * np.array([0., .013, .11, .333, .71, .999, 1.])
        fine.extend((lane, left, right, material) for left, right in zip(boundaries[:-1], boundaries[1:]))
    a, _, _ = load(saved, coarse, lengths=(6., 7.), kinetic=(30., 50.))
    b, _, _ = load(saved, fine, lengths=(6., 7.), kinetic=(30., 50.))
    for arm in ("inner", "outer"):
        pd.testing.assert_frame_equal(a["intervals"][arm], b["intervals"][arm])
        assert len(b["mapping"][arm]) > len(a["mapping"][arm])
    ra = pe.transport_pair(a, max_step_mm=.7, fractional_loss=fractional_loss, **REFS)
    rb = pe.transport_pair(b, max_step_mm=.7, fractional_loss=fractional_loss, **REFS)
    for arm in ("inner", "outer"):
        assert_same_numeric_result(ra["results"][arm], rb["results"][arm])
        pd.testing.assert_frame_equal(ra["finals"][arm], rb["finals"][arm])
    pd.testing.assert_frame_equal(ra["paired"], rb["paired"])


@pytest.mark.parametrize("flag", ["complete", "all_pairs_admitted"])
def test_summary_requires_boolean_admission_flags(saved, flag):
    directory, _ = saved()
    summary = json.loads((directory / "summary.json").read_text())
    target = summary if flag == "complete" else summary["candidates"][0]
    target[flag] = "false"
    with pytest.raises(ValueError):
        pe._summary(summary, 0.)


def test_csv_duplicate_identity_header_rejected():
    with pytest.raises(ValueError, match="unique nonempty raw CSV headers"):
        pe._read_csv(b"run,run\n1,2\n", 1, ["run"])


@pytest.mark.parametrize("case", ["duplicate_seed", "missing_decision", "reversed_decisions", "refusal", "bad_lane"])
def test_load_requires_every_original_lane_exactly_once_and_no_refusals(saved, case):
    def mutate(seeds, tables, decisions, refusals):
        if case == "duplicate_seed":
            seeds.loc[1, pe.KEY] = seeds.loc[0, pe.KEY].to_numpy()
        elif case == "missing_decision":
            decisions.drop(index=1, inplace=True)
        elif case == "reversed_decisions":
            decisions.iloc[:] = decisions.iloc[::-1].to_numpy()
        elif case == "refusal":
            refusals.loc[0] = [7, 2, 3, 2, 0, "unresolved"]
        else:
            tables["inner"].loc[0, "lane"] = 2

    _, pin = saved(lengths=(4., 4.), kinetic=(10., 20.), mutate=mutate)
    with pytest.raises(ValueError, match="seed count/identity|all original lanes|invalid lane"):
        pe.load_pinned(pin["pin"], pin["sha256"])


def test_zero_length_pair_retains_primary_state(saved, no_rng):
    bundle, _, _ = load(saved, [], lengths=(0.,), kinetic=(20.,))
    result = pe.transport_pair(bundle, **REFS)
    for arm in ("inner", "outer"):
        r = result["results"][arm]
        np.testing.assert_array_equal(r["final_ke_MeV"], bundle["kinetic_MeV"])
        np.testing.assert_array_equal(r["traveled_mm"], [0.])
        assert r["status"].tolist() == ["transmitted"]
        assert len(r["history"]) == 1
        assert r["history"]["applied_loss_MeV"][0] == 0.


def test_updated_energy_reaches_next_material_in_physical_order(saved, monkeypatch, no_rng):
    calls = []

    def mean(mat, momentum, cut):
        calls.append((mat.name, momentum.copy(), cut))
        return np.ones_like(momentum)

    monkeypatch.setattr(eloss, "dedx_restricted", mean)
    bundle, _, _ = load(saved, [(0, 0, 1, "G4_Cu"), (0, 1, 2, "G4_Al")], lengths=(2.,))
    bundle["kinetic_MeV"][:] = 20.
    result = pe.transport_pair(bundle, fractional_loss=1., **REFS)
    assert [name for name, _, _ in calls] == ["G4_Cu", "G4_Al", "G4_Cu", "G4_Al"]
    for index in (1, 3):
        np.testing.assert_allclose(calls[index][1], [np.sqrt(19. * (19. + 2 * eloss.MMU))])
    assert all(cut == 1e9 for _, _, cut in calls)
    np.testing.assert_array_equal(result["results"]["inner"]["final_ke_MeV"], [18.])


def test_mean_can_exhaust_without_resurrecting_in_later_vacuum(saved, monkeypatch, no_rng):
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.ones_like(p))
    bundle, _, _ = load(saved, [(0, 0, 3, "G4_Al"), (0, 3, 4, "vacuum")], kinetic=(2.,))
    bundle["kinetic_MeV"][:] = 2.
    result = pe.transport_pair(bundle, fractional_loss=1., max_step_mm=3., **REFS)
    for arm in ("inner", "outer"):
        r = result["results"][arm]
        np.testing.assert_array_equal(r["final_ke_MeV"], [0.])
        np.testing.assert_array_equal(r["traveled_mm"], [2.])
        assert r["status"].tolist() == ["exhausted"]
        assert r["history"]["terminal"][-1] == "exhausted"
        assert r["history"]["applied_loss_MeV"].sum() == 2.


def test_mean_step_cap_never_silently_transmits(saved, monkeypatch, no_rng):
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.ones_like(p))
    bundle, _, _ = load(saved)
    with pytest.raises(RuntimeError, match="max_steps"):
        pe.transport_pair(bundle, max_step_mm=.1, max_steps=1, **REFS)
