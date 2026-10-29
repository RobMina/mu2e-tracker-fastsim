"""Synthetic full-sidecar partition contracts; no captures, builds or remote IO."""
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import f1_transition_audit as D


PARTITIONS = ("initial", "same_meta", *D.LABELS, "new_extension")
CAPTURE_COUNTS = (2538, 31600, 7520, 2478, 17405, 81)


def receipts(per_partition=1):
    """One target, differing iteration/effect keys: event counts cannot prove coverage."""
    identities = np.array([
        [1, 2, 3, 4, group, lane]
        for group in range(len(PARTITIONS)) for lane in range(per_partition)
    ], dtype=np.int64)
    rows = ak.Array({name: identities[:, i] for i, name in enumerate(D.HIT_KEYS)})
    parts = identities.reshape(len(PARTITIONS), per_partition, len(D.HIT_KEYS))
    cad = {"population": {
        "hit_keys": parts[0].copy(),
        "reference_geometry": {field: {"exact": {"passed": True}}
                               for field in D.P.STATE_FIELDS},
    }}
    later = {"hits": {"following_hit_keys": parts[1].copy(),
                      "final_ca_agreement": {"exact": True}}}
    retained = {group: {"following_keys": parts[i + 2].copy(),
                        "final": {"agreement": {"exact": True}}}
                for i, group in enumerate(D.LABELS)}
    new = {"hit_keys": parts[-1].copy(), "final": {"agreement": {"exact": True}}}
    return rows, cad, later, retained, new


def partition_slot(case, name):
    _, cad, later, retained, new = case
    if name == "initial":
        return cad["population"], "hit_keys"
    if name == "same_meta":
        return later["hits"], "following_hit_keys"
    if name == "new_extension":
        return new, "hit_keys"
    return retained[name], "following_keys"


def exact_slot(case, name, field=None):
    _, cad, later, retained, new = case
    if name == "initial":
        return cad["population"]["reference_geometry"][field], "exact"
    if name == "same_meta":
        return later["hits"], "final_ca_agreement"
    return (new if name == "new_extension" else retained[name])["final"], "agreement"


def test_six_partitions_cover_every_full_snapshot_key_without_count_inference():
    assert sum(CAPTURE_COUNTS) == 61622
    result = D.snapshot_coverage(*receipts())
    assert result["n_snapshots"] == 6
    assert result["n_targets"] == 1
    assert result["by_partition"] == dict.fromkeys(PARTITIONS, 1)
    assert result["n_uncompared"] == 0
    assert result["disjoint_full_key_partition"] is True
    assert result["retained_geometry_exact"] is True
    assert "not autonomous" in result["scope"]


def test_sidecar_and_partition_order_do_not_matter_or_mutate_inputs():
    case = receipts(2)
    baseline = D.snapshot_coverage(*case)
    case = (case[0][[11, 2, 9, 0, 7, 4, 1, 10, 5, 8, 3, 6]], *case[1:])
    before = {}
    for name in PARTITIONS:
        obj, key = partition_slot(case, name)
        obj[key] = obj[key][::-1].copy()
        before[name] = obj[key].copy()
    rows_before = ak.to_list(case[0])
    assert D.snapshot_coverage(*case) == baseline
    assert ak.to_list(case[0]) == rows_before
    for name in PARTITIONS:
        obj, key = partition_slot(case, name)
        np.testing.assert_array_equal(obj[key], before[name])


@pytest.mark.parametrize("name", PARTITIONS)
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "cross_partition"])
def test_missing_duplicate_and_cross_partition_receipts_are_rejected(name, mutation):
    case = receipts(2)
    obj, key = partition_slot(case, name)
    if mutation == "missing":
        obj[key] = obj[key][1:]
    elif mutation == "duplicate":
        obj[key] = np.repeat(obj[key][:1], 2, axis=0)
    else:
        other = PARTITIONS[(PARTITIONS.index(name) + 1) % len(PARTITIONS)]
        source, source_key = partition_slot(case, other)
        obj[key][0] = source[source_key][0]
    with pytest.raises(ValueError, match="partition"):
        D.snapshot_coverage(*case)


@pytest.mark.parametrize("field", D.HIT_KEYS)
@pytest.mark.parametrize("name", PARTITIONS)
def test_every_key_component_participates_in_the_join(name, field):
    case = receipts()
    obj, key = partition_slot(case, name)
    obj[key][0, D.HIT_KEYS.index(field)] += 100
    with pytest.raises(ValueError, match="missing full identity join"):
        D.snapshot_coverage(*case)


@pytest.mark.parametrize("mutation", ["extra", "missing", "duplicate"])
def test_sidecar_population_is_authoritative(mutation):
    case = receipts()
    rows = case[0]
    if mutation == "extra":
        extra = ak.with_field(rows[:1], [99], "effect_index")
        rows = ak.concatenate([rows, extra])
    elif mutation == "missing":
        rows = rows[1:]
    else:
        rows = ak.concatenate([rows, rows[:1]])
    with pytest.raises(ValueError, match="partition|missing full identity join|duplicate identities"):
        D.snapshot_coverage(rows, *case[1:])


@pytest.mark.parametrize("name", PARTITIONS)
@pytest.mark.parametrize("mutation", [
    "empty", "flat", "short_key", "wide_key", "float", "nan", "infinity",
    "negative", "bool", "string", "uint_overflow", "masked",
])
def test_malformed_partition_keys_are_rejected(name, mutation):
    case = receipts()
    obj, key = partition_slot(case, name)
    value = obj[key].copy()
    if mutation == "empty":
        value = value[:0]
    elif mutation == "flat":
        value = value[0]
    elif mutation == "short_key":
        value = value[:, :-1]
    elif mutation == "wide_key":
        value = np.column_stack([value, [0]])
    elif mutation in ("float", "nan", "infinity"):
        value = value.astype(float)
        if mutation != "float":
            value[0, 0] = np.nan if mutation == "nan" else np.inf
    elif mutation == "negative":
        value[0, 0] = -1
    elif mutation == "bool":
        value = value.astype(bool)
    elif mutation == "string":
        value = value.astype(str)
    elif mutation == "uint_overflow":
        value = value.astype(np.uint64)
        value[0, 0] = np.uint64(2**63)
    else:
        value = np.ma.array(value, mask=False)
        value.mask[0, 0] = True
    obj[key] = value
    with pytest.raises(ValueError, match="invalid snapshot coverage keys|missing values"):
        D.snapshot_coverage(*case)


@pytest.mark.parametrize("mutation", ["negative", "bool", "float", "nan", "infinity", "uint_overflow", "masked"])
@pytest.mark.parametrize("field", D.HIT_KEYS)
def test_malformed_sidecar_identities_are_rejected(field, mutation):
    case = receipts()
    values = ak.to_numpy(case[0][field]).copy()
    if mutation == "negative":
        values[0] = -1
    elif mutation == "bool":
        values = values.astype(bool)
    elif mutation in ("float", "nan", "infinity"):
        values = values.astype(float)
        if mutation != "float":
            values[0] = np.nan if mutation == "nan" else np.inf
    elif mutation == "uint_overflow":
        values = values.astype(np.uint64)
        values[0] = np.uint64(2**63)
    else:
        values = ak.mask(values, np.arange(len(values)) != 0)
    rows = ak.with_field(case[0], values, field)
    with pytest.raises(ValueError, match="invalid integer identity|invalid dtype|missing values"):
        D.snapshot_coverage(rows, *case[1:])


@pytest.mark.parametrize("representation", ["list", "uint", "unmasked", "int64_max"])
def test_valid_key_representations_and_integer_boundary_are_accepted(representation):
    case = receipts()
    if representation == "int64_max":
        rows = ak.with_field(case[0], np.full(6, np.iinfo(np.int64).max, dtype=np.int64), "run")
        case = (rows, *case[1:])
    for name in PARTITIONS:
        obj, key = partition_slot(case, name)
        if representation == "list":
            obj[key] = obj[key].tolist()
        elif representation == "uint":
            obj[key] = obj[key].astype(np.uint64)
        elif representation == "unmasked":
            # A mask container is harmless when no value is missing.
            obj[key] = np.ma.array(obj[key], mask=False)
        else:
            obj[key][:, 0] = np.iinfo(np.int64).max
    assert D.snapshot_coverage(*case)["n_snapshots"] == 6


@pytest.mark.parametrize("name", PARTITIONS[1:])
@pytest.mark.parametrize("flag", [False, None, 0, 1, "true", np.bool_(True)])
def test_prior_receipt_exact_flag_must_be_literal_true(name, flag):
    case = receipts()
    obj, key = exact_slot(case, name)
    obj[key] = {"exact": flag, "criterion": True}
    with pytest.raises(ValueError, match="does not pass exact equality"):
        D.snapshot_coverage(*case)


@pytest.mark.parametrize("field", D.P.STATE_FIELDS)
@pytest.mark.parametrize("flag", [False, None, 0, 1, "true", np.bool_(True)])
def test_initial_geometry_exact_flag_must_be_literal_true(field, flag):
    case = receipts()
    obj, key = exact_slot(case, "initial", field)
    obj[key]["passed"] = flag
    obj["criterion"] = {"passed": True}
    with pytest.raises(ValueError, match="initial geometry receipt does not pass exact equality"):
        D.snapshot_coverage(*case)


@pytest.mark.parametrize("name", PARTITIONS)
def test_missing_partition_receipt_fails_closed(name):
    case = receipts()
    obj, key = partition_slot(case, name)
    del obj[key]
    with pytest.raises(KeyError):
        D.snapshot_coverage(*case)


def test_coverage_does_not_substitute_for_prior_exact_agreement():
    case = receipts()
    obj, key = partition_slot(case, "initial")
    obj[key] = None
    case[2]["hits"]["final_ca_agreement"]["exact"] = False
    with pytest.raises(ValueError, match="same-meta CA receipt does not pass exact equality"):
        D.snapshot_coverage(*case)
