"""Selection scope cannot turn a favorable reference history into complete-birth efficiency."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import cecontract as contract


def births():
    return pd.DataFrame({"run": np.full(3, 1430, dtype=np.uint64), "subrun": np.zeros(3, dtype=np.uint64),
                         "event": np.array([2**54 + 1, 2**54 + 3, 2**54 + 5], dtype=np.uint64)})


def receipt():
    return dict(mode="chain", generated_starts=3, reference_selection="unfiltered", expected_starts=3,
                conditional_on_dts_selection=False)


def test_selected_and_absent_receipts_remain_diagnostics_not_complete_birth_claims():
    for provenance in ({}, dict(mode="chain", reference_selection="selected-dts", generated_starts=3)):
        r = contract.efficiency_scope(births(), provenance)
        assert r["conditional_on_dts_selection"] and not r["complete_birth_eligible"]
        with pytest.raises(ValueError, match="complete-birth efficiency"):
            contract.efficiency_scope(births(), provenance, "complete-birth")


def test_complete_birth_eligibility_is_not_physics_closure_and_entry_control_is_separate():
    r = contract.efficiency_scope(births(), receipt(), "complete-birth")
    assert r["denominator_events"] == 3 and r["complete_birth_eligible"]
    assert r["validity_verdict"].startswith("not evaluated")
    assert r["production_filter_parity"] == "not certified"
    assert r["receipt_identity_binding"] == "legacy count-only or absent"
    p = {**receipt(), "mode": "entry"}
    r = contract.efficiency_scope(births(), p)
    assert r["interpretation"] == "shared-G4-entry tracker control"
    assert not r["complete_birth_eligible"] and not r["conditional_on_dts_selection"]
    with pytest.raises(ValueError, match="complete-birth efficiency"):
        contract.efficiency_scope(births(), p, "complete-birth")


@pytest.mark.parametrize("updates,match", [
    ({"expected_starts": None}, "externally audited"), ({"expected_starts": 2}, "externally audited"),
    ({"expected_starts": 3.0}, "externally audited"), ({"expected_starts": True}, "externally audited"),
    ({"conditional_on_dts_selection": True}, "contradictory"),
    ({"conditional_on_dts_selection": 0}, "contradictory"),
    ({"generated_starts": 2}, "count differs"), ({"reference_selection": "invented"}, "unknown"),
    ({"birth_keys_sha256": "wrong"}, "identities differ"), ({"mode": "invented"}, "unknown")])
def test_unfiltered_receipts_fail_closed(updates, match):
    with pytest.raises(ValueError, match=match):
        contract.efficiency_scope(births(), {**receipt(), **updates})


def test_missing_mode_or_generated_count_cannot_request_complete_birth():
    for field in ("mode", "generated_starts"):
        p = receipt()
        del p[field]
        assert not contract.efficiency_scope(births(), p)["complete_birth_eligible"]
        with pytest.raises(ValueError, match="complete-birth efficiency"):
            contract.efficiency_scope(births(), p, "complete-birth")


def test_fingerprint_preserves_large_ids_and_is_storage_order_independent():
    d = births()
    digest = contract.fingerprint(d)
    signed = d.astype(np.int64).iloc[::-1]
    assert contract.fingerprint(signed) == digest
    changed = d.copy()
    changed.loc[0, "event"] += np.uint64(1)
    assert contract.fingerprint(changed) != digest
    r = contract.efficiency_scope(d, {**receipt(), "birth_keys_sha256": digest}, "complete-birth")
    assert r["receipt_identity_binding"] == "sha256"


def test_denominator_and_subset_reject_duplicates_foreign_float_negative_and_empty():
    d = births()
    with pytest.raises(ValueError, match="duplicate"):
        contract.efficiency_scope(pd.concat([d, d.iloc[:1]]), receipt())
    with pytest.raises(ValueError, match="empty"):
        contract.efficiency_scope(d.iloc[:0], {})
    for invalid in (d.astype(float), d.astype(np.int64).assign(event=[-1, 2, 3])):
        with pytest.raises(ValueError, match="invalid event"):
            contract.subset(invalid, d, "invalid")
    with pytest.raises(ValueError, match="outside birth"):
        contract.subset(d.assign(event=np.array([1, 2, 3], dtype=np.uint64)), d, "foreign")
    assert len(contract.subset(d.iloc[:0], d, "missing all")) == 0
    pd.testing.assert_frame_equal(contract.subset(d.astype(np.int64), d, "mixed storage"), d)
