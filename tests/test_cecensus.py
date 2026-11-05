"""Closed inclusive references cannot silently drop failures or change their birth identities."""
import numpy as np
import pytest

from trkfastsim.validation import cecensus as C


def text(event=1, status=0, stock="fail", inclusive="pass"):
    return f""" ###############  PrintModule Run/Subrun/Event 1430 6200 {event}
ProductPrint mu2e::StatusG4_g4run__CaloLibraryCE
  status: {status} nG4Tracks: 5 overflowSimP: 0 nStepLimit: 0 CPU: 0.01 Wall: 0.02
ProductPrint art::TriggerResults_TriggerResults__CaloLibraryCE
  {inclusive} 0 PrimaryPath
  {stock} 1 StockSelectionPath
"""


def test_complete_status_selection_and_independent_metadata_reconcile():
    saved = C.read_log(text(2, stock="pass") + text(1), 2)
    generation = C.read_log(text(1) + text(2), 2, triggers=False)
    expected = saved[C.KEY].copy()
    r = C.reconcile(generation, saved, expected, expected)
    assert r["events"] == 2 and r["physics_eligible"]
    assert r["stock_selected_events"] == 1 and r["stock_rejected_events"] == 1
    assert r["inclusive_path_pass_events"] == 2
    assert saved.event.dtype == np.uint64


@pytest.mark.parametrize("change,match", [
    (lambda s: s.replace("status: 0", "status: -1"), "status records"),
    (lambda s: s.replace("status: 0", "status: 0.5"), "status records"),
    (lambda s: s.replace("overflowSimP: 0", "overflowSimP: 2"), "overflow"),
    (lambda s: s.replace("fail 1 StockSelectionPath", "error 1 StockSelectionPath"), "trigger state"),
    (lambda s: s.replace("fail 1 StockSelectionPath", "pass 1 StockSelectionPath\n fail 1 StockSelectionPath"), "trigger state"),
    (lambda s: s.replace("__CaloLibraryCE", "__Other"), "product/process"),
    (lambda s: s.replace("ProductPrint art::TriggerResults", "ProductPrint mu2e::StatusG4"), "product/process"),
    (lambda s: s.replace("###############  PrintModule", "Wrong PrintModule"), "headers"),
])
def test_malformed_missing_wrong_product_and_unknown_trigger_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        C.read_log(change(text()), 1)


def test_failures_are_preserved_not_treated_as_zero_response():
    saved = C.read_log(text(status=10, inclusive="fail"), 1)
    gen = C.read_log(text(status=10), 1, triggers=False)
    r = C.reconcile(gen, saved, saved[C.KEY], saved[C.KEY])
    assert not r["physics_eligible"] and r["validity_failure_events"] == 1
    assert r["status_counts"] == {"10": 1} and r["inclusive_path_pass_events"] == 0
    with pytest.raises(AssertionError):
        C.reconcile(C.read_log(text(status=1), 1, triggers=False), saved, saved[C.KEY], saved[C.KEY])


def test_duplicate_missing_foreign_ids_and_cross_event_records_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        C.read_log(text() + text(), 2)
    with pytest.raises(ValueError, match="headers"):
        C.read_log(text(), 2)
    saved = C.read_log(text(), 1)
    with pytest.raises(AssertionError):
        C.reconcile(saved, saved, saved[C.KEY].assign(event=np.uint64(2)), saved[C.KEY])
    altered = text().replace("ProductPrint mu2e::StatusG4", "############### PrintModule Run/Subrun 1430 6200\nProductPrint mu2e::StatusG4")
    with pytest.raises(ValueError, match="outside an event"):
        C.read_log(altered, 1)


def test_status_zero_cannot_silently_fail_inclusive_transport_path():
    saved = C.read_log(text(inclusive="fail"), 1)
    with pytest.raises(ValueError, match="contradictory"):
        C.reconcile(saved, saved, saved[C.KEY], saved[C.KEY])


def test_raw_product_presence_does_not_certify_response():
    names = list(C.REQUIRED_PRODUCTS.values())
    assert C.products(names)["all_required_products_present"]
    for missing in names:
        with pytest.raises(ValueError, match="missing exact typed raw"):
            C.products([n for n in names if n != missing])
    with pytest.raises(ValueError, match="missing exact typed raw"):
        C.products([n.replace("mu2e::StatusG4", "OtherType") for n in names])


@pytest.mark.parametrize("path_id", ["1", "-1", "18446744073709551616", "1.5"])
def test_declared_path_ids_fail_closed(path_id):
    with pytest.raises(ValueError, match="trigger"):
        C.read_log(text().replace("pass 0 PrimaryPath", f"pass {path_id} PrimaryPath"), 1)


@pytest.mark.parametrize("event", [0, 2**32 - 1, 2**32])
def test_native_art_event_identity_boundaries(event):
    with pytest.raises(ValueError, match="native art identity"):
        C.read_log(text(event), 1)


def test_additional_malformed_decision_is_not_ignored():
    with pytest.raises(ValueError, match="trigger state"):
        C.read_log(text() + "error 0 PrimaryPath trailing-junk\n", 1)
