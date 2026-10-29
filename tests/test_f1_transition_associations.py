"""Source-conditioned partner inference is distinct from captured constructor history."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_transition_audit as D


def fixture():
    return dict(hits=dict(source_keys=np.array([[1, 2, 3, 0, 2], [1, 2, 3, 0, 8]]),
                         initial=np.array([True, False]), first_ordinal=np.array([0, 8])),
                crossings=dict(source_keys=np.array([[1, 2, 3, 0, 0], [1, 2, 3, 0, 1], [1, 2, 3, 0, 2]]),
                               initial=np.array([True, False, False]), partner_hit_rows=np.array([0, 1, -1]),
                               first_ordinal=np.array([0, 8, 8]), strawid=np.array([11, 12, 13])))


def test_inference_does_not_claim_snapshots_or_offsets():
    result = D.association_contract(fixture())
    assert result["source_conditioned_constructor_partner_agreement"] is True
    assert result["native_iteration_local_partner_snapshots"] is False
    assert result["private_constructor_offsets_observed"] is False
    assert result["n_hits"] == 2 and result["n_crossings"] == 3
    assert result["n_hits_without_exactly_one_crossing"] == 0
    assert result["n_duplicate_crossing_straw_keys"] == 0


@pytest.mark.parametrize("change", ["unpaired_initial", "two_partners", "initial_extension", "different_birth"])
def test_disagreement_retained_not_silently_excluded(change):
    data = fixture()
    if change == "unpaired_initial":
        data["crossings"]["partner_hit_rows"][0] = -1
    elif change == "two_partners":
        data["crossings"]["partner_hit_rows"][2] = 1
    elif change == "initial_extension":
        data["hits"]["initial"][1] = True
    else:
        data["crossings"]["first_ordinal"][1] += 1
    result = D.association_contract(data)
    assert result["source_conditioned_constructor_partner_agreement"] is False
    assert result["n_hits"] == 2 and result["n_crossings"] == 3


def test_duplicate_straw_census_is_full_target_keyed():
    data = fixture()
    data["crossings"]["strawid"][2] = 12
    assert D.association_contract(data)["n_duplicate_crossing_straw_keys"] == 1
    data["crossings"]["source_keys"][2, 2] = 999
    assert D.association_contract(data)["n_duplicate_crossing_straw_keys"] == 0


@pytest.mark.parametrize("family,field", [("hits", "source_keys"), ("hits", "initial"), ("hits", "first_ordinal"),
    ("crossings", "source_keys"), ("crossings", "initial"), ("crossings", "first_ordinal"),
    ("crossings", "partner_hit_rows"), ("crossings", "strawid")])
def test_true_masks_rejected_without_changing_input(family, field):
    data = fixture()
    original = deepcopy(data)
    values = data[family][field]
    mask = np.zeros(values.shape, bool)
    mask.flat[0] = True
    data[family][field] = np.ma.array(values, mask=mask)
    with pytest.raises(ValueError, match="missing"):
        D.association_contract(data)
    np.testing.assert_array_equal(data[family][field].data, original[family][field])


def test_cross_target_partner_rejected():
    data = fixture()
    data["crossings"]["source_keys"][1, 2] = 999
    with pytest.raises(ValueError, match="partner target"):
        D.association_contract(data)


@pytest.mark.parametrize("field", ["source_keys", "first_ordinal", "strawid"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -1.])
def test_nonfinite_or_negative_identity_rejected(field, value):
    data = fixture()
    values = data["crossings"][field].astype(float)
    values.flat[0] = value
    data["crossings"][field] = values
    with pytest.raises(ValueError):
        D.association_contract(data)


@pytest.mark.parametrize("index", [-2, 2, np.iinfo(np.int64).max])
def test_partner_bounds(index):
    data = fixture()
    data["crossings"]["partner_hit_rows"][1] = index
    with pytest.raises(ValueError, match="bounds"):
        D.association_contract(data)
