"""All-primary handoff coverage cannot be silently subset by gas or acceptance."""
import numpy as np
import pandas as pd
import pytest

from scripts import ordered_guarded_audit as A


def frames():
    inventory = pd.DataFrame(dict(run=[1, 1], subrun=[2, 2], event=[3, 3], sp=[1, 2]))
    handoff = inventory.assign(alive_world=[True, False], accepted_forward=[True, False],
        disposition=["forward-handoff", "not-delivered-by-world"], x=[0., 0.], y=[870., 1000.], z=[0., 0.],
        ux=[0., 0.], uy=[-1., -1.], uz=[0., 0.], p_handoff_MeV_c=[80., 80.], path_offset_mm=[130., 0.],
        world_dt_ns=[1., np.nan], plane_roundoff_guard_mm=[1e-11, 1e-11])
    return inventory, handoff


def test_full_inventory_with_world_undelivered_and_multi_primary_keys():
    result = A.audit_handoff(*frames(), 870.)
    assert result["input_primaries"] == 2 and result["world_delivered"] == 1 and result["world_undelivered"] == 1


@pytest.mark.parametrize("fault", ["subset", "reordered", "upward", "offplane", "infinite", "refused"])
def test_handoff_coverage_and_domain_fail_closed(fault):
    inventory, handoff = frames()
    if fault == "subset":
        handoff = handoff.iloc[:1]
    elif fault == "reordered":
        handoff = handoff.iloc[::-1]
    elif fault == "upward":
        handoff.loc[0, "uy"] = 1.
    elif fault == "offplane":
        handoff.loc[0, "y"] -= 1.
    elif fault == "infinite":
        handoff.loc[0, "path_offset_mm"] = np.inf
    else:
        handoff.loc[0, "accepted_forward"] = False
    with pytest.raises(ValueError):
        A.audit_handoff(inventory, handoff, 870.)


def test_declared_forward_refusal_requires_full_ray_and_identity_evidence(tmp_path):
    _, handoff = frames()
    receipt = dict(forward_refusal_iteration=7, forward_refusal_primary_keys=[[1, 2, 3, 1]])
    path = tmp_path / "refusal.npz"
    with pytest.raises(ValueError, match="lacks ray"):
        A.audit_forward_refusal(path, handoff, receipt)
    np.savez_compressed(path, interior_lanes=np.array([0]), primary_keys=np.array([[1, 2, 3, 1]]), iteration=7,
        origin=np.array([[0., 870., 0.]]), direction=np.array([[0., 1., 0.]]), lengths=np.array([1.]), plane_y_mm=870.)
    assert A.audit_forward_refusal(path, handoff, receipt)["iteration"] == 7
    receipt["forward_refusal_primary_keys"] = [[1, 2, 3, 2]]
    with pytest.raises(ValueError, match="differ from receipt"):
        A.audit_forward_refusal(path, handoff, receipt)


def test_expanded_parent_checks_still_distinguish_multi_primary_inventory():
    inventory, handoff = frames()
    assert A.identity_frame(inventory, "inventory").tolist() == [(1, 2, 3, 1), (1, 2, 3, 2)]
