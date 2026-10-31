"""Conservation and exact slice clock refusal tests for saved conditional histories."""
import numpy as np
import pytest

from scripts import ordered_torus_evolution_audit as A
from trkfastsim import ledger
from trkfastsim.flighttime import linear_kinetic_dt_ns


def saved():
    ke = float(ledger.kinetic(80.))
    before = np.array([ke, ke - .1])
    after = before - .1
    dt = linear_kinetic_dt_ns(np.ones(2), before, after)
    return dict(slice_start=np.array([0., 1.]), slice_end=np.array([1., 2.]),
        slice_ke0=before, slice_ke1=after, slice_t0=np.array([0., dt[0]]), slice_t1=np.array([dt[0], dt.sum()]),
        slice_pos0=np.array([[0., 0., 0.], [1., 0., 0.]]), slice_pos1=np.array([[1., 0., 0.], [2., 0., 0.]]),
        slice_dir=np.array([[1., 0., 0.], [1., 0., 0.]]), slice_it=np.zeros(2, np.int64),
        slice_kind=np.array(["support", "thin"]), slice_material=np.array(["steel", A.steps.STRAW_GAS.name]),
        slice_deposited=np.array([np.nan, .04]), slice_unassigned=np.array([np.nan, .06]), gas_eion=np.array([.04]),
        final_ke=np.array([after[-1]]), final_lost_support=np.array([.1]), final_lost_thin=np.array([.1]),
        final_path=np.array([2.]), final_dt=np.array([dt.sum()]), final_o=np.array([[2., 0., 0.]]),
        slice_stopped=np.zeros(2, bool), final_stopped=np.array([False]))


def test_exact_saved_evolution_balance_clock_and_first_gas():
    result = A.audit_slices(saved(), np.array([80.]))
    assert result["primary_balance_MeV"] < 1e-12 and result["clock_error_ns"] < 1e-12
    assert result["first_gas_path_mm"] == 1.
    assert result["first_gas_position_mm"] == [1., 0., 0.]


@pytest.mark.parametrize("fault", ["gap", "clock", "balance", "tail", "deposition", "support", "position", "stop", "direction"])
def test_saved_unphysical_evolution_refuses(fault):
    values = saved()
    if fault == "gap":
        values["slice_start"][1] += .01
    elif fault == "clock":
        values["slice_t1"][-1] += .01
        values["final_dt"][0] += .01
    elif fault == "balance":
        values["final_lost_thin"][0] += .01
    elif fault == "tail":
        values["final_path"][0] += .01
    elif fault == "deposition":
        values["slice_unassigned"][1] += .01
    elif fault == "support":
        values["final_lost_support"][0] += .01
        values["final_lost_thin"][0] -= .01
    elif fault == "position":
        values["slice_pos1"][0, 1] += .01
    elif fault == "stop":
        values["slice_stopped"][0] = True
    else:
        values["slice_dir"][0, 0] = .5
    with pytest.raises(ValueError):
        A.audit_slices(values, np.array([80.]))


def test_legacy_string_labels_decode_restricted_without_generic_pickle(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, slice_kind=np.array(["support", "thin"], object),
        slice_material=np.array(["steel", "gas"], object), numeric=np.array([1., 2.]))
    result = A.load_saved(path)
    assert result["slice_kind"].tolist() == ["support", "thin"]
    assert result["slice_kind"].dtype.kind == "U"
    assert result["numeric"].tolist() == [1., 2.]
    with pytest.raises(ValueError, match="unsupported global"):
        A.LabelUnpickler(A.io.BytesIO(A.pickle.dumps(eval))).load()
    np.savez_compressed(path, slice_kind=np.array([1, "support"], object))
    with pytest.raises(ValueError, match="only strings"):
        A.load_saved(path)
