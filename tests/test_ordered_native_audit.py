"""Saved rational brackets must bind complete full-owner footprint, including unknown rounding bands."""
from fractions import Fraction

import pandas as pd
import pytest

from scripts.ordered_native_audit import audit_roots
from trkfastsim.exactplanar import serialize_roots


def fixture():
    roots = serialize_roots(pd.DataFrame(dict(lane=[0, 0], source=["a", "b"],
        q0=[Fraction(1, 3), Fraction(2, 3)], q1=[Fraction(2, 3), Fraction(4, 3)])))
    footprint = roots[["lane", "source"]].assign(s0_mm=roots.q0_lower_mm, s1_mm=roots.q1_upper_mm,
        ancestors_verified=True, leaf_verified=True)
    return roots, footprint


def test_saved_exact_shared_roots_keep_two_owners():
    result = audit_roots(*fixture())
    assert result["root_rows"] == result["sources"] == 2 and result["exact_positive_leaf_overlaps"] == 0


@pytest.mark.parametrize("fault", ["rational", "float", "owner", "footprint", "hierarchy"])
def test_mutated_saved_certificate_fails(fault):
    roots, footprint = fixture()
    if fault == "rational":
        roots.loc[0, "q0_numerator"] = "2"
    elif fault == "float":
        roots.loc[0, "q0_lower_mm"] = .1
    elif fault == "owner":
        footprint.loc[0, "source"] = "b"
    elif fault == "footprint":
        footprint.loc[0, "s0_mm"] = roots.q0_upper_mm.iloc[0]
    else:
        footprint.loc[0, "ancestors_verified"] = False
    with pytest.raises(ValueError):
        audit_roots(roots, footprint)
