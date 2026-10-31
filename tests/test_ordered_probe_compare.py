"""Probe pairing cannot silently drop identities or incomplete replays."""
import copy

import pytest

from scripts.ordered_probe_compare import compare


def test_compare_full_identity_and_refusal():
    a = dict(status="bounded_transport_complete", identities=[[1, 2, 3, 1]], input_sha256={"a": "hash"},
        requested_length_mm=10., final_ke_MeV=[20.], path_mm=[10.], boundary_relabel_mm=[0.],
        stopped=[False], entries=1, scatter=False)
    b = copy.deepcopy(a)
    b["final_ke_MeV"] = [19.]
    result = compare(a, b)
    assert result["deltas"]["final_ke_MeV"]["max_abs"] == 1.
    b["identities"] = [[1, 2, 3, 2]]
    with pytest.raises(ValueError, match="identities"):
        compare(a, b)
    b = dict(a, status="refused")
    with pytest.raises(ValueError, match="complete"):
        compare(a, b)
