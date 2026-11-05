"""Effective CE geometry diagnostic serialization and exact comparison."""
import numpy as np

from trkfastsim.validation.cegeometry import differences, plain


def test_geometry_exact_differences_include_missing_keys_and_nested_arrays():
    a = plain({"beam": {"foil_z": np.array([1.0, 2.0])}, "ring_t": 0.1})
    b = plain({"beam": {"foil_z": np.array([1.0, 2.0])}, "ring_t": 0.2, "extra": True})
    assert set(differences(a, b)) == {"ring_t", "extra"}
    assert differences(a, a) == {}
    b["beam"]["foil_z"][0] = 1.00000000001
    assert "beam" in differences(a, b)
