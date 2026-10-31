"""Cancellation is per verified physical owner, never generic same-material sibling cancellation."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim import placementownership as P


def terms(rows):
    return pd.DataFrame(rows, columns=["lane", "s0_mm", "s1_mm", "material", "source", "sign", "operation", "daughter"])


def test_direct_daughter_void_cancels_only_its_own_parent_and_keeps_material_field():
    raw = terms([(0, 0., 10., "steel", "p[0]", 1, "add-own-shape", "p[0]"),
                 (0, 2., 8., "steel", "p[0]", -1, "subtract-direct-daughter", "p[0]/void[0]")])
    result, atoms = P.normalize(raw, np.array([10.]))
    assert result.source.tolist() == ["p[0]", "p[0]"]
    assert result[["s0_mm", "s1_mm"]].to_numpy().tolist() == [[0., 2.], [8., 10.]]
    assert (atoms.original == atoms.normalized).all()


def test_cross_sibling_cancellation_never_repaired_even_if_material_net_valid():
    raw = terms([(0, 0., 10., "steel", "p[0]/a[0]", 1, "add-own-shape", "p[0]/a[0]"),
                 (0, 2., 8., "steel", "p[0]/b[0]", -1, "subtract-direct-daughter", "p[0]/b[0]/void[0]")])
    with pytest.raises(ValueError):
        P.normalize(raw, np.array([10.]))


def test_forged_sibling_daughter_label_refuses():
    raw = terms([(0, 0., 10., "steel", "p[0]/a[0]", 1, "add-own-shape", "p[0]/a[0]"),
                 (0, 2., 8., "steel", "p[0]/a[0]", -1, "subtract-direct-daughter", "p[0]/b[0]")])
    with pytest.raises(ValueError, match="direct physical daughter"):
        P.normalize(raw, np.array([10.]))


def test_same_material_positive_physical_sibling_overlap_still_refuses():
    raw = terms([(0, 0., 6., "steel", "p[0]/a[0]", 1, "add-own-shape", "p[0]/a[0]"),
                 (0, 4., 10., "steel", "p[0]/b[0]", 1, "add-own-shape", "p[0]/b[0]")])
    with pytest.raises(ValueError):
        P.normalize(raw, np.array([10.]))


def test_actual_gdml_physical_owners_and_void_subtraction_match_raw_materials(tmp_path):
    path = tmp_path / "fixture.gdml"
    path.write_text('''<gdml><solids><box name="world" x="100" y="100" z="100"/>
    <box name="mother" x="40" y="40" z="40"/><box name="parent" x="20" y="20" z="20"/>
    <box name="void" x="4" y="4" z="4"/></solids><structure>
    <volume name="World"><materialref ref="Vacuum"/><solidref ref="world"/>
    <physvol name="calo"><volumeref ref="CalorimeterMother"/></physvol></volume>
    <volume name="CalorimeterMother"><materialref ref="Vacuum"/><solidref ref="mother"/>
    <physvol name="foam"><volumeref ref="Foam"/></physvol></volume>
    <volume name="Foam"><materialref ref="AluminumHoneycomb"/><solidref ref="parent"/>
    <physvol name="hole"><volumeref ref="Hole"/></physvol></volume>
    <volume name="Hole"><materialref ref="Vacuum"/><solidref ref="void"/></volume>
    </structure><setup><world ref="World"/></setup></gdml>''')
    query = P.PlacementSources(path, (np.full(3, -50.), np.full(3, 50.)))
    raw = query(np.array([[-15., 0., 0.]]), np.array([[1., 0., 0.]]), np.array([30.]))
    assert set(raw.source) == {"calo[0]/foam[0]"}
    assert "calo[0]/foam[0]/hole[0]" in raw.daughter.to_numpy()
    result, atoms = query.normalize(raw, np.array([30.]))
    np.testing.assert_allclose((result.s1_mm - result.s0_mm).sum(), 16.)
    assert (atoms.original == atoms.normalized).all()
    forged = raw.copy()
    forged.loc[forged.sign == -1, "operation"] = "add-own-shape"
    forged.loc[forged.sign == -1, "daughter"] = forged.loc[forged.sign == -1, "source"]
    with pytest.raises(ValueError, match="provenance differs"):
        query.normalize(forged, np.array([30.]))


def test_matching_material_wrong_surviving_physical_parent_still_refuses():
    cells = pd.DataFrame(dict(lane=[0], s0_mm=[2.], s1_mm=[4.], scope=["calorimeter"],
        source=["calo[0]/parent[0]/ring[0]"], replaces_material=["steel"]))
    raw = terms([(0, 0., 10., "steel", "calo[0]/sibling[0]", 1, "add-own-shape", "calo[0]/sibling[0]")])
    normalized, _ = P.normalize(raw, np.array([10.]))
    with pytest.raises(ValueError, match="actual omitted leaf parent"):
        P.parent_owners(cells, normalized, max_pairs=100)
    normalized["source"] = "calo[0]/parent[0]"
    assert len(P.parent_owners(cells, normalized, max_pairs=100)) == 1


def test_tube_approximation_is_not_eligible_for_physical_cancellation():
    raw = terms([(0, 0., 10., "steel", "tube:0", 1, "add-own-shape", "tube:0")])
    normalized, _ = P.normalize(raw, np.array([10.]))
    assert normalized.source.tolist() == ["tube:0"]
    raw.loc[0, "sign"] = -1
    with pytest.raises(ValueError, match="never enter physical cancellation"):
        P.normalize(raw, np.array([10.]))


def test_empty_all_vacuum_query_does_not_require_string_accessor():
    query = P.PlacementSources.__new__(P.PlacementSources)
    query.relationships = pd.DataFrame()
    raw = pd.DataFrame({key: pd.Series(dtype=float) for key in P.COLS})
    raw["lane"] = raw.lane.astype("int64")
    normalized, atoms = query.normalize(raw, np.array([5.]))
    assert normalized.empty and atoms.empty
    assert P.parent_owners(pd.DataFrame(columns=["scope"]), normalized, max_pairs=100).empty


def test_missing_actual_parent_span_refuses_even_without_intersecting_owner():
    cells = pd.DataFrame(dict(lane=[0], s0_mm=[2.], s1_mm=[4.], scope=["calorimeter"],
        source=["calo[0]/parent[0]/ring[0]"], replaces_material=["steel"]))
    normalized = pd.DataFrame(dict(lane=pd.Series(dtype="int64"), s0_mm=pd.Series(dtype=float),
        s1_mm=pd.Series(dtype=float), material=pd.Series(dtype=str), source=pd.Series(dtype=str), sign=pd.Series(dtype=int)))
    with pytest.raises(ValueError, match="complete possible leaf footprint"):
        P.parent_owners(cells, normalized, max_pairs=100)
