"""Synthetic read-only loader inventory; no extracted geometry or transport run."""
from collections import Counter
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from mu2efastsim import world
from trkfastsim.navigationcensus import _geometry_bounds, census
from trkfastsim.steps import INTERIOR_SKIP

BBOX = ([-10, -10, -10], [10, 10, 10])


@pytest.fixture
def gdml(tmp_path):
    root = ET.fromstring('''<gdml><define><position name="far" x="500" y="0" z="0" unit="mm"/></define>
    <materials/><solids>
    <box name="box" x="4" y="6" z="8" lunit="mm"/>
    <tube name="tube" rmin="1" rmax="3" z="8" aunit="deg" deltaphi="360"/>
    <cone name="cone" rmax1="2" rmax2="4" z="6"/>
    <ellipsoid name="ellipsoid" ax="2" by="3" cz="4"/>
    <subtraction name="boolean"><first ref="tube"/><second ref="box"/></subtraction>
    <union name="union"><first ref="tube"/><second ref="box"/><position x="1000"/></union>
    <cutTube name="cut" rmin="1" rmax="3" z="8" lowX="0.5" lowY="0" lowZ="-1"
      highX="0.5" highY="0" highZ="1"/>
    <polycone name="taper" aunit="deg" deltaphi="360">
      <zplane z="-8" rmin="1" rmax="2"/><zplane z="0" rmin="2" rmax="4"/>
      <zplane z="8" rmin="2" rmax="4"/></polycone>
    <box name="cm_box" x="4" y="6" z="8" lunit="cm"/>
    </solids><structure/><setup><world ref="World"/></setup></gdml>''')
    structure = root.find("structure")

    def volume(name, solid="box", material="G4_Al"):
        v = ET.SubElement(structure, "volume", name=name)
        ET.SubElement(v, "materialref", ref=material)
        ET.SubElement(v, "solidref", ref=solid)
        return v

    def place(parent, child, x=0, extra=""):
        pv = ET.SubElement(parent, "physvol", name="pv_" + child)
        ET.SubElement(pv, "volumeref", ref=child)
        if x:
            ET.SubElement(pv, "position", x=str(x), unit="mm")
        if extra:
            pv.append(ET.fromstring(extra))
        return pv

    top = volume("World", material="G4_Galactic")
    tracker = volume("TrackerMother", material="DSVacuum")
    place(top, "TrackerMother")
    for name, solid, material, x in [
        ("SupportBox", "box", "G4_Al", 0),
        ("BooleanTube", "boolean", "G4_Al", 0),
        ("SimpleTube", "tube", "G4_Al", 0),
        ("BoundedCone", "cone", "G4_Al", 1000),
        ("VacuumVolume", "box", "DSVacuum", 0),
        ("StrawGasManifold", "tube", "StrawGas", 0),
        ("FarTube", "tube", "G4_Al", 100),
        ("CutTube", "cut", "G4_Al", 0),
        ("Taper", "taper", "G4_Al", 0),
        ("UnionTube", "union", "G4_Al", 0),
        ("CentimeterBox", "cm_box", "G4_Al", 100),
    ]:
        volume(name, solid, material)
        place(tracker, name, x)
    straw = volume("TrackerStraw_1", "tube")
    wire = volume("TrackerWire_1", "tube")
    volume("HiddenStrawChild", "cone")
    place(straw, "HiddenStrawChild")
    place(wire, "HiddenStrawChild")
    place(tracker, "TrackerStraw_1")
    place(tracker, "TrackerWire_1")
    for name, solid, x in [("ExternalNear", "box", 0), ("ExternalFar", "box", 100),
                           ("ExternalUnknown", "ellipsoid", 1000)]:
        volume(name, solid)
        place(top, name, x)
    calo = volume("CalorimeterMother", material="DSVacuum")
    place(top, "CalorimeterMother")
    for name, solid in [("CaloBox", "box"), ("CaloTube", "tube"), ("CaloCone", "cone")]:
        volume(name, solid)
        place(calo, name)
    path = tmp_path / "geometry.gdml"
    ET.ElementTree(root).write(path)
    return path, root, volume, place


def rows(gdml):
    return {r["leaf"]: r for r in census(gdml[0], BBOX)["records"]}


def save(gdml):
    ET.ElementTree(gdml[1]).write(gdml[0])


def test_omission_retention_and_unknown_are_distinct(gdml):
    r = rows(gdml)
    assert r["SupportBox"]["classification"] == "unsupported"
    assert r["SupportBox"]["reason"] == "non_tube_support_omitted"
    assert r["SupportBox"]["geometry_bbox_relation"] == "overlap"
    assert r["SupportBox"]["solid_tag"] == "box"
    assert r["SupportBox"]["material"] == "G4_Al"
    assert r["SupportBox"]["path"] == "TrackerMother/SupportBox"
    assert r["SimpleTube"]["classification"] == "retained"
    assert r["SimpleTube"]["tube_loader_selected"]
    assert r["BoundedCone"]["classification"] == "unsupported"
    assert r["BoundedCone"]["geometry_bbox_relation"] == "disjoint"
    assert r["BoundedCone"]["geometry_bounds_mm"] == [[996, -4, -3], [1004, 4, 3]]
    assert r["FarTube"]["classification"] == "bbox_excluded"
    assert r["FarTube"]["geometry_bbox_relation"] == "disjoint"
    assert not r["FarTube"]["tube_loader_selected"]


def test_boolean_cut_and_taper_approximations(gdml):
    r = rows(gdml)
    assert r["BooleanTube"]["classification"] == "approximate"
    assert r["BooleanTube"]["solid_tag"] == "subtraction"
    assert r["BooleanTube"]["loader_base_tag"] == "tube"
    assert "boolean_first_only:subtraction" in r["BooleanTube"]["approximations"]
    assert r["BooleanTube"]["geometry_bbox_relation"] == "overlap"
    assert r["UnionTube"]["geometry_bbox_relation"] == "overlap"
    assert r["UnionTube"]["geometry_bounds_mm"] == [[-3, -3, -4], [1002, 3, 4]]
    assert r["UnionTube"]["tube_loader_selected"]
    assert "cutTube_planes_ignored" in r["CutTube"]["approximations"]
    assert r["CutTube"]["geometry_bbox_relation"] == "unknown"
    assert "tapered_polycone_section_envelopes" in r["Taper"]["approximations"]
    assert r["Taper"]["tube_sections_selected"] == 2


def test_vacuum_gas_and_pruned_straws(gdml):
    result = census(gdml[0], BBOX)
    r = {r["leaf"]: r for r in result["records"]}
    assert "VacuumVolume" not in r
    assert result["counts"]["vacuum_placements"] == 3
    assert result["pruned"] == {"TrackerStraw": 1, "TrackerWire": 1}
    assert not {"TrackerStraw_1", "TrackerWire_1", "HiddenStrawChild"} & r.keys()
    assert r["StrawGasManifold"]["classification"] == "intentional_exclusion"
    assert r["StrawGasManifold"]["tube_loader_selected"]
    assert not r["StrawGasManifold"]["interior_selected"]


def test_external_bbox_scope_and_json_read_only(gdml):
    before = gdml[0].read_bytes()
    result = census(gdml[0], BBOX)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert gdml[0].read_bytes() == before
    r = {r["leaf"]: r for r in result["records"]}
    assert r["ExternalNear"]["scope"] == "external"
    assert "ExternalFar" not in r
    assert r["ExternalUnknown"]["geometry_bbox_relation"] == "unknown"
    assert result["counts"]["external_known_disjoint"] == 1


def test_calo_uses_signed_route_not_tube_omission(gdml):
    r = rows(gdml)
    assert r["CaloBox"]["classification"] == "retained"
    assert r["CaloBox"]["scope"] == "calorimeter"
    assert r["CaloBox"]["signed_pieces_selected"] == 1
    assert r["CaloBox"]["interior_selected"]
    assert r["CaloTube"]["classification"] == "approximate"
    assert r["CaloTube"]["tube_loader_excluded_by_name"]
    assert not r["CaloTube"]["tube_loader_selected"]
    assert r["CaloCone"]["classification"] == "unsupported"
    assert r["CaloCone"]["reason"] == "signed_local_unsupported"
    assert r["CaloCone"]["geometry_bounds_mm"] == [[-4, -4, -3], [4, 4, 3]]
    assert r["CaloCone"]["geometry_bbox_relation"] == "overlap"
    pcs, skipped = world.signed_subtree(gdml[0], "CalorimeterMother", bbox=BBOX)
    assert sum(r[n]["signed_pieces_selected"] for n in ("CaloBox", "CaloTube", "CaloCone")) == len(pcs)
    assert ("CaloCone", "cone", "G4_Al") in skipped


def test_tube_counts_match_actual_loader(gdml):
    names = []
    tubes = world.load_tubes(gdml[0], bbox=BBOX, skip_names=INTERIOR_SKIP, names=names)
    expected = Counter(names)
    result = census(gdml[0], BBOX)
    actual = Counter()
    for r in result["records"]:
        actual[r["path"]] += r["tube_sections_selected"]
    assert +actual == expected
    assert sum(actual.values()) == len(tubes)


@pytest.mark.parametrize("extra,note", [
    ('<positionref ref="far"/>', "positionref_ignored:far"),
    ('<rotationref ref="turn"/>', "rotationref_ignored:turn"),
    ('<scaleref ref="identity"/>', "scaleref_ignored:identity"),
    ('<position x="100" unit="cm"/>', "position_unit_ignored:cm"),
    ('<rotation x="1" unit="mrad"/>', "rotation_unit_treated_as_radians:mrad"),
    ('<scale x="2" y="2" z="2"/>', "placement_scale_ignored"),
])
def test_placement_assumptions_invalidate_descendant_bounds(gdml, extra, note):
    root, volume, place = gdml[1:]
    parent = volume("ReferenceParent", material="DSVacuum")
    volume("ReferenceChild", "tube")
    place(parent, "ReferenceChild")
    place(root.find("structure/volume"), "ReferenceParent", extra=extra)
    save(gdml)
    result = census(gdml[0], BBOX)
    child = next(r for r in result["records"] if r["leaf"] == "ReferenceChild")
    assert child["geometry_bbox_relation"] == "unknown"
    assert child["geometry_bounds_mm"] is None
    assert note in child["placement_assumptions"]
    assert note in result["placement_warnings"]
    assert rows(gdml)["CentimeterBox"]["geometry_bbox_relation"] == "unknown"


def test_repeated_logical_placements_have_distinct_physical_paths(gdml):
    root, _, place = gdml[1:]
    tracker = root.find("structure/volume[@name='TrackerMother']")
    place(tracker, "SimpleTube")
    save(gdml)
    r = [r for r in census(gdml[0], BBOX)["records"] if r["leaf"] == "SimpleTube"]
    assert len(r) == 2
    assert r[0]["path"] == r[1]["path"]
    assert r[0]["placement_path"] != r[1]["placement_path"]


def test_rotated_nested_bounds_are_conservative(gdml):
    root, volume, place = gdml[1:]
    parent = volume("RotatedMother", material="DSVacuum")
    volume("RotatedBox")
    place(parent, "RotatedBox", x=2)
    place(root.find("structure/volume"), "RotatedMother", x=5, extra='<rotation z="90" unit="deg"/>')
    save(gdml)
    r = rows(gdml)["RotatedBox"]
    np.testing.assert_allclose(r["geometry_bounds_mm"], [[2, -4, -4], [8, 0, 4]], atol=1e-14)
    assert r["geometry_bbox_relation"] == "overlap"


@pytest.mark.parametrize("bbox", [None, [0, 1], [[1, 0, 0], [0, 1, 1]], [[0, 0, 0], [float("inf"), 1, 1]]])
def test_invalid_bbox_refused(gdml, bbox):
    with pytest.raises(ValueError, match="bbox"):
        census(gdml[0], bbox)


def test_unimplemented_placement_construct_is_not_silently_omitted(gdml):
    ET.SubElement(gdml[1].find("structure/volume"), "replicavol")
    save(gdml)
    with pytest.raises(ValueError, match="unsupported GDML placement construct"):
        census(gdml[0], BBOX)


def test_equal_z_polycone_order_matches_loader(gdml):
    root, volume, place = gdml[1:]
    solid = ET.SubElement(root.find("solids"), "polycone", name="stepped")
    for z, ri, ro in [(-8, 0, 2), (0, 1, 9), (0, 2, 3), (8, 0, 2)]:
        ET.SubElement(solid, "zplane", z=str(z), rmin=str(ri), rmax=str(ro))
    volume("Stepped", "stepped")
    place(root.find("structure/volume[@name='TrackerMother']"), "Stepped")
    save(gdml)
    bbox = ([5, -1, -7], [6, 1, -1])
    names = []
    world.load_tubes(gdml[0], bbox=bbox, skip_names=INTERIOR_SKIP, names=names)
    row = next(r for r in census(gdml[0], bbox)["records"] if r["leaf"] == "Stepped")
    assert row["tube_sections_selected"] == names.count("TrackerMother/Stepped") == 1


def test_calo_boolean_is_not_first_only(gdml):
    root, volume, place = gdml[1:]
    volume("CaloBoolean", "boolean")
    place(root.find("structure/volume[@name='CalorimeterMother']"), "CaloBoolean")
    save(gdml)
    row = rows(gdml)["CaloBoolean"]
    assert row["interior_selected"]
    assert "equal_area_polygonal_tube" in row["approximations"]
    assert not any("first_only" in note for note in row["approximations"])


@pytest.mark.parametrize("xml,expected", [
    ('<torus rtor="20" rmin="1" rmax="3" startphi="20" deltaphi="30" aunit="deg"/>',
     [[-23, -23, -3], [23, 23, 3]]),
    ('<cone rmax1="7" rmax2="2" z="10" startphi="0" deltaphi="1" aunit="rad"/>',
     [[-7, -7, -5], [7, 7, 5]]),
    ('<cone rmax1="2" rmax2="7" z="10"/>', [[-7, -7, -5], [7, 7, 5]]),
    ('<arb8 dz="9" v1x="-2" v1y="-3" v2x="4" v2y="-3" v3x="4" v3y="5" v4x="-2" v4y="5" '
     'v5x="-7" v5y="-1" v6x="6" v6y="-1" v7x="6" v7y="8" v8x="-7" v8y="8"/>',
     [[-7, -3, -9], [6, 8, 9]]),
])
def test_new_shape_envelopes(xml, expected):
    np.testing.assert_allclose(_geometry_bounds(ET.fromstring(xml), {}), expected)


def test_xtru_bounds_include_every_scaled_offset_section():
    solid = ET.fromstring('''<xtru lunit="mm">
      <twoDimVertex x="-2" y="-1"/><twoDimVertex x="3" y="-1"/>
      <twoDimVertex x="3" y="4"/><twoDimVertex x="-2" y="4"/>
      <section zOrder="0" zPosition="-10" scalingFactor="2" xOffset="10" yOffset="-5"/>
      <section zOrder="1" zPosition="0" scalingFactor="3" xOffset="-20" yOffset="7"/>
      <section zOrder="2" zPosition="12" scalingFactor="0.5" xOffset="1" yOffset="2"/>
    </xtru>''')
    bounds = _geometry_bounds(solid, {})
    np.testing.assert_allclose(bounds, [[-26, -7, -10], [16, 19, 12]])
    assert world.signed_local(solid, {}) is None
    vertices = np.array([[-2, -1], [3, -1], [3, 4], [-2, 4]])
    sections = [(2, [10, -5], -10), (3, [-20, 7], 0), (0.5, [1, 2], 12)]
    points = [np.column_stack((vertices * scale + offset, np.full(4, z))) for scale, offset, z in sections]
    for start, end in zip(points[:-1], points[1:]):
        for fraction in np.linspace(0, 1, 11):
            sample = start * (1 - fraction) + end * fraction
            assert np.all(sample >= bounds[0]) and np.all(sample <= bounds[1])


@pytest.mark.parametrize("angle,unit,sides", [(360, "deg", 4), (180, "deg", 3), (2 * np.pi, "rad", 6)])
def test_polyhedra_envelope_includes_corners_beyond_apothem(angle, unit, sides):
    solid = ET.fromstring(f'''<polyhedra numsides="{sides}" startphi="10" deltaphi="{angle}" aunit="{unit}">
      <zplane z="-6" rmin="0" rmax="2"/><zplane z="9" rmin="0" rmax="5"/>
    </polyhedra>''')
    radians = angle * np.pi / 180 if unit == "deg" else angle
    r = 5 / np.cos(radians / (2 * sides))
    np.testing.assert_allclose(_geometry_bounds(solid, {}), [[-r, -r, -6], [r, r, 9]])
    assert r > 5


@pytest.mark.parametrize("xml", [
    '<torus rtor="20" rmax="3" lunit="cm"/>',
    '<cone rmax1="2" rmax2="4" z="6" aunit="mrad"/>',
    '<xtru/>', '<polyhedra/>', '<ellipsoid ax="2" by="3" cz="4"/>',
    '<polyhedra numsides="2" deltaphi="360"><zplane z="0" rmax="5"/></polyhedra>',
    '<polyhedra numsides="4" deltaphi="360" aunit="unknown"><zplane z="0" rmax="5"/></polyhedra>',
])
def test_unknown_or_unresolved_shape_bounds_stay_unknown(xml):
    assert _geometry_bounds(ET.fromstring(xml), {}) is None


def test_boolean_union_transforms_second_and_contains_both_operands():
    solids = {s.get("name"): s for s in ET.fromstring('''<solids>
      <box name="a" x="2" y="4" z="6"/><box name="b" x="4" y="8" z="2"/>
      <union name="u"><first ref="a"/><second ref="b"/>
        <position x="10" y="20" z="30" unit="mm"/><rotation z="90" unit="deg"/>
        <scale x="1" y="1" z="1"/></union>
      <union name="nested"><first ref="u"/><second ref="a"/><position x="-10"/></union>
    </solids>''')}
    np.testing.assert_allclose(_geometry_bounds(solids["u"], solids), [[-1, -2, -3], [14, 22, 31]])
    np.testing.assert_allclose(_geometry_bounds(solids["nested"], solids), [[-11, -2, -3], [14, 22, 31]])
    for tag in ("subtraction", "intersection"):
        solid = ET.fromstring(f'<{tag}><first ref="a"/><second ref="b"/><position x="100"/></{tag}>')
        np.testing.assert_allclose(_geometry_bounds(solid, solids), [[-1, -2, -3], [1, 2, 3]])


@pytest.mark.parametrize("extra", [
    '<positionref ref="far"/>', '<rotationref ref="turn"/>', '<scaleref ref="identity"/>',
    '<position x="10" unit="cm"/>', '<rotation z="1" unit="mrad"/>',
    '<scale x="1" y="2" z="1"/>', '<firstposition x="10"/>', '<firstrotationref ref="turn"/>',
])
def test_boolean_unresolved_transform_bounds_are_unknown(extra):
    solids = {"a": ET.fromstring('<box x="2" y="4" z="6"/>')}
    solid = ET.fromstring(f'<union><first ref="a"/><second ref="a"/>{extra}</union>')
    assert _geometry_bounds(solid, solids) is None


@pytest.mark.parametrize("unknown_operand", ["first", "second"])
def test_union_with_unknown_operand_cannot_use_only_known_bounds(unknown_operand):
    solids = {"known": ET.fromstring('<box x="2" y="4" z="6"/>'), "unknown": ET.fromstring('<ellipsoid/>')}
    refs = {operand: "unknown" if operand == unknown_operand else "known" for operand in ("first", "second")}
    solid = ET.fromstring(f'<union><first ref="{refs["first"]}"/><second ref="{refs["second"]}"/></union>')
    assert _geometry_bounds(solid, solids) is None


def test_identity_scale_preserves_descendant_bounds(gdml):
    root, volume, place = gdml[1:]
    parent = volume("IdentityMother", material="DSVacuum")
    volume("IdentityChild", "tube")
    place(parent, "IdentityChild", extra='<scale x="1" y="1" z="1"/>')
    place(root.find("structure/volume"), "IdentityMother", extra='<scale x="1" y="1" z="1"/>')
    save(gdml)
    row = rows(gdml)["IdentityChild"]
    assert row["geometry_bbox_relation"] == "overlap"
    assert row["geometry_bounds_mm"] == [[-3, -3, -4], [3, 3, 4]]
    assert row["classification"] == "retained"
    assert not row["placement_assumptions"]


def test_new_bounds_do_not_enable_unsupported_calo_transport(gdml):
    root, volume, place = gdml[1:]
    ET.SubElement(root.find("solids"), "torus", name="torus", rtor="20", rmax="3")
    volume("CaloTorus", "torus")
    place(root.find("structure/volume[@name='CalorimeterMother']"), "CaloTorus")
    save(gdml)
    row = rows(gdml)["CaloTorus"]
    assert row["geometry_bounds_mm"] == [[-23, -23, -3], [23, 23, 3]]
    assert row["geometry_bbox_relation"] == "overlap"
    assert row["classification"] == "unsupported"
    assert not row["interior_selected"]
    assert row["signed_pieces_selected"] == 0



def test_external_union_retained_when_only_transformed_second_overlaps(gdml):
    root, volume, place = gdml[1:]
    root.find("solids").append(ET.fromstring('''<union name="external_union">
      <first ref="tube"/><second ref="box"/><position x="-100"/>
      <rotation z="90" unit="deg"/></union>'''))
    volume("ExternalUnion", "external_union")
    place(root.find("structure/volume"), "ExternalUnion", x=100)
    save(gdml)
    row = rows(gdml)["ExternalUnion"]
    assert row["geometry_bbox_relation"] == "overlap"
    np.testing.assert_allclose(row["geometry_bounds_mm"], [[-3, -3, -4], [103, 3, 4]])
    assert row["classification"] == "bbox_excluded"
    assert not row["tube_loader_selected"]
    assert "boolean_first_only:union" in row["approximations"]


def test_xtru_geometry_bounds_do_not_follow_loader_ignored_offsets(gdml):
    root, volume, place = gdml[1:]
    root.find("solids").append(ET.fromstring('''<xtru name="offset_xtru">
      <twoDimVertex x="-2" y="-1"/><twoDimVertex x="3" y="-1"/><twoDimVertex x="0" y="4"/>
      <section zOrder="0" zPosition="-2" xOffset="100" yOffset="200"/>
      <section zOrder="1" zPosition="2" xOffset="110" yOffset="220"/>
    </xtru>'''))
    volume("CaloOffsetXtru", "offset_xtru")
    place(root.find("structure/volume[@name='CalorimeterMother']"), "CaloOffsetXtru")
    save(gdml)
    row = rows(gdml)["CaloOffsetXtru"]
    assert row["geometry_bbox_relation"] == "disjoint"
    assert row["geometry_bounds_mm"] == [[98, 199, -2], [113, 224, 2]]
    assert row["classification"] == "approximate"
    assert row["interior_selected"]
    assert row["signed_pieces_selected"] == 1
    assert "xtru_section_offsets_ignored" in row["approximations"]
