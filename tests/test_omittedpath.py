"""Synthetic omission probes with explicit hierarchy and nonconvex-prism checks."""
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from trkfastsim.omittedpath import PlacementIndex, contains_segments, hierarchy_check, xtru_intervals, xtru_probe, length_bound


def prism():
    return ET.fromstring('''<xtru lunit="mm" name="elbow">
    <twoDimVertex x="0" y="0"/><twoDimVertex x="3" y="0"/>
    <twoDimVertex x="3" y="1"/><twoDimVertex x="1" y="1"/>
    <twoDimVertex x="1" y="3"/><twoDimVertex x="0" y="3"/>
    <section zPosition="-1" scalingFactor="1"/><section zPosition="1" scalingFactor="1"/>
    </xtru>''')


def test_nonconvex_prism_excludes_false_bbox_crossings():
    spans = xtru_intervals(prism(), [[-1., 2., 0.], [1.5, 2., 0.]], [[1., 0., 0.]]*2, [5., 1.])
    assert spans.lane.unique().tolist() == [0]
    np.testing.assert_allclose(spans[["s0_mm", "s1_mm"]], [[1., 2.]])


def test_prism_self_intersection_and_incomplete_triangulation_refused(monkeypatch):
    solid = prism()
    vertices = solid.findall("twoDimVertex")
    vertices[1].set("x", "0")
    vertices[1].set("y", "2")
    with pytest.raises(ValueError, match="self-intersecting"):
        xtru_intervals(solid, [[0., 0., 0.]], [[1., 0., 0.]], [1.])
    from mu2efastsim import world
    monkeypatch.setattr(world, "_convex_pieces_2d", lambda v: [])
    with pytest.raises(ValueError, match="incomplete"):
        xtru_intervals(prism(), [[0., 0., 0.]], [[1., 0., 0.]], [1.])


def test_prism_start_inside_multiple_rays_and_empty():
    spans = xtru_intervals(prism(), [[.5, .5, 0.], [.5, .5, -2.]], [[0., 1., 0.], [0., 0., 1.]], [4., 4.])
    np.testing.assert_allclose(spans[["s0_mm", "s1_mm"]], [[0., 2.5], [1., 3.]])
    assert xtru_intervals(prism(), np.empty((0, 3)), np.empty((0, 3)), []).empty
    assert xtru_intervals(prism(), [[0., 0., 0.]], [[1., 0., 0.]], [0.]).empty


@pytest.mark.parametrize("key,value", [("scalingFactor", "2"), ("xOffset", "1"), ("yOffset", "1")])
def test_unsupported_prism_details_not_silently_ignored(key, value):
    solid = prism()
    solid.find("section").set(key, value)
    with pytest.raises(ValueError, match="unscaled/unshifted"):
        xtru_intervals(solid, [[0., 0., 0.]], [[1., 0., 0.]], [1.])


def test_tube_parent_checks_radial_minimum_not_endpoints_only():
    solid = ET.fromstring('<tube rmin="2" rmax="4" z="4" deltaphi="360" aunit="deg"/>')
    a = np.array([[-3., 0., 0.], [3., -.1, 0.]])
    b = np.array([[3., 0., 0.], [3., .1, 0.]])
    np.testing.assert_array_equal(contains_segments(solid, {}, a, b), [False, True])


def test_near_full_parent_sector_is_not_silently_full():
    solid = ET.fromstring(f'<tube rmin="0" rmax="4" z="4" deltaphi="{2*np.pi-5e-13}" aunit="rad"/>')
    a, b = np.array([[3., -3e-13, 0.]]), np.array([[3., -3e-13, 1.]])
    assert not contains_segments(solid, {}, a, b).any()


def test_guarded_prism_bounds_cover_known_chord_and_false_bbox():
    spans, info = xtru_probe(prism(), np.array([[-1., 2., 0.], [1.5, 2., 0.]]),
                             np.array([[1., 0., 0.]]*2), [5., 1.])
    assert spans.lane.unique().tolist() == [0]
    inside = spans[spans.state == "inside"]
    assert length_bound(inside, upper=False) <= 1. <= length_bound(spans, upper=True)
    assert info["guard_mm"] > 0
    assert (spans.state == "unresolved").any()
    empty, _ = xtru_probe(prism(), np.empty((0, 3)), np.empty((0, 3)), [])
    assert empty.empty and length_bound(empty, upper=True) == 0.


def test_directed_aggregation_surrounds_exact_decimal_width():
    spans = pd.DataFrame(dict(s0_mm=[0., 1., 2.], s1_mm=[.1, 1.2, 2.3]))
    from decimal import Decimal
    exact = sum(Decimal(float(b)) - Decimal(float(a)) for a, b in zip(spans.s0_mm, spans.s1_mm))
    assert Decimal(length_bound(spans, upper=False)) <= exact <= Decimal(length_bound(spans, upper=True))


def test_tapered_parent_not_replaced_by_envelope():
    solid = ET.fromstring('''<polycone><zplane z="-2" rmin="0" rmax="3"/>
                            <zplane z="2" rmin="0" rmax="4"/></polycone>''')
    assert not contains_segments(solid, {}, np.array([[1., 0., 0.]]), np.array([[1., 0., 1.]])).any()


def test_polycone_repeated_z_order_never_invents_cylinder():
    solid = ET.fromstring('''<polycone><zplane z="-2" rmax="2"/>
        <zplane z="0" rmax="4"/><zplane z="0" rmax="2"/><zplane z="2" rmax="4"/></polycone>''')
    a, b = np.array([[3.5, 0., .5]]), np.array([[3.5, 0., 1.]])
    assert not contains_segments(solid, {}, a, b).any()
    solid = ET.fromstring('''<polycone><zplane z="-2" rmax="4"/>
        <zplane z="0" rmax="4"/><zplane z="0" rmax="2"/><zplane z="2" rmax="2"/></polycone>''')
    assert not contains_segments(solid, {}, a, b).any()
    assert contains_segments(solid, {}, a - [0., 0., 1.5], b - [0., 0., 1.5]).all()
    solid.findall("zplane")[-1].set("z", "-1")
    assert not contains_segments(solid, {}, a, b).any()


def test_transformed_boolean_parent_union_is_conservative():
    box = ET.fromstring('<box name="b" x="2" y="2" z="2"/>')
    union = ET.fromstring('<union><first ref="b"/><second ref="b"/><position x="4"/></union>')
    a, b = np.array([[4., 0., 0.], [0., 0., 0.]]), np.array([[4.5, 0., 0.], [4., 0., 0.]])
    np.testing.assert_array_equal(contains_segments(union, {"b": box}, a, b), [True, False])


@pytest.fixture
def index_fixture(tmp_path):
    gdml = tmp_path / "geometry.gdml"
    gdml.write_text('''<gdml><solids><box name="world" x="100" y="100" z="100"/>
    <box name="parent" x="10" y="10" z="10"/><torus name="ring" rmin="0" rmax="1" rtor="2"/>
    </solids><structure><volume name="World"><materialref ref="Vacuum"/><solidref ref="world"/>
    <physvol name="p"><volumeref ref="Parent"/><position x="7"/><rotation z="90" unit="deg"/></physvol>
    </volume><volume name="Parent"><materialref ref="Honeycomb"/><solidref ref="parent"/>
    <physvol name="daughter"><volumeref ref="Leaf"/><position x="1"/></physvol>
    </volume><volume name="Leaf"><materialref ref="Al"/><solidref ref="ring"/></volume>
    </structure><setup><world ref="World"/></setup></gdml>''')
    return gdml, "p[0]/daughter[0]"


def test_index_retains_ancestry_and_material_replacement(index_fixture):
    path, key = index_fixture
    index = PlacementIndex(path, [key])
    placed = index.placements[key]
    np.testing.assert_allclose(placed["position"], [7., -1., 0.], atol=1e-14)
    assert placed["is_leaf"] and placed["ancestors"][-1]["material"] == "Honeycomb"
    spans = pd.DataFrame(dict(lane=[0], s0_mm=[0.], s1_mm=[.5]))
    checked, unresolved = hierarchy_check(index, placed, spans, np.array([[7., -1., 0.]]), np.array([[0., -1., 0.]]))
    assert checked.ancestors_verified.all() and not unresolved
    assert checked.replaces_material.tolist() == ["Honeycomb"]
    checked, unresolved = hierarchy_check(index, placed, spans, np.array([[7., -10., 0.]]), np.array([[0., -1., 0.]]))
    assert not checked.ancestors_verified.any() and unresolved


def test_index_rejects_unknown_path(index_fixture):
    path, _ = index_fixture
    with pytest.raises(ValueError, match="do not match"):
        PlacementIndex(path, ["wrong[0]"])
