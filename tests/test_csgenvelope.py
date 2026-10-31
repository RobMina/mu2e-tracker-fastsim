"""Signed membership bounds preserve unknown boundary bands and Boolean analytic holes."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from trkfastsim import csgenvelope as P


def test_actual_subtraction_keeps_hole_and_analytic_occupied_length():
    a = ET.fromstring('<box name="a" x="10" y="10" z="10"/>')
    b = ET.fromstring('<box name="b" x="2" y="2" z="20"/>')
    solid = ET.fromstring('<subtraction name="s"><first ref="a"/><second ref="b"/></subtraction>')
    cells, report = P.probe(solid, {"a": a, "b": b}, np.array([[-10., 0., 0.]]), np.array([[1., 0., 0.]]), np.array([20.]))
    inside = cells[cells.state == "inside"]
    assert (inside.s1_mm - inside.s0_mm).sum() <= 8.
    assert (cells.s1_mm - cells.s0_mm).sum() >= 8.
    assert not report["capped"] and report["certificate_scheme"] == P.POLICY
    assert (cells.state == "unresolved").any()
    assert not ((inside.s0_mm < 10.) & (inside.s1_mm > 10.)).any()


def test_coplanar_box_boundary_is_never_certified_inside():
    box = ET.fromstring('<box name="a" x="10" y="10" z="10"/>')
    cells, _ = P.probe(box, {}, np.array([[-10., 5., 0.]]), np.array([[1., 0., 0.]]), np.array([20.]))
    assert not (cells.state == "inside").any()
    assert (cells.state == "unresolved").all()


def test_invalid_units_or_piece_cap_refuse():
    box = ET.fromstring('<box name="a" x="10" y="10" z="10" lunit="cm"/>')
    with pytest.raises(ValueError, match="units"):
        P.probe(box, {}, np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([1.]))
    box.set("lunit", "mm")
    with pytest.raises(ValueError, match="cap"):
        P.probe(box, {}, np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([1.]), max_pairs=0)


@pytest.mark.parametrize("guard", [np.nan, np.inf, -1.])
def test_nonfinite_or_negative_guard_refuses_not_silent_outside(guard):
    box = ET.fromstring('<box name="a" x="10" y="10" z="10"/>')
    with pytest.raises(ValueError, match="coordinate guard"):
        P.probe(box, {}, np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([1.]), coordinate_guard_mm=guard)


def test_bad_xtru_axial_extent_and_nested_construction_cap():
    xtru = ET.fromstring('''<xtru name="a"><twoDimVertex x="0" y="0"/><twoDimVertex x="2" y="0"/>
        <twoDimVertex x="0" y="2"/><section zPosition="nan"/><section zPosition="1"/></xtru>''')
    with pytest.raises(ValueError, match="coordinates"):
        P.probe(xtru, {}, np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([1.]))
    box = ET.fromstring('<box name="a" x="10" y="10" z="10"/>')
    solid = ET.fromstring('<union name="u"><first ref="a"/><second ref="a"/></union>')
    with pytest.raises(ValueError, match="construction"):
        P.probe(solid, {"a": box}, np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([1.]), max_pairs=10)


def test_empty_and_zero_length_have_complete_miss_counts():
    box = ET.fromstring('<box name="a" x="10" y="10" z="10"/>')
    for origin, direction, lengths in ((np.empty((0, 3)), np.empty((0, 3)), np.array([])),
                                        (np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([0.]))):
        cells, report = P.probe(box, {}, origin, direction, lengths)
        assert cells.empty and report["unresolved_cells"] == 0 and not report["capped"]


def test_nonfinite_boolean_translation_refuses_before_empty_intersection():
    box = ET.fromstring('<box name="a" x="10" y="10" z="10"/>')
    solid = ET.fromstring('<intersection name="i"><first ref="a"/><second ref="a"/><position x="inf"/></intersection>')
    with pytest.raises(ValueError, match="finite CSG transform"):
        P.probe(solid, {"a": box}, np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([1.]))
