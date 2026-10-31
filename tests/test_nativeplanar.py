"""Literal geometry sensitivity is explicit and does not snap unequal decimal faces."""
from fractions import Fraction
from types import SimpleNamespace
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from trkfastsim import nativeplanar as P
from scripts.ordered_torus_pair_probe import load_pair_sources


def placement(position=(Fraction(0), Fraction(0), Fraction(0))):
    return dict(rotation=np.eye(3, dtype=int).astype(object), position=np.array(position, object), nonquarter_rotations=0)


def test_literal_transform_quarter_turns_do_not_leave_cosine_residue():
    node = ET.fromstring('<physvol><position x="0.1" y="0.2"/><rotation x="90" z="90"/></physvol>')
    r, t, count = P._transform(node)
    assert count == 0 and t[0] + t[1] == Fraction(3, 10)
    np.testing.assert_array_equal(r.T @ r, np.eye(3, dtype=int))
    assert set(r.ravel()) <= {-1, 0, 1}


def test_native_boxes_exact_contact_gap_overlap_and_coplanar():
    box = ET.fromstring('<box name="a" x="2" y="2" z="2"/>')
    o, d, length = np.array([[-2., 0., 0.]]), np.array([[1., 0., 0.]]), np.array([6.])
    def run(p, origin=o):
        return P.intervals(box, {}, p, origin, d, length, reviewed=SimpleNamespace())
    a, _, _ = run(placement())
    for delta in (Fraction(0), Fraction(1, 10**20), Fraction(-1, 10**20)):
        b, _, _ = run(placement((Fraction(2)+delta, Fraction(0), Fraction(0))))
        assert b.q0.iloc[0] - a.q1.iloc[-1] == delta
    rows, _, report = run(placement(), np.array([[-2., 1., 0.]]))
    assert rows.empty and report["coplanar_lanes"] == [0]


def test_native_boolean_subtraction_does_not_fill_hole():
    a = ET.fromstring('<box name="a" x="4" y="4" z="4"/>')
    b = ET.fromstring('<box name="b" x="2" y="2" z="8"/>')
    solid = ET.fromstring('<subtraction name="s"><first ref="a"/><second ref="b"/></subtraction>')
    rows, faults, _ = P.intervals(solid, {"a": a, "b": b}, placement(), np.array([[-3., 0., 0.]]),
        np.array([[1., 0., 0.]]), np.array([6.]), reviewed=SimpleNamespace())
    assert faults.empty and sum(rows.q1 - rows.q0) == Fraction(2)


@pytest.fixture(scope="module")
def modules():
    source = os.environ.get("TRKFS_REVIEWED_GEOMETRY_SOURCE")
    if source is None:
        pytest.skip("explicit reviewed geometry source required")
    return load_pair_sources(source)


def test_native_concave_xtru_uses_polygon_not_triangulation_seams(modules):
    solid = ET.fromstring('''<xtru name="l"><twoDimVertex x="0" y="0"/><twoDimVertex x="4" y="0"/>
      <twoDimVertex x="4" y="1"/><twoDimVertex x="1" y="1"/><twoDimVertex x="1" y="4"/>
      <twoDimVertex x="0" y="4"/><section zPosition="-1"/><section zPosition="1"/></xtru>''')
    o = np.array([[-1., .5, 0.], [-1., 2., 0.]])
    d = np.tile([1., 0., 0.], (2, 1))
    rows, faults, report = P.intervals(solid, {}, placement(), o, d, np.array([6., 6.]), reviewed=modules["omittedpath"])
    widths = (rows.q1 - rows.q0).groupby(rows.lane).sum()
    assert widths[0] == Fraction(4) and widths[1] == Fraction(1)
    assert faults.empty and not report["coplanar_lanes"]


@pytest.mark.parametrize("field,value", [("scalingFactor", "1.00000000000000000001"), ("xOffset", "1e-400")])
def test_literal_nonidentity_sections_cannot_round_to_identity(field, value):
    solid = ET.fromstring(f'''<xtru name="bad"><twoDimVertex x="0" y="0"/><twoDimVertex x="1" y="0"/>
      <twoDimVertex x="0" y="1"/><section zPosition="-1" {field}="{value}"/><section zPosition="1"/></xtru>''')
    with pytest.raises(ValueError, match="exactly identity"):
        P._polygon(solid, SimpleNamespace(_xtru_pieces=lambda s: []))


def test_exact_literal_zero_edge_not_hidden_by_float_validation():
    solid = ET.fromstring('''<xtru name="bad"><twoDimVertex x="0" y="0"/><twoDimVertex x="0" y="0"/>
      <twoDimVertex x="1" y="0"/><twoDimVertex x="0" y="1"/><section zPosition="-1"/><section zPosition="1"/></xtru>''')
    with pytest.raises(ValueError, match="nonzero edges"):
        P._polygon(solid, SimpleNamespace(_xtru_pieces=lambda s: []))


def test_cap_units_and_nonquarter_model_label():
    box = ET.fromstring('<box name="a" x="2" y="2" z="2"/>')
    with pytest.raises(ValueError, match="cap"):
        P.intervals(box, {}, placement(), np.zeros((1, 3)), np.array([[1., 0., 0.]]), np.array([1.]), reviewed=SimpleNamespace(), max_work=1)
    _, _, n = P._transform(ET.fromstring('<physvol><rotation x="45"/></physvol>'))
    assert n == 1
    with pytest.raises(ValueError, match="references"):
        P._transform(ET.fromstring('<physvol><positionref ref="x"/></physvol>'))
