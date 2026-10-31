"""Exact stored predicates distinguish sub-budget physical gaps/overlaps from contacts."""
from fractions import Fraction
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from trkfastsim import exactplanar as P


def box(position=(0., 0., 0.), rotation=None):
    return ET.fromstring('<box name="a" x="2" y="2" z="2"/>'), dict(assumptions=[], is_leaf=True,
        rotation=np.eye(3) if rotation is None else rotation, position=np.array(position))


def probe(solid, placement, *, y=0., max_work=2000000, lengths=None):
    return P.intervals(solid, {}, placement, np.array([[-2., y, 0.]]), np.array([[1., 0., 0.]]),
        np.array([6.]) if lengths is None else lengths, reviewed=SimpleNamespace(), max_work=max_work)


def test_binary64_decode_and_directed_root_export():
    values = np.array([0., -0., 1., -2.5, np.nextafter(0., 1.), np.finfo(float).max])
    got = P.rational(values)
    expected = np.array([Fraction.from_float(v) for v in values], object)
    np.testing.assert_array_equal(got, expected)
    roots = np.array([Fraction(1, 3), Fraction(-1, 3), Fraction(2), Fraction(1, 10**325)], object)
    assert (P.rational(P.export(roots, False)) <= roots).all()
    assert (P.rational(P.export(roots, True)) >= roots).all()


@pytest.mark.parametrize("delta,relation", [(0., "contact"), (np.spacing(2.), "gap"), (-np.spacing(2.), "overlap")])
def test_exact_joint_roots_do_not_snap_subbudget_physical_gap_or_overlap(delta, relation):
    a, place_a = box()
    b, place_b = box((2. + delta, 0., 0.))
    left, faults, _ = probe(a, place_a)
    right, other_faults, _ = probe(b, place_b)
    assert faults.empty and other_faults.empty
    end, start = left.q1.iloc[-1], right.q0.iloc[0]
    if relation == "contact":
        assert end == start
    elif relation == "gap":
        assert start > end and start - end < Fraction.from_float(1e-10)
    else:
        assert end > start and end - start < Fraction.from_float(1e-10)


def test_coplanar_positive_ray_length_is_explicit_not_admitted():
    solid, placement = box()
    occupied, faults, result = probe(solid, placement, y=1.)
    assert len(occupied) == 1 and faults.empty
    assert result["coplanar_lanes"] == [0]


def test_promotes_before_translation_and_dot_not_rounded_local_ray():
    solid, placement = box((1., 0., 0.))
    origin = np.array([[1e16, 0., 0.]])
    direction = np.array([[-1., 0., 0.]])
    rows, faults, _ = P.intervals(solid, {}, placement, origin, direction, np.array([1e16]), reviewed=SimpleNamespace())
    assert faults.empty and rows.q0.iloc[0] == Fraction(10**16 - 2)
    assert rows.q1.iloc[0] == Fraction(10**16)


def test_exact_signed_boolean_hole_never_filled():
    a = ET.fromstring('<box name="a" x="4" y="4" z="4"/>')
    b = ET.fromstring('<box name="b" x="2" y="2" z="8"/>')
    solid = ET.fromstring('<subtraction name="s"><first ref="a"/><second ref="b"/></subtraction>')
    _, placement = box()
    rows, faults, _ = P.intervals(solid, {"a": a, "b": b}, placement, np.array([[-3., 0., 0.]]),
        np.array([[1., 0., 0.]]), np.array([6.]), reviewed=SimpleNamespace())
    assert faults.empty and sum(rows.q1 - rows.q0) == Fraction(2)
    assert not ((rows.q0 < Fraction(3)) & (rows.q1 > Fraction(3))).any()


def test_caps_nonleaf_nonfinite_and_zero_length():
    solid, placement = box()
    with pytest.raises(ValueError, match="cap"):
        probe(solid, placement, max_work=1)
    placement["is_leaf"] = False
    with pytest.raises(ValueError, match="leaf"):
        probe(solid, placement)
    placement["is_leaf"] = True
    placement["position"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        probe(solid, placement)
    solid, placement = box()
    rows, faults, result = probe(solid, placement, lengths=np.array([0.]))
    assert rows.empty and faults.empty and not result["coplanar_lanes"]
