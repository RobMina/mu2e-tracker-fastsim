"""Wide local transforms and unchanged exported hierarchy contracts on actual GDML fixtures."""
import os

import numpy as np
import pytest

from scripts.ordered_torus_pair_probe import load_pair_sources
from trkfastsim import wideomission as P


@pytest.fixture(scope="module")
def modules():
    source = os.environ.get("TRKFS_REVIEWED_GEOMETRY_SOURCE")
    if source is None:
        pytest.skip("explicit reviewed geometry source required")
    return load_pair_sources(source)


def test_promoted_transform_real_torus_stored_translation_and_ancestry(tmp_path, modules):
    gdml = tmp_path / "wide.gdml"
    gdml.write_text('''<gdml><solids><box name="w" x="100000" y="100000" z="100000"/>
    <box name="p" x="100" y="100" z="100"/>
    <torus name="ring" rtor="10" rmin="0" rmax="2" deltaphi="360" aunit="deg"/></solids>
    <structure><volume name="World"><materialref ref="Vacuum"/><solidref ref="w"/>
    <physvol name="p"><volumeref ref="Parent"/><position x="-3904" z="24171"/></physvol></volume>
    <volume name="Parent"><materialref ref="Vacuum"/><solidref ref="p"/>
    <physvol name="r"><volumeref ref="Leaf"/></physvol></volume>
    <volume name="Leaf"><materialref ref="G4_Al"/><solidref ref="ring"/></volume></structure>
    <setup><world ref="World"/></setup></gdml>''')
    record = dict(classification="unsupported", solid_tag="torus", placement_path="p[0]/r[0]", material="G4_Al", scope="support", is_leaf=True)
    origin = np.array([[-3894., 0., 24166.], [-3904., 0., 24166.]])
    direction, length = np.tile([0., 0., 1.], (2, 1)), np.array([10., 10.])
    wide, reports = P.WideOmission(modules["omittedpath"]).probe(gdml, [record], origin, direction, length)
    assert len(reports) == 1 and reports[0]["resolved"] and not reports[0]["method"]["cap_reached"]
    assert reports[0]["method"]["arithmetic_policy"] == P.POLICY
    assert wide.ancestors_verified.all() and wide.leaf_verified.all()
    assert set(wide.replaces_material) == {"vacuum"}
    assert modules["omittedpath"].length_bound(wide[wide.state == "inside"], upper=False) <= 4.
    assert modules["omittedpath"].length_bound(wide, upper=True) >= 4.
    assert set(wide.lane) == {0}
    control, control_reports = modules["omittedpath"].probe(gdml, [record], origin, direction, length)
    assert reports[0]["solid_lower_mm"] <= control_reports[0]["solid_upper_mm"]
    assert control_reports[0]["solid_lower_mm"] <= reports[0]["solid_upper_mm"]
