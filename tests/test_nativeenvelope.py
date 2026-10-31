"""Native directed shared roots retain source identities and all strict controls."""
import os

import numpy as np
import pandas as pd
import pytest

from scripts.ordered_torus_pair_probe import load_pair_sources
from trkfastsim import inclusiveenvelope, nativeenvelope as P, wideomission


@pytest.fixture(scope="module")
def modules():
    source = os.environ.get("TRKFS_REVIEWED_GEOMETRY_SOURCE")
    if source is None:
        pytest.skip("explicit reviewed geometry source required")
    return load_pair_sources(source)


def fixture(tmp_path, modules, *, separation="2", coplanar=False):
    gdml = tmp_path / "g.gdml"
    gdml.write_text(f'''<gdml><solids><box name="world" x="1000" y="1000" z="1000"/>
    <xtru name="prism"><twoDimVertex x="-1" y="-1"/><twoDimVertex x="1" y="-1"/>
    <twoDimVertex x="1" y="1"/><twoDimVertex x="-1" y="1"/><section zPosition="-1"/><section zPosition="1"/></xtru>
    </solids><structure><volume name="World"><materialref ref="Vacuum"/><solidref ref="world"/>
    <physvol name="a"><volumeref ref="A"/></physvol>
    <physvol name="b"><volumeref ref="B"/><position x="{separation}"/></physvol></volume>
    <volume name="A"><materialref ref="CONCRETE_MARS"/><solidref ref="prism"/></volume>
    <volume name="B"><materialref ref="MBOverburden"/><solidref ref="prism"/></volume>
    </structure><setup><world ref="World"/></setup></gdml>''')
    records = modules["navigationcensus"].census(gdml, np.array([[-10.]*3, [10.]*3]))["records"]
    base = inclusiveenvelope.CertifiedOmission(wideomission.WideOmission(modules["omittedpath"], authenticate_xtru=True), modules["omittedpath"])
    probe = P.NativeOmission(base, modules["omittedpath"])
    o, d, length = np.array([[-2., 1. if coplanar else 0., 0.]]), np.array([[1., 0., 0.]]), np.array([6.])
    cells, reports = probe.probe(gdml, records, o, d, length)
    raw = pd.DataFrame({k: pd.Series(dtype=t) for k, t in dict(lane="int64", s0_mm="float64", s1_mm="float64", material="object", sign="int64").items()})
    return P.NativeEnvelope(modules["supportenvelope"], modules["supportnavigation"], probe), [raw, length, cells, reports, records]


def compose(envelope, args):
    return envelope.compose(*args, endpoint_budget_mm=0., min_cell_mm=1e-9, max_component_mm=1e-6, max_ray_unknown_mm=1e-5)


def test_joint_shared_root_contacts_not_priority_and_old_control_preserved(tmp_path, modules):
    envelope, args = fixture(tmp_path, modules)
    result = compose(envelope, args)
    assert result["summary"]["all_pairs_admitted"]
    assert result["summary"]["native_placements"] == 2
    assert result["native_possible_footprint"].source.nunique() == 2
    assert not result["original_inclusive_control"]["summary"]["all_pairs_admitted"]
    assert not result["summary"]["admitted_for_transport"]
    assert result["summary"]["mode"] == P.POLICY
    intervals = result["inner_intervals"]
    assert intervals.material.isin(["CONCRETE_MARS", "MBOverburden"]).sum() == 2


def test_subbudget_positive_overlap_is_not_snapped(tmp_path, modules):
    with pytest.raises(ValueError, match="positive exact overlap"):
        fixture(tmp_path, modules, separation="1.99999999999999")


def test_coplanar_shared_face_is_not_assumed_resolved(tmp_path, modules):
    with pytest.raises(ValueError, match="boundary contact"):
        fixture(tmp_path, modules, coplanar=True)


@pytest.mark.parametrize("key", ["footprint", "roots", "root_unknown"])
def test_mutated_full_footprint_evidence_refuses(tmp_path, modules, key):
    envelope, args = fixture(tmp_path, modules)
    envelope.probe.latest[key] = envelope.probe.latest[key].assign(forged=1)
    with pytest.raises(ValueError, match="mutated"):
        compose(envelope, args)


def test_returned_footprint_is_defensive_copy(tmp_path, modules):
    envelope, args = fixture(tmp_path, modules)
    first = compose(envelope, args)
    first["native_possible_footprint"].loc[0, "ancestors_verified"] = False
    assert compose(envelope, args)["summary"]["all_pairs_admitted"]


def test_mutated_native_source_selection_refuses(tmp_path, modules):
    envelope, args = fixture(tmp_path, modules)
    envelope.probe.latest["methods"] = []
    with pytest.raises(ValueError, match="selection"):
        compose(envelope, args)


def test_mutated_native_cells_refuse(tmp_path, modules):
    envelope, args = fixture(tmp_path, modules)
    args[2].loc[0, "ancestors_verified"] = False
    with pytest.raises(ValueError, match="unbound"):
        compose(envelope, args)
