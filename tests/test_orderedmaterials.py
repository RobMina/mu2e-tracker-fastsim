"""Explicit additional constants never alias a material or mutate the core registry."""
import numpy as np
import pytest

from mu2efastsim import eloss, materials
from trkfastsim.orderedmaterials import MaterialResolver


def fixture(tmp_path, *, density="0.935", fraction="0.86"):
    path = tmp_path / "materials.gdml"
    path.write_text(f'''<gdml><materials><element name="H0x1" Z="1"/><element name="C0x2" Z="6"/>
    <material name="Polyethylene09350x3" state="solid"><MEE unit="eV" value="56.982601836529"/>
    <D unit="g/cm3" value="{density}"/><fraction n="0.14" ref="H0x1"/><fraction n="{fraction}" ref="C0x2"/>
    </material></materials></gdml>''')
    return path


def test_explicit_constants_and_frozen_core_registry(tmp_path):
    before = materials.BY_NAME.copy()
    resolver = MaterialResolver(fixture(tmp_path), ["Polyethylene0935"])
    mat = resolver("Polyethylene0935")
    assert mat.rho == .935 and mat.I_eV == 56.982601836529 and mat.fractions == {"H": .14, "C": .86}
    assert resolver("G4_Al") is materials.ALUMINIUM
    assert materials.BY_NAME == before and "Polyethylene0935" not in materials.BY_NAME
    assert np.isfinite(eloss.dedx_restricted(mat, np.array([80., 1000.]), 1e9)).all()
    assert np.isfinite(eloss.radiation_length_mm(mat))
    with pytest.raises(KeyError):
        resolver("not-a-material")


@pytest.mark.parametrize("kwargs", [dict(density="nan"), dict(fraction="0.5")])
def test_invalid_constants_refuse_without_normalization(tmp_path, kwargs):
    with pytest.raises(ValueError):
        MaterialResolver(fixture(tmp_path, **kwargs), ["Polyethylene0935"])


def test_material_element_name_cannot_hide_wrong_charge(tmp_path):
    path = fixture(tmp_path)
    path.write_text(path.read_text().replace('name="H0x1" Z="1"', 'name="H0x1" Z="8"'))
    with pytest.raises(ValueError, match="charge"):
        MaterialResolver(path, ["Polyethylene0935"])


def test_no_replacement_or_implicit_material_names(tmp_path):
    path = fixture(tmp_path)
    with pytest.raises(ValueError, match="replace"):
        MaterialResolver(path, ["G4_Al"])
    with pytest.raises(ValueError, match="definition"):
        MaterialResolver(path, ["missing"])
