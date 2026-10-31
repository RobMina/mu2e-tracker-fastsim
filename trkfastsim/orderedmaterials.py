"""Explicit opt-in GDML constants for ordered materials absent from the frozen core registry."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from mu2efastsim import materials, world

POLICY = "ordered-explicit-gdml-material-constants-v1"


class MaterialResolver:
    """No aliases/fallback or global registry mutation; only named pure-element mass-fraction solids."""

    def __init__(self, gdml, names):
        self.policy = POLICY
        self.gdml = Path(gdml)
        self.hash = hashlib.sha256(self.gdml.read_bytes()).hexdigest()
        if not isinstance(names, (list, tuple)) or len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
            raise ValueError("explicit unique material names required")
        root = ET.parse(self.gdml).getroot()
        definitions = root.find("materials")
        self.loaded, self.receipts = {}, {}
        for name in names:
            if name in materials.BY_NAME:
                raise ValueError("explicit resolver may not replace core material constants")
            selected = [m for m in definitions.findall("material") if world._strip(m.get("name")) == name]
            if len(selected) != 1:
                raise ValueError("exactly one GDML material definition required")
            node = selected[0]
            if node.get("state") != "solid" or node.find("composite") is not None:
                raise ValueError("ordered explicit material must be a mass-fraction solid")
            density, mee = node.find("D"), node.find("MEE")
            if density is None or mee is None or density.get("unit") != "g/cm3" or mee.get("unit") != "eV":
                raise ValueError("explicit material requires GDML density g/cm3 and MEE eV")
            rho, excitation = float(density.get("value")), float(mee.get("value"))
            if not np.isfinite([rho, excitation]).all() or min(rho, excitation) <= 0:
                raise ValueError("finite positive density/excitation required")
            fractions = {}
            for part in node.findall("fraction"):
                symbol = world._strip(part.get("ref"))
                if symbol not in materials.ELEMENTS or symbol in fractions:
                    raise ValueError("explicit material requires distinct known elemental mass fractions")
                elements = [e for e in definitions.findall("element") if e.get("name") == part.get("ref")]
                if len(elements) != 1:
                    raise ValueError("material fraction must refer directly to one GDML element")
                element = elements[0]
                z_values = ([float(element.get("Z"))] if element.get("Z") is not None else [])
                for isotope in element.findall("fraction"):
                    matches = [i for i in definitions.findall("isotope") if i.get("name") == isotope.get("ref")]
                    if len(matches) != 1:
                        raise ValueError("explicit elemental isotope definition required")
                    z_values.append(float(matches[0].get("Z")))
                if not z_values or any(not np.isfinite(z) or z != materials.ELEMENTS[symbol][0] for z in z_values):
                    raise ValueError("GDML elemental charge disagrees with frozen core element key")
                value = float(part.get("n"))
                if not np.isfinite(value) or value <= 0:
                    raise ValueError("finite positive mass fractions required")
                fractions[symbol] = value
            if not fractions or abs(sum(fractions.values()) - 1.) > 1e-12:
                raise ValueError("elemental mass fractions must sum to one; no normalization")
            self.loaded[name] = materials.Material(name, rho, excitation, fractions)
            self.receipts[name] = dict(gdml_name=node.get("name"), rho_g_cm3=rho, I_eV=excitation,
                fractions=fractions, density_effect="existing core condensed-material formula, not a new fitted table",
                element_constants="frozen core ELEMENTS; GDML isotope precision not substituted",
                gdml_sha256=self.hash)
        if hashlib.sha256(self.gdml.read_bytes()).hexdigest() != self.hash:
            raise ValueError("material GDML changed during explicit load")

    def __call__(self, name):
        if name in self.loaded:
            return self.loaded[name]
        return materials.get(name)
