"""Explicit geometry source loading refuses replacement and records exactly selected modules."""
import sys
from types import ModuleType

import pytest

from scripts.ordered_omitted_solid_audit import load_geometry_sources


def test_geometry_module_sources_are_explicit_and_no_replacement(tmp_path, monkeypatch):
    names = ("navigationcensus", "torusprobe", "omittedpath")
    for name in names:
        (tmp_path / f"{name}.py").write_text(f'NAME = "{name}"\n')
        monkeypatch.delitem(sys.modules, f"trkfastsim.{name}", raising=False)
    loaded = load_geometry_sources(tmp_path)
    try:
        assert {name: module.NAME for name, module in loaded.items()} == {name: name for name in names}
        with pytest.raises(ValueError, match="refusing to replace"):
            load_geometry_sources(tmp_path)
    finally:
        for name in names:
            sys.modules.pop(f"trkfastsim.{name}", None)


def test_already_imported_geometry_is_not_overwritten(tmp_path, monkeypatch):
    sentinel = ModuleType("trkfastsim.navigationcensus")
    monkeypatch.setitem(sys.modules, sentinel.__name__, sentinel)
    with pytest.raises(ValueError, match="refusing to replace"):
        load_geometry_sources(tmp_path)
    assert sys.modules[sentinel.__name__] is sentinel
