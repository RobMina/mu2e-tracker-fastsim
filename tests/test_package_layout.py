"""Package layout: subpackages sim, digi, reco, validation, kit; the one-release aliases; scripts/ is not a library."""
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("old,new", [("trkfastsim.kit_task", "trkfastsim.kit.kit_task"),
                                     ("trkfastsim.kit_uva", "trkfastsim.kit.kit_uva"),
                                     ("trkfastsim.worldmods", "trkfastsim.sim.worldmods")])
def test_aliases_resolve_to_the_moved_module(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


def test_nothing_imports_from_scripts():
    """scripts/ holds entry points only; shared code lives in the package."""
    import ast
    entry = {p.stem for p in (ROOT / "scripts").glob("*.py")} | {"scripts"}
    bad = []
    for path in [*(ROOT / "tests").glob("*.py"), *(ROOT / "trkfastsim").rglob("*.py"), *(ROOT / "scripts").glob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else [])
            bad += [f"{path.relative_to(ROOT)}: {n}" for n in names if n.split(".")[0] in entry]
    assert not bad, bad
    assert str(ROOT / "scripts") not in sys.path


def test_script_wrapper_runs_the_package_module():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "digi_cert_report.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "usage" in r.stdout
