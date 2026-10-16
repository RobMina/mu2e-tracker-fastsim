"""Smoke test: the package and its core dependency import."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_import():
    import trkfastsim
    assert trkfastsim.__version__
