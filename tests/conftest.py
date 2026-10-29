"""Shared fastsim test hooks (mu2efastsim.testing): JAX caches released per module; @pytest.mark.slow opt-in.
scripts/ is importable, for the tests of the F1 closure modules that live there."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from mu2efastsim.testing import (_release_jax_caches, pytest_addoption,  # noqa: F401
                                     pytest_collection_modifyitems, pytest_configure)
except ImportError:   # an older core without the shared hooks
    pass
