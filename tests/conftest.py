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

import pytest

# Placeholder site configuration for the kit tests (mu2efastsim.site): nothing here names a real site, user or
# storage area. Tests that check fail-closed behaviour delete the variables themselves.
SITE_PLACEHOLDERS = {
    "M2FS_DATA": "/path/to/fnal/data",
    "M2FS_CODE": "/path/to/fnal/code",
    "FASTSIMKIT_SCRATCH": "/path/to/scratch",
    "FASTSIMKIT_FNAL_HOST": "fnal-host.example",
    "FASTSIMKIT_SLURM_HOST": "slurm-host.example",
    "FASTSIMKIT_SLURM_ACCOUNT": "test_account",
}
SITE_OPTIONAL = ("M2FS_INPUTS", "FASTSIMKIT_FNAL_DATA", "CRVFS_DATA", "FASTSIMKIT_WORKDIR", "FASTSIMKIT_CODE",
                 "FASTSIMKIT_SIF", "FASTSIMKIT_TRACKER_DATA", "FASTSIMKIT_TRACKER_CODE")


@pytest.fixture
def site_env(monkeypatch):
    """Set the placeholder site variables (and clear optional overrides) for one test."""
    for name in SITE_OPTIONAL:
        monkeypatch.delenv(name, raising=False)
    for name, value in SITE_PLACEHOLDERS.items():
        monkeypatch.setenv(name, value)
    return monkeypatch
