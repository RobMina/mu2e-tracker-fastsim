"""Shared fastsim test hooks (mu2efastsim.testing): JAX caches released per module; @pytest.mark.slow opt-in."""
try:
    from mu2efastsim.testing import (_release_jax_caches, pytest_addoption,  # noqa: F401
                                     pytest_collection_modifyitems, pytest_configure)
except ImportError:   # an older core without the shared hooks
    pass
