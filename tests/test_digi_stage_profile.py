"""Profiling wrappers restore production functions, including on failed diagnostics."""
import pytest


def test_disabled_stage_profile_does_not_import_jax():
    from digi_stage_profile import stage_profile
    with stage_profile(False) as receipt:
        assert receipt == {}


def test_stage_profile_restores_kernels_after_error():
    pytest.importorskip("jax")
    from digi_stage_profile import stage_profile
    from trkfastsim import digi_jax as dj, digi_clusters_jax as dc
    before = dj._charge_kernel, dj._charge_blocks_kernel, dj._electronics_kernel, dc.make_clusters, dc._counts, dc._sample
    with pytest.raises(RuntimeError, match="diagnostic"):
        with stage_profile(True):
            assert dj._charge_kernel is not before[0]
            raise RuntimeError("diagnostic")
    assert (dj._charge_kernel, dj._charge_blocks_kernel, dj._electronics_kernel, dc.make_clusters, dc._counts, dc._sample) == before
