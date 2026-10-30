"""Synchronised diagnostic stage timings for the host-packed JAX digitiser.

This intentionally synchronises every profiled kernel and is not a throughput
benchmark. Use alongside a separate uninstrumented warm repeat.
"""
from contextlib import contextmanager
import time


@contextmanager
def stage_profile(enabled):
    receipt = {}
    if not enabled:
        yield receipt
        return
    import jax
    from trkfastsim import digi_jax as dj, digi_clusters_jax as dc

    def measure(fun, name, device=True):
        def call(*args, **kwargs):
            start = time.perf_counter()
            result = fun(*args, **kwargs)
            if device:
                jax.block_until_ready(result)
            row = receipt.setdefault(name, {"calls": 0, "seconds": 0.})
            row["calls"] += 1
            row["seconds"] += time.perf_counter() - start
            return result
        return call

    charge, grouped_charge, electronics = dj._charge_kernel, dj._charge_blocks_kernel, dj._electronics_kernel
    clusters, counts, sample = dc.make_clusters, dc._counts, dc._sample
    dj._charge_kernel = lambda c: measure(charge(c), "charge")
    dj._charge_blocks_kernel = lambda c: measure(grouped_charge(c), "charge_group")
    dj._electronics_kernel = lambda c, cap, pair, batch: measure(electronics(c, cap, pair, batch), f"electronics/{cap}x{batch}/pairs{pair}")
    dc.make_clusters = measure(clusters, "make_clusters_total", device=False)
    dc._counts = measure(counts, "cluster_counts")
    dc._sample = measure(sample, "cluster_sample")
    try:
        yield receipt
    finally:
        dj._charge_kernel, dj._charge_blocks_kernel, dj._electronics_kernel = charge, grouped_charge, electronics
        dc.make_clusters, dc._counts, dc._sample = clusters, counts, sample
