"""Bounded, default-off snapshots of actual F1 operands, not a native-input replay."""
import numpy as np

MAX_BYTES = 64 * 1024 * 1024


def shapes(nhit, nxing):
    """Per-iteration payload dimensions; source identities live in the trace manifest."""
    e, p = nhit + nxing + 1, nxing + 1
    return {
        "valid": (), "iteration": (), "status": (), "swapped": (), "chi2": (), "ndof": (),
        "previous_cpn": (), "cpn": (), "endpoint_chi2": (2,), "avggap": (),
        "reference_parameters": (e - 1, 6), "reference_covariance": (e - 1, 6, 6),
        "reference_index": (e - 1,), "owner_parameters": (p, 6), "owner_covariance": (p, 6, 6),
        "owner_ranges": (p, 2), "owner_count": (),
        "candidate_parameters": (p, 6), "candidate_covariance": (p, 6, 6),
        "candidate_ranges": (p, 2), "candidate_count": (), "next_reference_index": (e - 1,),
        "order": (e,), "ordering_time": (e,), "physical_time": (e,), "reference_query_time": (e,), "is_material": (e,),
        "eff_xref": (e, 6), "eff_H": (e, 3, 6), "eff_r": (e, 3), "eff_w": (e, 3),
        "eff_mvar": (e, 3), "eff_D": (e, 6, 3), "eff_sv": (e,), "eff_dm": (e,), "eff_dmv": (e,),
        "bounds": (2,), "prior_parameters": (2, 6), "prior_covariance": (2, 6, 6),
        "deweight": (), "query": (2,), "status_reference_index": (2,),
        "status_reference_parameters": (2, 6), "status_reference_covariance": (2, 6, 6),
        "hit_active": (nhit,), "hit_lr": (nhit,), "crossing_active": (nxing,),
        "crossing_inbounds": (nxing,), "crossing_doca": (nxing,), "crossing_docavar": (nxing,),
        "crossing_wall": (nxing,), "crossing_gas": (nxing,), "crossing_dirdot": (nxing,),
        "crossing_offset": (nxing,), "crossing_transit": (nxing,),
    }


BOOL_FIELDS = {"valid", "swapped", "is_material", "hit_active", "crossing_active", "crossing_inbounds"}
INT_FIELDS = {"iteration", "status", "ndof", "reference_index", "owner_count", "candidate_count",
              "next_reference_index", "order", "status_reference_index"}


def dtype(name):
    return np.bool_ if name in BOOL_FIELDS else np.int32 if name in INT_FIELDS else np.float64


def allocation_bytes(nhit, nxing, nstage, maxniter, batch=1):
    """Conservative live-buffer estimate including the carry and scan output; reject invalid dimensions."""
    if any(not isinstance(n, (int, np.integer)) or n < 1 for n in (nhit, nxing, nstage, maxniter, batch)):
        raise ValueError("trace dimensions must be positive integers")
    record = sum(int(np.prod(shape)) * np.dtype(dtype(name)).itemsize for name, shape in shapes(nhit, nxing).items())
    return record * maxniter * (nstage + 2) * batch


def empty(nhit, nxing, nstage, maxniter, max_bytes=MAX_BYTES):
    """Allocate one stage buffer only after checking full-history size; no truncation."""
    import jax.numpy as jnp
    required = allocation_bytes(nhit, nxing, nstage, maxniter)
    if required > max_bytes:
        raise ValueError(f"F1 trace needs {required} bytes, exceeding limit {max_bytes}")
    return {name: jnp.zeros((maxniter,) + shape, dtype=dtype(name)) for name, shape in shapes(nhit, nxing).items()}
