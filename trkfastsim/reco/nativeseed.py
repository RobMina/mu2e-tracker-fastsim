"""Certified native seed operands for the downward, field-free F1 line model.

This changes seed parameters, seed covariance and initial owner range only.
The caller must explicitly supply the same native-time check for a comparison;
loading a seed does not select or change the immutable constraint time.
"""
import awkward as ak
import jax
import jax.numpy as jnp
import numpy as np
import uproot

from . import kalman_jax as K, nativetime

# Native muon mass tables differ at the last few published decimal places.
MUON_MASS_ATOL = 1e-5


def to_fitter6(parameters):
    """JAX KL (d0, phi0, z0, theta, t0, p) to (a0, b0, a1, b1, t_y0, p).

    This differentiable primitive assumes finite parameters, p > 0 and u_y < 0;
    use transform_seed or load_seed to validate external operands first.
    """
    d0, phi, z0, theta, t0, mom = parameters
    point = jnp.array([-d0 * jnp.sin(phi), d0 * jnp.cos(phi), z0])
    direction = jnp.array([jnp.sin(theta) * jnp.cos(phi),
                           jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)])
    return K.params_from(point, direction, t0, mom)


_transform = jax.jit(jax.vmap(to_fitter6))
_jacobian = jax.jit(jax.vmap(jax.jacfwd(to_fitter6)))


def transform_seed(parameters, covariance):
    """Validate dense native [N,6], [N,6,6] operands and return x, J C J.T.

    Captured covariance must be symmetric positive definite. No diagonal-only
    approximation, eigenvalue clipping, jitter or covariance regularization is used.
    """
    parameters = np.asarray(parameters, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[1] != 6:
        raise ValueError("native seed parameters must have shape [N,6]")
    if covariance.shape != (len(parameters), 6, 6):
        raise ValueError("native seed covariance must have shape [N,6,6]")
    if not np.all(np.isfinite(parameters)) or not np.all(np.isfinite(covariance)):
        raise ValueError("nonfinite native seed parameters or covariance")
    if np.any(parameters[:, 5] <= 0):
        raise ValueError("native seed momentum must be positive")
    uy = np.sin(parameters[:, 3]) * np.sin(parameters[:, 1])
    if np.any(uy >= 0):
        raise ValueError("native seed must be downward (u_y < 0)")
    if not np.array_equal(covariance, covariance.swapaxes(1, 2)):
        raise ValueError("native seed covariance must be symmetric")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("native seed covariance must be positive definite") from exc
    x = np.asarray(_transform(jnp.asarray(parameters)))
    jac = np.asarray(_jacobian(jnp.asarray(parameters)))
    cov = jac @ covariance @ jac.swapaxes(1, 2)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(cov)):
        raise ValueError("nonfinite transformed native seed")
    return x, cov


def _dense(seed, name, shape):
    try:
        values = ak.to_numpy(seed[name])
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid native seed shape: " + name) from exc
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError("missing native seed operand: " + name)
    if values.shape != shape or values.dtype.kind not in "iuf":
        raise ValueError("invalid native seed shape or type: " + name)
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite native seed operand: " + name)
    return values


def load_seed(check_path, baseline_path, keys):
    """Return dense x[N,6], cov[N,6,6], ranges[N,2], and descriptive metadata.

    nativetime.load_times verifies the full capture gate, evidence hashes, full
    identities and momentum-only mask. This loader reads the same highest tree
    cycle and rejects unsupported schema, mass, direction or constraint operands.
    Native arrays remain in the certified capture, not in JSON metadata. A caller
    comparing fits must require the same check for its explicit constraint_time.
    """
    times, verified = nativetime.load_times(check_path, baseline_path, keys)
    fields = [*nativetime.KEYS, "schema_version", "parameters", "covariance", "mass", "bnom",
              "range_begin", "range_end", "ph_parameters", "ph_covariance"]
    with uproot.open(verified["inputs"]["capture"]["path"]) as root:
        parent = root["kktrace"]
        cycles = [k for k in parent.keys(cycle=True, recursive=False) if k.split(";")[0] == "seed_contract"]
        cycle = max(cycles, key=lambda k: int(k.rsplit(";", 1)[1]))
        if "kktrace/" + cycle != verified["seed_tree"]:
            raise ValueError("native seed tree changed after time verification")
        tree = parent[cycle]
        if not set(fields).issubset(tree.keys()):
            raise ValueError("missing schema-1 native seed operands")
        seed = tree.arrays(fields)
    if len(seed) != verified["native_seeds"]:
        raise ValueError("native seed count changed after time verification")
    if not np.all(_dense(seed, "schema_version", (len(seed),)) == 1):
        raise ValueError("unsupported native seed schema_version (requires 1)")
    native_keys = np.column_stack([ak.to_numpy(seed[k]) for k in nativetime.KEYS])
    selected = seed[nativetime.join_keys(keys, native_keys)]
    n = len(selected)
    par = _dense(selected, "parameters", (n, 6))
    cov = _dense(selected, "covariance", (n, 36)).reshape(n, 6, 6)
    mass = _dense(selected, "mass", (n,))
    if not np.all(np.isclose(mass, K.MMU, rtol=0, atol=MUON_MASS_ATOL)):
        raise ValueError("unsupported native seed mass: requires muon mass")
    _dense(selected, "bnom", (n, 3))
    ranges = np.column_stack([_dense(selected, k, (n,)) for k in ("range_begin", "range_end")])
    if np.any(ranges[:, 0] > ranges[:, 1]):
        raise ValueError("reversed native seed range")
    if not np.array_equal(times, 0.5 * (ranges[:, 0] + ranges[:, 1])):
        raise ValueError("native seed range changed after time verification")
    ph_par = _dense(selected, "ph_parameters", (n, 6))
    ph_cov = _dense(selected, "ph_covariance", (n, 36)).reshape(n, 6, 6)
    if not np.array_equal(ph_par[:, 5], par[:, 5]):
        raise ValueError("native constraint momentum differs from seed momentum")
    variance = ph_cov[:, 5, 5]
    if not np.isfinite(K.MOM_CONSTRAINT) or K.MOM_CONSTRAINT <= 0 or np.any(variance <= 0):
        raise ValueError("native momentum constraint sigma must be positive")
    if not np.all(np.isclose(np.sqrt(variance), K.MOM_CONSTRAINT, rtol=1e-12, atol=0)):
        raise ValueError("native momentum constraint sigma differs from K.MOM_CONSTRAINT")
    x, cov = transform_seed(par, cov)
    meta = {k: verified[k] for k in ("check", "check_sha256", "inputs", "seed_tree", "native_seeds", "matched_seeds")}
    meta.update(schema_version=1, source_parameters="KinematicLine6 (d0,phi0,z0,theta,t0,mom)",
                target_parameters="fitter6 (a0,b0,a1,b1,time_at_y0,mom)",
                covariance_transform="full JAX forward Jacobian: J C J.T; no regularization",
                changed_operands=["seed parameters", "seed covariance", "initial owner range",
                                  "initial prior operands", "seed-derived momentum constraint target"],
                unchanged=["immutable constraint time", "hit and crossing candidates", "prior policy",
                           "status reference policy", "material", "ordering algorithm", "thresholds"],
                constraint_time_policy="caller must explicitly use the same native-time check for comparison",
                mass_policy=f"muon mass within {MUON_MASS_ATOL:g} MeV; fitter speed uses {K.MMU} MeV",
                bnom_policy="finite captured 3-vector required; not used in field-free geometry",
                stock_exact_fit=False)
    return x, cov, ranges, meta
