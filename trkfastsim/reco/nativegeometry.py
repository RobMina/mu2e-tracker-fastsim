"""Opt-in, inferred panel frames for captured-operand geometry diagnostics.

The assumed nominal midpoint grid is ``origin + floor(i/2)*pitch*V +
(i%2)*layer_offset`` for each contiguous, locally ordered group of 96 straws.
Wire midpoint displacements supply two additional independent regressors,
mean wire dV and dW. Their fitted vector coefficients identify the shared
aligned panel V and W, not a separate approximate frame for every straw.

This is a model-validation diagnostic, not independently native-exact geometry.
In particular, dumped endpoints use a rounded float halfLength. Correcting those
endpoints retains its small length error; it does not recover exact native straw
directions. No production caller or default is changed by importing this module.
"""
import numpy as np

NSTRAWS = 96


class GeometryValidationError(ValueError):
    """Inputs do not support the stated rigid, complete-panel inference model."""


def _array(value, name, shape=None):
    """Convert numeric input without accepting nonfinite or malformed geometry."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GeometryValidationError(f"{name}: expected a numeric array") from exc
    if shape is not None and array.shape != shape:
        raise GeometryValidationError(f"{name}: expected shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise GeometryValidationError(f"{name}: non-finite values")
    return array


def _require_panels(ok, message):
    """Report all failing panels, without processing panels individually."""
    if not np.all(ok):
        bad = np.flatnonzero(~ok).tolist()
        raise GeometryValidationError(f"{message}; failing panel indices {bad}")


def straw_axes(cond_straw, wire_cal, wire_hv, align, *, midpoint_atol=1e-8,
               rotation_atol=1e-7, pitch_atol=1e-8, direction_atol=1e-10,
               max_condition=1e8):
    """Return inferred straw axes, panel frames, closure metrics and limitations.

    Inputs are ordered by unique straw, beginning at a panel boundary, with
    exactly 96 rows per panel; row order cannot be certified without straw IDs.
    ``cond_straw`` provides midx/midy/midz and dirx/diry/dirz, each shape [S].
    Endpoints have shape [S,3], and align has shape [S,8] in wire-cal V/W,
    wire-hv V/W, straw-cal V/W, straw-hv V/W order. No padding or slicing is done.

    A single batched SVD solves the centered, RMS-column-scaled five-column
    regression. Rank and condition limits apply to that scaled design. Fitted
    V/W are validated, never projected onto a rotation to hide poor closure.
    Frame columns are U=V cross W, V, W. Pitch must be positive along V; the
    odd/even offset must lie in the V/W plane. Endpoint wire directions must
    point along positive U after subtracting their wire alignment differences.

    Tolerances are explicit absolute limits: midpoint/pitch in input length
    units, rotation for Gram/determinant and nominal-wire transverse fraction,
    direction for endpoint/conditions unit-vector distance. No tolerance is
    fitted to observed residuals. Failures raise GeometryValidationError.

    Return dictionary keys: midpoint [S,3], direction [S,3], panel_frames
    [P,3,3], panel_coefficients [P,5,3] (unscaled original design), panel_metrics
    (per-panel arrays), metrics (scalar summaries), metadata. Straw midpoints
    use conditions wire midpoints plus mean alignment differences; directions
    use corrected dumped endpoint differences and therefore approximate lengths.
    """
    limits = _array([midpoint_atol, rotation_atol, pitch_atol, direction_atol,
                     max_condition], "validation limits")
    if np.any(limits[:4] <= 0) or max_condition <= 1:
        raise GeometryValidationError("validation limits must be positive; max_condition must exceed 1")
    cal = _array(wire_cal, "wire_cal")
    if cal.ndim != 2 or cal.shape[1] != 3:
        raise GeometryValidationError("wire_cal: expected shape [S,3]")
    size = cal.shape[0]
    if size == 0 or size % NSTRAWS:
        raise GeometryValidationError("complete nonempty panels of 96 straws are required")
    hv = _array(wire_hv, "wire_hv", (size, 3))
    alignment = _array(align, "align", (size, 8)).reshape(-1, NSTRAWS, 8)
    try:
        mid = np.stack([_array(cond_straw[f"mid{c}"], f"mid{c}", (size,))
                        for c in "xyz"], axis=-1).reshape(-1, NSTRAWS, 3)
        direction = np.stack([_array(cond_straw[f"dir{c}"], f"dir{c}", (size,))
                              for c in "xyz"], axis=-1).reshape(-1, NSTRAWS, 3)
    except (KeyError, TypeError) as exc:
        raise GeometryValidationError("cond_straw: missing midpoint or direction columns") from exc
    panels = mid.shape[0]
    cal = cal.reshape(panels, NSTRAWS, 3)
    hv = hv.reshape(panels, NSTRAWS, 3)
    chord = cal - hv
    lengths = np.linalg.norm(chord, axis=-1)
    _require_panels(np.all(np.isfinite(lengths) & (lengths > 0), axis=1),
                    "wire endpoints have zero or non-finite length")
    wire_unit = chord / lengths[..., None]
    mid_closure = np.linalg.norm(0.5 * cal + 0.5 * hv - mid, axis=-1).max(axis=1)
    dir_closure = np.linalg.norm(wire_unit - direction, axis=-1).max(axis=1)
    _require_panels(mid_closure <= midpoint_atol, "conditions/endpoint midpoint closure exceeds midpoint_atol")
    _require_panels(dir_closure <= direction_atol, "conditions/endpoint direction closure exceeds direction_atol")

    index = np.arange(NSTRAWS)
    mean_wire = 0.5 * alignment[..., :2] + 0.5 * alignment[..., 2:4]
    regressors = np.concatenate((np.broadcast_to(
        np.stack((index // 2, index % 2), axis=-1), (panels, NSTRAWS, 2)), mean_wire), axis=-1)
    center = regressors.mean(axis=1, keepdims=True)
    centered = regressors - center
    scale = np.sqrt(np.mean(centered ** 2, axis=1))
    _require_panels(np.all(np.isfinite(scale) & (scale > 0), axis=1),
                    "degenerate alignment design: constant or invalid regressor")
    design = np.concatenate((np.ones((panels, NSTRAWS, 1)), centered / scale[:, None, :]), axis=-1)
    try:
        left, singular, right = np.linalg.svd(design, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise GeometryValidationError("panel design SVD failed") from exc
    rank = np.sum(singular > singular[:, :1] * (NSTRAWS * np.finfo(np.float64).eps), axis=1)
    _require_panels(rank == 5, "degenerate alignment design: rank below five")
    condition = singular[:, 0] / singular[:, -1]
    _require_panels(condition <= max_condition, "alignment design condition exceeds max_condition")
    mid_center = mid.mean(axis=1, keepdims=True)
    beta = np.swapaxes(right, -1, -2) @ (
        (np.swapaxes(left, -1, -2) @ (mid - mid_center)) / singular[..., None])
    residual = np.linalg.norm(mid - mid_center - design @ beta, axis=-1)
    panel_residual = residual.max(axis=1)
    _require_panels(panel_residual <= midpoint_atol, "nonrigid panel midpoint residual exceeds midpoint_atol")
    coefficients = beta.copy()
    coefficients[:, 1:] /= scale[..., None]
    coefficients[:, 0] += mid_center[:, 0] - np.sum(center[:, 0, :, None] * coefficients[:, 1:], axis=1)
    v, w = coefficients[:, 3], coefficients[:, 4]
    u = np.cross(v, w)
    frames = np.stack((u, v, w), axis=-1)
    orthogonality = np.max(np.abs(np.swapaxes(frames, -1, -2) @ frames - np.eye(3)), axis=(1, 2))
    determinant = np.linalg.det(frames)
    _require_panels((orthogonality <= rotation_atol) & (np.abs(determinant - 1) <= rotation_atol),
                    "inferred panel rotation is not orthonormal")
    pitch_vector = coefficients[:, 1]
    pitch = np.sum(pitch_vector * v, axis=-1) / np.sum(v * v, axis=-1)
    pitch_residual = np.linalg.norm(pitch_vector - pitch[:, None] * v, axis=-1)
    layer_u = np.abs(np.sum(coefficients[:, 2] * u, axis=-1))
    _require_panels((pitch > 0) & (pitch_residual <= pitch_atol),
                    "constant pitch is not positive and parallel to V")
    _require_panels(layer_u <= pitch_atol, "odd/even layer offset is outside the panel V/W plane")

    wire_delta = alignment[..., :2] - alignment[..., 2:4]
    nominal_chord = chord - np.einsum("psi,pji->psj", wire_delta, frames[..., 1:])
    nominal_length = np.sum(nominal_chord * u[:, None, :], axis=-1) / np.sum(u * u, axis=-1)[:, None]
    transverse = np.linalg.norm(nominal_chord - nominal_length[..., None] * u[:, None, :], axis=-1) / lengths
    _require_panels(np.all(nominal_length > 0, axis=1) & (transverse.max(axis=1) <= rotation_atol),
                    "wire orientation/nominal transverse closure failed")
    delta_cal = alignment[..., 4:6] - alignment[..., :2]
    delta_hv = alignment[..., 6:8] - alignment[..., 2:4]
    correction_cal = np.einsum("psi,pji->psj", delta_cal, frames[..., 1:])
    correction_hv = np.einsum("psi,pji->psj", delta_hv, frames[..., 1:])
    midpoint_correction = 0.5 * correction_cal + 0.5 * correction_hv
    straw_midpoint = mid + midpoint_correction
    straw_chord = chord + correction_cal - correction_hv
    straw_lengths = np.linalg.norm(straw_chord, axis=-1)
    _require_panels(np.all(np.isfinite(straw_lengths) & (straw_lengths > 0), axis=1),
                    "corrected straw endpoints have zero or non-finite length")
    straw_direction = straw_chord / straw_lengths[..., None]
    _require_panels(np.all(np.isfinite(straw_midpoint), axis=(1, 2)), "corrected straw midpoint is non-finite")
    direction_correction = np.linalg.norm(straw_direction - wire_unit, axis=-1)
    midpoint_correction_norm = np.linalg.norm(midpoint_correction, axis=-1)
    panel_metrics = {
        "midpoint_residual_max": panel_residual,
        "midpoint_residual_rms": np.sqrt(np.mean(residual ** 2, axis=1)),
        "design_rank": rank, "design_condition": condition, "design_singular_values": singular,
        "rotation_orthogonality_max": orthogonality, "rotation_determinant": determinant,
        "pitch": pitch, "pitch_transverse_residual": pitch_residual,
        "layer_u_residual": layer_u,
        "conditions_endpoint_midpoint_max": mid_closure,
        "conditions_endpoint_direction_max": dir_closure,
        "nominal_wire_transverse_fraction_max": transverse.max(axis=1),
        "midpoint_correction_max": midpoint_correction_norm.max(axis=1),
        "direction_correction_max": direction_correction.max(axis=1),
    }
    metrics = {
        "panel_count": panels, "straw_count": size,
        "max_panel_residual": float(panel_residual.max()),
        "max_design_condition": float(condition.max()),
        "max_rotation_orthogonality": float(orthogonality.max()),
        "max_rotation_determinant_error": float(np.abs(determinant - 1).max()),
        "max_pitch_transverse_residual": float(pitch_residual.max()),
        "max_layer_u_residual": float(layer_u.max()),
        "max_conditions_endpoint_midpoint": float(mid_closure.max()),
        "max_conditions_endpoint_direction": float(dir_closure.max()),
        "max_nominal_wire_transverse_fraction": float(transverse.max()),
        "max_midpoint_correction": float(midpoint_correction_norm.max()),
        "rms_midpoint_correction": float(np.sqrt(np.mean(midpoint_correction_norm ** 2))),
        "max_direction_correction": float(direction_correction.max()),
        "max_endpoint_correction": float(max(np.linalg.norm(correction_cal, axis=-1).max(),
                                               np.linalg.norm(correction_hv, axis=-1).max())),
        "min_dumped_wire_length": float(lengths.min()), "max_dumped_wire_length": float(lengths.max()),
    }
    return {
        "midpoint": straw_midpoint.reshape(size, 3), "direction": straw_direction.reshape(size, 3),
        "panel_frames": frames, "panel_coefficients": coefficients,
        "panel_metrics": panel_metrics, "metrics": metrics,
        "metadata": {
            "model": "inferred_validated_rigid_panel_midpoint_grid",
            "independently_native_exact": False,
            "frame_status": "inferred and validated against the assumed midpoint grid",
            "direction_status": "corrected dumped endpoints; rounded halfLength approximation retained",
            "ordering_assumption": "contiguous complete panels, local straw indices 0..95",
            "design_scaling": "centered regressors with unit RMS; intercept unscaled",
            "validation_limits": dict(zip(("midpoint_atol", "rotation_atol", "pitch_atol",
                                           "direction_atol", "max_condition"), map(float, limits))),
        },
    }
