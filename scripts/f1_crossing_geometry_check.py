"""Vectorized schema-1 crossing geometry/completion gate, not geometry reconstruction.

``validate(tables, expected_targets)`` consumes in-memory Awkward record arrays
keyed by the five full ``crossing_geometry_*`` tree names. It returns a JSON-safe
summary or raises ValueError. No I/O, native execution, normalization, endpoint
inference, or numerical/fit-closure claim is made. Read ROOT trees without field
projection: all writer fields are required, including in empty populations.
"""
import awkward as ak
import numpy as np

from f1_native_decision_audit import KEYS, array, column, dense, keys, match_rows, require

PREFIX = "crossing_geometry_"
COMMON = ("schema_version", "geometry_index")
IDENTITY = ("fullsid", "plane_id", "plane_ordinal")
FRAME = ("origin", "u_direction", "v_direction", "w_direction")
ENDS = ("tube_cal", "tube_hv", "tube_midpoint", "tube_direction",
        "wire_cal", "wire_hv", "wire_midpoint", "wire_direction")
COUNTS = ("nplanes", "npanels", "nstraws")
FIELDS = {
    "bindings": COMMON + KEYS + ("source_status", "error") + COUNTS,
    "tracker": COMMON + ("origin", "straw_inner_radius", "straw_outer_radius", "wire_radius"),
    "planes": COMMON + IDENTITY + FRAME + ("plane_exists", "ordered_panel_fullsid"),
    "panels": COMMON + IDENTITY + FRAME + ("panel_id", "panel_ordinal", "tracker_panel_ordinal",
               "ordered_straw_fullsid", "ds_to_panel_rotation", "ds_to_panel_translation"),
    "straws": COMMON + IDENTITY + ("panel_id", "panel_ordinal", "straw_ordinal",
               "tracker_straw_ordinal", "half_length") + ENDS,
}
I64_MAX = np.iinfo(np.int64).max
I32_MAX = np.iinfo(np.int32).max
LIMITS = [
    "Schema/completion and raw-geometry hierarchy validation only; no independent geometry reconstruction.",
    "No normalization, endpoint/midpoint derivation, rotation orthogonality test, or geometry tolerance.",
    "No crossing_capture_status, constructor/search/discovery, off/on passivity, or full-fit validation.",
    "One writer job per input: geometry_index is job-local; merged jobs with reused indices are rejected.",
]


def _integer(table, name, maximum=I64_MAX):
    """Guard BEFORE any int64 conversion/join (including unsigned overflow)."""
    value = column(table, name, "iu")
    require(value >= 0, name + ": negative integer")
    # Comparing two uint64 arrays avoids signed/unsigned float promotion at 2**63.
    require(value.astype(np.uint64) <= np.uint64(maximum), name + ": integer overflow/range")
    return value.astype(np.int64)


def _float(table, name, width=None, bits=64):
    value = column(table, name, "f") if width is None else dense(table, name, width)
    source = table[name]
    if width is not None:
        # dense's per-row width check is vacuous for zero-row regular arrays.
        shape = ak.type(ak.drop_none(source, axis=0)).content
        require(isinstance(shape, (ak.types.ListType, ak.types.RegularType)), name + ": vector type")
        if isinstance(shape, ak.types.RegularType):
            require(shape.size == width, name + ": empty regular vector width")
        source = ak.flatten(source, axis=1)
    primitive = ak.type(ak.drop_none(source, axis=0)).content
    # Awkward converts unknown empty leaves to float64 in to_numpy: they must
    # not masquerade as typed writer vectors/scalars.
    require(isinstance(primitive, ak.types.NumpyType) and primitive.primitive == "float" + str(bits),
            name + ": expected typed float" + str(bits) + " source")
    require(value.dtype.kind == "f" and value.dtype.itemsize == bits // 8,
            name + ": expected float" + str(bits))
    require(np.isfinite(value), name + ": nonfinite raw geometry")
    return value


def _children(table, name):
    value = table[name]
    require(value.ndim == 2, name + ": expected one vector per row")
    sizes = array(ak.num(value, axis=1))  # Reject missing lists BEFORE flattening.
    flat = array(ak.flatten(value, axis=1))
    require(flat.ndim == 1 and flat.dtype.kind in "iu", name + ": integer vector required")
    require(flat >= 0, name + ": negative child ID")
    require(flat.astype(np.uint64) <= np.uint64(65535), name + ": child ID overflow")
    return sizes, flat.astype(np.int64)


def _population(table, geometry, counts, ordinal=None):
    """Exact full binding coverage; optional flat ordinal uniqueness/contiguity."""
    parent = match_rows(keys(table, ("geometry_index",), unique=False), geometry)
    require(np.bincount(parent, minlength=len(geometry)) == counts, "geometry count mismatch")
    if ordinal is not None:
        identity = keys(table, ("geometry_index", ordinal))
        require(identity[:, 1] < counts[parent], ordinal + ": noncontiguous/outside count")
    return parent


def _hierarchy(child, parent, names, ordinal, children):
    """Ordered child-ID lists must describe exactly the complete child population."""
    parent_key = keys(parent, names)
    child_key = keys(child, names + (ordinal,))
    join = match_rows(child_key[:, :-1], parent_key)
    sizes, ids = _children(parent, children)
    require(np.bincount(join, minlength=len(parent)) == sizes, children + ": child count mismatch")
    index = child_key[:, -1]
    require(index < sizes[join], ordinal + ": noncontiguous/outside parent count")
    offsets = np.cumsum(sizes) - sizes
    require(column(child, "fullsid") == ids[offsets[join] + index], children + ": ordered ID mismatch")
    require(column(child, "plane_id") == column(parent, "plane_id")[join], "parent plane ID mismatch")
    return join


def _validate(tables, expected_targets):
    require(isinstance(tables, dict), "tables must be a dictionary")
    require(isinstance(expected_targets, np.ndarray) and not np.ma.isMaskedArray(expected_targets),
            "expected_targets must be an unmasked numpy integer matrix")
    require(expected_targets.ndim == 2 and expected_targets.shape[1] == 4 and
            len(expected_targets) > 0 and expected_targets.dtype.kind in "iu", "malformed/empty selected targets")
    require(expected_targets >= 0, "negative selected target")
    require(expected_targets.astype(np.uint64) <= np.uint64(I32_MAX), "selected target overflow")
    expected = expected_targets.astype(np.int64)
    require(len(np.unique(expected, axis=0)) == len(expected), "duplicate selected targets")
    # Additional unrelated trees are harmless; extra fields in these schema-1 trees are not.
    data = {}
    for name, fields in FIELDS.items():
        table = tables[PREFIX + name]
        require(isinstance(table, ak.Array) and table.ndim == 1, name + ": expected record array")
        require(set(ak.fields(table)) == set(fields), name + ": writer field set differs")
        require(_integer(table, "schema_version", I32_MAX) == 1, name + ": unsupported schema")
        _integer(table, "geometry_index")
        data[name] = table
    bindings, tracker, planes, panels, straws = (data[name] for name in FIELDS)

    for field in KEYS:
        _integer(bindings, field, I32_MAX)
    status = column(bindings, "source_status", "iu")
    require((status == 0) | (status == 1), "geometry binding failed/unknown status")
    error = column(bindings, "error", "U")
    require(error == "", "geometry binding error")
    counts = np.column_stack([_integer(bindings, field) for field in COUNTS])
    bk = keys(bindings, KEYS + ("source_status",))
    require(len(bindings) == 2 * len(expected), "missing/extra geometry bindings")
    start_rows = np.flatnonzero(status == 0)
    end_rows = np.flatnonzero(status == 1)
    start = start_rows[match_rows(expected, bk[start_rows, :4])]
    end = end_rows[match_rows(expected, bk[end_rows, :4])]
    require(start < end, "geometry completion precedes attempt")
    gi = _integer(bindings, "geometry_index")
    require(gi[start] == gi[end], "attempt/completion geometry_index mismatch")
    require(counts[start] == counts[end], "attempt/completion geometry counts differ")
    geometry = gi[end, None]
    require(len(np.unique(geometry, axis=0)) == len(expected), "geometry_index reused across targets/jobs")
    counts = counts[end]
    _population(tracker, geometry, np.ones(len(expected), dtype=np.int64))
    keys(tracker, ("geometry_index",))

    # All identity columns are checked even for empty child trees. The native
    # StrawId layout is plane:bits10..15, panel:bits7..9, straw:bits0..6;
    # uniquePanel=6*plane+panel, uniqueStraw=96*uniquePanel+straw.
    for name, table in (("planes", planes), ("panels", panels), ("straws", straws)):
        sid = _integer(table, "fullsid", 65535)
        plane = _integer(table, "plane_id", 35)
        _integer(table, "plane_ordinal")
        require((sid >> 10) == plane, name + ": encoded plane ID mismatch")
        keys(table, ("geometry_index", "fullsid"))
        if name == "planes":
            require((sid & 1023) == 0, "plane fullsid contains panel/straw bits")
            require(column(table, "plane_ordinal") == plane, "plane ID/ordinal mismatch")
        else:
            panel = _integer(table, "panel_id", 5)
            _integer(table, "panel_ordinal")
            require(((sid >> 7) & 7) == panel, name + ": encoded panel ID mismatch")
            if name == "panels":
                ordinal = _integer(table, "tracker_panel_ordinal")
                require((sid & 127) == 0, "panel fullsid contains straw bits")
                require(ordinal == 6 * plane + panel, "tracker panel ordinal/ID mismatch")
            else:
                _integer(table, "straw_ordinal")
                ordinal = _integer(table, "tracker_straw_ordinal")
                require((sid & 127) < 96, "invalid encoded straw ID")
                require(ordinal == 96 * (6 * plane + panel) + (sid & 127), "tracker straw ordinal/ID mismatch")
    _population(planes, geometry, counts[:, 0], "plane_ordinal")
    _population(panels, geometry, counts[:, 1], "tracker_panel_ordinal")
    _population(straws, geometry, counts[:, 2], "tracker_straw_ordinal")
    _hierarchy(panels, planes, ("geometry_index", "plane_ordinal"),
               "panel_ordinal", "ordered_panel_fullsid")
    parent = _hierarchy(straws, panels, ("geometry_index", "plane_ordinal", "panel_ordinal"),
                        "straw_ordinal", "ordered_straw_fullsid")
    require(column(straws, "panel_id") == column(panels, "panel_id")[parent], "parent panel ID mismatch")

    exists = column(planes, "plane_exists", "b")
    _float(tracker, "origin", 3)
    for field in ("straw_inner_radius", "straw_outer_radius", "wire_radius"):
        _float(tracker, field)
    for table in (planes, panels):
        for field in FRAME:
            _float(table, field, 3)
    _float(panels, "ds_to_panel_rotation", 9)
    _float(panels, "ds_to_panel_translation", 3)
    _float(straws, "half_length", bits=32)
    for field in ENDS:
        _float(straws, field, 3)

    return {"schema_version": 1, "passed": True, "n_targets": len(expected),
            "n_geometries": len(geometry), "n_bindings": len(bindings), "n_tracker": len(tracker),
            "n_planes": len(planes), "n_panels": len(panels), "n_straws": len(straws),
            "n_absent_planes": int((~exists).sum()), "target_key_fields": list(KEYS),
            "target_keys": expected.tolist(), "geometry_indices": geometry[:, 0].tolist(),
            "complete_target_coverage": True, "complete_geometry_bindings": True,
            "complete_geometry_hierarchy": True, "raw_geometry_finite": True,
            "half_length_float32": True, "geometry_normalized_or_derived": False,
            "full_numerical_validation": False, "full_closure": False, "limits": list(LIMITS)}


def validate(tables: dict, expected_targets: np.ndarray) -> dict:
    """Fail closed on missing, malformed, masked, or overflowing source data.

    Geometry rows may be reordered: joins use the actual job-local geometry_index
    and hierarchy ordinals, never row positions or target rank. Binding row order
    must still show attempt (0) before completion (1) separately for every target.
    Integer join identities are bounded to int64 before conversion; native double
    geometry and float32 half_length must retain their source precision. Empty
    child populations must carry the complete, correctly typed writer schema.
    """
    try:
        return _validate(tables, expected_targets)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        raise ValueError("malformed crossing geometry: " + str(error)) from error
