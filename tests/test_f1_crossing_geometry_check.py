"""Synthetic geometry/status gates only: no ROOT inputs, remote I/O or native run."""
import ast
import inspect
import json

import awkward as ak
import numpy as np
import pytest

import f1_crossing_geometry_check as check


@pytest.fixture
def sample():
    targets = np.array([[12, 3, 9, 1], [12, 3, 9, 2]], dtype=np.int32)
    geometries = np.array([41, 7], dtype=np.uint64)  # Neither target rank nor contiguous.

    def common(n):
        return {"schema_version": np.ones(2 * n, dtype=np.int32),
                "geometry_index": np.repeat(geometries, n)}

    def frame(n):
        # Deliberately non-unit, non-orthogonal frames. They are raw stored data.
        return {name: np.tile(np.array([1., 2., 3.]), (2 * n, 1)) for name in check.FRAME}

    b = common(2)
    b.update({name: np.repeat(targets[:, i], 2) for i, name in enumerate(check.KEYS)})
    b.update(source_status=np.tile(np.array([0, 1], dtype=np.int32), 2), error=[""] * 4,
             nplanes=np.full(4, 2, dtype=np.uint64), npanels=np.full(4, 2, dtype=np.uint64),
             nstraws=np.full(4, 98, dtype=np.uint64))
    t = common(1)
    t.update(origin=np.array([[1., 2., 3.], [4., 5., 6.]]),
             straw_inner_radius=np.full(2, 2.45), straw_outer_radius=np.full(2, 2.5),
             wire_radius=np.full(2, .0125))
    p = common(2)
    p.update(frame(2))
    p.update(fullsid=np.tile(np.array([0, 1024], dtype=np.uint32), 2),
             plane_id=np.tile(np.array([0, 1], dtype=np.uint32), 2),
             plane_ordinal=np.tile(np.array([0, 1], dtype=np.uint64), 2),
             plane_exists=np.tile([True, False], 2),
             ordered_panel_fullsid=ak.values_astype(ak.Array([[0, 128], [], [0, 128], []]), np.uint32))
    a = common(2)
    a.update(frame(2))
    a.update(fullsid=np.tile(np.array([0, 128], dtype=np.uint32), 2),
             plane_id=np.zeros(4, dtype=np.uint32), plane_ordinal=np.zeros(4, dtype=np.uint64),
             panel_id=np.tile(np.array([0, 1], dtype=np.uint32), 2),
             panel_ordinal=np.tile(np.array([0, 1], dtype=np.uint64), 2),
             tracker_panel_ordinal=np.tile(np.array([0, 1], dtype=np.uint64), 2),
             ordered_straw_fullsid=ak.unflatten(np.tile(np.r_[np.arange(96), [128, 129]], 2).astype(np.uint32), [96, 2, 96, 2]),
             ds_to_panel_rotation=np.tile(np.arange(9, dtype=np.float64), (4, 1)),
             ds_to_panel_translation=np.tile(np.array([.11, .22, .33]), (4, 1)))
    s = common(98)
    s.update(fullsid=np.tile(np.r_[np.arange(96), [128, 129]], 2).astype(np.uint32),
             plane_id=np.zeros(196, dtype=np.uint32), plane_ordinal=np.zeros(196, dtype=np.uint64),
             panel_id=np.tile(np.r_[np.zeros(96, dtype=np.uint32), [1, 1]], 2).astype(np.uint32),
             panel_ordinal=np.tile(np.r_[np.zeros(96, dtype=np.uint64), [1, 1]], 2).astype(np.uint64),
             straw_ordinal=np.tile(np.r_[np.arange(96), [0, 1]], 2).astype(np.uint64),
             tracker_straw_ordinal=np.tile(np.arange(98, dtype=np.uint64), 2),
             half_length=np.full(196, np.float32(100.12345), dtype=np.float32))
    s.update({field: np.tile(np.array([5.25, -2., 9.125]), (196, 1)) for field in check.ENDS})
    return {check.PREFIX + name: ak.Array(table) for name, table in
            (("bindings", b), ("tracker", t), ("planes", p), ("panels", a), ("straws", s))}, targets


def replace(tables, tree, field, value):
    name = check.PREFIX + tree
    tables[name] = ak.with_field(tables[name], value, field)


def change(tables, tree, field, index, value, dtype=None):
    values = ak.to_numpy(tables[check.PREFIX + tree][field]).copy()
    if dtype is not None:
        values = values.astype(dtype)
    values[index] = value
    replace(tables, tree, field, values)


def empty_geometry(sample):
    tables, targets = sample
    for name in ("planes", "panels", "straws"):
        tables[check.PREFIX + name] = tables[check.PREFIX + name][:0]
    for name in check.COUNTS:
        replace(tables, "bindings", name, np.zeros(4, dtype=np.uint64))
    return tables, targets


def test_complete_raw_geometry_and_json(sample):
    tables, targets = sample
    before = {name: ak.copy(table) for name, table in tables.items()}
    result = check.validate(tables, targets)
    assert result["passed"] and result["n_targets"] == 2
    assert result["geometry_indices"] == [41, 7]
    assert result["n_straws"] == 196 and result["n_absent_planes"] == 2
    assert result["half_length_float32"]
    assert not result["geometry_normalized_or_derived"]
    assert not result["full_closure"] and not result["full_numerical_validation"]
    json.dumps(result, allow_nan=False)
    for name, table in tables.items():
        assert ak.almost_equal(before[name], table, rtol=0, atol=0)
    # The intentionally inconsistent endpoints/directions/transforms are not
    # reconstructed, normalized, constrained to a tolerance or silently corrected.
    assert ak.to_numpy(tables[check.PREFIX + "straws"].half_length).dtype == np.float32


def test_actual_geometry_join_and_reordered_rows(sample):
    tables, targets = sample
    for name in ("tracker", "planes", "panels", "straws"):
        tables[check.PREFIX + name] = tables[check.PREFIX + name][::-1]
    tables[check.PREFIX + "bindings"] = tables[check.PREFIX + "bindings"][[2, 0, 3, 1]]
    assert check.validate(tables, targets[::-1])["geometry_indices"] == [7, 41]


def test_empty_typed_children_are_valid(sample):
    tables, targets = empty_geometry(sample)
    result = check.validate(tables, targets)
    assert result["n_straws"] == result["n_panels"] == result["n_planes"] == 0


@pytest.mark.parametrize("tree", list(check.FIELDS))
def test_missing_tree(sample, tree):
    tables, targets = sample
    del tables[check.PREFIX + tree]
    with pytest.raises(ValueError):
        check.validate(tables, targets)


@pytest.mark.parametrize("tree,field", [(tree, field) for tree, fields in check.FIELDS.items() for field in fields])
@pytest.mark.parametrize("empty", [False, True])
def test_every_source_field_mandatory_even_empty(sample, tree, field, empty):
    tables, targets = empty_geometry(sample) if empty else sample
    tables[check.PREFIX + tree] = ak.without_field(tables[check.PREFIX + tree], field)
    with pytest.raises(ValueError):
        check.validate(tables, targets)


@pytest.mark.parametrize("tree", list(check.FIELDS))
def test_extra_field_and_bad_schema(sample, tree):
    tables, targets = sample
    original = tables[check.PREFIX + tree]
    replace(tables, tree, "new_field", np.zeros(len(original)))
    with pytest.raises(ValueError):
        check.validate(tables, targets)
    tables[check.PREFIX + tree] = original
    change(tables, tree, "schema_version", 0, 2)
    with pytest.raises(ValueError):
        check.validate(tables, targets)


@pytest.mark.parametrize("targets", [np.empty((0, 4), dtype=int), np.ones((2, 3), dtype=int),
    np.ones(4, dtype=int), np.ones((1, 4)), np.ones((1, 4), dtype=bool),
    np.array([[12, 3, -1, 1]]), np.array([[12, 3, 9, 1], [12, 3, 9, 1]]),
    np.array([[12, 3, 9, 2**63]], dtype=np.uint64),
    np.array([[12, 3, 9, 2**31]], dtype=np.uint64),
    np.ma.array([[12, 3, 9, 1]], mask=[[False, False, False, True]]), [[12, 3, 9, 1]], None])
def test_bad_selected_targets(sample, targets):
    with pytest.raises(ValueError):
        check.validate(sample[0], targets)


@pytest.mark.parametrize("selection", [[0, 1], [0, 2, 3], [0, 1, 2], [1, 0, 2, 3],
    [0, 1, 0, 1], [0, 1, 2, 3, 3], [0, 2], []])
def test_missing_duplicate_reversed_extra_bindings(sample, selection):
    tables, targets = sample
    tables[check.PREFIX + "bindings"] = tables[check.PREFIX + "bindings"][np.array(selection, dtype=int)]
    with pytest.raises(ValueError):
        check.validate(tables, targets)


@pytest.mark.parametrize("field,index,value", [("source_status", 1, -1), ("source_status", 1, 2),
    ("source_status", 1, 0), ("source_status", 0, 1), ("error", 0, "failure"),
    ("error", 1, "failure"), ("geometry_index", 1, 42), ("run", 2, 13),
    ("event", 3, 10), ("iseed", 2, 8), ("nplanes", 1, 3), ("npanels", 1, 1),
    ("nstraws", 1, 197)])
def test_invalid_binding_payload(sample, field, index, value):
    tables, targets = sample
    change(tables, "bindings", field, index, value, dtype="U20" if field == "error" else None)
    with pytest.raises(ValueError):
        check.validate(tables, targets)


def test_geometry_index_reused_by_targets(sample):
    tables, targets = sample
    replace(tables, "bindings", "geometry_index", np.full(4, 41, dtype=np.uint64))
    with pytest.raises(ValueError, match="reused"):
        check.validate(tables, targets)


@pytest.mark.parametrize("tree", ["tracker", "planes", "panels", "straws"])
@pytest.mark.parametrize("mode", ["missing", "duplicate", "orphan", "wrong_target", "empty"])
def test_geometry_population_failures(sample, tree, mode):
    tables, targets = sample
    name = check.PREFIX + tree
    original = tables[name]
    if mode == "missing":
        tables[name] = original[1:]
    elif mode == "duplicate":
        tables[name] = ak.concatenate([original, original[:1]])
    elif mode == "empty":
        tables[name] = original[:0]
    else:
        change(tables, tree, "geometry_index", 0, 999 if mode == "orphan" else 7)
    with pytest.raises(ValueError):
        check.validate(tables, targets)


@pytest.mark.parametrize("field,value", [("nplanes", 3), ("npanels", 3), ("nstraws", 99)])
def test_consistent_but_wrong_binding_count(sample, field, value):
    tables, targets = sample
    replace(tables, "bindings", field, np.full(4, value, dtype=np.uint64))
    with pytest.raises(ValueError, match="count"):
        check.validate(tables, targets)


@pytest.mark.parametrize("tree,field,value", [
    ("planes", "plane_ordinal", 3), ("planes", "plane_id", 1), ("planes", "fullsid", 1),
    ("planes", "fullsid", 128), ("planes", "plane_id", 36),
    ("panels", "plane_ordinal", 1), ("panels", "plane_id", 1), ("panels", "panel_ordinal", 2),
    ("panels", "panel_id", 2), ("panels", "panel_id", 6), ("panels", "fullsid", 1),
    ("panels", "tracker_panel_ordinal", 2), ("panels", "fullsid", 256),
    ("straws", "plane_ordinal", 1), ("straws", "plane_id", 1), ("straws", "panel_ordinal", 1),
    ("straws", "panel_id", 1), ("straws", "straw_ordinal", 98),
    ("straws", "tracker_straw_ordinal", 198), ("straws", "fullsid", 96),
    ("straws", "fullsid", 65535), ("straws", "fullsid", 65536), ("straws", "fullsid", 1)])
def test_hierarchy_and_id_corruption(sample, tree, field, value):
    tables, targets = sample
    change(tables, tree, field, 0, value)
    with pytest.raises(ValueError):
        check.validate(tables, targets)


@pytest.mark.parametrize("field", ["panel_ordinal", "tracker_panel_ordinal"])
def test_duplicate_panel_ordinals(sample, field):
    tables, targets = sample
    change(tables, "panels", field, 1, 0)
    with pytest.raises(ValueError):
        check.validate(tables, targets)


@pytest.mark.parametrize("field", ["straw_ordinal", "tracker_straw_ordinal"])
def test_duplicate_straw_ordinals(sample, field):
    tables, targets = sample
    change(tables, "straws", field, 1, 0)
    with pytest.raises(ValueError):
        check.validate(tables, targets)


@pytest.mark.parametrize("value", [[[128, 0], [], [0, 128], []], [[0, 0], [], [0, 128], []],
    [[0], [], [0, 128], []], [[0, 128, 256], [], [0, 128], []],
    [[0, -1], [], [0, 128], []], [[0, 65536], [], [0, 128], []],
    [[0., 128.], [], [0., 128.], []], [[0, None], [], [0, 128], []],
    [None, [], [0, 128], []], [0, 0, 0, 0]])
def test_malformed_ordered_children(sample, value):
    tables, targets = sample
    replace(tables, "planes", "ordered_panel_fullsid", ak.Array(value))
    with pytest.raises(ValueError):
        check.validate(tables, targets)


def test_straw_child_list_order(sample):
    tables, targets = sample
    replace(tables, "panels", "ordered_straw_fullsid", tables[check.PREFIX + "panels"].ordered_straw_fullsid[:, ::-1])
    with pytest.raises(ValueError, match="ordered ID"):
        check.validate(tables, targets)


FLOAT_FIELDS = ([("tracker", "origin", 3)] +
    [("tracker", f, None) for f in ("straw_inner_radius", "straw_outer_radius", "wire_radius")] +
    [(tree, f, 3) for tree in ("planes", "panels") for f in check.FRAME] +
    [("panels", "ds_to_panel_rotation", 9), ("panels", "ds_to_panel_translation", 3),
     ("straws", "half_length", None)] + [("straws", f, 3) for f in check.ENDS])


@pytest.mark.parametrize("tree,field,width", FLOAT_FIELDS)
@pytest.mark.parametrize("mode", ["nan", "inf", "wrong_type", "mask", "wrong_shape"])
def test_every_raw_geometry_field_checked(sample, tree, field, width, mode):
    tables, targets = sample
    values = ak.to_numpy(tables[check.PREFIX + tree][field]).copy()
    if mode == "nan":
        values.flat[0] = np.nan
    elif mode == "inf":
        values.flat[0] = np.inf
    elif mode == "wrong_type":
        values = values.astype(np.float64 if field == "half_length" else np.float32)
    elif mode == "wrong_shape":
        values = values[:, None] if width is None else values[:, :-1]
    else:
        mask = np.zeros(values.shape, dtype=bool)
        mask.flat[0] = True
        values = ak.from_numpy(np.ma.array(values, mask=mask))
    replace(tables, tree, field, values)
    with pytest.raises(ValueError):
        check.validate(tables, targets)


@pytest.mark.parametrize("tree,field", [("bindings", "run"), ("bindings", "geometry_index"),
    ("bindings", "nstraws"), ("tracker", "geometry_index"), ("planes", "plane_ordinal"),
    ("panels", "panel_ordinal"), ("straws", "tracker_straw_ordinal")])
@pytest.mark.parametrize("value", [2**63, 2**63 + 1, 2**64 - 1])
def test_uint64_overflow_before_join(sample, tree, field, value):
    tables, targets = sample
    change(tables, tree, field, 0, value, dtype=np.uint64)
    with pytest.raises(ValueError, match="overflow"):
        check.validate(tables, targets)


@pytest.mark.parametrize("tree,field", [("bindings", "run"), ("bindings", "nstraws"),
    ("tracker", "geometry_index"), ("planes", "plane_id"), ("panels", "panel_ordinal"),
    ("straws", "tracker_straw_ordinal")])
@pytest.mark.parametrize("mode", ["negative", "float", "bool", "masked", "nested"])
def test_malformed_integer_columns(sample, tree, field, mode):
    tables, targets = sample
    values = ak.to_numpy(tables[check.PREFIX + tree][field]).astype(np.int64)
    if mode == "negative":
        values[0] = -1
    elif mode == "float":
        values = values.astype(float)
    elif mode == "bool":
        values = values.astype(bool)
    elif mode == "nested":
        values = values[:, None]
    else:
        values = ak.from_numpy(np.ma.array(values, mask=np.arange(len(values)) == 0))
    replace(tables, tree, field, values)
    with pytest.raises(ValueError):
        check.validate(tables, targets)


@pytest.mark.parametrize("tree", list(check.FIELDS))
def test_missing_record(sample, tree):
    tables, targets = sample
    original = tables[check.PREFIX + tree]
    tables[check.PREFIX + tree] = ak.mask(original, np.arange(len(original)) != 0)
    with pytest.raises(ValueError):
        check.validate(tables, targets)


@pytest.mark.parametrize("tree,field,width", [entry for entry in FLOAT_FIELDS if entry[0] != "tracker"])
def test_empty_geometry_source_type_checked(sample, tree, field, width):
    tables, targets = empty_geometry(sample)
    dtype = np.float64 if field == "half_length" else np.float32
    replace(tables, tree, field, np.empty((0,) if width is None else (0, width), dtype=dtype))
    with pytest.raises(ValueError):
        check.validate(tables, targets)


@pytest.mark.parametrize("field,value", [("plane_exists", [0, 1, 0, 1]),
    ("plane_exists", [True, None, True, False]), ("plane_exists", [[True]] * 4)])
def test_plane_exists_type_and_masks(sample, field, value):
    tables, targets = sample
    replace(tables, "planes", field, ak.Array(value))
    with pytest.raises(ValueError):
        check.validate(tables, targets)


def test_no_scalar_population_walk_or_awkward_to_list():
    source = inspect.getsource(check)
    assert "ak.to_list" not in source
    module = ast.parse(source)
    # Every explicit loop is over fixed tree/field names, never captured rows.
    loops = [node for node in ast.walk(module) if isinstance(node, (ast.For, ast.comprehension))]
    assert loops
    for node in loops:
        iterator = ast.unparse(node.iter)
        assert not any(word in iterator for word in ("range(", "targets", "np.unique", "enumerate(", "zip("))


@pytest.mark.parametrize("tree,field,width", [entry for entry in FLOAT_FIELDS if entry[0] != "tracker"])
def test_empty_geometry_unknown_type_rejected(sample, tree, field, width):
    tables, targets = empty_geometry(sample)
    unknown = ak.Array([]) if width is None else ak.unflatten(ak.Array([]), np.empty(0, dtype=np.int64))
    replace(tables, tree, field, unknown)
    with pytest.raises(ValueError):
        check.validate(tables, targets)


@pytest.mark.parametrize("tree,field,width", [entry for entry in FLOAT_FIELDS if entry[0] != "tracker" and entry[2] is not None])
def test_empty_regular_vector_wrong_width_rejected(sample, tree, field, width):
    tables, targets = empty_geometry(sample)
    replace(tables, tree, field, np.empty((0, width + 1), dtype=np.float64))
    with pytest.raises(ValueError):
        check.validate(tables, targets)


def test_empty_typed_jagged_geometry(sample):
    tables, targets = empty_geometry(sample)
    for tree, field, width in FLOAT_FIELDS:
        if tree != "tracker" and width is not None:
            replace(tables, tree, field, ak.unflatten(np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)))
    assert check.validate(tables, targets)["passed"]


def test_ordered_local_hierarchy_need_not_equal_native_id_order(sample):
    # Writer records ordered pointer collections and compact native IDs separately.
    # Local ordinals may be permuted consistently; tracker ordinals remain native.
    tables, targets = sample
    replace(tables, "planes", "ordered_panel_fullsid", ak.Array([[128, 0], [], [128, 0], []]))
    for tree in ("panels", "straws"):
        replace(tables, tree, "panel_ordinal", 1 - tables[check.PREFIX + tree].panel_ordinal)
    assert check.validate(tables, targets)["passed"]


def test_wrong_plane_parent_even_with_globally_valid_counts(sample):
    tables, targets = sample
    change(tables, "panels", "plane_ordinal", 0, 1)
    replace(tables, "planes", "ordered_panel_fullsid", ak.Array([[128], [0], [0, 128], []]))
    change(tables, "panels", "panel_ordinal", 1, 0)
    with pytest.raises(ValueError, match="parent plane"):
        check.validate(tables, targets)


def test_max_supported_geometry_index_is_not_rounded(sample):
    tables, targets = sample
    for tree in check.FIELDS:
        index = ak.to_numpy(tables[check.PREFIX + tree].geometry_index)
        replace(tables, tree, "geometry_index", np.where(index == 41, np.uint64(2**63 - 1), index))
    assert check.validate(tables, targets)["geometry_indices"] == [2**63 - 1, 7]


def test_all_five_typed_empty_trees_cannot_satisfy_selected_targets(sample):
    tables, targets = sample
    tables = {name: table[:0] for name, table in tables.items()}
    with pytest.raises(ValueError):
        check.validate(tables, targets)


@pytest.mark.parametrize("tree,field", [("bindings", "error"), ("bindings", "source_status")])
def test_binding_mask(sample, tree, field):
    tables, targets = sample
    values = tables[check.PREFIX + tree][field]
    replace(tables, tree, field, ak.mask(values, np.arange(len(values)) != 0))
    with pytest.raises(ValueError):
        check.validate(tables, targets)
