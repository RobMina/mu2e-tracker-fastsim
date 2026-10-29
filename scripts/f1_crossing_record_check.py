"""Vectorized schema/lifecycle audit of passive crossing sidecars, not physics replay.

validate raises ValueError for malformed/incomplete/inconsistent evidence. No ROOT IO,
CLI, tolerance matching, record/target Python loops, or native calculations occur here.
"""
import awkward as ak
import numpy as np

from f1_native_decision_audit import array, column, dense, keys, match_rows, require

KEYS = ("run", "subrun", "event", "iseed")
EFFECT_KEYS = KEYS + ("iteration_ordinal", "effect_index")
PRECISION = float(np.float32(1e-4))
PIECE = {"parameters": 6, "covariance": 36, "range": 2, "bnom": 3}
DERIVED = {"particle_poca": 4, "sensor_poca": 4, "particle_direction": 3,
           "sensor_direction": 3, "d_dd_p": 6, "d_td_p": 6}
LINE = {"line_start": 3, "line_end": 3, "line_direction": 3}
CA_FLOAT = ("doca", "doca_variance", "toca_variance", "lsign")
SEARCH_INT = ("search_initial_index", "search_final_index", "search_solution_index", "search_max_moves")
SEARCH_FLOAT = ("search_particle_hint", "search_sensor_hint", "search_precision")
SEARCH_VECTORS = {"search_solution_indices": "iu", "search_input_particle_hints": "f",
                  "search_input_sensor_hints": "f", "search_particle_times": "f",
                  "search_sensor_times": "f", "search_statuses": "iu", "search_usable": "biu"}
COMMON = {name: ("iu", 1) for name in ("schema_version",) + KEYS}
STATE = {"ca_status": ("iu", 1), "ca_usable": ("biu", 1), "precision": ("f", 1),
         "reference_mass": ("f", 1), "reference_charge": ("iu", 1),
         "line_measurement_time": ("f", 1), "line_length": ("f", 1),
         **{name: ("f", 1) for name in CA_FLOAT},
         **{name: ("f", 2) for name in (*DERIVED, *LINE)},
         **{"reference_" + name: ("f", 2) for name in PIECE}}
SEARCH = {"has_search": ("biu", 1), "search_failed": ("biu", 1), "search_exhausted": ("biu", 1),
          **{name: ("iu", 1) for name in SEARCH_INT},
          **{name: ("f", 1) for name in SEARCH_FLOAT},
          **{name: (kind, 2) for name, kind in SEARCH_VECTORS.items()}}
RECORD = {**COMMON, **STATE, **SEARCH,
          **{name: ("iu", 1) for name in ("object", "sequence", "stage", "category", "hint_kind", "strawid",
                                          "partner", "trajectory_index", "iteration_ordinal", "effect_index", "input_index")},
          **{name: ("f", 1) for name in ("offset", "effect_time", "radius_operand", "speed_operand")},
          "intrinsic_active": ("biu", 1), "active": ("biu", 1)}
TRAJECTORY = {**COMMON, **{name: ("iu", 1) for name in ("trajectory_index", "category", "piece_index", "charge")},
              "mass": ("f", 1), **{name: ("f", 2) for name in PIECE}}
DISCOVERY = {**COMMON, **STATE, **SEARCH,
             **{name: ("iu", 1) for name in ("sequence", "stage", "trajectory_index", "plane", "panel", "straw", "object")},
             "passed": ("biu", 1), "integers": ("iu", 2), "values": ("f", 2)}
STATUS = {**COMMON, **{name: ("iu", 1) for name in ("status", "n_records", "n_objects", "n_trajectories", "n_discovery")},
          "error": ("US", 1)}
EFFECT = {**COMMON, **{name: ("iu", 1) for name in ("iteration_ordinal", "effect_index", "identity_kind", "kind",
                                                 "input_index", "strawid", "partner_shindex")},
          **{name: ("biu", 1) for name in ("identity_valid", "identity_from_final_collections", "has_reference",
                                          "has_change", "payload_supported", "active")},
          "time": ("f", 1), **{"reference_" + name: ("f", 2) for name in PIECE}}
SCHEMAS = {"crossing_records": RECORD, "crossing_trajectories": TRAJECTORY,
           "crossing_discovery": DISCOVERY, "crossing_capture_status": STATUS}
LIMITS = [
    "Structural, exact captured-payload and constructor quotient checks only; no independent CA/material/trajectory physics replay.",
    "Discovery checks stage/order/length/search/object binding, NOT detailed physical cuts or geometry-universe completeness.",
    "Original effects partner metadata may be final-collection stamped; historical partners are checked in the contemporaneous record stream only.",
    "Unusable CA effect times are deliberately absent in the new sidecar and are not compared to unguarded original effect times.",
    "No full-fit closure, runtime passivity, predecessor comparison, geometry validation or native-build certification.",
]


def _schema(table, schema, label, exact=True):
    require(isinstance(table, ak.Array), label + ": expected awkward array")
    fields = set(ak.fields(table))
    require(set(schema) <= fields, label + ": missing required fields")
    if exact:
        require(fields == set(schema), label + ": unexpected schema fields")
    for name, (kind, ndim) in schema.items():
        x = table[name]
        require(x.ndim == ndim, label + "/" + name + ": shape")
        for axis in range(ndim):
            require(not ak.any(ak.is_none(x, axis=axis), axis=None), label + "/" + name + ": missing/masked value")
        source_type = ak.type(x).content
        if ndim == 2:
            require(isinstance(source_type, (ak.types.ListType, ak.types.RegularType)), label + "/" + name + ": vector type")
            source_type = source_type.content
        if kind != "US":
            require(isinstance(source_type, ak.types.NumpyType), label + "/" + name + ": untyped/masked primitive")
        value = array(x if ndim == 1 else ak.flatten(x, axis=1))
        # Typed empty arrays are required too; untyped [] is not schema evidence.
        require(value.dtype.kind in kind, label + "/" + name + ": dtype")
        if kind == "f":
            require(value.dtype.itemsize == 8, label + "/" + name + ": expected double")
        if value.dtype.kind == "u":
            require(value <= np.uint64(np.iinfo(np.int64).max), label + "/" + name + ": integer overflow")
        if kind == "iu" and name in (*COMMON, "ca_status", "category", "hint_kind", "stage", "status", "charge", "reference_charge"):
            require((value >= np.iinfo(np.int32).min) & (value <= np.iinfo(np.int32).max), label + "/" + name + ": native int32 overflow")
        if kind == "biu":
            require((value == 0) | (value == 1), label + "/" + name + ": invalid boolean")
    require(column(table, "schema_version") == 1, label + ": unsupported schema")


def _i(table, name):
    return column(table, name).astype(np.int64)


def _b(table, name):
    return column(table, name).astype(bool)


def _ordinal(parent, ordinal, label):
    """Require zero-based contiguous ordinals in each parent's physical row order."""
    order = np.argsort(parent, kind="stable")
    p = parent[order]
    ix = np.arange(len(p))
    start = np.r_[True, p[1:] != p[:-1]][:len(p)]
    expected = ix - np.maximum.accumulate(np.where(start, ix, 0))
    require(ordinal[order] == expected, label + ": noncontiguous/order")


def _pieces(table, prefix=""):
    for name, width in PIECE.items():
        dense(table, prefix + name, width)
    r = dense(table, prefix + "range", 2)
    require(r[:, 0] <= r[:, 1], prefix + "range reversed")
    mass = column(table, prefix + "mass")
    require(np.isfinite(mass) & (mass >= 0), prefix + "mass invalid")


def _equal_fields(a, b, fields, label):
    for name in fields:
        x, y = a[name], b[name]
        require(x.ndim == y.ndim, label + "/" + name + ": shape")
        if x.ndim == 2:
            require(array(ak.num(x, axis=1)) == array(ak.num(y, axis=1)), label + "/" + name + ": lengths")
            x, y = ak.flatten(x, axis=1), ak.flatten(y, axis=1)
        xx, yy = array(x), array(y)
        require(np.array_equal(xx, yy, equal_nan=xx.dtype.kind == "f"), label + "/" + name + ": exact mismatch")


def _state(table, present):
    status, usable = _i(table, "ca_status"), _b(table, "ca_usable")
    require((status >= 0) & (status <= 6), "invalid ca_status")
    require(usable == (status < 3), "ca_usable/status mismatch")
    require(~usable | present, "CA state without presence")
    precision = column(table, "precision")
    require(precision[present] == PRECISION, "unsupported exact CA precision")
    require(np.isnan(precision[~present]) & (status[~present] == 6), "absent state defaults")
    for name, width in {**DERIVED, **LINE, **{"reference_" + n: w for n, w in PIECE.items()}}.items():
        mask = usable if name in DERIVED else present
        require(array(ak.num(table[name], axis=1)) == width * mask, name + ": usable/present/empty shape")
        dense(table[mask], name, width)
    for name in CA_FLOAT:
        x = column(table, name)
        require(np.isfinite(x[usable]), name + ": nonfinite usable field")
        require(np.isnan(x[~usable]), name + ": unusable must be NaN")
    require(np.isin(column(table, "lsign")[usable], [-1., 1.]), "invalid lsign")
    _pieces(table[present], "reference_")
    for name in ("line_measurement_time", "line_length", "reference_mass"):
        x = column(table, name)
        require(np.isfinite(x[present]), name + ": nonfinite state")
        require(np.isnan(x[~present]), name + ": absent state default")
    require(column(table, "line_length")[present] > 0, "invalid line_length")
    require(_i(table, "reference_charge")[~present] == 0, "absent reference_charge")
    return usable


def _search(table, trajectories, trajectory_keys, trajectory_counts):
    has = _b(table, "has_search")
    n = array(ak.num(table["search_solution_indices"], axis=1))
    require((n > 0) == has, "search missing/nonempty defaults")
    for name in SEARCH_VECTORS:
        require(array(ak.num(table[name], axis=1)) == n, name + ": search vector alignment")
    require(~_b(table, "search_failed"), "failed search")
    require(~_b(table, "search_exhausted")[~has], "absent exhausted search")
    for name in SEARCH_INT:
        x = _i(table, name)
        require(x >= 0, name + ": negative index/count")
        require(x[~has] == 0, name + ": absent search default")
    for name in SEARCH_FLOAT:
        x = column(table, name)
        require(np.isfinite(x[has]), name + ": nonfinite search hint")
        require(np.isnan(x[~has]), name + ": absent search default")
    s = table[has]
    if not len(s):
        return
    n = n[has]
    parent = match_rows(keys(s, KEYS + ("trajectory_index",), unique=False), trajectory_keys)
    counts = trajectory_counts[parent]
    require(column(s, "search_precision") == PRECISION, "unsupported search precision")
    require(column(s, "search_precision") == column(s, "precision"), "state/search precision differs")
    initial, final, solution, cap = (_i(s, name) for name in SEARCH_INT)
    require((initial < counts) & (final < counts) & (solution < counts), "search index outside trajectory")
    require(cap <= np.iinfo(np.uint32).max, "search max_moves overflow")
    require(n - 1 <= cap, "search exceeds move cap")
    flat = {name: array(ak.flatten(s[name], axis=1)) for name in SEARCH_VECTORS}
    row = np.repeat(np.arange(len(s)), n)
    starts, ends = np.cumsum(n) - n, np.cumsum(n) - 1
    idx = flat["search_solution_indices"].astype(np.int64)
    status = flat["search_statuses"].astype(np.int64)
    usable = flat["search_usable"].astype(bool)
    require((idx >= 0) & (idx < counts[row]), "solved index outside trajectory")
    require((status >= 0) & (status <= 6) & (usable == (status < 3)), "search status/usability")
    require((idx[starts] == initial) & (idx[ends] == solution), "initial/last solved index mismatch")
    for incoming, hint, time in (("search_input_particle_hints", "search_particle_hint", "search_particle_times"),
                                 ("search_input_sensor_hints", "search_sensor_hint", "search_sensor_times")):
        v, t = flat[incoming], flat[time]
        require(np.isfinite(v), "nonfinite search input hints")
        require(v[starts] == column(s, hint), "initial search hints differ")
        require(np.isfinite(t[usable]), "nonfinite usable search times")
        require(np.isnan(t[~usable]), "unusable search times must be NaN")
        same = row[1:] == row[:-1]
        require(v[1:][same] == t[:-1][same], "search hint continuation differs")
    same = row[1:] == row[:-1]
    require(usable[:-1][same] & (idx[1:][same] != idx[:-1][same]), "invalid search continuation")
    exhausted = _b(s, "search_exhausted")
    require(~exhausted | ((cap > 0) & (n - 1 == cap) & usable[ends] & (final != solution)), "invalid exhausted search")
    require((cap != 0) | ((n == 1) & (initial == final) & ~exhausted), "zero-cap search semantics")
    require(exhausted | ~usable[ends] | (final == solution), "stable usable search pointer differs")
    expected_status = np.where(exhausted & (status[ends] == 0), 1, status[ends])
    require(_i(s, "ca_status") == expected_status, "post-cap CA status differs")
    u = _b(s, "ca_usable")
    for poca, time in (("particle_poca", "search_particle_times"), ("sensor_poca", "search_sensor_times")):
        require(dense(s[u], poca, 4)[:, 3] == flat[time][ends][u], "search solution TOCA differs")
    # The CA reference is the FINAL POINTER, not necessarily the last solved piece.
    wanted = np.column_stack((keys(s, KEYS + ("trajectory_index",), unique=False), final))
    p = match_rows(wanted, keys(trajectories, KEYS + ("trajectory_index", "piece_index")))
    for name, width in PIECE.items():
        require(dense(s, "reference_" + name, width) == dense(trajectories[p], name, width), "search final-pointer reference " + name)
    for name in ("mass", "charge"):
        require(column(s, "reference_" + name) == column(trajectories[p], name), "search final-pointer reference " + name)


def _discovery(d, ctors, trajectory_keys, trajectory_category):
    stage, passed = _i(d, "stage"), _b(d, "passed")
    require((stage >= 0) & (stage <= 8), "invalid discovery stage")
    require(_b(d, "has_search") == (stage == 7), "discovery search stage")
    parent = match_rows(keys(d, KEYS + ("trajectory_index",), unique=False), trajectory_keys)
    require(trajectory_category[parent] == 2, "discovery trajectory category")
    require(array(ak.num(d["values"], axis=1)) == np.array([17, 0, 6, 11, 1, 0, 0, 5, 0])[stage], "discovery values length")
    ni = array(ak.num(d["integers"], axis=1))
    expected = np.array([2, 0, 0, 3, 1, 2, 1, 0, 0])[stage]
    require(np.where(stage == 0, ni >= expected, ni == expected), "discovery integers length")
    require(np.isfinite(array(ak.flatten(d[stage != 7]["values"], axis=1))), "nonfinite discovery operands")
    # Stage 7's native unguarded variance/sqrt operands may legitimately be NaN.
    for name, first in (("plane", 1), ("panel", 3), ("straw", 6)):
        value = _i(d, name)
        require(np.where(stage >= first, (value >= 0) & (value <= 65535), value == -1), "discovery " + name + " identity")
    require(passed[np.isin(stage, [0, 5, 8])], "unsupported/invalid discovery branch")
    require(_i(d, "object")[stage != 8] == -1, "unexpected discovery object")
    # Every supplied material-discovery trajectory has its stage-0 header, even
    # when no straw is accepted. Detailed geometry-universe completeness is separate.
    require(len(np.unique(parent)) == int(np.sum(trajectory_category == 2)), "missing discovery trajectory header")
    # Source order within each supplied discovery trajectory, independent of target interleaving.
    order = np.argsort(parent, kind="stable")
    p, st, ok = parent[order], stage[order], passed[order]
    first = np.r_[True, p[1:] != p[:-1]][:len(p)]
    require((st == 0) == first, "discovery must start once at stage 0")
    # Nonmandatory transitions can return to the next straw/panel/plane, but
    # cannot fabricate descendants of an older successful parent.
    allowed = np.zeros((9, 9), dtype=bool)
    allowed[:, 1] = True
    allowed[2:, 3] = True
    allowed[5:, 6] = True
    allowed[1, 2] = allowed[3, 4] = allowed[4, 5] = True
    allowed[6, 7] = allowed[7, 8] = True
    require(first[1:] | allowed[st[:-1], st[1:]], "discovery stage transition")
    pos = np.arange(len(d))
    # Certain passed stages have an unconditional immediately emitted child.
    for ancestor, child in ((1, 2), (3, 4), (4, 5), (6, 7), (7, 8)):
        reached = np.flatnonzero((st == ancestor) & ok)
        require(reached + 1 < len(st), "discovery missing mandatory child")
        require((st[reached + 1] == child) & (p[reached + 1] == p[reached]), "discovery mandatory child order")
    for child, ancestor, identities in ((1, 0, ()), (2, 1, ("plane",)), (3, 2, ("plane",)),
                                        (4, 3, ("plane", "panel")), (5, 4, ("plane", "panel")),
                                        (6, 5, ("plane", "panel")), (7, 6, ("plane", "panel", "straw")),
                                        (8, 7, ("plane", "panel", "straw"))):
        take = st == child
        previous = np.maximum.accumulate(np.where(st == ancestor, pos, -1))[take]
        require(previous >= 0, "discovery missing ancestor")
        current = pos[take]
        require((p[previous] == p[current]) & ok[previous], "discovery parent/order/pass")
        if child in (2, 4, 5, 7, 8):
            require(previous + 1 == current, "discovery immediate stage order")
        for name in identities:
            value = _i(d, name)[order]
            require(value[previous] == value[current], "discovery parent " + name)
    # Every accepted attempted PCA must immediately bind one constructor, and vice versa.
    accept = np.flatnonzero((st == 7) & ok)
    require(accept + 1 < len(st), "accepted discovery missing constructor link")
    require((st[accept + 1] == 8) & (p[accept + 1] == p[accept]), "accepted discovery missing object stage")
    linked = d[stage == 8]
    ck = keys(ctors, KEYS + ("object",))
    lk = keys(linked, KEYS + ("object",))
    ci = match_rows(lk, ck)
    require(len(linked) == int(np.sum(_i(ctors, "category") == 2)), "material constructor discovery coverage")
    require((_i(ctors, "category")[ci] == 2) & (_i(ctors, "trajectory_index")[ci] == _i(linked, "trajectory_index")) &
            (_i(ctors, "strawid")[ci] == _i(linked, "straw")), "discovery constructor identity")
    # Order accepted PCA rows in exactly the stage-8 rows' physical order.
    previous_by_row = np.empty(len(d), dtype=np.int64)
    previous_by_row[order] = np.maximum(pos - 1, 0)
    attempted = d[order[previous_by_row[np.flatnonzero(stage == 8)]]]
    _equal_fields(attempted, ctors[ci], (*STATE, *SEARCH), "discovery constructor search/state")


def validate(tables, effects, expected_targets):
    """Return a JSON-safe successful summary; raise ValueError on any gate failure."""
    require(isinstance(expected_targets, np.ndarray) and not np.ma.isMaskedArray(expected_targets), "expected_targets must be unmasked numpy")
    require(expected_targets.ndim == 2 and expected_targets.shape[1] == 4 and expected_targets.dtype.kind in "iu", "expected_targets Nx4 integer shape/type")
    require((expected_targets >= 0) & (expected_targets <= np.uint64(np.iinfo(np.int64).max)), "expected_targets invalid/overflow")
    targets = expected_targets.astype(np.int64)
    require(len(np.unique(targets, axis=0)) == len(targets), "duplicate expected targets")
    require(set(SCHEMAS) <= set(tables), "missing crossing tables")
    for name, schema in SCHEMAS.items():
        _schema(tables[name], schema, name)
    _schema(effects, EFFECT, "effects", exact=False)
    r, t, d, status = (tables[name] for name in SCHEMAS)
    rp, tp, dp, sp = (match_rows(keys(x, unique=False), targets) for x in (r, t, d, status))
    match_rows(keys(effects, unique=False), targets)
    require(len(status) == 2 * len(targets), "status population must be attempt then complete")
    _ordinal(sp, _i(status, "status"), "status attempt then complete")
    require(np.bincount(sp, minlength=len(targets)) == 2, "missing target status pair")
    require(column(status, "error") == "", "capture status error")
    _ordinal(rp, _i(r, "sequence"), "record sequence")
    _ordinal(dp, _i(d, "sequence"), "discovery sequence")
    tkeys = keys(t, KEYS + ("trajectory_index", "piece_index"))
    trajectory_keys, first, inverse, counts = np.unique(tkeys[:, :5], axis=0, return_index=True, return_inverse=True, return_counts=True)
    physical_first = np.sort(first)
    _ordinal(tp[physical_first], _i(t, "trajectory_index")[physical_first], "trajectory IDs")
    _ordinal(inverse, _i(t, "piece_index"), "trajectory piece IDs")
    serialization = np.argsort(tp, kind="stable")
    same_target = tp[serialization[1:]] == tp[serialization[:-1]]
    require(~same_target | (_i(t, "trajectory_index")[serialization[1:]] >= _i(t, "trajectory_index")[serialization[:-1]]),
            "trajectory pieces interleaved/out of source order")
    category = _i(t, "category")
    require((category >= 0) & (category <= 5) & (category == category[first][inverse]), "trajectory category")
    _pieces(t)
    rt = match_rows(keys(r, KEYS + ("trajectory_index",), unique=False), trajectory_keys)
    stage = _i(r, "stage")
    require((stage >= 0) & (stage <= 4), "invalid record stage")
    ctor = stage == 0
    c = r[ctor]
    ck = keys(c, KEYS + ("object",))
    _ordinal(rp[ctor], _i(c, "object"), "constructor object IDs")
    cp = match_rows(keys(r, KEYS + ("object",), unique=False), ck)
    require(_i(r, "sequence") >= _i(c, "sequence")[cp], "record precedes constructor")
    for name in ("category", "strawid", "offset"):
        require(column(r, name) == column(c, name)[cp], "fixed constructor " + name + " differs")
    require((_i(r, "category") >= 0) & (_i(r, "category") <= 2), "record category")
    require((_i(r, "strawid") >= 0) & (_i(r, "strawid") <= 65535), "record straw identity")
    require(_i(c, "category") == category[first][rt[ctor]], "constructor trajectory category")
    require(np.isin(category[first][rt[stage == 1]], [4, 5]), "reference update trajectory category")
    require(category[first][rt[stage == 2]] == 3, "wrapper trajectory category")
    require(_b(r, "has_search") == np.isin(stage, [0, 1, 2]), "record search stage")
    hint = _i(r, "hint_kind")
    require(np.where(np.isin(stage, [1, 2]), (hint >= 0) & (hint <= 2), hint == -1), "record hint_kind")
    for name in ("effect_index", "input_index"):
        require(np.where(stage == 3, _i(r, name) >= 0, _i(r, name) == -1), name + ": prefilter identity")
    require(np.where(stage == 3, _i(r, "iteration_ordinal") >= 0, _i(r, "iteration_ordinal") >= -1), "iteration ordinal")
    offset = column(r, "offset")
    require(np.isfinite(offset) & (offset > 0), "invalid stored offset")
    for name in ("radius_operand", "speed_operand"):
        v = column(r, name)
        require(np.isfinite(v[ctor]) & (v[ctor] > 0), "invalid constructor " + name)
        require(np.isnan(v[~ctor]), "nonconstructor operand must be NaN")
    with np.errstate(all="ignore"):
        quotient = column(c, "radius_operand") / column(c, "speed_operand")
    require(np.isfinite(quotient) & (quotient > 0) & (quotient == column(c, "offset")), "constructor radius/speed exact quotient")
    usable = _state(r, np.ones(len(r), dtype=bool))
    with np.errstate(all="ignore"):
        time = dense(r[usable], "particle_poca", 4)[:, 3] + offset[usable]
    require(np.isfinite(time) & (time == column(r, "effect_time")[usable]), "exact TOCA plus offset effect_time")
    require(np.isnan(column(r, "effect_time")[~usable]), "unusable effect_time must be NaN")
    # setHit alone changes partner; updateState can change activity, not CA/reference.
    partner = _i(r, "partner")
    require(partner >= -1, "invalid contemporaneous partner")
    require((_i(c, "category") == 2) == (_i(c, "partner") == -1), "constructor partner/category")
    require(~_b(r, "intrinsic_active") | _b(r, "active"), "intrinsic active implies active")
    require((partner >= 0) | (_b(r, "active") == _b(r, "intrinsic_active")), "unassociated activity")
    order = np.lexsort((_i(r, "sequence"), cp))
    same = cp[order[1:]] == cp[order[:-1]]
    update = np.isin(stage[order[1:]], [1, 2])
    previous, current = order[:-1][update], order[1:][update]
    require(same[update], "reference update without preceding object state")
    require((hint[current] == 1) == usable[previous], "own usable CA hint_kind mismatch")
    require((hint[current] != 2) | (partner[current] >= 0), "associated hit hint without partner")
    own = hint[current] == 1
    for poca, name in (("particle_poca", "search_particle_hint"), ("sensor_poca", "search_sensor_hint")):
        require(dense(r[previous[own]], poca, 4)[:, 3] == column(r, name)[current[own]], "own CA search hint mismatch")
    require(~same | (stage[order[1:]] == 4) | (partner[order[1:]] == partner[order[:-1]]), "partner changed outside setHit")
    _equal_fields(r, c[cp], (*LINE, "line_measurement_time", "line_length"), "fixed sensor line")
    pos = np.arange(len(r))
    last_search = np.maximum.accumulate(np.where(_b(r, "has_search")[order], pos, -1))
    require(last_search >= 0, "record without preceding CA search")
    _equal_fields(r[order], r[order[last_search]], STATE, "retained CA state")
    _search(r, t, trajectory_keys, counts)
    _state(d, _i(d, "stage") == 7)
    _search(d, t, trajectory_keys, counts)
    _discovery(d, c, trajectory_keys, category[first])
    complete = _i(status, "status") == 1
    for name, parent in (("n_records", rp), ("n_objects", rp[ctor]), ("n_trajectories", tp[first]), ("n_discovery", dp)):
        v = _i(status, name)
        require(v[~complete] == 0, "attempt " + name + " must be zero")
        require(v[complete] == np.bincount(parent, minlength=len(targets))[sp[complete]], "completion exact " + name + " mismatch")
    ek = keys(effects, EFFECT_KEYS)
    cross = _i(effects, "identity_kind") == 5
    require(cross == (_i(effects, "kind") == 2), "material effect identity_kind mismatch")
    pre = r[stage == 3]
    pk = keys(pre, EFFECT_KEYS)
    require(len(pre) == int(cross.sum()), "all prefilter crossing effect coverage")
    e = effects[cross][match_rows(pk, ek[cross])]
    for name in ("identity_valid", "has_reference", "has_change", "payload_supported"):
        require(_b(e, name), "invalid crossing effect " + name)
    for name in ("input_index", "strawid", "active"):
        require(column(pre, name) == column(e, name), "prefilter effect " + name + " differs")
    keys(pre, KEYS + ("iteration_ordinal", "object"))
    keys(pre, KEYS + ("iteration_ordinal", "input_index"))
    pp = cp[stage == 3]
    low, high = np.full(len(c), np.iinfo(np.int64).max), np.full(len(c), -1, dtype=np.int64)
    np.minimum.at(low, pp, _i(pre, "input_index")); np.maximum.at(high, pp, _i(pre, "input_index"))
    require((high < 0) | (low == high), "object generic input_index changed")
    represented = high >= 0
    input_keys = np.column_stack((ck[represented, :4], high[represented]))
    require(len(np.unique(input_keys, axis=0)) == int(represented.sum()), "generic input_index reused by different objects")
    for name, width in PIECE.items():
        require(dense(pre, "reference_" + name, width) == dense(e, "reference_" + name, width), "prefilter exact reference " + name)
    pu = _b(pre, "ca_usable")
    require(column(pre, "effect_time")[pu] == column(e, "time")[pu], "prefilter exact effect time")
    final_stamp = _b(e, "identity_from_final_collections")
    require(_i(e, "partner_shindex") >= -1, "invalid effect partner")
    require(_i(pre, "partner")[~final_stamp] == _i(e, "partner_shindex")[~final_stamp], "contemporaneous effect partner differs")
    return {"schema_version": 1, "passed": True, "n_targets": len(targets), "n_records": len(r),
            "n_objects": len(c), "n_trajectories": len(first), "n_trajectory_pieces": len(t),
            "n_discovery": len(d), "n_prefilter": len(pre), "n_expected_prefilter": int(cross.sum()),
            "n_unusable_ca": int((~usable).sum()), "n_searches": int(_b(r, "has_search").sum() + _b(d, "has_search").sum()),
            "n_exhausted_searches": int(_b(r, "search_exhausted").sum() + _b(d, "search_exhausted").sum()),
            "n_final_stamped_partners_not_compared": int(final_stamp.sum()), "n_excluded_prefilter": 0,
            "constructor_quotient_exact": True, "stored_offsets_constant": True,
            "ca_precision": PRECISION, "detailed_physical_discovery_replay": False,
            "full_closure": False, "full_numerical_validation": False, "limits": LIMITS}
