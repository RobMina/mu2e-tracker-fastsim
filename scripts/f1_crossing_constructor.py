"""Vectorized associated-extension crossing construction on supplied native trajectories.

Geometry endpoints, native mass/ranges/parameters and independently reconstructed
upstream hit-constructor hints are inputs. Captured crossing outcomes are not inputs
to generate(); comparison is separate. No fitter integration, initial/material
constructor replay, history propagation, candidate discovery or full-fit claim.
"""
import awkward as ak
import numpy as np

import f1_later_ca as L
from f1_native_decision_audit import array, column, dense, keys, match_rows, require
from trkfastsim.reco import nativearith as N, nativeca as C

KEYS = ("run", "subrun", "event", "iseed")
OBJECT = KEYS + ("object",)
TRAJECTORY = KEYS + ("trajectory_index",)
RTOL, ATOL = 1e-5, 1e-7


def integer_keys(table, names, unique=True):
    for name in names:
        x = column(table, name, "iu")
        require(x >= 0, "negative identity: " + name)
        require(x.astype(np.uint64) <= np.uint64(np.iinfo(np.int64).max), "identity overflow")
    return keys(table, names, unique=unique)


def integer_matrix(value, width, label):
    try:
        value = array(value)
    except (TypeError, RuntimeError, OverflowError) as error:
        raise ValueError(label + ": invalid integer representation") from error
    require(value.ndim == 2 and value.shape[1] == width and value.dtype.kind in "iu", label + ": integer matrix required")
    require(value >= 0, label + ": negative key")
    require(value.astype(np.uint64) <= np.uint64(np.iinfo(np.int64).max), label + ": overflow")
    value = value.astype(np.int64)
    require(len(np.unique(value, axis=0)) == len(value), label + ": duplicate keys")
    return value


def associated_inputs(records, hits, hints):
    """Full-key object/source mapping, with captured fields used only to identify inputs."""
    ctors = records[column(records, "stage", "iu") == 0]
    integer_keys(ctors, OBJECT)
    selected = ctors[column(ctors, "category", "iu") == 1]
    require(len(selected) > 0, "empty associated constructor population")
    ck = integer_keys(selected, OBJECT)
    first = records[(column(records, "stage") == 3) & (column(records, "category") == 1)]
    fk = integer_keys(first, OBJECT, unique=False)
    parent = match_rows(fk, ck)
    seq = column(first, "sequence", "iu")
    order = np.lexsort((seq, parent))
    take = order[np.r_[True, parent[order][1:] != parent[order][:-1]]]
    require(len(take) == len(selected), "missing first prefilter record")
    first = first[take][match_rows(ck, fk[take])]
    source = integer_keys(first, KEYS + ("input_index",))
    hk = integer_matrix(hints["source_keys"], 5, "upstream source")
    require(len(hk) == len(selected), "upstream constructor census differs")
    join = match_rows(source, hk)
    hit_keys = integer_matrix(hints["partner_hit_keys"], 6, "upstream hit")
    require(len(hit_keys) == len(hk), "upstream partner census differs")
    require(np.array_equal(hit_keys[:, :4], hk[:, :4]), "upstream partner target differs")
    hi = match_rows(hit_keys[join], integer_keys(hits, KEYS + ("iteration_ordinal", "effect_index")))
    partner = column(hits[hi], "shindex", "iu")
    require(column(selected, "partner") == partner, "constructor partner differs from upstream hit")
    require(column(selected, "strawid") == column(hits[hi], "strawid"), "constructor/partner straw differs")
    require(column(selected, "partner") == column(first, "partner"), "partner changes before first prefilter")
    values = {}
    for name, field in (("particle_hint", "reconstructed_hit_constructor_particle_hint"),
                        ("sensor_hint", "reconstructed_hit_constructor_sensor_hint"),
                        ("measurement_time", "straw_line_measurement_time")):
        x = L._real(hints[field], (len(hk),), field)
        values[name] = x[join]
    require(values["measurement_time"] == values["particle_hint"], "source line anchor differs from hit-constructor particle time")
    return selected, source, values


def pack_trajectories(trajectories, constructors):
    """Preserve explicit piece order/ranges, including zero-duration pieces."""
    t = trajectories[column(trajectories, "category", "iu") == 1]
    tk = integer_keys(t, TRAJECTORY + ("piece_index",))
    owner, parent = np.unique(tk[:, :-1], axis=0, return_inverse=True)
    wanted = integer_keys(constructors, TRAJECTORY, unique=False)
    row = match_rows(wanted, owner)
    counts = np.bincount(parent, minlength=len(owner))
    require(len(counts) > 0, "no construction trajectories")
    index = tk[:, -1]
    require(index < counts[parent], "noncontiguous piece index")
    width = int(counts.max())
    q = np.zeros((len(owner), width, 6))
    ranges = np.zeros((len(owner), width, 2))
    q[parent, index] = dense(t, "parameters", 6)
    ranges[parent, index] = dense(t, "range", 2)
    mass = L._real(column(t, "mass", "f"), (len(t),), "trajectory mass")
    require(mass > 0, "nonpositive native mass")
    first = np.flatnonzero(index == 0)
    require(len(first) == len(owner), "missing piece zero")
    owner_mass = np.zeros(len(owner))
    owner_mass[parent[first]] = mass[first]
    require(mass == owner_mass[parent], "mass varies along trajectory")
    L._pieces(q, ranges, counts)
    return q, ranges, counts, row, owner_mass[row]


def geometry_inputs(constructors, bindings, tracker, straws):
    done = bindings[column(bindings, "source_status") == 1]
    b = match_rows(integer_keys(constructors, KEYS, unique=False), integer_keys(done, KEYS))
    geometry = column(done, "geometry_index", "iu")[b]
    sk = integer_keys(straws, ("geometry_index", "fullsid"))
    j = match_rows(np.column_stack((geometry, column(constructors, "strawid", "iu"))), sk)
    ti = match_rows(geometry[:, None], integer_keys(tracker, ("geometry_index",)))
    radius = L._real(column(tracker, "wire_radius", "f")[ti], (len(j),), "wire radius")
    require(radius > 0, "nonpositive wire radius")
    return dense(straws[j], "tube_cal", 3), dense(straws[j], "tube_hv", 3), radius


def generate(parameters, ranges, counts, parent, mass, cal, hv, particle_hint, sensor_hint, radius):
    """Source-grouped SensorLine + native piece search, no captured CA/line/offset input."""
    n = len(parent)
    cal, hv = L._real(cal, (n, 3), "tube cal"), L._real(hv, (n, 3), "tube hv")
    ph, sh = L._real(particle_hint, (n,), "particle hint"), L._real(sensor_hint, (n,), "sensor hint")
    radius = L._real(radius, (n,), "wire radius")
    require(radius > 0, "nonpositive wire radius")
    line = N.sensor(cal, hv, ph, np.full(n, N.LIGHT))
    state = L.piecewise(parameters, ranges, counts, parent, line, ph, sh, mass,
                        precision=L.NATIVE_CA_PRECISION, max_moves=10)
    require(state["usable"], "unusable generated constructor; unsupported branch, no exclusions")
    pointer_q = parameters[parent, state["index"]]
    reference_speed = C.particle(pointer_q, state["particle_time"], mass)[2]
    offset = radius/reference_speed
    require(np.isfinite(offset) & (offset > 0), "invalid generated offset")
    return dict(line=line, line_start=hv.copy(), state=state, radius=radius, reference_speed=reference_speed,
                offset=offset, effect_time=state["particle_time"]+offset,
                reference_parameters=pointer_q, reference_range=ranges[parent, state["index"]],
                particle_hint=ph, sensor_hint=sh)


def boolean(table, name):
    return column(table, name, "b")


def comparison(actual, expected, ids, exact=False):
    a, e = array(actual), array(expected)
    require(a.shape == e.shape and a.shape[:1] == (len(ids),), "comparison shape")
    require(np.isfinite(a) & np.isfinite(e), "nonfinite comparison")
    if exact:
        require((a.dtype.kind == e.dtype.kind == "b") or (a.dtype.kind in "iu" and e.dtype.kind in "iu"),
                "exact discrete comparison requires boolean/boolean or integer/integer")
    equal = a == e
    try:
        with np.errstate(over="raise", invalid="raise"):
            close = equal if exact else np.isclose(a, e, rtol=RTOL, atol=ATOL)
            delta = np.abs(a.astype(np.float64)-e.astype(np.float64))
    except FloatingPointError as error:
        raise ValueError("comparison residual overflow; no finite summary") from error
    rowgood = close.reshape(len(ids), -1).all(axis=1)
    rowexact = equal.reshape(len(ids), -1).all(axis=1)
    return dict(passed=bool(rowgood.all()), exact=bool(rowexact.all()), n_rows=len(ids),
                n_components=int(a.size), n_failed_components=int((~close).sum()),
                n_failed_rows=int((~rowgood).sum()), n_inexact_rows=int((~rowexact).sum()),
                max_absolute_residual=float(delta.max()) if a.size else None,
                failed_keys=ids[~rowgood].tolist(), inexact_keys=ids[~rowexact].tolist())


def compare(generated, constructors, ids):
    """Only here may recorded constructor outcomes enter numerical comparisons."""
    require(len(constructors) == len(ids) > 0, "empty/mismatched constructor comparison")
    line, state = generated["line"], generated["state"]
    require(boolean(constructors, "ca_usable"), "unusable captured constructor; no exclusions")
    result = {}
    expected = {
        "line_start": (generated["line_start"], dense(constructors, "line_start", 3)),
        "line_end": (line["end"], dense(constructors, "line_end", 3)),
        "line_direction": (line["direction"], dense(constructors, "line_direction", 3)),
        "line_length": (line["length"], column(constructors, "line_length")),
        "line_measurement_time": (line["measurement_time"], column(constructors, "line_measurement_time")),
        "particle_hint": (generated["particle_hint"], column(constructors, "search_particle_hint")),
        "sensor_hint": (generated["sensor_hint"], column(constructors, "search_sensor_hint")),
        "particle_time": (state["particle_time"], dense(constructors, "particle_poca", 4)[:, 3]),
        "sensor_time": (state["sensor_time"], dense(constructors, "sensor_poca", 4)[:, 3]),
        "particle_poca": (state["particle_poca"], dense(constructors, "particle_poca", 4)[:, :3]),
        "sensor_poca": (state["sensor_poca"], dense(constructors, "sensor_poca", 4)[:, :3]),
        "particle_direction": (state["direction"], dense(constructors, "particle_direction", 3)),
        "sensor_direction": (state["sensor_direction"], dense(constructors, "sensor_direction", 3)),
        "reference_parameters": (generated["reference_parameters"], dense(constructors, "reference_parameters", 6)),
        "reference_range": (generated["reference_range"], dense(constructors, "reference_range", 2)),
        "radius": (generated["radius"], column(constructors, "radius_operand")),
        "reference_speed": (generated["reference_speed"], column(constructors, "speed_operand")),
        "offset": (generated["offset"], column(constructors, "offset")),
        "effect_time": (generated["effect_time"], column(constructors, "effect_time")),
    }
    for name, (a, e) in expected.items():
        result[name] = comparison(a, e, ids)
    for name, a, e in (("status", state["status"], column(constructors, "ca_status")),
                       ("usable", state["usable"], column(constructors, "ca_usable")),
                       ("index", state["index"], column(constructors, "search_final_index")),
                       ("solution_index", state["solution_index"], column(constructors, "search_solution_index")),
                       ("initial_index", state["solution_history"][:, 0], column(constructors, "search_initial_index")),
                       ("exhausted", state["exhausted"], column(constructors, "search_exhausted"))):
        result[name] = comparison(a, e, ids, exact=True)
    require(boolean(constructors, "has_search"), "missing constructor search")
    require(column(constructors, "search_max_moves", "iu") == 10, "constructor move cap differs")
    require(column(constructors, "precision", "f") == L.NATIVE_CA_PRECISION, "constructor precision differs")
    require(column(constructors, "search_precision", "f") == L.NATIVE_CA_PRECISION, "constructor search precision differs")
    length = array(ak.num(constructors["search_solution_indices"], axis=1))
    require((length > 0) & (length <= 11), "invalid constructor search length")
    raw_history = constructors["search_solution_indices"]
    flat_history = array(ak.flatten(raw_history, axis=1))
    require(flat_history.dtype.kind in "iu" and np.all(flat_history >= 0), "invalid solved-piece identity")
    require(flat_history.astype(np.uint64) <= np.uint64(np.iinfo(np.int64).max), "solved-piece identity overflow")
    signed_history = ak.values_astype(raw_history, np.int64)
    history = array(ak.fill_none(ak.pad_none(signed_history, 11, clip=True), -1))
    result["solution_history"] = comparison(state["solution_history"], history, ids, exact=True)
    result["search_length"] = comparison(state["moves"]+1, length, ids, exact=True)
    return dict(passed=all(item["passed"] for item in result.values()),
                exact=all(item["exact"] for item in result.values()), metrics=result,
                n_constructors=len(ids), n_excluded=0,
                n_moved=int(np.sum(state["moves"] > 0)), n_exhausted=int(state["exhausted"].sum()))
