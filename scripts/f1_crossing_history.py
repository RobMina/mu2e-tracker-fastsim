"""Conditional crossing-CA history replay; native trajectories and material seeds supplied."""
import awkward as ak
import numpy as np

import f1_crossing_constructor as B
from f1_crossing_constructor import L, N, C
from f1_native_decision_audit import array, column, dense, match_rows, require

STATE = ("particle_time", "sensor_time", "particle_poca", "sensor_poca", "direction", "sensor_direction", "status", "usable")


def pack(trajectories):
    """Pack all source contexts, preserving zero-duration pieces and full target identities."""
    tk = B.integer_keys(trajectories, B.TRAJECTORY+("piece_index",))
    owner, inverse, counts = np.unique(tk[:, :-1], axis=0, return_inverse=True, return_counts=True)
    require(len(counts) > 0, "empty trajectories")
    index = tk[:, -1]
    require(index < counts[inverse], "noncontiguous trajectory pieces")
    q = np.zeros((len(owner), int(counts.max()), 6))
    ranges = np.zeros(q.shape[:2]+(2,))
    q[inverse, index], ranges[inverse, index] = dense(trajectories, "parameters", 6), dense(trajectories, "range", 2)
    q, ranges, counts = L._pieces(q, ranges, counts)
    first = np.flatnonzero(index == 0)
    mass = np.zeros(len(owner))
    category = np.zeros(len(owner), dtype=np.int64)
    mass[inverse[first]] = column(trajectories, "mass", "f")[first]
    category[inverse[first]] = column(trajectories, "category", "iu")[first]
    require(np.isfinite(mass) & (mass > 0), "invalid trajectory mass")
    require(column(trajectories, "mass", "f") == mass[inverse], "mass changes within trajectory")
    require(column(trajectories, "category", "iu") == category[inverse], "trajectory category changes")
    return dict(keys=owner, q=q, ranges=ranges, counts=counts, mass=mass, category=category)


def schedule(records):
    """Full object/sequence topology; no captured hint/state/search result enters scheduling."""
    seqkeys = B.integer_keys(records, B.KEYS+("sequence",))
    stage = column(records, "stage", "iu")
    require(np.isin(stage, np.arange(5)), "invalid record stage")
    ctors = records[stage == 0]
    ck = B.integer_keys(ctors, B.OBJECT)
    require(len(ck) > 0, "empty constructors")
    parent = match_rows(B.integer_keys(records, B.OBJECT, unique=False), ck)
    order = np.lexsort((seqkeys[:, -1], parent))
    p, st = parent[order], stage[order]
    starts = np.r_[True, p[1:] != p[:-1]]
    require((st == 0) == starts, "object must start exactly once at constructor")
    for name in ("category", "strawid"):
        require(column(records, name, "iu") == column(ctors, name, "iu")[parent], "fixed object identity changed: "+name)
    search = np.isin(st, [1, 2])
    accumulated = np.cumsum(search)
    origin = np.maximum.accumulate(np.where(starts, accumulated, 0))
    rank = accumulated-origin-1
    update_rows = order[search]
    up = parent[update_rows]
    depth = rank[search]
    require(np.all(depth >= 0), "invalid update depth")
    waves = np.full((len(ctors), int(depth.max())+1 if len(depth) else 0), -1, dtype=np.int64)
    waves[up, depth] = update_rows
    position = np.arange(len(order))
    last = np.maximum.accumulate(np.where(np.isin(st, [0, 1, 2]), position, -1))
    require(p[last] == p, "retained search crosses object boundary")
    retained = np.empty(len(records), dtype=np.int64)
    retained[order] = order[last]
    return dict(constructors=ctors, parent=parent, waves=waves, retained=retained, update_rows=update_rows,
                constructor_rows=np.flatnonzero(stage == 0), order=order)


def signed_partner(value):
    x = array(value)
    require(x.ndim == 1 and x.dtype.kind in "iu", "integer partner required")
    if x.dtype.kind == "u":
        require(x <= np.uint64(np.iinfo(np.int64).max), "partner overflow")
    require(x >= -1, "invalid partner")
    return x.astype(np.int64)


def partners(records, plan):
    """Propagate supplied constructor/setHit identities; do not infer association discovery."""
    order, parent = plan["order"], plan["parent"]
    stage = column(records, "stage", "iu")[order]
    take = np.isin(stage, [0, 4])
    source = np.maximum.accumulate(np.where(take, np.arange(len(order)), -1))
    require(parent[order[source]] == parent[order], "partner mutation crosses object")
    supplied = signed_partner(records["partner"])
    result = np.empty(len(order), dtype=np.int64)
    result[order] = supplied[order[source]]
    return result


def bootstrap(constructors, geometry, trajectories, initial, associated):
    """Generated initial/associated states; explicitly captured material constructor state."""
    n = len(constructors)
    keys = B.integer_keys(constructors, B.OBJECT)
    cat = column(constructors, "category", "iu")
    require(np.isin(cat, [0, 1, 2]), "invalid constructor category")
    state = {name: np.zeros((n, 3) if name in ("particle_poca", "sensor_poca", "direction", "sensor_direction") else n,
                            dtype=bool if name == "usable" else np.int64 if name == "status" else float) for name in STATE}
    anchor, offset = np.zeros(n), np.zeros(n)
    index = np.zeros(n, dtype=np.int64)
    for category, report in ((0, initial), (1, associated)):
        selected = np.flatnonzero(cat == category)
        ids = B.integer_matrix(report["identities"]["object_keys"], 5, "constructor receipt keys")
        require(len(selected) == len(ids), "constructor receipt census differs")
        join = match_rows(keys[selected], ids)
        generated = report["generated"]
        source = generated["crossing"] if category == 0 else generated["state"]
        for name in STATE:
            if name == "usable":
                continue
            shape = (len(ids), 3) if name in ("particle_poca", "sensor_poca", "direction", "sensor_direction") else (len(ids),)
            value = L._integer(source[name], shape, name, 0, 6) if name == "status" else L._real(source[name], shape, name)
            state[name][selected] = value[join]
        state["usable"][selected] = state["status"][selected] < 3
        anchor[selected] = L._real(generated["particle_hint"], (len(ids),), "line anchor")[join]
        offset[selected] = L._real(generated["offset"], (len(ids),), "fixed offset")[join]
        index[selected] = L._integer(source["index"], (len(ids),), "constructor index", 0, trajectories["q"].shape[1]-1)[join]
    material = np.flatnonzero(cat == 2)
    c = constructors[material]
    state["status"][material], state["usable"][material] = column(c, "ca_status", "iu"), B.boolean(c, "ca_usable")
    for name, field in (("particle_poca", "particle_poca"), ("sensor_poca", "sensor_poca")):
        value = dense(c, field, 4)
        state[name][material] = value[:, :3]
        state[name.replace("poca", "time")][material] = value[:, 3]
    state["direction"][material], state["sensor_direction"][material] = dense(c, "particle_direction", 3), dense(c, "sensor_direction", 3)
    anchor[material], offset[material] = column(c, "line_measurement_time", "f"), column(c, "offset", "f")
    index[material] = L._integer(column(c, "search_final_index", "iu"), (len(material),), "material constructor index", 0, trajectories["q"].shape[1]-1)
    require((state["status"] >= 0) & (state["status"] <= 6), "invalid constructor status")
    require(state["usable"] == (state["status"] < 3), "constructor status/usability differs")
    require(state["usable"], "unusable constructor unsupported; no exclusions")
    require(np.isfinite(anchor) & np.isfinite(offset) & (offset > 0), "invalid bootstrap anchor/offset")
    line = N.sensor(geometry["cal"], geometry["hv"], anchor, np.full(n, N.LIGHT))
    parent = match_rows(B.integer_keys(constructors, B.TRAJECTORY, unique=False), trajectories["keys"])
    require(index < trajectories["counts"][parent], "constructor index outside trajectory")
    return dict(state=state, line=line, line_start=L._real(geometry["hv"], (n, 3), "tube start"), offset=offset, index=index,
                reference_parameters=trajectories["q"][parent, index], reference_range=trajectories["ranges"][parent, index])


def solve_trace(q, ranges, counts, parent, line, ph, sh, mass, max_moves=10):
    """Trace only generated visited pieces, using unchanged local and piecewise solvers."""
    state = L.piecewise(q, ranges, counts, parent, line, ph, sh, mass, precision=L.NATIVE_CA_PRECISION, max_moves=max_moves)
    history = state["solution_history"]
    n, width = history.shape
    trace = {k: np.zeros((n, width), dtype=bool if k == "usable" else np.int64 if k == "status" else float)
             for k in ("particle_hint", "sensor_hint", "particle_time", "sensor_time", "status", "usable")}
    pt, st = ph.copy(), sh.copy()
    for depth in range(width):
        take = np.flatnonzero(history[:, depth] >= 0)
        if not len(take):
            break
        trace["particle_hint"][take, depth], trace["sensor_hint"][take, depth] = pt[take], st[take]
        local = L._solve(q[parent[take], history[take, depth]], {k: v[take] for k, v in line.items()}, pt[take], st[take], L.NATIVE_CA_PRECISION, mass[take])
        for name in ("particle_time", "sensor_time", "status", "usable"):
            trace[name][take, depth] = local[name]
        pt[take], st[take] = local["particle_time"], local["sensor_time"]
    require(pt == state["particle_time"], "generated trace/final particle time differs")
    require(st == state["sensor_time"], "generated trace/final sensor time differs")
    return state, trace


def replay(records, plan, trajectories, seed):
    """Batch each same-object dependency depth; later captured outcomes never feed updates."""
    n, nobj = len(records), len(plan["constructors"])
    out = {name: np.zeros((n, 3) if seed["state"][name].ndim == 2 else n, dtype=seed["state"][name].dtype) for name in STATE}
    for name, width in (("reference_parameters", 6), ("reference_range", 2)):
        out[name] = np.zeros((n, width))
    for name in ("particle_hint", "sensor_hint"):
        out[name] = np.zeros(n)
    for name in ("index", "solution_index", "moves", "hint_kind"):
        out[name] = np.zeros(n, dtype=np.int64)
    out["exhausted"] = np.zeros(n, dtype=bool)
    out["solution_history"] = np.full((n, 11), -1, dtype=np.int64)
    traces = {name: np.zeros((n, 11), dtype=bool if name == "usable" else np.int64 if name == "status" else float)
              for name in ("particle_hint", "sensor_hint", "particle_time", "sensor_time", "status", "usable")}
    current = {k: v.copy() for k, v in seed["state"].items()}
    cr = plan["constructor_rows"]
    for name in STATE:
        out[name][cr] = current[name]
    for name in ("reference_parameters", "reference_range", "index"):
        out[name][cr] = seed[name]
    out["solution_index"][cr] = seed["index"]
    updates = plan["update_rows"]
    target = match_rows(B.integer_keys(records[updates], B.TRAJECTORY, unique=False), trajectories["keys"])
    parent = np.full(n, -1, dtype=np.int64)
    parent[updates] = target
    stage = column(records, "stage", "iu")
    category = trajectories["category"][target]
    require(np.where(stage[updates] == 2, category == 3, np.isin(category, [4, 5])), "update trajectory category differs")
    for wave in range(plan["waves"].shape[1]):
        rows = plan["waves"][:, wave]
        objects = np.flatnonzero(rows >= 0)
        rows = rows[objects]
        require(current["usable"][objects], "own CA unusable: partner/midpoint fallback not observed/replayed; no exclusions")
        line = {k: v[objects] for k, v in seed["line"].items()}
        own = {k: current[k][objects] for k in ("usable", "particle_time", "sensor_time")}
        ph, sh = L.crossing_hint(own, None, line)
        p = parent[rows]
        state, trace = solve_trace(trajectories["q"], trajectories["ranges"], trajectories["counts"], p, line,
                                  ph, sh, trajectories["mass"][p])
        require(state["usable"], "unusable generated crossing history; no exclusions")
        for name in STATE:
            current[name][objects], out[name][rows] = state[name], state[name]
        for name in ("index", "solution_index", "moves", "exhausted", "solution_history"):
            out[name][rows] = state[name]
        out["particle_hint"][rows], out["sensor_hint"][rows], out["hint_kind"][rows] = ph, sh, 1
        out["reference_parameters"][rows] = trajectories["q"][p, state["index"]]
        out["reference_range"][rows] = trajectories["ranges"][p, state["index"]]
        for name in traces:
            traces[name][rows] = trace[name]
    retain = plan["retained"]
    for name in (*STATE, "reference_parameters", "reference_range"):
        out[name] = out[name][retain]
    out["effect_time"] = out["particle_time"]+seed["offset"][plan["parent"]]
    require(np.isfinite(out["effect_time"]), "nonfinite effect time")
    return out, traces


def compare(records, plan, seed, out, traces):
    """Recorded states and every per-piece search trace enter comparisons only here."""
    ids = B.integer_keys(records, B.KEYS+("sequence",))
    result = {}
    for name, field, width in (("particle_poca", "particle_poca", 4), ("sensor_poca", "sensor_poca", 4),
                              ("direction", "particle_direction", 3), ("sensor_direction", "sensor_direction", 3),
                              ("reference_parameters", "reference_parameters", 6), ("reference_range", "reference_range", 2)):
        expected = dense(records, field, width)
        result[name] = B.comparison(out[name], expected[:, :3] if width == 4 else expected, ids)
    for name, field in (("particle_time", "particle_poca"), ("sensor_time", "sensor_poca")):
        result[name] = B.comparison(out[name], dense(records, field, 4)[:, 3], ids)
    for name, field in (("status", "ca_status"), ("usable", "ca_usable")):
        result[name] = B.comparison(out[name], column(records, field), ids, exact=True)
    result["effect_time"] = B.comparison(out["effect_time"], column(records, "effect_time", "f"), ids)
    result["offset"] = B.comparison(seed["offset"][plan["parent"]], column(records, "offset", "f"), ids)
    result["partner"] = B.comparison(partners(records, plan), signed_partner(records["partner"]), ids, exact=True)
    result["line_start"] = B.comparison(seed["line_start"][plan["parent"]], dense(records, "line_start", 3), ids)
    line = {k: v[plan["parent"]] for k, v in seed["line"].items()}
    for name, field in (("end", "line_end"), ("direction", "line_direction"), ("length", "line_length"), ("measurement_time", "line_measurement_time")):
        expected = dense(records, field, 3) if line[name].ndim == 2 else column(records, field, "f")
        result[field] = B.comparison(line[name], expected, ids)
    rows = plan["update_rows"]
    r, rid = records[rows], ids[rows]
    require(B.boolean(r, "has_search") & ~B.boolean(r, "search_failed"), "missing/failed update search")
    require(column(r, "search_max_moves", "iu") == 10, "unsupported update move cap")
    require(column(r, "precision", "f") == L.NATIVE_CA_PRECISION, "update precision differs")
    require(column(r, "search_precision", "f") == L.NATIVE_CA_PRECISION, "search precision differs")
    for name, field in (("index", "search_final_index"), ("solution_index", "search_solution_index"), ("exhausted", "search_exhausted"),
                        ("hint_kind", "hint_kind"), ("particle_hint", "search_particle_hint"), ("sensor_hint", "search_sensor_hint")):
        result[name] = B.comparison(out[name][rows], column(r, field), rid, exact=out[name].dtype.kind in "biu")
    result["initial_index"] = B.comparison(out["solution_history"][rows, 0], column(r, "search_initial_index", "iu"), rid, exact=True)
    count = array(ak.num(r["search_solution_indices"], axis=1))
    result["solve_count"] = B.comparison(out["moves"][rows]+1, count, rid, exact=True)
    require((count >= 1) & (count <= 11), "invalid captured search length")
    valid = np.arange(11)[None, :] < count[:, None]
    for name, field in (("solution_history", "search_solution_indices"), ("particle_hint", "search_input_particle_hints"),
                        ("sensor_hint", "search_input_sensor_hints"), ("particle_time", "search_particle_times"),
                        ("sensor_time", "search_sensor_times"), ("status", "search_statuses"), ("usable", "search_usable")):
        require(array(ak.num(r[field], axis=1)) == count, "search trace length mismatch")
        raw = array(ak.flatten(r[field], axis=1))
        discrete = name in ("solution_history", "status", "usable")
        if name in ("solution_history", "status"):
            require(raw.dtype.kind in "iu" and np.all(raw >= 0), "invalid trace integer")
            require(raw.astype(np.uint64) <= np.uint64(np.iinfo(np.int64).max), "trace integer overflow")
        expected = array(ak.fill_none(ak.pad_none(ak.values_astype(r[field], np.int64) if name in ("solution_history", "status") else r[field], 11, clip=True), False if name == "usable" else 0))
        actual = out[name][rows] if name == "solution_history" else traces[name][rows]
        result["trace_"+name] = B.comparison(np.where(valid, actual, False if name == "usable" else 0), expected, rid, exact=discrete)
    return dict(passed=all(x["passed"] for x in result.values()), exact=all(x["exact"] for x in result.values()), metrics=result,
                n_records=len(records), n_updates=len(rows), n_excluded=0, n_moved=int(np.sum(out["moves"][rows] > 0)),
                n_exhausted=int(out["exhausted"][rows].sum()), n_waves=plan["waves"].shape[1])
