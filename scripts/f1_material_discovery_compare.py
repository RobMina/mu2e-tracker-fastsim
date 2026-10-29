"""Comparison-only adapter for independently generated material discovery stages."""
import awkward as ak
import numpy as np

import f1_material_discovery as M
from f1_material_discovery import B, H, L, N, array, column, dense, match_rows, require


def stage_keys(stage, trajectories):
    return np.column_stack((trajectories["keys"][stage["parent"]], stage["plane"], stage["panel"], stage["straw"]))


def compare(out, trajectories, discovery, constructors):
    stage = column(discovery, "stage", "iu")
    require(np.isin(stage, np.arange(9)), "invalid discovery stage")
    metrics, census, orderkeys, recorded_sequence = {}, {}, [], []
    for number in range(9):
        generated = out["stages"][number]
        captured = discovery[stage == number]
        ids = stage_keys(generated, trajectories)
        require(len(captured) == len(ids), "stage census differs: "+str(number))
        target = B.integer_keys(captured, B.TRAJECTORY, unique=False)
        full = np.column_stack((target, column(captured, "plane"), column(captured, "panel"), column(captured, "straw"))).astype(np.int64)
        require(len(np.unique(full, axis=0)) == len(full), "duplicate discovery identity")
        join = match_rows(ids, full)
        captured = captured[join]
        label = "stage"+str(number)+"/"
        metrics[label+"passed"] = B.comparison(generated["passed"], B.boolean(captured, "passed"), ids, exact=True)
        metrics[label+"values"] = B.comparison(generated["values"], dense(captured, "values", generated["values"].shape[1]), ids)
        integer = captured["integers"]
        if number == 0:
            require(array(ak.num(integer, axis=1)) == 2+array(ak.num(generated["old"], axis=1)), "oldstraw set size differs")
            require(np.array_equal(array(ak.flatten(integer[:, 2:])), array(ak.flatten(generated["old"]))), "oldstraw identities differ")
            integer = integer[:, :2]
        metrics[label+"integers"] = B.comparison(generated["integers"], array(integer), ids, exact=True)
        census[str(number)] = dict(n_rows=len(ids), n_passed=int(generated["passed"].sum()), n_rejected=int((~generated["passed"]).sum()))
        orderkeys.append(np.column_stack((generated["parent"], generated["plane"], generated["panel"], generated["straw"], np.full(len(ids), number))))
        recorded_sequence.append(column(captured, "sequence", "iu"))
    ordering = np.concatenate(orderkeys)
    seq = np.concatenate(recorded_sequence)
    order = np.lexsort(tuple(ordering[:, k] for k in range(4, -1, -1)))
    owner = ordering[order, 0]
    index = np.arange(len(order))
    start = np.r_[True, owner[1:] != owner[:-1]]
    expected = index-np.maximum.accumulate(np.where(start, index, 0))
    require(seq[order] == expected, "generated hierarchy order differs from discovery chronology")
    attempt = out["stages"][7]
    ids = stage_keys(attempt, trajectories)
    captured = discovery[stage == 7]
    full = np.column_stack((B.integer_keys(captured, B.TRAJECTORY, unique=False), column(captured, "plane"), column(captured, "panel"), column(captured, "straw"))).astype(np.int64)
    captured = captured[match_rows(ids, full)]
    generated, state = out["generated"], out["generated"]["state"]
    for name, width, value in (("particle_poca", 4, np.column_stack((state["particle_poca"], state["particle_time"]))),
        ("sensor_poca", 4, np.column_stack((state["sensor_poca"], state["sensor_time"]))),
        ("particle_direction", 3, state["direction"]), ("sensor_direction", 3, state["sensor_direction"]),
        ("line_start", 3, generated["line_start"]), ("line_end", 3, generated["line"]["end"]),
        ("line_direction", 3, generated["line"]["direction"]), ("reference_parameters", 6, generated["reference_parameters"]),
        ("reference_range", 2, generated["reference_range"]), ("d_dd_p", 6, out["derivatives"]["H"][:, 0]),
        ("d_td_p", 6, -out["derivatives"]["H"][:, 1])):
        metrics["CA/"+name] = B.comparison(value, dense(captured, name, width), ids)
    scalar = dict(doca=out["derivatives"]["doca"], doca_variance=out["docavar"],
                  line_length=generated["line"]["length"], line_measurement_time=generated["particle_hint"],
                  search_particle_hint=generated["particle_hint"], search_sensor_hint=generated["sensor_hint"])
    for name, value in scalar.items():
        metrics["CA/"+name] = B.comparison(value, column(captured, name), ids)
    for name, value in dict(ca_status=state["status"], ca_usable=state["usable"], search_initial_index=state["solution_history"][:, 0],
        search_final_index=state["index"], search_solution_index=state["solution_index"], search_exhausted=state["exhausted"]).items():
        metrics["CA/"+name] = B.comparison(value, column(captured, name), ids, exact=True)
    require(B.boolean(captured, "has_search"), "missing discovery search")
    require(column(captured, "search_max_moves", "iu") == 10, "search cap differs")
    require(column(captured, "search_precision", "f") == L.NATIVE_CA_PRECISION, "search precision differs")
    raw = captured["search_solution_indices"]
    flat = array(ak.flatten(raw))
    require(flat.dtype.kind in "iu" and np.all(flat >= 0), "invalid trace index")
    require(flat.astype(np.uint64) <= np.uint64(np.iinfo(np.int64).max), "trace index overflow")
    history = array(ak.fill_none(ak.pad_none(ak.values_astype(raw, np.int64), 11, clip=True), -1))
    metrics["CA/history"] = B.comparison(state["solution_history"], history, ids, exact=True)
    metrics["CA/trace_length"] = B.comparison(state["moves"]+1, array(ak.num(raw)), ids, exact=True)
    _, trace = H.solve_trace(trajectories["q"], trajectories["ranges"], trajectories["counts"], attempt["parent"],
                            generated["line"], generated["particle_hint"], generated["sensor_hint"], trajectories["mass"][attempt["parent"]])
    for name, field in dict(particle_hint="search_input_particle_hints", sensor_hint="search_input_sensor_hints",
                            particle_time="search_particle_times", sensor_time="search_sensor_times", status="search_statuses", usable="search_usable").items():
        captured_trace = array(ak.fill_none(ak.pad_none(captured[field], 11, clip=True), False if name == "usable" else 0))
        metrics["trace/"+name] = B.comparison(trace[name], captured_trace, ids, exact=name in ("status", "usable"))
    accepted = out["accepted"]
    accepted_keys = np.column_stack((trajectories["keys"][attempt["parent"][accepted], :4], attempt["straw"][accepted]))
    require(len(constructors) == len(accepted_keys), "accepted constructor census differs")
    join = match_rows(accepted_keys, B.integer_keys(constructors, B.KEYS+("strawid",)))
    constructors = constructors[join]
    object_keys = B.integer_keys(constructors, B.OBJECT)
    linked = discovery[stage == 8]
    links = B.integer_keys(linked, B.KEYS+("straw",))
    linked = linked[match_rows(accepted_keys, links)]
    require(column(linked, "object") == column(constructors, "object"), "accepted object link differs")
    require(column(constructors, "partner") == -1, "material constructor has partner")
    subset = {k: ({name: v[accepted] for name, v in value.items()} if isinstance(value, dict) else value[accepted]) for k, value in generated.items()}
    ctor_comparison = B.compare(subset, constructors, object_keys)
    return dict(passed=all(m["passed"] for m in metrics.values()) and ctor_comparison["passed"],
                exact=all(m["exact"] for m in metrics.values()) and ctor_comparison["exact"],
                metrics=metrics, stages=census, constructors=ctor_comparison, object_keys=object_keys, generated=subset,
                n_records=len(discovery), n_attempts=len(ids), n_piece_solves=int((state["moves"]+1).sum()))
