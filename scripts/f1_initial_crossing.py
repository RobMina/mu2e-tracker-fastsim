"""Initial hit-CA and tube-crossing reconstruction on supplied construction trajectories."""
from pathlib import Path
import re

import awkward as ak
import numpy as np

import f1_crossing_constructor as B
from f1_crossing_constructor import L, N, C
from f1_native_decision_audit import array, column, dense, match_rows, require

EVENT_FIELDS = ("sid", "time", "et0", "et1", "edep", "eend")


def speed_configuration(path):
    """Read the literal expanded speed calibration without importing fitter/JAX modules."""
    text = Path(path).read_text()
    require(not re.search(r"(?m)^\s*#\s*include\b|\bstrawResponse\s*\.", text), "unexpanded speed configuration")
    blocks = re.findall(r"\bstrawResponse\s*:\s*\{([^{}]*)\}", text)
    require(len(blocks) == 1, "ambiguous expanded strawResponse table")
    block = blocks[0]
    def scalar(name):
        values = re.findall(r"(?m)^\s*"+name+r"\s*:\s*([^\s]+)\s*$", block)
        require(len(values) == 1, "ambiguous speed scalar "+name)
        return values[0]
    def sequence(name):
        values = re.findall(r"\b"+name+r"\s*:\s*\[([^\]]*)\]", block)
        require(len(values) == 1, "ambiguous speed sequence "+name)
        return [] if not values[0].strip() else [float(x.strip()) for x in values[0].split(",")]
    require(scalar("useDb") == "false", "database speed unsupported")
    require(not sequence("strawHalfPropVelocity"), "per-straw speed unsupported")
    bins = int(scalar("eBins"))
    config = dict(bin_width=float(scalar("eBinWidth")), mean_half_velocity=float(scalar("defaultHalfPropVelocity")),
                  scale=sequence("eHalfPVScale"))
    require(bins == len(config["scale"]), "speed grid size differs")
    signal_speed(np.zeros(1), config)
    return config


def signal_speed(energy, config):
    """Source PieceLine slope-first arithmetic at measured energy, no response inversion."""
    energy = L._real(energy, (len(energy),), "energy")
    require(energy >= 0, "negative measured energy")
    scale = L._real(config["scale"], (len(config["scale"]),), "speed scale")
    require(len(scale) >= 2 and np.all(scale > 0), "invalid speed scale")
    width = L._real(config["bin_width"], (), "speed bin width")
    mean = L._real(config["mean_half_velocity"], (), "half velocity")
    require(width > 0 and mean > 0, "nonpositive speed constants")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        grid = np.arange(len(scale), dtype=float)*width
        width = (grid[-1]-grid[0])/(len(grid)-1)
        x = 1000.*energy
        ib = np.clip(np.floor((x-grid[0])/width), 0, len(grid)-2).astype(np.int64)
        slope = (scale[ib+1]-scale[ib])/width
        interpolated = scale[ib]+(x-grid[ib])*slope
        speed = 2.*(interpolated*mean)
    require(np.isfinite(speed) & (speed > 0), "invalid source signal speed")
    return speed


def initial_inputs(records, seeds):
    """Join contemporaneous constructor partners to immutable initial hit membership."""
    constructors = records[(column(records, "stage", "iu") == 0) & (column(records, "category", "iu") == 0)]
    require(len(constructors) > 0, "empty initial constructors")
    B.integer_keys(constructors, B.OBJECT)
    wanted = B.integer_keys(constructors, B.KEYS+("partner",))
    target = np.unique(wanted[:, :4], axis=0)
    chosen = seeds[match_rows(target, B.integer_keys(seeds, B.KEYS))]
    require(B.boolean(chosen, "valid"), "invalid initial seed")
    counts = array(ak.num(chosen["sh_shindex"], axis=1))
    require(counts > 0, "empty initial seed hits")
    fields = {}
    for name in ("sh_shindex", "sh_strawid", "sh_time"):
        require(array(ak.num(chosen[name], axis=1)) == counts, "initial membership lengths differ")
        fields[name] = array(ak.flatten(chosen[name], axis=1))
    source_index = fields["sh_shindex"]
    require(source_index.dtype.kind in "iu" and np.all(source_index >= 0), "invalid seed hit index dtype/range")
    require(source_index.astype(np.uint64) <= np.uint64(np.iinfo(np.int64).max), "seed hit index overflow")
    available = B.integer_matrix(np.column_stack((np.repeat(target, counts, axis=0), source_index.astype(np.int64))), 5, "seed hit keys")
    require(len(available) == len(wanted), "initial constructor membership census differs")
    join = match_rows(wanted, available)
    sid = fields["sh_strawid"]
    require(sid.dtype.kind in "iu" and np.all(sid >= 0), "invalid initial straw identity")
    require(sid.astype(np.uint64) <= np.uint64(65535), "initial straw overflow")
    require(column(constructors, "strawid", "iu") == sid[join], "initial partner straw differs")
    owner = match_rows(wanted[:, :4], target)
    ctime = L._real(fields["sh_time"], (len(available),), "captured initial hit time")[join]
    return constructors, wanted, chosen[owner], ctime


def construction_inputs(trajectories, constructors, seeds):
    """The frozen initial population has exactly one piece per construction context."""
    t = trajectories[column(trajectories, "category", "iu") == 0]
    tk = B.integer_keys(t, B.TRAJECTORY+("piece_index",))
    require(len(t) > 0 and np.all(tk[:, -1] == 0), "initial construction requires one piece")
    owner = B.integer_keys(t, B.TRAJECTORY)
    require(len(np.unique(owner[:, :4], axis=0)) == len(owner), "multiple initial contexts per target")
    parent = match_rows(B.integer_keys(constructors, B.TRAJECTORY, unique=False), owner)
    require(len(np.unique(parent)) == len(t), "unreferenced initial construction context")
    q, ranges = dense(t, "parameters", 6)[:, None, :], dense(t, "range", 2)[:, None, :]
    counts = np.ones(len(t), dtype=np.int64)
    q, ranges, counts = L._pieces(q, ranges, counts)
    mass = L._real(column(t, "mass", "f")[parent], (len(constructors),), "construction mass")
    require(mass > 0, "nonpositive construction mass")
    require(q[parent, 0] == dense(seeds, "parameters", 6), "construction/seed parameters differ")
    require(mass == column(seeds, "mass", "f"), "construction/seed mass differs")
    seed_range = np.column_stack((column(seeds, "range_begin", "f"), column(seeds, "range_end", "f")))
    require(np.isfinite(seed_range) & (seed_range[:, :1] <= seed_range[:, 1:]), "invalid post-construction seed range")
    return q, ranges, counts, parent, mass


def geometry_inputs(constructors, bindings, tracker, straws):
    """Raw target-specific wire and tube endpoints remain separate measured geometry inputs."""
    done = bindings[column(bindings, "source_status", "iu") == 1]
    bi = match_rows(B.integer_keys(constructors, B.KEYS, unique=False), B.integer_keys(done, B.KEYS))
    geom = column(done, "geometry_index", "iu")[bi]
    selected = straws[match_rows(np.column_stack((geom, column(constructors, "strawid", "iu"))),
                                B.integer_keys(straws, ("geometry_index", "fullsid")))]
    cal, hv, radius = B.geometry_inputs(constructors, bindings, tracker, straws)
    return dict(cal=cal, hv=hv, radius=radius, wire_cal=dense(selected, "wire_cal", 3),
                wire_hv=dense(selected, "wire_hv", 3))


def measurements(events, hit_keys, strawid):
    """Only independently measured ComboHit fields enter the initial time reconstruction."""
    hit_keys = B.integer_matrix(hit_keys, 5, "initial hit keys")
    event = match_rows(hit_keys[:, :3], B.integer_keys(events, B.KEYS[:3]))
    counts = array(ak.num(events["s_sid"], axis=1))
    require(hit_keys[:, 4] < counts[event], "source hit index outside event")
    flat = np.r_[0, np.cumsum(counts)][event]+hit_keys[:, 4]
    measured = {}
    for name in EVENT_FIELDS:
        require(array(ak.num(events["s_"+name], axis=1)) == counts, "measured lengths differ: "+name)
        value = array(ak.flatten(events["s_"+name], axis=1))[flat]
        if name in ("sid", "eend"):
            require(value.dtype.kind in "iu", "measured integer required: "+name)
        else:
            require(value.dtype == np.dtype("float32"), "measured float32 required: "+name)
        require(np.isfinite(value), "nonfinite measured field: "+name)
        measured[name] = value
    require(measured["sid"] == strawid, "measured straw differs")
    require(np.isin(measured["eend"], [0, 1]), "invalid measured early end")
    require(measured["eend"] == (measured["et1"] < measured["et0"]), "measured early end disagrees with times")
    require(measured["edep"] >= 0, "negative measured energy")
    return {k: v.astype(float) if k not in ("sid", "eend") else v for k, v in measured.items()}


def generate(q, ranges, counts, parent, mass, geometry, measured, configuration):
    """Generate both searches without recorded crossing line, hints, state, speed or offset."""
    n = len(parent)
    q, ranges, counts = L._pieces(q, ranges, counts)
    parent = L._integer(parent, (n,), "parent", 0, len(q)-1)
    require(counts == 1, "initial hit hint requires one construction piece")
    mass = L._real(mass, (n,), "mass")
    require(mass > 0, "nonpositive mass")
    values = {k: L._real(measured[k], (n,), k) for k in ("time", "et0", "et1", "edep")}
    wire_cal = L._real(geometry["wire_cal"], (n, 3), "wire cal")
    wire_hv = L._real(geometry["wire_hv"], (n, 3), "wire hv")
    early_hv = values["et1"] < values["et0"]
    early, late = np.where(early_hv[:, None], wire_hv, wire_cal), np.where(early_hv[:, None], wire_cal, wire_hv)
    speed = signal_speed(values["edep"], configuration)
    line = N.sensor(early, late, np.where(early_hv, values["et1"], values["et0"]), speed)
    ph, sh = C.constructor_hint(q[parent, 0], line, values["time"], mass)
    hit = L.piecewise(q, ranges, counts, parent, line, ph, sh, mass,
                      precision=L.NATIVE_CA_PRECISION, max_moves=10)
    require(hit["usable"], "unusable initial hit constructor; no exclusions")
    crossing = B.generate(q, ranges, counts, parent, mass, geometry["cal"], geometry["hv"],
                          hit["particle_time"], hit["sensor_time"], geometry["radius"])
    return dict(hit=hit, wire_line=line, hit_particle_hint=ph, hit_sensor_hint=sh, crossing=crossing)
