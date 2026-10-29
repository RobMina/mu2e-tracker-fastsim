"""Literal-FCL drift diagnostics for exactly zero B only; no fitter execution.

All phi time vectors coincide when tempGamma=1, so only one is stored. The
caller must establish exactly zero B independently; this API neither accepts
nor models arbitrary fields or Lorentz drift. Calibration uses vectorized
NumPy online regression, not a certified native-GSL implementation.
"""
from pathlib import Path
import re

import numpy as np

from f1_initial_measurements import _flat_response, NUMBER
from f1_native_decision_audit import require
from trkfastsim.reco.nativearith import _checked

MAX_GRID_POINTS = 1_000_000
CALIBRATION_METHOD = "NumPy online regression (not certified native GSL)"
DRIFT_FIELDS = ("useDb", "wireVoltage", "kVcm", "cmus", "deltaDistance", "deltaTime", "phiBins")
RESPONSE_FIELDS = ("driftOffBins", "driftOffset", "driftRMSBins", "signedDriftRMS",
                   "unsignedDriftRMS", "dRdTScale", "driftIgnorePhi")


def _flat_drift(path):
    """Narrow flat strawDrift counterpart of the frozen _flat_response parser."""
    text = Path(path).read_text()
    require(not re.search(r"(?m)^\s*#\s*include\b|\b(?:BEGIN_PROLOG|END_PROLOG)\b", text),
            "unexpanded FCL directives unsupported")
    masked = re.sub(r'"(?:\\.|[^"\\])*"|#[^\n]*', lambda m: " " * len(m[0]), text)
    require(not re.search(r"\bstrawDrift\s*\.", masked), "strawDrift overrides unsupported")
    starts = list(re.finditer(r"\bstrawDrift\s*:", masked))
    require(len(starts) == 1, "ambiguous expanded strawDrift table")
    begin = starts[0].end()
    opening = re.match(r"\s*\{", masked[begin:])
    require(opening is not None, "literal expanded strawDrift table required")
    begin += opening.end()
    end = masked.find("}", begin)
    require(end >= begin and "{" not in masked[begin:end], "nested/unterminated strawDrift table unsupported")
    block, result, position = text[begin:end], {}, 0
    field = re.compile(r'\s*([A-Za-z_][A-Za-z_0-9]*)\s*:\s*(\[[^\[\]{}]*\]|"(?:\\.|[^"\\])*"|[^\s\[\]{}]+)')
    while position < len(block) and block[position:].strip():
        match = field.match(block, position)
        require(match is not None, "unsupported flat strawDrift syntax")
        name, token = match.group(1, 2)
        require(name not in result, "duplicate strawDrift field " + name)
        if token.startswith("["):
            body = token[1:-1].strip()
            require(not body or re.fullmatch(NUMBER + r"(?:\s*,\s*" + NUMBER + r")*", body),
                    "invalid strawDrift sequence " + name)
            value = [] if not body else [float(v) for v in body.split(",")]
        elif token in ("true", "false"):
            value = token == "true"
        elif re.fullmatch(NUMBER, token):
            value = int(token) if re.fullmatch(r"[+-]?\d+", token) else float(token)
        elif token.startswith('"'):
            value = token
        else:
            raise ValueError("unsupported strawDrift literal " + name)
        result[name] = value
        position = match.end()
    return result


def _array(value, name):
    require(not np.any(np.ma.getmask(value)), "masked values: " + name)
    try:
        value = np.asarray(value)
        require(value.dtype.kind in "fiu", "invalid real values: " + name)
        value = value.astype(np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError("invalid real values: " + name) from exc
    require(np.isfinite(value), "nonfinite " + name)
    return value


def _positive(value, name):
    value = _array(value, name)
    require(value.ndim == 0 and value > 0, "invalid positive scalar: " + name)
    return float(value)


@_checked
def _validated(config):
    require(isinstance(config, dict), "drift configuration must be a dictionary")
    require(all(k in config for k in DRIFT_FIELDS + RESPONSE_FIELDS), "missing drift configuration field")
    require(type(config["useDb"]) is bool and not config["useDb"], "database drift unsupported")
    require(type(config["driftIgnorePhi"]) is bool, "invalid driftIgnorePhi boolean")
    phi = config["phiBins"]
    require(not np.any(np.ma.getmask(phi)), "masked values: phiBins")
    if np.ma.isMaskedArray(phi) and phi.ndim == 0:
        phi = phi.item()
    require(type(phi) is int and 2 <= phi <= MAX_GRID_POINTS, "invalid phiBins")
    out = dict(useDb=False, driftIgnorePhi=config["driftIgnorePhi"], phiBins=phi)
    for name in ("wireVoltage", "deltaDistance", "deltaTime", "dRdTScale"):
        out[name] = _positive(config[name], name)
    for name in ("kVcm", "cmus", "driftOffBins", "driftOffset", "driftRMSBins",
                 "signedDriftRMS", "unsignedDriftRMS"):
        value = _array(config[name], name)
        require(value.ndim == 1 and 2 <= value.size <= MAX_GRID_POINTS, "invalid dimensions: " + name)
        out[name] = value
    require(out["kVcm"].shape == out["cmus"].shape, "electric-field/speed dimensions differ")
    require((out["kVcm"] > 0) & (out["cmus"] > 0), "nonpositive electric field or speed")
    require(np.diff(out["kVcm"]) > 0, "electric fields must strictly increase")
    for name in ("driftOffBins", "driftRMSBins"):
        bins = out[name]
        require(bins.size == 2 and bins[1] > bins[0], "invalid calibration range: " + name)
    for bins, name in (("driftOffBins", "driftOffset"), ("driftRMSBins", "signedDriftRMS"),
                       ("driftRMSBins", "unsignedDriftRMS")):
        width = (out[bins][1] - out[bins][0]) / len(out[name])
        require(np.isfinite(width) and width > 0, "invalid calibration width: " + name)
    return out


def drift_configuration(path):
    """Return JSONable literal strawDrift and response calibration fields."""
    drift, response = _flat_drift(path), _flat_response(path)
    config = {k: drift[k] for k in DRIFT_FIELDS if k in drift}
    config.update({k: response[k] for k in RESPONSE_FIELDS if k in response})
    _validated(config)
    return config


@_checked
def build_zero_field(config, outer_radius, wire_radius):
    """Build the sampled zero-B source tables (mm, ns, mm/ns).

    Accumulation is deliberately sequential, not deltaD*arange or a pairwise
    sum. Time inversion deliberately uses the FIRST STRICTLY GREATER distance
    time, then caps at nDistance-2; it is not ordinary lower-bracket inversion.
    Dense allocations are bounded independently of untrusted FCL bin widths.
    """
    c = _validated(config)
    outer = _positive(outer_radius, "outer radius")
    wire = _positive(wire_radius, "wire radius")
    require(outer > wire, "outer radius must exceed wire radius")
    deltaD, deltaT = c["deltaDistance"], c["deltaTime"]
    efield, speed_data = c["kVcm"] * 100., c["cmus"] * .01
    data_dist = c["wireVoltage"] / (efield * np.log(outer / wire))
    require(np.isfinite(data_dist) & (data_dist > 0), "invalid electric-field distance")
    require(np.diff(data_dist) < 0, "electric-field distances must strictly descend")
    require(np.isfinite(speed_data) & (speed_data > 0), "invalid converted drift speeds")
    ratio = outer / deltaD
    require(np.isfinite(ratio) and ratio < MAX_GRID_POINTS - 16, "distance grid exceeds memory guard")
    # A small bounded reserve covers accumulated rounding at the outer edge.
    count = int(np.ceil(ratio)) + 8
    candidates = np.add.accumulate(np.full(count, deltaD, dtype=np.float64))
    require(np.isfinite(candidates), "nonfinite distance grid")
    require(candidates[-1] >= outer, "distance grid rounding exhausted reserve")
    distance = candidates[candidates < outer]
    require(len(distance) >= 1, "distance grid requires two points")
    index = np.clip(len(data_dist) - np.searchsorted(data_dist[::-1], distance, side="left") - 1,
                    0, len(data_dist) - 2)
    dist0, dist1 = data_dist[index + 1], data_dist[index]
    speed0, speed1 = speed_data[index + 1], speed_data[index]
    speed = speed0 + (speed1 - speed0) * (distance - dist0) / (dist1 - dist0)
    require(np.isfinite(speed) & (speed > 0), "invalid interpolated drift speed")
    distance_grid = np.r_[0., distance]
    distance_times = np.r_[0., np.add.accumulate(deltaD / speed)]
    instant_speed = np.r_[speed[0], speed]
    require(np.isfinite(distance_times), "nonfinite distance times")
    require(np.diff(distance_times) > 0, "distance times must strictly increase")
    time_bins = np.ceil(distance_times[-1] / deltaT)
    require(np.isfinite(time_bins) and 2 <= time_bins <= MAX_GRID_POINTS, "invalid/bounded time grid")
    time_grid = deltaT * np.arange(int(time_bins), dtype=np.float64)
    times = time_grid[1:]
    index = np.minimum(np.searchsorted(distance_times, times, side="right"), len(distance_grid) - 2)
    time_distances = np.r_[0., distance_grid[index] + (times - distance_times[index]) /
                           (distance_times[index + 1] - distance_times[index]) * deltaD]
    require(np.isfinite(time_grid) & np.isfinite(time_distances), "nonfinite time lookup")
    return dict(distance_grid=distance_grid, instant_speed=instant_speed, distance_times=distance_times,
                time_grid=time_grid, time_distances=time_distances, deltaD=deltaD, deltaT=deltaT)


@_checked
def _table(table, config):
    require(isinstance(table, dict), "drift table must be a dictionary")
    keys = ("distance_grid", "instant_speed", "distance_times", "time_grid", "time_distances", "deltaD", "deltaT")
    require(all(k in table for k in keys), "missing drift table field")
    out = {k: _array(table[k], k) for k in keys[:5]}
    for key, value in out.items():
        require(value.ndim == 1 and 2 <= value.size <= MAX_GRID_POINTS, "invalid table dimensions: " + key)
    d, t = out["distance_grid"], out["time_grid"]
    require(out["instant_speed"].shape == d.shape == out["distance_times"].shape, "distance table shapes differ")
    require(out["time_distances"].shape == t.shape, "time table shapes differ")
    for name in ("distance_grid", "distance_times", "time_grid"):
        require(out[name][0] == 0 and np.all(np.diff(out[name]) > 0), "invalid increasing table: " + name)
    require(out["time_distances"][0] == 0, "invalid time-distance origin")
    require(out["instant_speed"] > 0, "invalid instant speeds")
    for key, source in (("deltaD", "deltaDistance"), ("deltaT", "deltaTime")):
        out[key] = _positive(table[key], key)
        require(out[key] == config[source], "inconsistent table spacing: " + key)
    require(np.array_equal(t, out["deltaT"] * np.arange(len(t))), "inconsistent time grid")
    expected_d = np.r_[0., np.add.accumulate(np.full(len(d) - 1, out["deltaD"]))]
    require(np.array_equal(d, expected_d), "inconsistent sequential distance grid")
    return out


@_checked
def _interpolate_calib(x, bins, values):
    """NumPy regression, not certified native GSL; five terms maximum per lane.

    Mirrors GSL's two online passes: means, then mean dx2 and dxdy. No
    per-pulse loops, polynomial fitter, normal-equation solve, or clipping of
    signed extrapolations. The window is centered on the saturated bin index.
    """
    x, bins, values = (_array(v, n) for v, n in ((x, "calibration x"), (bins, "calibration bins"),
                                               (values, "calibration values")))
    require(x.ndim == 1 and bins.shape == (2,) and values.ndim == 1 and len(values) >= 2,
            "invalid regression dimensions")
    width = (bins[1] - bins[0]) / len(values)
    require(np.isfinite(width) and width > 0, "invalid regression width")
    index = np.clip(np.floor((x - bins[0]) / width), 0, len(values) - 1).astype(np.int64)
    imin, imax = np.maximum(0, index - 2), np.minimum(len(values) - 1, index + 2)
    mean_x, mean_y = np.zeros_like(x), np.zeros_like(x)
    for offset in range(5):
        active = imin + offset <= imax
        j = np.minimum(imin + offset, imax)
        xf, yf = bins[0] + width * (j + .5), values[j]
        mean_x += np.where(active, (xf - mean_x) / (offset + 1), 0.)
        mean_y += np.where(active, (yf - mean_y) / (offset + 1), 0.)
    mean_dx2, mean_dxdy = np.zeros_like(x), np.zeros_like(x)
    for offset in range(5):
        active = imin + offset <= imax
        j = np.minimum(imin + offset, imax)
        dx, dy = bins[0] + width * (j + .5) - mean_x, values[j] - mean_y
        mean_dx2 += np.where(active, (dx * dx - mean_dx2) / (offset + 1), 0.)
        mean_dxdy += np.where(active, (dx * dy - mean_dxdy) / (offset + 1), 0.)
    require(mean_dx2 > 0, "degenerate calibration regression")
    slope = mean_dxdy / mean_dx2
    intercept = mean_y - mean_x * slope
    value = intercept + slope * x
    require(np.isfinite(value) & np.isfinite(slope), "nonfinite calibration regression")
    return value, slope


@_checked
def drift_info(times, table, config):
    """Return 1-D cdrift, rdrift, velocity, sderr, uderr for finite input times.

    Negative times and end-segment extrapolations are intentional. Signed RMS
    regression results are returned as-is, not squared, absolutized or clipped.
    """
    c = _validated(config)
    t = _table(table, c)
    times = _array(times, "times")
    require(times.ndim == 1, "times must be one-dimensional")
    index = np.clip(np.floor(times / t["deltaT"]), 0, len(t["time_grid"]) - 2).astype(np.int64)
    cdrift = t["time_distances"][index] + (times - t["time_grid"][index]) / t["deltaT"] * (
        t["time_distances"][index + 1] - t["time_distances"][index])
    correction, slope = _interpolate_calib(cdrift, c["driftOffBins"], c["driftOffset"])
    rdrift = cdrift - correction
    index = np.clip(np.floor(cdrift / t["deltaD"]), 0, len(t["distance_grid"]) - 2).astype(np.int64)
    velocity = t["instant_speed"][index] + (cdrift - t["distance_grid"][index]) / t["deltaD"] * (
        t["instant_speed"][index + 1] - t["instant_speed"][index])
    velocity = (velocity * (1. - slope)) * c["dRdTScale"]
    sderr, _ = _interpolate_calib(rdrift, c["driftRMSBins"], c["signedDriftRMS"])
    uderr, _ = _interpolate_calib(rdrift, c["driftRMSBins"], c["unsignedDriftRMS"])
    result = dict(cdrift=cdrift, rdrift=rdrift, velocity=velocity, sderr=sderr, uderr=uderr)
    for name, value in result.items():
        require(np.isfinite(value), "nonfinite drift info: " + name)
    return result
