"""Source-calibration single-straw ComboHit operands; no fitter/default changes or FCL resolution."""
from pathlib import Path
import re

import numpy as np

from f1_captured_ca_audit import source_speed_configuration, source_signal_speed
from f1_native_decision_audit import require
from trkfastsim.reco.nativearith import _checked

BINS = ("eBins", "totTBins", "totEBins")
POSITIVE = ("eBinWidth", "defaultHalfPropVelocity", "strawLengthFactor", "errorFactor",
            "totTBinWidth", "totEBinWidth")
NONNEGATIVE = ("centralWirePos", "wireLengthBuffer")
BOOLEANS = ("rmsLongErrors", "truncateLongitudinal")
SEQUENCES = ("eHalfPVScale", "tdCentralRes", "tdResSlope", "totDriftTime", "totDriftError")
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _flat_response(path):
    """Accept literal flat fields only, never includes, references, overrides, or evaluation."""
    text = Path(path).read_text()
    require(not re.search(r"(?m)^\s*#\s*include\b|\b(?:BEGIN_PROLOG|END_PROLOG)\b", text),
            "unexpanded FCL directives unsupported")
    masked = re.sub(r'"(?:\\.|[^"\\])*"|#[^\n]*', lambda m: " " * len(m[0]), text)
    require(not re.search(r"\bstrawResponse\s*\.", masked), "strawResponse overrides unsupported")
    starts = list(re.finditer(r"\bstrawResponse\s*:", masked))
    require(len(starts) == 1, "ambiguous expanded strawResponse table")
    begin = starts[0].end()
    opening = re.match(r"\s*\{", masked[begin:])
    require(opening is not None, "literal expanded strawResponse table required")
    begin += opening.end()
    end = masked.find("}", begin)
    require(end >= begin and "{" not in masked[begin:end], "nested/unterminated strawResponse table unsupported")
    block, result, position = text[begin:end], {}, 0
    field = re.compile(r'\s*([A-Za-z_][A-Za-z_0-9]*)\s*:\s*(\[[^\[\]{}]*\]|"(?:\\.|[^"\\])*"|[^\s\[\]{}]+)')
    while position < len(block) and block[position:].strip():
        match = field.match(block, position)
        require(match is not None, "unsupported flat strawResponse syntax")
        name, token = match.group(1, 2)
        require(name not in result, "duplicate strawResponse field " + name)
        if token.startswith("["):
            body = token[1:-1].strip()
            require(not body or re.fullmatch(NUMBER + r"(?:\s*,\s*" + NUMBER + r")*", body),
                    "invalid strawResponse sequence " + name)
            value = [] if not body else [float(v) for v in body.split(",")]
        elif token in ("true", "false"):
            value = token == "true"
        elif re.fullmatch(NUMBER, token):
            value = int(token) if re.fullmatch(r"[+-]?\d+", token) else float(token)
        elif token.startswith('"'):
            value = token
        else:
            raise ValueError("unsupported strawResponse literal " + name)
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


@_checked
def _validated(config):
    """Validate both parsed and caller-supplied configurations without allocating unchecked grids."""
    require(isinstance(config, dict), "response configuration must be a dictionary")
    require(all(k in config for k in BINS + POSITIVE + NONNEGATIVE + BOOLEANS + SEQUENCES),
            "missing response configuration field")
    out = {}
    for name in BINS:
        value = config[name]
        require(type(value) is int and 0 < value <= np.iinfo(np.int64).max, "invalid bins: " + name)
        out[name] = value
    require(out["eBins"] >= 2, "energy grid needs at least two bins")
    for name in POSITIVE + NONNEGATIVE:
        value = _array(config[name], name)
        require(value.ndim == 0, "invalid scalar: " + name)
        require(value > 0 if name in POSITIVE else value >= 0, "invalid constant: " + name)
        out[name] = float(value)
    for name in BOOLEANS:
        require(type(config[name]) is bool, "invalid boolean: " + name)
        out[name] = config[name]
    for name in SEQUENCES:
        value = _array(config[name], name)
        size = out["eBins"] if name in SEQUENCES[:3] else out["totTBins"] * out["totEBins"]
        require(value.ndim == 1 and value.size == size, "invalid dimensions: " + name)
        if name in ("eHalfPVScale", "tdCentralRes", "totDriftError"):
            require(value > 0, "invalid positive calibration: " + name)
        out[name] = value
    grid = np.arange(out["eBins"], dtype=np.float64) * out["eBinWidth"]
    require(np.isfinite(grid) & np.r_[True, np.diff(grid) > 0], "invalid energy grid")
    speed = dict(bin_width=out["eBinWidth"], mean_half_velocity=out["defaultHalfPropVelocity"],
                 scale=out["eHalfPVScale"].tolist())
    for name in speed:
        if name in config:
            require(np.array_equal(config[name], speed[name]), "inconsistent speed configuration: " + name)
    out["speed"], out["grid"] = speed, grid
    return out


def response_configuration(path):
    """Return JSONable literal response fields and the existing source-speed provenance/aliases."""
    fields = _flat_response(path)
    speed = source_speed_configuration(path)
    config = {name: fields[name] for name in BINS + POSITIVE + NONNEGATIVE + BOOLEANS + SEQUENCES
              if name in fields}
    config.update(speed)
    _validated(config)
    return config


@_checked
def _piece_line(grid, values, x):
    """StrawResponse::PieceLine with slope-first arithmetic and end-segment extrapolation."""
    width = (grid[-1] - grid[0]) / (len(grid) - 1)
    index = np.clip(np.floor((x - grid[0]) / width), 0, len(grid) - 2).astype(np.int64)
    slope = (values[index + 1] - values[index]) / width
    result = values[index] + (x - grid[index]) * slope
    require(np.isfinite(result), "nonfinite PieceLine result")
    return result


def _stored(value):
    return value.astype(np.float32).astype(np.float64)


@_checked
def reconstruct(measured, geometry, config):
    """Vectorized measured float energy/end-times/TOT plus source geometry to initial ComboHit operands."""
    c = _validated(config)
    require(isinstance(measured, dict) and isinstance(geometry, dict), "input dictionaries required")
    require(all(k in measured for k in ("energy", "et0", "et1", "tot0", "tot1")), "missing measured field")
    require(all(k in geometry for k in ("hlen", "direction")), "missing geometry field")
    m = {}
    for name in ("energy", "et0", "et1", "tot0", "tot1"):
        value = _array(measured[name], "measured " + name)
        require(value.ndim == 1, "invalid measured dimensions: " + name)
        if name in ("energy", "tot0", "tot1"):
            require(value >= 0, "negative measured " + name)
        m[name] = value.astype(np.float32)
    shape = m["energy"].shape
    require(all(v.shape == shape for v in m.values()), "measured shapes differ")
    hlen, direction = (_array(geometry[k], "geometry " + k) for k in ("hlen", "direction"))
    require(hlen.shape == shape and direction.shape == shape + (3,), "invalid geometry dimensions")
    require(hlen > 0, "invalid geometry half length")
    udir = _stored(direction)
    require(np.any(udir != 0, axis=1), "invalid zero geometry direction")
    energy = m["energy"].astype(np.float64)
    dt = (m["et1"] - m["et0"]).astype(np.float64)
    speed = source_signal_speed(energy, c["speed"])
    wdist = (speed / 2.) * dt
    x = 1000. * energy
    wderr = _piece_line(c["grid"], c["tdCentralRes"], x)
    if c["rmsLongErrors"]:
        outside = np.abs(wdist) > c["centralWirePos"]
        slope = _piece_line(c["grid"], c["tdResSlope"], x[outside])
        wderr[outside] += (np.abs(wdist[outside]) - c["centralWirePos"]) * slope
    else:
        slope = _piece_line(c["grid"], c["tdResSlope"], x)
        wderr += (slope * np.abs(wdist)) * np.abs(wdist)
    wderr = np.maximum(30., wderr)
    truncated = (np.abs(wdist) > hlen + c["wireLengthBuffer"] * wderr) & c["truncateLongitudinal"]
    wdist[truncated] = np.copysign(hlen[truncated] * c["strawLengthFactor"], wdist[truncated])
    wderr[truncated] *= c["errorFactor"]
    early_end = (m["et1"] < m["et0"]).astype(np.int64)
    tot = np.where(early_end == 1, m["tot1"], m["tot0"]).astype(np.float64)
    timebin = np.minimum(c["totTBins"] - 1, tot / c["totTBinWidth"]).astype(np.int64)
    ebin = np.minimum(c["totEBins"] - 1, energy / c["totEBinWidth"]).astype(np.int64)
    index = timebin * c["totEBins"] + ebin
    error = c["totDriftError"][index]
    result = dict(dtime=_stored(c["totDriftTime"][index] - 0.1), timevar=_stored(error * error),
                  wdist=_stored(wdist), uvar=_stored(wderr * wderr), udir=udir,
                  early_end=early_end, signal_speed=speed, truncated=truncated)
    for name, value in result.items():
        require(np.isfinite(value), "nonfinite reconstructed " + name)
    require((result["timevar"] > 0) & (result["uvar"] > 0), "invalid reconstructed variance")
    return result
