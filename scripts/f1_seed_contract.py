"""Source-grouped native seed construction from saved LineFinder inputs, without JAX."""
from pathlib import Path
import re

import awkward as ak
import numpy as np

import f1_crossing_constructor as B
from f1_native_decision_audit import array, column, match_rows, require

SEED_FIELDS = ("a0", "b0", "a1", "b1", "t0")
ERRORS = np.array([15., .05, 10., .02, 2., 1.])
CONSTRAINTS = np.array([0., 0., 0., 0., 0., .0001])


def configuration(path):
    """Accept only the frozen literal seed/error/constraint configuration."""
    text = Path(path).read_text()
    require(not re.search(r"(?m)^\s*#\s*include\b", text), "unexpanded seed configuration")
    text = re.sub(r"(?m)#.*$", "", text)
    require(not re.search(r"\b\w+(?:\.\w+)+\s*:", text), "dotted seed configuration override")
    for name, expected in (("SeedErrors", ERRORS), ("ParameterConstraints", CONSTRAINTS)):
        values = re.findall(r"\b"+name+r"\s*:\s*\[([^\]]*)\]", text)
        require(len(values) == 2, "ambiguous seed sequence: "+name)
        for value in values:
            actual = np.array([float(x.strip()) for x in value.split(",")])
            require(actual.shape == (6,) and np.array_equal(actual, expected), "unsupported "+name)
    for name, expected in (("SeedMomentum", 2000.), ("FitParticle", 13.)):
        values = re.findall(r"(?m)^\s*"+name+r"\s*:\s*([^\s]+)\s*$", text)
        require(len(values) == 2 and all(float(x) == expected for x in values), "unsupported "+name)
    require('module_label: "LineFinder"' in text and 'module_type: "LineFinder"' in text,
            "LineFinder source equivalence unavailable")
    return dict(momentum=2000., errors=ERRORS.tolist(), constraints=CONSTRAINTS.tolist())


def inputs(events, target_keys):
    """Join full seed keys to raw LineFinder values and ordered source-hit membership."""
    target_keys = B.integer_matrix(target_keys, 4, "seed keys")
    ev = match_rows(target_keys[:, :3], B.integer_keys(events, B.KEYS[:3]))
    ns = array(ak.num(events["sd_a0"], axis=1))
    require(target_keys[:, 3] < ns[ev], "seed index outside event")
    loc = np.r_[0, np.cumsum(ns)][ev]+target_keys[:, 3]
    values = []
    for name in SEED_FIELDS:
        branch = events["sd_"+name]
        require(array(ak.num(branch, axis=1)) == ns, "raw seed lengths differ")
        flat = array(ak.flatten(branch, axis=1))
        require(flat.dtype == np.dtype("float64") and np.all(np.isfinite(flat)), "invalid raw seed field: "+name)
        values.append(flat[loc])
    require(array(ak.num(events["sd_n"], axis=1)) == ns, "seed membership lengths differ")
    counts = array(ak.flatten(events["sd_n"], axis=1))
    require(counts.dtype.kind in "iu" and np.all(counts >= 0), "invalid seed hit counts")
    require(array(ak.sum(events["sd_n"], axis=1)) == array(ak.num(events["sd_ind"], axis=1)), "seed hit flattening differs")
    indices = array(ak.flatten(events["sd_ind"], axis=1))
    require(indices.dtype.kind in "iu" and np.all(indices >= 0), "invalid source hit indices")
    selected_counts = counts[loc]
    require(selected_counts > 0, "empty source seed")
    parent = np.repeat(np.arange(len(loc)), selected_counts)
    local = np.arange(len(parent))-np.repeat(np.r_[0, np.cumsum(selected_counts)[:-1]], selected_counts)
    source = indices[np.r_[0, np.cumsum(counts)][loc[parent]]+local]
    hit_keys = B.integer_matrix(np.column_stack((target_keys[parent], source)), 5, "source hit keys")
    nh = array(ak.num(events["s_time"], axis=1))
    require(source < nh[ev[parent]], "source hit index outside event")
    hitloc = np.r_[0, np.cumsum(nh)][ev[parent]]+source
    times = array(ak.flatten(events["s_time"], axis=1))
    require(times.dtype == np.dtype("float32") and np.all(np.isfinite(times)), "source corrected time must be float32")
    require(array(ak.num(events["s_sid"], axis=1)) == nh, "source straw lengths differ")
    sid = array(ak.flatten(events["s_sid"], axis=1))[hitloc]
    require(sid.dtype.kind in "iu" and np.all((sid >= 0) & (sid <= 65535)), "invalid source straw id")
    return dict(raw=np.column_stack(values), hit_keys=hit_keys, parent=parent, strawid=sid,
                construction_range=bounds(parent, times[hitloc].astype(float), len(loc)))


def bounds(parent, times, n):
    """Extrema by parent with no missing seed, sorting assumption, or activity filter."""
    parent = np.asarray(parent)
    require(parent.ndim == 1 and parent.dtype.kind in "iu", "range parent: integer vector required")
    require(parent >= 0, "range parent: negative key")
    require(parent.astype(np.uint64) <= np.uint64(np.iinfo(np.int64).max), "range parent overflow")
    parent = parent.astype(np.int64)
    times = np.asarray(times)
    require(times.shape == parent.shape and times.dtype.kind == "f" and np.all(np.isfinite(times)), "invalid range times")
    require(n > 0 and np.all(parent < n), "range parent outside seed population")
    require(np.bincount(parent, minlength=n) > 0, "empty range population")
    lo, hi = np.full(n, np.inf), np.full(n, -np.inf)
    np.minimum.at(lo, parent, times)
    np.maximum.at(hi, parent, times)
    return np.column_stack((lo, hi))


def dot(a, b):
    """ROOT Cartesian three-term grouping, preserving input precision."""
    return (a[..., 0]*b[..., 0]+a[..., 1]*b[..., 1])+a[..., 2]*b[..., 2]


def unit(v):
    """ROOT Unit uses reciprocal multiplication, not componentwise division."""
    norm = np.sqrt(dot(v, v))
    require(np.isfinite(norm) & (norm > 0), "invalid vector norm")
    reciprocal = np.asarray(1, dtype=v.dtype)/norm
    return v*reciprocal[..., None]


def generate(raw, mass, config):
    """Generate KL6 and configured covariance from (A0,B0,A1,B1,T0) and mass."""
    raw, mass = np.asarray(raw), np.asarray(mass)
    require(raw.ndim == 2 and raw.shape[1] == 5 and raw.dtype.kind == "f" and np.all(np.isfinite(raw)), "invalid raw seed")
    n = len(raw)
    require(n > 0 and mass.shape == (n,) and np.all(np.isfinite(mass) & (mass > 0)), "invalid seed mass")
    require(config == dict(momentum=2000., errors=ERRORS.tolist(), constraints=CONSTRAINTS.tolist()), "unsupported seed configuration")
    raw = raw.astype(float)
    slopes = np.column_stack((raw[:, 2], -np.ones(n), raw[:, 3])).astype(np.float32)
    mom3 = (unit(slopes)*np.float32(config["momentum"])).astype(float)
    momentum = np.sqrt(dot(mom3, mom3))
    speed = (momentum/np.sqrt(dot(mom3, mom3)+mass*mass))*299.792458
    direction = unit(mom3)
    poca_direction = unit(direction)
    position = np.column_stack((raw[:, 0], np.zeros(n), raw[:, 1]))
    c = poca_direction[:, 2]
    sinsq = 1.-c*c
    parallel = sinsq < 1.e-8
    denominator = np.where(parallel, 1., sinsq)
    d1, d2 = dot(position, poca_direction), position[:, 2]
    s1 = np.where(parallel, 0., (d2*c-d1)/denominator)
    s2 = np.where(parallel, 0., -(d1*c-d2)/denominator)
    p1 = position+poca_direction*s1[:, None]
    p2 = np.column_stack((np.zeros(n), np.zeros(n), s2))
    delta = p1-p2
    angular = -((-delta[:, 1])*direction[:, 0]+delta[:, 0]*direction[:, 1])
    d0 = np.copysign(1., angular)*np.sqrt(dot(delta, delta))
    flight = dot(direction, position-p1)
    q = np.column_stack((d0, np.arctan2(direction[:, 1], direction[:, 0]), p1[:, 2],
                         np.arccos(direction[:, 2]), raw[:, 4]-flight/speed, momentum))
    covariance = np.broadcast_to(np.diag(ERRORS*ERRORS), (n, 6, 6)).copy()
    require(np.all(np.isfinite(q)), "nonfinite generated seed")
    return dict(parameters=q, covariance=covariance, momentum3=mom3, speed=speed, parallel=parallel)


def constraint(parameters, ranges, config):
    """Native ParameterHit copies q, resets C, masks positive configured sigmas."""
    q, ranges = np.asarray(parameters), np.asarray(ranges)
    require(q.ndim == 2 and q.shape[1] == 6 and np.all(np.isfinite(q)), "invalid constraint parameters")
    require(ranges.shape == (len(q), 2) and np.all(np.isfinite(ranges)) and np.all(ranges[:, 0] <= ranges[:, 1]), "invalid constraint range")
    sigmas = np.asarray(config["constraints"])
    require(np.array_equal(sigmas, CONSTRAINTS), "unsupported constraint sigmas")
    covariance = np.diag(np.where(sigmas > 0, sigmas*sigmas, 1.))
    return dict(parameters=q.copy(), covariance=np.broadcast_to(covariance, (len(q), 6, 6)).copy(),
                mask=np.broadcast_to((sigmas > 0).astype(np.int64), q.shape).copy(), time=.5*(ranges[:, 0]+ranges[:, 1]))
