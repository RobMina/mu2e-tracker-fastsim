"""Full-universe normal-branch material discovery on supplied native trajectories/geometry."""
import re

import awkward as ak
import numpy as np

import f1_crossing_history as H
import f1_crossing_constructor as B
from f1_crossing_constructor import L, N, C
from f1_native_decision_audit import array, column, dense, match_rows, require

CONFIG = dict(AddStrawMinDz=0., MaxDStraw=2, MaxStrawDOCA=5., MaxStrawDOCAConsistency=1.,
              MaxStrawUposBuffer=0., StrawNBuffer=0, SkipStrawCheck=False)


def configuration(path):
    text = path.read_text()
    for name, expected in CONFIG.items():
        values = re.findall(r"\b"+name+r"\s*:\s*([^\s,}\]]+)", text)
        require(bool(values), "missing source config: "+name)
        if type(expected) is bool:
            require(all(v == "false" for v in values), "unsupported config: "+name)
        else:
            require(all(float(v) == expected for v in values), "unsupported config: "+name)
    return dict(CONFIG)


def pack(table):
    table = table[column(table, "category", "iu") == 2]
    out = H.pack(table)
    require(len(np.unique(out["keys"][:, :4], axis=0)) == len(out["keys"]), "multiple material contexts per target unsupported")
    k = B.integer_keys(table, B.TRAJECTORY+("piece_index",))
    parent = match_rows(k[:, :-1], out["keys"])
    covariance = np.zeros(out["q"].shape[:2]+(6, 6))
    covariance[parent, k[:, -1]] = dense(table, "covariance", 36).reshape(-1, 6, 6)
    require(np.isfinite(covariance), "nonfinite covariance")
    require(covariance == covariance.swapaxes(-1, -2), "nonsymmetric covariance")
    out["covariance"] = covariance
    return out


def geometry(tables):
    """Retain complete ordered geometry; no discovery candidate list is an input."""
    done = tables["bindings"][column(tables["bindings"], "source_status") == 1]
    target = B.integer_keys(done, B.KEYS)
    gid = B.integer_keys(done, ("geometry_index",))
    result = dict(target=target, gid=gid)
    tracker = tables["tracker"][match_rows(gid, B.integer_keys(tables["tracker"], ("geometry_index",)))]
    result["radius"] = column(tracker, "straw_outer_radius", "f")
    result["wire_radius"] = column(tracker, "wire_radius", "f")
    require((result["radius"] > 0) & (result["wire_radius"] > 0), "invalid geometry radii")
    fields = dict(planes=(36, "plane_ordinal", ("fullsid", "plane_exists", "origin")),
                  panels=(216, "tracker_panel_ordinal", ("fullsid", "origin", "v_direction", "ds_to_panel_rotation", "ds_to_panel_translation", "panel_ordinal")),
                  straws=(20736, "tracker_straw_ordinal", ("fullsid", "tube_cal", "tube_hv", "wire_cal", "wire_midpoint", "wire_direction", "half_length", "straw_ordinal")))
    for kind, (count, ordinal, names) in fields.items():
        table = tables[kind]
        key = B.integer_keys(table, ("geometry_index", ordinal))
        parent = match_rows(key[:, :1], gid)
        require(len(table) == len(gid)*count, "incomplete geometry universe: "+kind)
        require(np.bincount(parent, minlength=len(gid)) == count, "geometry count differs: "+kind)
        require(key[:, 1] < count, "geometry ordinal out of range")
        order = np.lexsort((key[:, 1], parent))
        result[kind] = {}
        for name in names:
            value = array(table[name])
            require(np.isfinite(value), "nonfinite geometry: "+name)
            result[kind][name] = value[order].reshape((len(gid), count)+value.shape[1:])
    require(result["straws"]["half_length"].dtype == np.float32, "halfLength must retain float32 source")
    require(result["panels"]["panel_ordinal"] == np.tile(np.arange(6), 36), "nonstandard ordered panel hierarchy")
    require(result["straws"]["straw_ordinal"] == np.tile(np.arange(96), 216), "nonstandard ordered straw hierarchy")
    return result


def exclusions(trajectories, geo, initial, associated):
    """Only pinned initial/associated membership seeds the single-extension oldstraw set."""
    target = trajectories["keys"][:, :4]
    match_rows(target, geo["target"])
    supplied = []
    for receipt in (initial, associated):
        identities = receipt["identities"]
        obj = B.integer_matrix(identities["object_keys"], 5, "prior constructor")
        sid = L._integer(identities["strawid"], (len(obj),), "prior straw", 0, 65535)
        supplied.append(np.column_stack((obj[:, :4], sid)))
    old = np.unique(np.concatenate(supplied), axis=0)
    g = match_rows(old[:, :4], geo["target"])
    exists = np.zeros((len(geo["gid"]), 65536), dtype=bool)
    exists[g, old[:, -1]] = True
    gcontext = match_rows(target, geo["target"])
    return exists[gcontext]


def transform(position, rotation, translation):
    r = rotation.reshape(-1, 3, 3)
    return translation + ((r[:, :, 0]*position[:, 0, None] + r[:, :, 1]*position[:, 1, None]) + r[:, :, 2]*position[:, 2, None])


def tzero(t):
    q, ranges, counts = t["q"], t["ranges"], t["counts"]
    parent = np.arange(len(q))
    time = q[:, 0, 4].copy()
    index = L._nearest(ranges, counts, parent, time)
    old = np.full(len(q), -1, dtype=np.int64)
    moves = np.zeros(len(q), dtype=np.int64)
    for _ in range(10):
        active = index != old
        if not active.any():
            break
        time[active] = q[parent[active], index[active], 4]
        old[active] = index[active]
        index[active] = L._nearest(ranges, counts, parent[active], time[active])
        moves += active
    return time, index, moves


def ztime(t, parent, z):
    """KKFitUtilities zTime; the last pointer may be unsolved at the 100-step cap."""
    parent = L._integer(parent, (len(parent),), "z parent", 0, len(t["q"])-1)
    z = L._real(z, (len(parent),), "plane z")
    time = t["ranges"][parent, 0, 0].copy()
    index = L._nearest(t["ranges"], t["counts"], parent, time)
    active = np.ones(len(parent), dtype=bool)
    tries = np.zeros(len(parent), dtype=np.int64)
    for _ in range(100):
        rows = np.flatnonzero(active)
        if not len(rows):
            break
        p, u, speed = C.particle(t["q"][parent[rows], index[rows]], time[rows], t["mass"][parent[rows]])
        vz = u[:, 2]*speed
        require(vz != 0, "zero-z velocity unsupported")
        time[rows] = time[rows]+(z[rows]-p[:, 2])/vz
        next_index = L._nearest(t["ranges"], t["counts"], parent[rows], time[rows])
        active[rows] = next_index != index[rows]
        index[rows] = next_index
        tries[rows] += 1
    require(np.isfinite(time), "nonfinite zTime")
    return time, index, tries, active


def similarity(row, covariance):
    projected = N._matvec6(covariance, row)
    products = row*projected
    value = products[:, 0].copy()
    for i in range(1, 6):
        value = value+products[:, i]
    require(np.isfinite(value), "nonfinite variance")
    return value


def material_cut(doca, variance, du, radius, half_length):
    dsig = np.maximum(0., np.abs(doca)-radius)/np.sqrt(np.maximum(variance, np.finfo(np.float64).tiny))
    passed = (np.abs(doca) < 5.) & (dsig < 1.) & (du < half_length.astype(np.float64))
    return dsig, passed


def generate(t, g, old):
    """Generate every reached stage from all planes/panels/straws, not captured outcomes."""
    q, ranges, counts = t["q"], t["ranges"], t["counts"]
    context = np.arange(len(q))
    gi = match_rows(t["keys"][:, :4], g["target"])
    require(old.dtype.kind == "b" and old.shape == (len(q), 65536), "invalid exclusion set")
    p, a, s = g["planes"], g["panels"], g["straws"]
    first = np.arange(len(g["gid"]))
    inner = transform(s["wire_midpoint"][first, 0], a["ds_to_panel_rotation"][first, 0], a["ds_to_panel_translation"][first, 0])
    outer = transform(s["wire_midpoint"][first, 95], a["ds_to_panel_rotation"][first, 0], a["ds_to_panel_translation"][first, 0])
    ymin, ymax = inner[:, 1], outer[:, 1]
    rmin = ymin-2*g["radius"]
    wirecal = s["wire_cal"][:, 95]
    rmax = np.sqrt(N._dot3(wirecal, wirecal))+2*g["radius"]
    spitch = 95/(ymax-ymin)
    constants = np.column_stack((g["radius"], rmin, rmax, ymin, ymax, spitch))
    require(np.isfinite(constants), "invalid tracker limits")
    require(constants == constants[:1], "cached tracker-info geometry changes unsupported")
    tz, iz, ttries = tzero(t)
    pos, direction, _ = C.particle(q[context, iz], tz, t["mass"])
    stages = {}
    def emit(stage, parent, plane=None, panel=None, straw=None, passed=None, values=None, integers=None):
        n = len(parent)
        pl = np.full(n, -1, dtype=np.int64) if plane is None else plane
        pa = np.full(n, -1, dtype=np.int64) if panel is None else panel
        st = np.full(n, -1, dtype=np.int64) if straw is None else straw
        stages[stage] = dict(parent=parent, plane=pl, panel=pa, straw=st,
            passed=np.ones(n, bool) if passed is None else passed,
            values=np.empty((n, 0)) if values is None else values,
            integers=np.empty((n, 0), dtype=np.int64) if integers is None else integers)
    values = np.column_stack((pos, direction, np.zeros(len(q)), constants[gi], np.full(len(q), 5.), np.ones(len(q)), np.zeros((len(q), 2))))
    emit(0, context, values=values, integers=np.tile([2, 0], (len(q), 1)))
    oi, os = np.nonzero(old)
    stages[0]["old"] = ak.unflatten(os, np.bincount(oi, minlength=len(q)))
    cp = np.repeat(context, 36)
    gp = gi[cp]
    planeord = np.tile(np.arange(36), len(q))
    planeid = p["fullsid"][gp, planeord]
    exists = p["plane_exists"][gp, planeord]
    require(exists.dtype.kind == "b", "nonboolean plane existence")
    emit(1, cp, planeid, passed=exists)
    cp, gp, planeord, planeid = (x[exists] for x in (cp, gp, planeord, planeid))
    plz = p["origin"][gp, planeord, 2]
    zt, zi, tries, zcap = ztime(t, cp, plz)
    plpos, tdir, _ = C.particle(q[cp, zi], zt, t["mass"][cp])
    rho = np.sqrt(plpos[:, 0]*plpos[:, 0]+plpos[:, 1]*plpos[:, 1])
    radial = (rho > rmin[gp]) & (rho < rmax[gp])
    emit(2, cp, planeid, passed=radial, values=np.column_stack((plz, zt, plpos, rho)))
    selected = np.repeat(np.flatnonzero(radial), 6)
    cp, gp, plz, zt, plpos, tdir, planeid = (x[selected] for x in (cp, gp, plz, zt, plpos, tdir, planeid))
    panelord = planeord[selected]*6+np.tile(np.arange(6), int(radial.sum()))
    panelid = a["fullsid"][gp, panelord]
    dz = a["origin"][gp, panelord, 2]-plz
    require(tdir[:, 2] != 0, "zero-z panel direction unsupported")
    papos = plpos+(dz/tdir[:, 2])[:, None]*tdir
    ppos = transform(papos, a["ds_to_panel_rotation"][gp, panelord], a["ds_to_panel_translation"][gp, panelord])
    dirv = np.abs(N._dot3(tdir, a["v_direction"][gp, panelord])/tdir[:, 2])
    rough = np.rint((ppos[:, 1]-ymin[gp])*spitch[gp])
    require(np.isfinite(rough) & (rough > -2**31) & (rough < 2**31-1), "straw-number conversion overflow")
    istraw = rough.astype(np.int64)
    ibuffer = (dirv*0*2*g["radius"][gp]*spitch[gp]).astype(np.int64)
    window = (istraw >= -2-ibuffer) & (istraw < 96+2+ibuffer)
    emit(3, cp, planeid, panelid, passed=window, values=np.column_stack((tdir, dz, papos, ppos, dirv)),
         integers=np.column_stack((ibuffer, istraw, np.full(len(cp), 96))))
    cp, gp, panelord, planeid, panelid, zt, ppos, istraw, ibuffer = (x[window] for x in (cp, gp, panelord, planeid, panelid, zt, ppos, istraw, ibuffer))
    imin = np.maximum(istraw-2-ibuffer, 0)
    half = s["half_length"][gp, panelord*96+imin]
    length = np.abs(ppos[:, 0]) < half.astype(np.float64)
    emit(4, cp, planeid, panelid, passed=length, values=half[:, None].astype(np.float64), integers=imin[:, None])
    cp, gp, panelord, planeid, panelid, zt, istraw, ibuffer, imin = (x[length] for x in (cp, gp, panelord, planeid, panelid, zt, istraw, ibuffer, imin))
    imax = np.minimum(istraw+2+ibuffer, 95)
    emit(5, cp, planeid, panelid, integers=np.column_stack((imin, imax)))
    sizes = imax-imin+1
    require(sizes > 0, "empty candidate window")
    expand = np.repeat(np.arange(len(cp)), sizes)
    straword = np.arange(int(sizes.sum()))-np.repeat(np.cumsum(sizes)-sizes, sizes)+imin[expand]
    cp, gp, panelord, planeid, panelid, zt = (x[expand] for x in (cp, gp, panelord, planeid, panelid, zt))
    flat = panelord*96+straword
    sid = s["fullsid"][gp, flat].astype(np.int64)
    require(len(np.unique(np.column_stack((cp, sid)), axis=0)) == len(cp), "duplicate straw visit unsupported")
    fresh = ~old[cp, sid]
    emit(6, cp, planeid, panelid, sid, passed=fresh, integers=straword[:, None])
    cp, gp, flat, planeid, panelid, sid, zt = (x[fresh] for x in (cp, gp, flat, planeid, panelid, sid, zt))
    generated = B.generate(q, ranges, counts, cp, t["mass"][cp], s["tube_cal"][gp, flat], s["tube_hv"][gp, flat], zt, zt, g["wire_radius"][gp])
    state = generated["state"]
    rows = N.rows_at_state(q[cp, state["solution_index"]], generated["line"], state, t["mass"][cp])
    covariance = t["covariance"][cp, state["solution_index"]]
    docavar = similarity(rows["H"][:, 0], covariance)
    du = np.abs(N._dot3(state["sensor_poca"]-s["wire_midpoint"][gp, flat], s["wire_direction"][gp, flat]))
    half = s["half_length"][gp, flat]
    dsig, accepted = material_cut(rows["doca"], docavar, du, g["radius"][gp], half)
    emit(7, cp, planeid, panelid, sid, passed=accepted, values=np.column_stack((du, np.abs(rows["doca"]), dsig, docavar, half)))
    emit(8, cp[accepted], planeid[accepted], panelid[accepted], sid[accepted])
    return dict(stages=stages, generated=generated, derivatives=rows, docavar=docavar, accepted=accepted,
                tzero_tries=ttries, ztime_tries=tries, ztime_caps=zcap,
                universe=dict(contexts=len(q), planes=len(q)*36, panels=len(q)*216, straws=len(q)*20736))
