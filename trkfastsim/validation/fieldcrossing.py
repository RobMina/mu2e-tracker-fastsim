"""Report-only crossing audit on saved gas-step endpoint secants; never rerun transport."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from mu2efastsim import tables
from trkfastsim import crossings, geometry
from trkfastsim.sim import fieldsteps as fs
from trkfastsim.validation import cecontract
from trkfastsim.validation.fieldmultiplicity import TRACK, STRAW, track_keys

FILES = dict(fast="fasttrk_field.root", g4_all="g4prim_field.root", g4_clock="g4fp_field.root")
COVERAGE = ("saved gas-step endpoint secants only; not actual stochastic chords/gaps, physical G4 first passage, "
            "or cosmic deficit closure")
ATOL_MM = 1e-7
EXAMPLES = 20
COLUMNS = TRACK + STRAW + [f"{axis}{end}" for end in (0, 1) for axis in "xyz"]


def _positive_integer(value, label, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < (0 if allow_zero else 1):
        raise ValueError(f"invalid {label}")


def validate_rays(x, c, length):
    """Reject nonfinite, nonunit or zero directions rather than silently skipping them."""
    x, c, length = np.asarray(x, float), np.asarray(c, float), np.asarray(length, float)
    if x.ndim != 2 or x.shape[1] != 3 or c.shape != x.shape or length.shape != (len(x),):
        raise ValueError("invalid secant shapes")
    if not (np.isfinite(x).all() and np.isfinite(c).all() and np.isfinite(length).all()):
        raise ValueError("nonfinite secant input rejected")
    if (length <= 0).any() or not np.allclose(np.linalg.norm(c, axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("invalid zero/nonunit direction or nonpositive secant length")
    return x, c, length


def endpoint_secants(start, end, origin):
    """Return positive secants and their original row mask; exact zero lengths are counted separately."""
    start, end, origin = np.asarray(start, float), np.asarray(end, float), np.asarray(origin, float)
    if start.ndim != 2 or start.shape[1] != 3 or end.shape != start.shape or origin.shape != (3,):
        raise ValueError("invalid endpoint/origin shapes")
    if not (np.isfinite(start).all() and np.isfinite(end).all() and np.isfinite(origin).all()):
        raise ValueError("nonfinite saved endpoints/origin rejected; no rows dropped")
    with np.errstate(over="ignore", invalid="ignore"):
        x = geometry.to_mu2e(start, origin)
        delta = end - start
        length = np.linalg.norm(delta, axis=1)
    if not (np.isfinite(x).all() and np.isfinite(length).all()):
        raise ValueError("nonfinite derived secant rejected")
    keep = length > 0
    c = delta[keep] / length[keep, None]
    return (*validate_rays(x[keep], c, length[keep]), keep)


def canonical(rows, length, nstraws, label):
    """Validate unique lane/straw keys, clip to the finite secant, and sort by lane/straw."""
    r = np.asarray(rows, float).copy()
    if r.ndim != 2 or r.shape[1] != 4 or not np.isfinite(r).all():
        raise ValueError(f"invalid/nonfinite crossing rows: {label}")
    keys = r[:, :2]
    if (keys != np.floor(keys)).any() or (keys < 0).any():
        raise ValueError(f"invalid crossing identities: {label}")
    if (keys[:, 0] >= len(length)).any() or (keys[:, 1] >= nstraws).any():
        raise ValueError(f"foreign crossing identities: {label}")
    r = r[np.lexsort((r[:, 1], r[:, 0]))]
    if len(r) > 1 and np.any(np.all(r[1:, :2] == r[:-1, :2], axis=1)):
        raise ValueError(f"duplicate lane/straw crossings: {label}")
    lane = r[:, 0].astype(np.int64)
    r[:, 2] = np.maximum(r[:, 2], 0)
    r[:, 3] = np.minimum(r[:, 3], length[lane])
    return r[r[:, 3] > r[:, 2]]


def oracle_crossings(x, c, length, geom, batch=64, straw_block=256):
    """Independent all-straw finite-cylinder intersections, without any panel or bounding-box search."""
    x, c, length = validate_rays(x, c, length)
    _positive_integer(batch, "oracle batch")
    _positive_integer(straw_block, "straw block")
    out = []
    for i in range(0, len(x), batch):
        for j in range(0, geom.nstraws, straw_block):
            axis = geom.dir[j:j + straw_block]
            projector = np.eye(3)[None] - axis[:, :, None] * axis[:, None, :]
            displacement = x[i:i + batch, None, :] - geom.mid[None, j:j + straw_block]
            transverse = np.einsum("sij,bsj->bsi", projector, displacement)
            velocity = np.einsum("sij,bj->bsi", projector, c[i:i + batch])
            speed2 = np.einsum("bsi,bsi->bs", velocity, velocity)
            dot = np.einsum("bsi,bsi->bs", velocity, transverse)
            offset = np.einsum("bsi,bsi->bs", transverse, transverse) - geom.radius ** 2
            parallel = speed2 <= 1e-15
            discriminant = dot ** 2 - speed2 * offset
            side_ok = np.where(parallel, offset <= 0, discriminant > 0)
            root = np.sqrt(np.maximum(discriminant, 0))
            denominator = np.where(parallel, 1, speed2)
            lower = np.where(parallel, -np.inf, (-dot - root) / denominator)
            upper = np.where(parallel, np.inf, (-dot + root) / denominator)
            axial_position = np.einsum("bsi,si->bs", displacement, axis)
            axial_speed = c[i:i + batch] @ axis.T
            stationary = np.abs(axial_speed) <= 1e-12
            cap_denominator = np.where(stationary, 1, axial_speed)
            half = geom.halflen[None, j:j + straw_block]
            cap_a = (-half - axial_position) / cap_denominator
            cap_b = (half - axial_position) / cap_denominator
            cap_low = np.where(stationary, -np.inf, np.minimum(cap_a, cap_b))
            cap_high = np.where(stationary, np.inf, np.maximum(cap_a, cap_b))
            cap_ok = ~stationary | (np.abs(axial_position) <= half)
            entry = np.maximum(np.maximum(lower, cap_low), 0)
            exit = np.minimum(np.minimum(upper, cap_high), length[i:i + batch, None])
            lanes, straws = np.nonzero(side_ok & cap_ok & (exit > entry))
            out.append(np.column_stack((lanes + i, straws + j, entry[lanes, straws], exit[lanes, straws])))
    rows = np.concatenate(out) if out else np.empty((0, 4))
    return canonical(rows, length, geom.nstraws, "C")


def compare_rows(left, right, names):
    """Outer-join the complete straw sets and compare both interval endpoints with absolute tolerance only."""
    cols = ["lane", "straw_index", "s_in", "s_out"]
    a, b = pd.DataFrame(left, columns=cols), pd.DataFrame(right, columns=cols)
    for frame in (a, b):
        frame[["lane", "straw_index"]] = frame[["lane", "straw_index"]].astype(np.int64)
    d = a.merge(b, on=["lane", "straw_index"], how="outer", suffixes=("_left", "_right"),
                indicator=True, validate="one_to_one")
    interval = (d._merge == "both") & (
        (np.abs(d.s_in_left - d.s_in_right) > ATOL_MM) | (np.abs(d.s_out_left - d.s_out_right) > ATOL_MM))
    d["kind"] = np.select([d._merge == "left_only", d._merge == "right_only", interval],
                          [f"missing_{names[1]}", f"missing_{names[0]}", "interval"], default="equal")
    return d[d.kind != "equal"].drop(columns="_merge").sort_values(["lane", "straw_index"]).reset_index(drop=True)


def _particles(frame, label):
    p = track_keys(frame, label)[TRACK]
    if p.empty or p.duplicated(TRACK).any():
        raise ValueError(f"empty or duplicate entry particle identities: {label}")
    return p.sort_values(TRACK).reset_index(drop=True)


def verify_particles(particles):
    """Require the identical unique complete saved entry-particle denominator in all three arms."""
    if set(particles) != set(FILES):
        raise ValueError("all three particle tables required")
    normalized = {label: _particles(p, label) for label, p in particles.items()}
    den = normalized["fast"]
    for label, p in normalized.items():
        if not den.equals(p):
            raise ValueError(f"different entry particle identities: {label}")
    return den


def _steps(frame, particle_index, straw_index, label):
    s = track_keys(frame, label).reset_index(drop=True)
    if not pd.MultiIndex.from_frame(s[TRACK]).isin(particle_index).all():
        raise ValueError(f"step particle outside entry denominator: {label}")
    if (s[STRAW].isna().any().any() or (s[STRAW] < 0).any().any()
            or not all(pd.api.types.is_integer_dtype(s[k].dtype) for k in STRAW)):
        raise ValueError(f"invalid current straw identities: {label}")
    if not pd.MultiIndex.from_frame(s[STRAW]).isin(straw_index).all():
        raise ValueError(f"foreign current straw identities: {label}")
    return s


def audit_arm(batches, denominator, geom, origin, oracle_segments=256, chord_index=None, panel_index=None,
              example_limit=EXAMPLES):
    """Audit row batches in saved order, retaining only bounded examples and aggregate counts."""
    _positive_integer(oracle_segments, "oracle segments", allow_zero=True)
    _positive_integer(example_limit, "example limit", allow_zero=True)
    chord_index = fs.ChordIndex(geom) if chord_index is None else chord_index
    panel_index = crossings.panel_index(geom) if panel_index is None else panel_index
    particle_index = pd.MultiIndex.from_frame(_particles(denominator, "audit denominator"))
    straw_index = pd.MultiIndex.from_frame(pd.DataFrame({k: getattr(geom, k) for k in STRAW}))
    result = dict(rows=0, zero_length=0, secants=0, nonfinite_rejected=0,
                  nonfinite_policy="reject entire audit; never drop",
                  oracle_secants=0, crossings=dict(A=0, B=0, C=0), comparisons={}, examples=[])
    for pair in ("A_B", "A_C", "B_C"):
        result["comparisons"][pair] = dict(mismatches=0, segments=0, kinds={})
    for frame in batches:
        s = _steps(frame, particle_index, straw_index, "saved step batch")
        x, c, length, keep = endpoint_secants(s[["x0", "y0", "z0"]], s[["x1", "y1", "z1"]], origin)
        ids = s.loc[keep, TRACK + STRAW].copy()
        ids["segment_row"] = result["rows"] + np.flatnonzero(keep)
        ids["lane"] = np.arange(len(ids))
        ids = ids.rename(columns={k: f"current_{k}" for k in STRAW})
        result["rows"] += len(s)
        result["zero_length"] += int((~keep).sum())
        result["secants"] += len(x)
        a = canonical(chord_index.crossings(x, c, length, geom), length, geom.nstraws, "A")
        b = canonical(crossings.straw_crossings(x, c, geom, index=panel_index), length, geom.nstraws, "B")
        n = min(oracle_segments - result["oracle_secants"], len(x))
        oracle = oracle_crossings(x[:n], c[:n], length[:n], geom)
        result["oracle_secants"] += n
        for name, rows in (("A", a), ("B", b), ("C", oracle)):
            result["crossings"][name] += len(rows)
        batch_examples = []
        for pair, left, right, count in (("A_B", a, b, len(x)), ("A_C", a[a[:, 0] < n], oracle, n),
                                         ("B_C", b[b[:, 0] < n], oracle, n)):
            mismatches = compare_rows(left, right, pair.split("_"))
            stat = result["comparisons"][pair]
            stat["segments"] += count
            stat["mismatches"] += len(mismatches)
            for kind, total in mismatches.kind.value_counts().items():
                stat["kinds"][kind] = stat["kinds"].get(kind, 0) + int(total)
            room = example_limit - len(result["examples"])
            if room > 0 and len(mismatches):
                examples = mismatches.head(room).merge(ids, on="lane", validate="many_to_one")
                straw_keys = pd.DataFrame({"straw_index": np.arange(geom.nstraws),
                                           **{f"crossed_{k}": getattr(geom, k) for k in STRAW}})
                examples = examples.merge(straw_keys, on="straw_index", validate="many_to_one").assign(comparison=pair)
                batch_examples.append(examples)
        if batch_examples:
            examples = pd.concat(batch_examples, ignore_index=True).sort_values(["segment_row", "comparison", "straw_index"])
            examples = examples.head(example_limit - len(result["examples"]))
            result["examples"].extend(json.loads(examples.drop(columns="lane").to_json(orient="records", double_precision=15)))
    result["total_mismatches"] = sum(v["mismatches"] for v in result["comparisons"].values())
    result["agreement"] = result["total_mismatches"] == 0
    return result


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(field_dir, gdml, outdir, chunk=512, oracle_segments=256):
    """Read only closed saved inputs and write a small report into an exclusively fresh directory."""
    _positive_integer(chunk, "chunk")
    _positive_integer(oracle_segments, "oracle segments", allow_zero=True)
    field_dir, gdml, outdir = Path(field_dir), Path(gdml), Path(outdir)
    paths = {"fieldcompare": field_dir / "fieldcompare.json", "gdml": gdml,
             **{label: field_dir / filename for label, filename in FILES.items()}}
    hashes = {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in paths.items()}
    provenance = json.loads(paths["fieldcompare"].read_text())
    origin = np.asarray(provenance["origin_mm"], float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("invalid saved origin_mm")
    denominator = verify_particles({label: tables.read(paths[label], "particles", tables.KEY + ["key"])
                                   .rename(columns={"key": "sp"}) for label in FILES})
    if "entries" in provenance and provenance["entries"] != len(denominator):
        raise ValueError("fieldcompare entries disagree with complete saved particle denominator")
    geom = geometry.from_gdml(str(gdml))
    chord_index, panel_index = fs.ChordIndex(geom), crossings.panel_index(geom)
    outdir.mkdir(parents=True, exist_ok=False)
    result = dict(coverage=COVERAGE, origin_mm=origin.tolist(), inputs=hashes, chunk=chunk,
                  oracle_segments_requested=oracle_segments, interval_atol_mm=ATOL_MM, interval_rtol=0,
                  denominator_particles=len(denominator),
                  particle_keys_sha256=hashlib.sha256(denominator[TRACK].to_numpy(dtype="<u8").tobytes()).hexdigest(),
                  event_keys_sha256=cecontract.fingerprint(denominator[tables.KEY].drop_duplicates()),
                  oracle_selection="first min(oracle_segments, nonzero finite secants) per arm in saved row order",
                  mismatch_unit="one lane/straw disagreement per comparison; not unique segments across comparisons",
                  verdict="report-only search agreement; no count correction, reconstruction or physics closure", arms={})
    for label in FILES:
        with uproot.open(paths[label]) as root:
            tree_name = f"{tables.DIR}/sgsteps"
            if tree_name in root:
                batches = root[tree_name].iterate(COLUMNS, step_size=chunk, library="pd")
            else:
                batches = []
            room = EXAMPLES - sum(len(arm["examples"]) for arm in result["arms"].values())
            result["arms"][label] = audit_arm(batches, denominator, geom, origin, oracle_segments, chord_index, panel_index, room)
            result["arms"][label]["saved_step_tree_present"] = tree_name in root
    result["total_mismatches"] = sum(arm["total_mismatches"] for arm in result["arms"].values())
    result["agreement"] = result["total_mismatches"] == 0
    if any(_sha256(paths[name]) != receipt["sha256"] for name, receipt in hashes.items()):
        raise ValueError("saved inputs changed during crossing audit")
    with (outdir / "fieldcrossing.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field-dir", required=True)
    ap.add_argument("--gdml", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--oracle-segments", type=int, default=256)
    args = ap.parse_args(argv)
    result = run(args.field_dir, args.gdml, args.outdir, args.chunk, args.oracle_segments)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
