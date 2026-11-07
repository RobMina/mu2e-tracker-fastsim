"""Current-main NumPy tracker transport of the complete typed entry-replay census."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import uproot

from mu2efastsim import ehelix, tables
from mu2efastsim.tables import KEY
from trkfastsim import geometry
from trkfastsim.sim import fieldsteps as fs
from trkfastsim.validation import entry_replay as er

ORIGIN = (-3904.0, 0.0, 10170.999666398546)
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
STEP_INTS = KEY + ["nsgs", "sid", "plane", "panel", "straw", "upanel", "shape", "ion", "sp"]
STEP_FLOATS = ["eion", "len", "width", "t", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"]
STEP_TYPES = {**dict.fromkeys(STEP_INTS, "int64"), **dict.fromkeys(STEP_FLOATS, "float64")}


def load_field(header):
    """Use fieldcompare's stock G4BL/trilinear route, rejecting out-of-map evaluations."""
    header = str(header)
    if Path(header).name != "DSMap.header":
        raise ValueError("--map must name the stock BFieldMaps/Mau13/DSMap.header")
    lo, spacing, shape, grid = ehelix.read_g4bl(header)
    hi = lo + spacing * (shape - 1)
    interpolate = ehelix.grid_field(lo, spacing, shape, grid)

    def field(x):
        x = np.asarray(x, float)
        if not np.isfinite(x).all() or np.any((x < lo) | (x >= hi)):
            raise ValueError("transport queried outside the DS map; a matched field is not established")
        return interpolate(x)

    info = {"configuration": "bfgeom_no_tsu_ps_v01.txt", "map": str(Path(header).resolve()),
            "binary": str(Path(header).with_suffix(".bin").resolve()), "scale": 1.0,
            "interpolation": "trilinear", "lo_mm": lo.tolist(), "hi_exclusive_mm": hi.tolist(),
            "outside_policy": "error; no WorldMap/TSd/DSExtension fallback"}
    return field, info


def injection_tables(events, particles, fast, final):
    """Vectorized keyed export; preserve all primaries, including those producing no gas steps."""
    identity = KEY + ["sp"]
    mapping = particles[KEY + ["index", "source_sp"]].copy()
    mapping["sp"] = particles["index"].to_numpy(dtype=np.int64) + 1
    expected = mapping[identity]
    if final.duplicated(identity).any() or len(final) != len(mapping):
        raise ValueError("final states must cover every entry exactly once")
    aligned = expected.merge(final, on=identity, how="outer", validate="one_to_one", indicator=True)
    if not aligned._merge.eq("both").all():
        raise ValueError("final states have missing or foreign primary identities")
    if not aligned.status.isin([0, 1, 2]).all():
        raise ValueError("unknown final transport status")
    if not np.isfinite(aligned[er.STATE + ["path", "chords"]].to_numpy()).all():
        raise ValueError("nonfinite final transport state")
    aligned = aligned.drop(columns="_merge")
    fast = fast[list(STEP_TYPES)].astype(STEP_TYPES).copy()
    known = fast[identity].merge(expected, on=identity, how="left", validate="many_to_one", indicator=True)
    if not known._merge.eq("both").all():
        raise ValueError("gas step has unknown primary identity")
    if not np.isfinite(fast[STEP_FLOATS].to_numpy()).all():
        raise ValueError("nonfinite gas step")
    fast["nsgs"] = fast.groupby(KEY).sid.transform("size").astype("int64")
    parts = particles[KEY + ["pdg"]].copy()
    parts["key"] = mapping.sp.to_numpy()
    parts["genidx"] = particles["index"].to_numpy(dtype=np.int64)
    parts = parts.assign(parent=np.int64(-1), isprim=np.int64(1), ismcprim=np.int64(1))
    for state, name in ((particles, "s"), (aligned, "e")):
        for column in er.STATE:
            parts[name + column] = state[column].to_numpy(dtype=np.float64)
    counts = fast.groupby(KEY).size().rename("nsgs").reset_index()
    census = events.merge(counts, on=KEY, how="left", validate="one_to_one")
    census["nsgs"] = census.nsgs.fillna(0).astype("int64")
    ends = mapping.merge(aligned, on=identity, validate="one_to_one")
    return {"sgsteps": fast, "particles": parts, "events": census, "entry_map": mapping, "final": ends}


def write_tables(path, frames):
    """Reuse the injector TTree exporter and supply schemas for its skipped empty frames."""
    tables.write(str(path), frames, as_ttree=True)
    with uproot.update(path) as root:
        for name, frame in frames.items():
            if frame.empty:
                root.mktree(f"dump/{name}", {column: frame[column].dtype for column in frame})


def run(entries, *, gdml, field_map, origin, outdir, metadata=None, seed=1,
        expected_events=52037, expected_particles=43156):
    """Transport every prepared entry once; never read a Geant4 response table."""
    if any(os.environ.get(name) != "1" for name in THREAD_VARS):
        raise ValueError("set all single-thread variables to 1 before Python starts: " + ", ".join(THREAD_VARS))
    origin = np.asarray(origin, dtype=np.float64)
    if not np.array_equal(origin, ORIGIN):
        raise ValueError("origin must equal the saved benchmark origin, without refitting")
    events, particles = er.read(entries)
    if len(events) != expected_events or len(particles) != expected_particles:
        raise ValueError("prepared event/entry census differs from expected counts")
    geom = geometry.from_gdml(gdml)
    envelope = fs.Envelope(geom, axis_xy=origin[:2])
    boundary = dict(type="outwardCylinder", axisXY=envelope.axis.tolist(), radius=envelope.r,
                    zMin=envelope.z0, zMax=envelope.z1)
    if metadata is not None:
        prepared = json.loads(Path(metadata).read_text())
        if prepared["origin_mm"] != origin.tolist() or prepared["envelope"] != boundary:
            raise ValueError("prepared origin/envelope differs from fast geometry")
    field, field_info = load_field(field_map)
    support = fs.interior(gdml, geom, envelope, origin=origin)
    bnom = ehelix.field_at(field, origin[None])[0]
    start = particles[KEY + ["pdg"] + er.STATE].copy()
    start["sp"] = particles["index"].to_numpy(dtype=np.int64) + 1
    rng = np.random.default_rng(seed)
    begin = time.perf_counter()
    fast, final = fs.transport(rng, start, geom, field, support, envelope=envelope,
                               origin=origin, bnom=bnom, complete_outer_wall=False)
    transport_s = time.perf_counter() - begin
    frames = injection_tables(events, particles, fast, final)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    write_tables(out / "fasttrk_field.root", frames)
    counts = final.status.value_counts()
    report = dict(source_events=len(events), entry_particles=len(particles),
                  empty_entry_events=int((events.n_particles == 0).sum()),
                  multi_entry_events=int((events.n_particles > 1).sum()),
                  gas_steps=len(fast), events_with_gas_steps=int((frames["events"].nsgs > 0).sum()),
                  zero_hit_primaries=int(len(particles) - len(fast[KEY + ["sp"]].drop_duplicates())),
                  final_status=dict(exited=int(counts.get(0, 0)), stopped=int(counts.get(1, 0)),
                                    capped=int(counts.get(2, 0))),
                  transport_s=transport_s, transport_timing_s=getattr(fs.transport, "timing", None),
                  backend="numpy", threads={name: os.environ[name] for name in THREAD_VARS},
                  seed=seed, complete_outer_wall=False, chord_mm=20.0, sagitta_mm=0.02,
                  max_path_mm=30000.0, max_chords=20000, origin_mm=origin.tolist(),
                  envelope=boundary, bnom_T=bnom.tolist(), field=field_info,
                  entries=str(Path(entries).resolve()), gdml=str(Path(gdml).resolve()),
                  scope="all prepared S2-selected events; primary muons only; no G4 response or time-window cut",
                  full_process_timing="external /usr/bin/time; not transport_s")
    (out / "entry-fast.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entries", help="prepared entries.root (typed entry/events and entry/particles)")
    parser.add_argument("--gdml", required=True, help="saved source geometry, not a fresh export")
    parser.add_argument("--map", dest="field_map", required=True, help="stock BFieldMaps/Mau13/DSMap.header (+ .bin)")
    parser.add_argument("--origin", nargs=3, type=float, required=True)
    parser.add_argument("--metadata", required=True, help="prepared entries.json: verify origin and envelope")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--expect-events", dest="expected_events", type=int, default=52037)
    parser.add_argument("--expect-particles", dest="expected_particles", type=int, default=43156)
    print(json.dumps(run(**vars(parser.parse_args(argv))), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
