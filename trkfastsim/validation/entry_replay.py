"""Prepare event-keyed cosmic tracker entries and verify the art reader's optional readback."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from mu2efastsim.tables import KEY
from trkfastsim import geometry, tables
from trkfastsim.sim.fieldsteps import Envelope
from trkfastsim.validation import fieldcompare

STATE = ["x", "y", "z", "px", "py", "pz", "t"]
EVENT_TYPES = {**dict.fromkeys(KEY, "uint32"), "n_particles": "uint32"}
PARTICLE_TYPES = {**dict.fromkeys(KEY, "uint32"), "index": "uint32", "source_sp": "int64",
                  "pdg": "int32", **dict.fromkeys(STATE, "float64")}


def integers(frame, names, dtype, minimum=None, maximum=None):
    """Convert integer columns without truncation, float mediation or unsigned wrapping."""
    info = np.iinfo(dtype)
    lo, hi = info.min if minimum is None else minimum, info.max if maximum is None else maximum
    for name in names:
        col = frame[name]
        if not pd.api.types.is_integer_dtype(col.dtype) or col.isna().any():
            raise ValueError(f"{name} must contain integers")
        if ((col < lo) | (col > hi)).any():
            raise ValueError(f"{name} outside {dtype} range")
        frame[name] = col.astype(dtype)


def event_keys(frame):
    out = frame.copy()
    integers(out, KEY, "uint32", 0, np.iinfo(np.uint32).max - 1)
    if (out[["run", "event"]] == 0).any().any():
        raise ValueError("run and event must be positive")
    return out


def canonical(events, particles):
    """Validate explicit empty events and deterministic per-event particle identity/order."""
    ev = event_keys(events[list(EVENT_TYPES)])
    pt = event_keys(particles[list(PARTICLE_TYPES)])
    integers(ev, ["n_particles"], "uint32")
    integers(pt, ["index"], "uint32")
    integers(pt, ["source_sp"], "int64")
    integers(pt, ["pdg"], "int32")
    if ev.duplicated(KEY).any():
        raise ValueError("duplicate event keys")
    if pt.duplicated(KEY + ["index"]).any() or pt.duplicated(KEY + ["source_sp"]).any():
        raise ValueError("duplicate particle identity")
    if not pt.pdg.isin([-13, 13]).all():
        raise ValueError("entry replay supports muons only")
    pt[STATE] = pt[STATE].astype(np.float64)
    if not np.isfinite(pt[STATE].to_numpy()).all():
        raise ValueError("nonfinite particle state")
    momentum = np.linalg.norm(pt[["px", "py", "pz"]].to_numpy(), axis=1)
    if ((momentum <= 0) | ~np.isfinite(momentum)).any():
        raise ValueError("nonpositive or nonfinite momentum")
    pt = pt.sort_values(KEY + ["index"]).reset_index(drop=True)
    if not np.array_equal(pt["index"], pt.groupby(KEY).cumcount().to_numpy()):
        raise ValueError("particle index must be contiguous from zero")
    counts = pt.groupby(KEY).size().rename("actual").reset_index()
    matched = ev.merge(counts, on=KEY, how="outer", indicator=True, validate="one_to_one")
    if (matched._merge == "right_only").any():
        raise ValueError("particle event missing from event census")
    if not np.array_equal(matched.n_particles, matched.actual.fillna(0)):
        raise ValueError("particle counts differ from event census")
    return ev.sort_values(KEY).reset_index(drop=True), pt


def write(path, events, particles, directory="entry"):
    ev, pt = canonical(events, particles)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as root:
        for name, frame, schema in (("events", ev, EVENT_TYPES), ("particles", pt, PARTICLE_TYPES)):
            tree = root.mktree(f"{directory}/{name}", schema)
            if len(frame):
                tree.extend({key: frame[key].to_numpy() for key in schema})


def read(path, directory="entry"):
    with uproot.open(path) as root:
        frames = []
        for name, schema in (("events", EVENT_TYPES), ("particles", PARTICLE_TYPES)):
            tree = root[f"{directory}/{name}"]
            if tree.classname != "TTree":
                raise ValueError(f"{directory}/{name} must be a TTree")
            cols = tree.arrays(list(schema), library="np")
            if any(cols[key].dtype != np.dtype(dtype) for key, dtype in schema.items()):
                raise ValueError(f"unexpected {name} branch dtype")
            frames.append(pd.DataFrame(cols))
    return canonical(*frames)


def select_entries(vd, surf, origin):
    """The saved S56 earliest-VD-or-surface selector, without a straw-response selection."""
    cols = KEY + ["sp", "pdg"] + STATE + ["vdid", "x0", "y0", "z0", "x1", "y1", "z1"]
    ent = pd.concat([fieldcompare.entry_states(vd).reindex(columns=cols),
                     fieldcompare.entry_states_surface(surf, origin).reindex(columns=cols)])
    return ent.sort_values("t").groupby(KEY + ["sp"], sort=False).first().reset_index()


def prepare(input_table, gdml, origin, outdir, source_art=None):
    """Write the complete source-event census and variable-multiplicity tracker-entry table."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    events = tables.read(input_table, "gen", KEY).drop_duplicates().reset_index(drop=True)
    _, _, vd, surf, _ = fieldcompare.primaries([str(input_table)], 13)
    ent = select_entries(vd, surf, np.asarray(origin, float))
    pt = ent[KEY + ["sp", "pdg"] + STATE].rename(columns={"sp": "source_sp"})
    pt = pt.sort_values(KEY + ["source_sp"]).reset_index(drop=True)
    pt["index"] = pt.groupby(KEY).cumcount().astype("uint32")
    counts = pt.groupby(KEY).size().rename("n_particles").reset_index()
    ev = events.merge(counts, on=KEY, how="left", validate="one_to_one")
    ev["n_particles"] = ev.n_particles.fillna(0).astype("uint32")
    ev, pt = canonical(ev, pt)
    source_count = None
    if source_art is not None:
        with uproot.open(source_art) as root:
            source_count = root["Events"].num_entries
        if source_count != len(ev):
            raise ValueError("source art Events count differs from generated-event keys")
    geom = geometry.from_gdml(gdml)
    env = Envelope(geom, axis_xy=np.asarray(origin)[:2])
    write(out / "entries.root", ev, pt)
    cut = dict(type="outwardCylinder", axisXY=env.axis.tolist(), radius=env.r, zMin=env.z0, zMax=env.z1)
    report = dict(input_table=str(input_table), source_art=str(source_art) if source_art is not None else None,
                  source_art_entries=source_count, source_events=len(ev), entry_particles=len(pt),
                  entry_events=int((ev.n_particles > 0).sum()), empty_events=int((ev.n_particles == 0).sum()),
                  multi_particle_events=int((ev.n_particles > 1).sum()), origin_mm=list(origin), gdml=str(gdml),
                  envelope=cut, scope="S2-selected source events; not generated CRY", source_sp="signed key as persisted in the saved dumper table")
    (out / "entries.json").write_text(json.dumps(report, indent=2) + "\n")
    text = ("physics.producers.g4run.Mu2eG4CommonCut: {\n"
            "  type: outwardCylinder\n"
            f"  axisXY: [{env.axis[0]:.17g}, {env.axis[1]:.17g}]\n"
            f"  radius: {env.r:.17g}\n  zMin: {env.z0:.17g}\n  zMax: {env.z1:.17g}\n}}\n")
    (out / "envelope.fcl").write_text(text)
    print(json.dumps(report, indent=2))
    return report


def check(expected, observed, directory="entryReplay", source_art=None):
    """Compare actual art readback, including empty/multi-particle events, at exact precision."""
    want = read(expected)
    got = read(observed, directory)
    for left, right in zip(want, got):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    report = dict(events=len(want[0]), particles=len(want[1]), exact=True)
    if source_art is not None:
        with uproot.open(source_art) as root:
            count = root["Events"].num_entries
        if count != len(got[0]):
            raise ValueError("source art Events count differs from replayed events")
        report.update(source_art_entries=count, empty_events=int((got[0].n_particles == 0).sum()),
                      entry_events=int((got[0].n_particles > 0).sum()),
                      multi_particle_events=int((got[0].n_particles > 1).sum()))
    return report


def smoke(outdir):
    """Synthetic input for EmptyEvent replay tests; no Geant4 transport is involved."""
    out = Path(outdir)
    ev = pd.DataFrame(dict(run=[1, 1, 1], subrun=[0, 0, 0], event=[1, 2, 3], n_particles=[0, 2, 1]))
    pt = pd.DataFrame(dict(run=[1, 1, 1], subrun=[0, 0, 0], event=[2, 2, 3], index=[0, 1, 0],
                           source_sp=[-2147483647, 2**53 + 7, 9], pdg=[13, -13, 13],
                           x=[-3904., -3903., -3904.], y=[0., 1., 2.], z=[9000., 9100., 9200.],
                           px=[1., 2., 3.], py=[4., 5., 6.], pz=[1000., 2000., 3000.], t=[0., 2.5, 3.5]))
    write(out / "entries.root", ev, pt)
    bad = ev.copy()
    bad.loc[0, "n_particles"] = 1
    out.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(out / "bad-count.root") as root:
        for name, frame, schema in (("events", bad, EVENT_TYPES), ("particles", pt, PARTICLE_TYPES)):
            root.mktree(f"entry/{name}", schema).extend({key: frame[key].to_numpy(dtype=dtype)
                                                        for key, dtype in schema.items()})
    return dict(events=len(ev), particles=len(pt), directory=str(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("input_table", type=Path)
    p.add_argument("--gdml", type=Path, required=True)
    p.add_argument("--origin", type=float, nargs=3, required=True)
    p.add_argument("--source-art", type=Path)
    p.add_argument("--outdir", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("expected", type=Path)
    c.add_argument("observed", type=Path)
    c.add_argument("--directory", default="entryReplay")
    c.add_argument("--source-art", type=Path)
    s = sub.add_parser("smoke")
    s.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.command == "prepare":
        prepare(a.input_table, a.gdml, a.origin, a.outdir, a.source_art)
    elif a.command == "check":
        print(json.dumps(check(a.expected, a.observed, a.directory, a.source_art), indent=2))
    else:
        print(json.dumps(smoke(a.outdir), indent=2))


if __name__ == "__main__":
    main()
