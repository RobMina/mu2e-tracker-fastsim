"""Validate a bounded Geant4 entry pilot using the existing flat truth and replay TTrees."""
import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from mu2efastsim.tables import KEY
from trkfastsim import geometry
from trkfastsim.validation import entry_replay as er

KILLER_VOLUME = 54
MAX_STEPS = 55


def frame(path, tree, columns=None, entry_stop=None):
    with uproot.open(path) as root:
        return pd.DataFrame(root[tree].arrays(columns, entry_stop=entry_stop, library="np"))


def geometry_check(reference, observed):
    """Compare parsed straw geometry rather than GDML names containing memory addresses."""
    want, got = geometry.from_gdml(reference), geometry.from_gdml(observed)
    differences = {}
    for field in fields(want):
        left, right = getattr(want, field.name), getattr(got, field.name)
        if left is None or right is None:
            if left != right:
                raise ValueError(f"geometry field {field.name} differs")
            continue
        left, right = np.asarray(left), np.asarray(right)
        if left.shape != right.shape:
            raise ValueError(f"geometry shape {field.name} differs")
        differences[field.name] = float(np.max(np.abs(left - right)))
    return dict(straws=want.nstraws, max_abs_differences=differences,
                within_1e_minus_6=all(v <= 1.e-6 for v in differences.values()))


def endpoints(particles, cut):
    """Evaluate the strict step-end outward predicate; do not interpolate a crossing."""
    xyz = particles[["ex", "ey", "ez"]].to_numpy()
    mom = particles[["epx", "epy", "epz"]].to_numpy(dtype=float)
    if not np.isfinite(xyz).all() or not np.isfinite(mom).all():
        raise ValueError("nonfinite endpoint")
    radial = xyz[:, :2] - np.asarray(cut["axisXY"])
    r = np.linalg.norm(radial, axis=1)
    dr = r - cut["radius"]
    low, high = cut["zMin"] - xyz[:, 2], xyz[:, 2] - cut["zMax"]
    side = (dr > 0) & ((radial * mom[:, :2]).sum(axis=1) > 0)
    bottom = (low > 0) & (mom[:, 2] < 0)
    top = (high > 0) & (mom[:, 2] > 0)
    distance = np.linalg.norm(np.column_stack([np.maximum(dr, 0), np.maximum.reduce([low, high, np.zeros(len(low))])]), axis=1)
    return side | bottom | top, distance


def check(entries, coverage, pilot, pilot_art, metadata, reference_gdml, observed_gdml, events):
    ev, pt = er.read(pilot, "entryReplay")
    source = frame(coverage, "entryReplay/events", list(er.EVENT_TYPES), entry_stop=events)
    want_ev, want_pt = er.read(entries)
    selected = want_ev.merge(source[KEY], on=KEY, how="inner", validate="one_to_one")
    selected_pt = want_pt.merge(source[KEY], on=KEY, how="inner", validate="many_to_one")
    selected, selected_pt = er.canonical(selected, selected_pt)
    if len(source) != events:
        raise ValueError("coverage file has fewer than requested source events")
    pd.testing.assert_frame_equal(selected, ev, check_exact=True)
    pd.testing.assert_frame_equal(selected_pt, pt, check_exact=True)
    particles = frame(pilot, "dump/particles")
    if not (particles.isprim == 1).all() or not particles.pdg.isin([-13, 13]).all():
        raise ValueError("pilot includes transported non-primary-muon particles")
    joined = pt.merge(particles.rename(columns={"genidx": "index"}), on=KEY + ["index"],
                      how="outer", validate="one_to_one", indicator=True, suffixes=("", "_sim"))
    if not (joined._merge == "both").all():
        raise ValueError("GenParticle to SimParticle identity coverage differs")
    np.testing.assert_array_equal(joined.pdg, joined.pdg_sim)
    np.testing.assert_array_equal(joined[["x", "y", "z"]], joined[["sx", "sy", "sz"]])
    np.testing.assert_array_equal(joined[["px", "py", "pz", "t"]].to_numpy().astype(np.float32),
                                  joined[["spx", "spy", "spz", "st"]].to_numpy())
    cut = json.loads(Path(metadata).read_text())["envelope"]
    leaving, excess = endpoints(particles, cut)
    killed = particles.scode.to_numpy() == KILLER_VOLUME
    if not leaving[killed].all():
        raise ValueError("cylinder-killed primary does not satisfy outward predicate at its endpoint")
    boundary = frame(pilot, "dump/vdsteps")
    exits = particles.loc[killed, KEY + ["key"]].rename(columns={"key": "sp"})
    match = exits.merge(boundary[KEY + ["sp"]], on=KEY + ["sp"], how="outer", indicator=True, validate="one_to_one")
    if not (match._merge == "both").all():
        raise ValueError("boundary write product does not cover exactly the cylinder-killed primaries")
    steps = frame(pilot, "dump/sgsteps")
    smatch = steps.merge(particles[KEY + ["key"]].rename(columns={"key": "sp"}),
                        on=KEY + ["sp"], how="left", indicator=True, validate="many_to_one")
    if not (smatch._merge == "both").all():
        raise ValueError("straw steps reference a particle outside the new primary collection")
    with uproot.open(pilot_art) as root:
        tree = root["Events"]
        art_count = tree.num_entries
        branches = tree.keys(recursive=False)
    if art_count != len(ev):
        raise ValueError("pilot art highest Events cycle disagrees with readback census")
    products = [b for b in branches if "StrawGasStep" in b and "StrawGasStepMaker" in b and "TRK2EntryG4" in b]
    if not products:
        raise ValueError("new StrawGasStep product absent from pilot art")
    stopping = particles.scode.value_counts().sort_index()
    overshoot = excess[killed]
    return dict(source_events=len(ev), source_event_ids=ev[KEY].to_dict("records"),
                empty_events=int((ev.n_particles == 0).sum()), primary_muons=len(pt), replay_exact=True,
                start_states_match_at_dumper_precision=True, pilot_art_events=art_count,
                stopping_codes={str(k): int(v) for k, v in stopping.items()},
                cylinder_exits=int(killed.sum()), capped=int((particles.scode == MAX_STEPS).sum()),
                all_cylinder_exits_outward=True, boundary_hits=len(boundary),
                straw_steps=len(steps), straw_step_events=len(steps[KEY].drop_duplicates()),
                straw_step_primaries=len(steps[KEY + ["sp"]].drop_duplicates()), straw_products=products,
                endpoint_outside_distance_mm=dict(max=float(overshoot.max()) if len(overshoot) else None,
                                                 median=float(np.median(overshoot)) if len(overshoot) else None),
                tracker_geometry=geometry_check(reference_gdml, observed_gdml),
                scope="primary-only conditional entry pilot; not physics closure or a throughput measurement")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("entries", "coverage", "pilot", "pilot-art", "metadata", "reference-gdml", "observed-gdml", "output"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--events", type=int, required=True)
    args = vars(ap.parse_args(argv))
    output = args.pop("output")
    report = check(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
