"""Paired primary-filter census on a completed bounded smoke and saved G4 steps."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.transport_capture_audit import digest
from scripts.ordered_g4_state_compare import check_generation
from trkfastsim import tables
from trkfastsim.analyses.ratearms import _paired


def primary_flags(steps, particles):
    primary = particles[(particles.isprim == 1) & (particles.pdg.abs() == 13)][tables.KEY + ["key"]]
    if primary.duplicated(tables.KEY + ["key"]).any():
        raise ValueError("duplicate primary particle identity")
    selected = steps.merge(primary.rename(columns={"key": "sp"}), on=tables.KEY + ["sp"], validate="many_to_one")
    decision = tables.trk_filter(selected).reset_index()
    return decision.loc[decision.pass_trk, tables.KEY]


def run(directory, g4, out):
    directory, g4 = Path(directory), Path(g4)
    sources = [directory / "complete.json", directory / "ordered_steps.root", g4]
    before = {str(p): digest(p) for p in sources}
    report = json.loads(sources[0].read_text())
    if report["status"] != "bounded_full_chain_complete":
        raise ValueError("smoke did not complete")
    gen = tables.read(sources[1], "gen")
    events = gen[tables.KEY].drop_duplicates()
    if len(events) != report["generated_events"]:
        raise ValueError("full generated-event coverage mismatch")
    generation_check = check_generation(gen, tables.read(g4, "gen"))
    if generation_check["reference_events"] == 0:
        raise ValueError("G4 reference has no matching generated events")
    decisions = []
    for path in (sources[1], g4):
        decision = primary_flags(tables.read(path, "sgsteps"), tables.read(path, "particles"))
        decisions.append(pd.MultiIndex.from_frame(events).isin(pd.MultiIndex.from_frame(decision)))
    paired = _paired(*decisions)
    result = dict(scope="bounded saved-sample primary-filter comparison; not precision rate closure",
        generated_events=len(events), navigation=report["navigation"], paired=paired, generation_check=generation_check,
        discordant_normal_z=(paired["diff"] / np.sqrt(paired["arm_only"] + paired["ref_only"])
                             if paired["arm_only"] + paired["ref_only"] else None), input_sha256=before)
    if before != {str(p): digest(p) for p in sources}:
        raise ValueError("input changed")
    with Path(out).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--g4", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.directory, args.g4, args.out)
