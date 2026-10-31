"""Gate complete source-prefix generator identity and primary keys before stock table injection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mu2efastsim.tables import all_primary_muons
from scripts.transport_capture_audit import digest
from trkfastsim import tables


def validate(generated, source, particles, straw, surface, *, expected_events):
    if not isinstance(expected_events, int) or expected_events <= 0:
        raise ValueError("explicit independently requested source prefix count required")
    for frame in (generated, source):
        if len(frame[tables.KEY].drop_duplicates()) != expected_events:
            raise ValueError("generator prefix event count differs from requested source bound")
    identity = tables.KEY + ["idx"]
    if generated.duplicated(identity).any() or source.duplicated(identity).any():
        raise ValueError("duplicate generator identity")
    cols = identity + ["pdg", "x", "y", "z", "px", "py", "pz", "e", "t"]
    pd.testing.assert_frame_equal(generated[cols].reset_index(drop=True), source[cols].reset_index(drop=True),
                                  check_exact=True, check_dtype=False)
    expected = all_primary_muons(generated)[tables.KEY + ["mukey", "pdg"]].rename(columns={"mukey": "key"})
    key = tables.KEY + ["key"]
    if particles.duplicated(key).any():
        raise ValueError("duplicate table primary identity")
    pd.testing.assert_frame_equal(expected.sort_values(key).reset_index(drop=True),
        particles[key + ["pdg"]].sort_values(key).reset_index(drop=True), check_exact=True, check_dtype=False)
    if not ((particles.parent == -1) & (particles.isprim == 1)).all():
        raise ValueError("table particles are not parentless primaries")
    for frame in (straw, surface):
        linked = frame[tables.KEY + ["sp"]].merge(particles[key], left_on=tables.KEY + ["sp"], right_on=key,
                                                 how="left", validate="many_to_one", indicator=True)
        if (linked._merge != "both").any():
            raise ValueError("unlinked step/surface primary identity")
    keys = generated[tables.KEY].drop_duplicates()
    selected = tables.trk_filter(straw).reset_index()
    return dict(generated_events=len(keys), generated_particles=len(generated), primary_muons=len(particles),
        straw_rows=len(straw), surface_rows=len(surface), tracker_filter_passes=int(selected.pass_trk.sum()),
        generator_prefix_exact=True, all_primary_and_step_keys_valid=True,
        primary_link_policy="existing producer same-PDG generator ordering fallback; no genidx or remapping introduced")


def run(table, source_dump, out, *, expected_events, source_log, source_art, trusted_source_sha256):
    paths = list(map(Path, (table, source_dump, source_log, source_art)))
    before = {str(p): digest(p) for p in paths}
    if before[str(paths[3])] != trusted_source_sha256:
        raise ValueError("source art differs from independently trusted staged input")
    text = paths[2].read_text()
    if f"TrigReport Events total = {expected_events} passed = {expected_events} failed = 0" not in text or "Art has completed and will exit with status 0." not in text:
        raise ValueError("source dumper did not finish the requested complete prefix")
    result = validate(tables.read(table, "gen"), tables.read(source_dump, "gen"), tables.read(table, "particles"),
                      tables.read(table, "sgsteps"), tables.read(table, "surfsteps"), expected_events=expected_events)
    result["source_acquisition"] = dict(expected_events=expected_events, completed_art_source_log=True,
                                        trusted_source_art_sha256=trusted_source_sha256)
    result.update(input_sha256=before, scope="source-prefix and injection identity gate, not stock response validation",
                  source_sha256={str(Path(__file__)): digest(__file__)})
    if before != {str(p): digest(p) for p in paths}:
        raise ValueError("stock injection input changed")
    with Path(out).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table")
    parser.add_argument("--source-dump", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-events", required=True, type=int)
    parser.add_argument("--source-log", required=True)
    parser.add_argument("--source-art", required=True)
    parser.add_argument("--trusted-source-sha256", required=True)
    args = parser.parse_args()
    run(args.table, args.source_dump, args.out, expected_events=args.expected_events, source_log=args.source_log,
        source_art=args.source_art, trusted_source_sha256=args.trusted_source_sha256)
