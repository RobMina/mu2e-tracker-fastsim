"""Check a CE table's generated-event denominator and primary-identity/VD13 coverage before transport.

An external completed-art count and successful transport-status audit are still required; this table cannot certify
lossless precision or full incident-particle retention. Tracker steps without VD13 are reported, never silently
counted as no-response events for an unrestricted tracker-efficiency claim.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from mu2efastsim import tables
from trkfastsim.validation import ceaudit, fieldce

KEY = tables.KEY


def census(paths, expected_events, audited_events=None):
    g = ceaudit.unique(fieldce.starts(paths), "CE starts")
    if len(g) != expected_events:
        raise ValueError(f"CE starts {len(g)} != externally audited Events {expected_events}")
    if audited_events is not None:
        audited = ceaudit.unique(audited_events[KEY], "closed reference census")
        pd.testing.assert_frame_equal(g[KEY], audited[KEY], check_exact=True)
    p = tables.read_many(paths, "particles")
    p = p[(p.pdg == 11) & ((p.isprim == 1) | (p.ismcprim == 1))].rename(columns={"key": "sp"})
    p = ceaudit.unique(p, "primary electron identities")
    pd.testing.assert_frame_equal(g[KEY], p[KEY])
    v = ceaudit.unique(fieldce.primary_entries(paths), "G4 VD13 entries")
    s = ceaudit.integer_keys(tables.read_many(paths, "sgsteps"), "G4 steps")
    primary = s.merge(p[KEY + ["sp"]], on=KEY + ["sp"], validate="many_to_one")
    n = primary.groupby(KEY).size().rename("primary_steps").reset_index()
    d = g[KEY].merge(v[KEY].assign(vd13=True), on=KEY, how="outer", validate="one_to_one", indicator=True)
    if (d._merge == "right_only").any():
        raise ValueError("VD13 event outside generated denominator")
    d = d.drop(columns="_merge").merge(n, on=KEY, how="left", validate="one_to_one")
    d["vd13"] = d.vd13.fillna(False).astype(bool)
    d["primary_steps"] = d.primary_steps.fillna(0).astype(int)
    after = primary.merge(v[KEY + ["t"]].rename(columns={"t": "entry_t"}), on=KEY, validate="many_to_one")
    before = int((after.t < after.entry_t - 1e-6).sum())
    report = {"generated_starts": len(g), "external_expected_events": expected_events, "primary_electrons": len(p),
              "primary_vd13": len(v), "primary_steps": len(primary),
              "primary_step_events": int((d.primary_steps > 0).sum()),
              "primary_step_events_without_vd13": int(((d.primary_steps > 0) & ~d.vd13).sum()),
              "steps_before_first_vd13_in_crossing_events": before,
              "all_steps": len(s), "other_particle_steps": len(s) - len(primary),
              "birth_keys_sha256": fieldce.cecontract.fingerprint(g),
              "closed_census_keys_exact": audited_events is not None,
              "scope": "one CE start per audited event; integer identity coverage only, not lossless retention certification"}
    return report, d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--expected-events", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--census-events", help="closed cecensus events.csv.gz for exact table identity binding")
    a = ap.parse_args(argv)
    audited = pd.read_csv(a.census_events, dtype={k: "uint64" for k in KEY}) if a.census_events else None
    report, events = census(tables.expand(a.inputs), a.expected_events, audited)
    if a.census_events:
        from trkfastsim.validation.cestopwindow import checksum
        report["census_events_sha256"] = checksum(a.census_events)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ceinput.json").write_text(json.dumps(report, indent=2) + "\n")
    events.to_csv(out / "events.csv.gz", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
