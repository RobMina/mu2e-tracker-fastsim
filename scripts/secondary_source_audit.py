"""Count true primary steps entering a single-MC-primary secondary-template population.

Read saved G4 step tables only; no library is built or modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from trkfastsim import tables as T


def audit(steps, particles):
    p = particles.rename(columns={"key": "sp"})
    if p.duplicated(T.KEY + ["sp"]).any():
        raise ValueError("duplicate particle identities")
    anchors = p[(p.ismcprim == 1) & (p.pdg.abs() == 13)][T.KEY + ["sp"]].rename(columns={"sp": "anchor"})
    multiplicity = anchors.groupby(T.KEY).size()
    s = steps.assign(source_row=np.arange(len(steps))).merge(anchors, on=T.KEY, how="inner", validate="many_to_many")
    s = s.merge(p[T.KEY + ["sp", "pdg", "isprim"]], on=T.KEY + ["sp"], how="left", validate="many_to_one")
    if s.pdg.isna().any() or s.isprim.isna().any():
        raise ValueError("steps missing particle metadata")
    sec = s[s.sp != s.anchor]
    primary = (sec.isprim == 1) & (sec.pdg.abs() == 13)
    result = {"anchor_events": len(multiplicity), "anchor_particles": len(anchors),
              "multi_anchor_events": int((multiplicity > 1).sum()),
              "template_unique_source_steps": int(sec.source_row.nunique()), "template_steps": len(sec),
              "template_eion": float(sec.eion.sum()), "primary_muon_steps_in_template": int(primary.sum()),
              "primary_muon_eion_in_template": float(sec.loc[primary, "eion"].sum()),
              "events_with_primary_muon_contamination": len(sec.loc[primary, T.KEY].drop_duplicates())}
    eligible_keys = s.loc[s.sp == s.anchor, T.KEY].drop_duplicates()
    eligible = sec.merge(eligible_keys, on=T.KEY, validate="many_to_one")
    eligible_primary = (eligible.isprim == 1) & (eligible.pdg.abs() == 13)
    result.update({"library_eligible_events": len(eligible_keys), "eligible_template_steps": len(eligible),
                   "eligible_primary_muon_steps": int(eligible_primary.sum())})
    for name, mask in (("all_primary", primary), ("all_template", np.ones(len(sec), dtype=bool))):
        x = sec.loc[mask]
        mu_straw = s[s.sp == s.anchor][T.KEY + ["sid"]].drop_duplicates().assign(in_straw=True)
        x = x.merge(mu_straw, on=T.KEY + ["sid"], how="left", validate="many_to_one")
        result[name + "_offaxis_steps"] = int(x.in_straw.isna().sum())
        y = x.merge(eligible_keys, on=T.KEY, validate="many_to_one")
        result[name + "_eligible_offaxis_steps"] = int(y.in_straw.isna().sum())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="*")
    p.add_argument("--library-meta", help="audit only the exact ordered donors from this library sidecar")
    p.add_argument("--source-dir", help="directory holding the sidecar's donor basenames")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.library_meta:
        if args.inputs or not args.source_dir:
            p.error("--library-meta requires --source-dir and no positional inputs")
        from secondary_library_topology import donor_paths
        paths = [str(x) for x in donor_paths(json.loads(Path(args.library_meta).read_text()), args.source_dir)]
    else:
        if not args.inputs or args.source_dir:
            p.error("supply input files, or --library-meta with --source-dir")
        paths = T.expand(args.inputs)
    rows = []
    for path in paths:
        row = audit(T.read(path, "sgsteps"), T.read(path, "particles", T.KEY + ["key", "pdg", "isprim", "ismcprim"]))
        rows.append({"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), **row})
    if not rows:
        raise ValueError("no input tables")
    result = {"files": rows, "totals": {k: sum(r[k] for r in rows) for k in rows[0] if k not in ("path", "sha256")},
              "scope": ("exact library-sidecar donor population" if args.library_meta else
                        "saved input population; not the training-library population unless those exact files are supplied"),
              "library_metadata": args.library_meta,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with open(args.out, "x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
