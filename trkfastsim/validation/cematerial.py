"""Partial G4 target/absorber material ledger from an audited existing raw-step export.

Exact generated StageParticle parent -> reseated raw G4 parent -> primary electron identity is required. No proximity
matching, new G4 or correction. SD rows are not a complete all-material ledger, and saved fast encounter counts are
not symmetric material path lengths. Clock-prefix records are not clipped if a G4 step straddles the VD13 clock.
"""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from trkfastsim.validation import cecontract, ceaudit

KEY = cecontract.KEY
SIM = KEY + ["sim_product_id", "sim_key"]
ME = 0.51099895


def ids(d, fields, label):
    """Validate persistent ProductID/key fields without narrowing or floating-point conversion."""
    d = cecontract.integer_keys(d, label)
    for col in fields:
        if not pd.api.types.is_integer_dtype(d[col].dtype) or d[col].isna().any() or (d[col] < 0).any():
            raise ValueError(f"invalid persistent identity: {label}/{col}")
        d[col] = d[col].astype(np.uint64)
    return d


def primary_roots(events, registry, particles, generated):
    """Identify one generated primary per event using persistent parent reseating, never phase-space similarity."""
    events = cecontract.unique(events, "export events")
    if (events.g4_status != 0).any():
        raise ValueError("material diagnostic requires audited status-zero transport")
    g = cecontract.unique(generated, "generated StageParticles")
    pd.testing.assert_frame_equal(events[KEY], g[KEY], check_exact=True)
    if (g.pdg != 11).any() or not g.parent_valid.all():
        raise ValueError("one parent-linked generated CE electron required")
    g = ids(g, ["parent_product_id", "parent_key"], "generated parents")
    reg = registry[(registry.kind == -1) & registry.branch_name.str.endswith("_g4run__CaloLibraryCE.")]
    reg = cecontract.unique(ids(reg, ["product_id"], "raw registry"), "raw registry")
    pd.testing.assert_frame_equal(events[KEY], reg[KEY], check_exact=True)
    p = ids(particles, ["sim_product_id", "sim_key", "parent_product_id", "parent_key"], "particles")
    if p.duplicated(SIM).any():
        raise ValueError("duplicate persistent particle identity")
    target = g.merge(reg[KEY + ["product_id"]].rename(columns={"product_id": "raw_id"}), on=KEY, validate="one_to_one")
    for label, product in (("original", "parent_product_id"), ("reseated", "raw_id")):
        parent = target[KEY + [product, "parent_key"]].rename(columns={product: "sim_product_id", "parent_key": "sim_key"})
        check = parent.merge(p[SIM], on=SIM, how="left", indicator=True, validate="one_to_one")
        if (check._merge != "both").any():
            raise ValueError(f"missing {label} StageParticle parent")
    candidates = p[(p.pdg == 11) & p.parent_valid].copy()
    target = target[KEY + ["raw_id", "parent_key", "creation"]].rename(columns={"creation": "generated_creation"})
    roots = candidates.merge(target, on=KEY + ["parent_key"], validate="many_to_one")
    roots = roots[(roots.sim_product_id == roots.raw_id) & (roots.parent_product_id == roots.raw_id)]
    roots = cecontract.unique(roots, "raw primary roots")
    pd.testing.assert_frame_equal(events[KEY], roots[KEY], check_exact=True)
    if (roots.creation != roots.generated_creation).any():
        raise ValueError("primary creation code differs from generated StageParticle")
    return roots


def ledger(roots, steps):
    """Summarize direct-primary target/absorber SD records preceding its first recorded VD13, retaining zeros."""
    s = ids(steps, ["sim_product_id", "sim_key", "product_id", "index"], "raw steps")
    if s.duplicated(KEY + ["product_id", "index"]).any():
        raise ValueError("duplicate raw step identity")
    s = s.merge(roots[SIM], on=SIM, validate="many_to_one")
    if not np.isfinite(s.time).all():
        raise ValueError("nonfinite raw step time")
    vd = s[(s.kind == 0) & (s.volume == 13)].sort_values("time").drop_duplicates(KEY)
    d = roots[SIM].merge(vd[KEY + ["time"]].rename(columns={"time": "g4_vd13_t"}), on=KEY, how="left", validate="one_to_one")
    material = s[s.kind.isin([3, 4])].merge(d[KEY + ["g4_vd13_t"]], on=KEY, validate="many_to_one")
    m = material[material.g4_vd13_t.isna() | (material.time < material.g4_vd13_t)].copy()
    if not np.isfinite(m[["length", "energy_g4"]]).all().all() or (m.length < 0).any():
        raise ValueError("invalid material path or deposited energy")
    before = np.linalg.norm(m[["momentum_x", "momentum_y", "momentum_z"]].to_numpy(float), axis=1)
    after = np.linalg.norm(m[["post_momentum_x", "post_momentum_y", "post_momentum_z"]].to_numpy(float), axis=1)
    m["primary_delta_energy"] = np.sqrt(before**2 + ME**2) - np.sqrt(after**2 + ME**2)
    for kind, label in ((3, "target"), (4, "absorber")):
        subset = m[m.kind == kind]
        summary = subset.groupby(KEY).agg(records=("index", "size"), length_mm=("length", "sum"),
                                           edep_MeV=("energy_g4", "sum"), primary_delta_MeV=("primary_delta_energy", "sum"),
                                           distinct_volumes=("volume", "nunique")).reset_index()
        names = {k: f"g4_{label}_{k}" for k in summary if k not in KEY}
        d = d.merge(summary.rename(columns=names), on=KEY, how="left", validate="one_to_one")
        d[list(names.values())] = d[list(names.values())].fillna(0)
    return d, {"raw_primary_material_rows": len(material), "prefix_material_rows": len(m),
               "rows_after_or_at_vd13": len(material) - len(m), "direct_primary_vd13_events": len(vd)}


def attach(events, material):
    """Require the exact same complete birth set before describing material in reciprocal migration classes."""
    d = cecontract.unique(events, "saved upstream events")
    material = cecontract.unique(material, "material ledger")
    pd.testing.assert_frame_equal(d[KEY], material[KEY], check_exact=True)
    joined = d.merge(material, on=KEY, validate="one_to_one")
    if "entry_has_vd13" in joined:
        if not np.array_equal(joined.entry_has_vd13.to_numpy(bool), joined.g4_vd13_t.notna().to_numpy()):
            raise ValueError("raw-primary VD13 coverage differs from saved reference")
        if "entry_t" in joined:
            valid = joined.g4_vd13_t.notna()
            if not np.allclose(joined.loc[valid, "entry_t"], joined.loc[valid, "g4_vd13_t"], rtol=0, atol=1e-5):
                raise ValueError("raw-primary VD13 clock differs from saved reference")
    return joined


def report(d, counts):
    g, f = d.g4_steps > 0, d.chain_steps > 0
    columns = [k for k in d if k.startswith(("g4_target_", "g4_absorber_", "upstream_"))]
    columns += [k for k in ("fast_entry_loss", "g4_entry_loss", "entry_p_fast_minus_g4") if k in d]
    classes, common = {}, {}
    both_entries = d.g4_vd13_t.notna()
    if "chain_has_vd13" in d:
        both_entries &= d.chain_has_vd13.to_numpy(bool)
    else:
        both_entries &= False
    for label, mask in (("all_births", np.ones(len(d), dtype=bool)), ("g4_only_steps", g & ~f),
                        ("fast_only_steps", ~g & f), ("lost_good_track", d.g4_track & ~d.chain_track),
                        ("gained_good_track", ~d.g4_track & d.chain_track)):
        classes[label] = ceaudit.describe(d, mask, columns)
        common[label] = ceaudit.describe(d, mask & both_entries, columns)
    return {"events": len(d), "record_counts": counts, "classes": classes, "common_vd13_classes": common,
            "scope": "existing raw direct-primary target/absorber SD records; prefix by first VD13 start time, whole history if absent",
            "limits": "not all-material coverage; SD records not encounters; no fast per-material path ledger; no causal correction",
            "verdict": "diagnostic only; no tuning or physics closure"}


def load(path):
    """Read fixed columns only; exact identity selection precedes any material aggregate."""
    with uproot.open(path) as f:
        def read(name, fields):
            return f[f"dump/{name}"].arrays(fields, library="pd")
        events = read("events", KEY + ["g4_status"])
        reg = read("products", KEY + ["kind", "product_id"])
        reg["branch_name"] = ak.to_numpy(f["dump/products/branch_name"].array())
        p = read("particles", SIM + ["parent_valid", "parent_product_id", "parent_key", "pdg", "creation"])
        g = read("generated", KEY + ["pdg", "creation", "parent_valid", "parent_product_id", "parent_key"])
        roots = primary_roots(events, reg, p, g)
        s = read("steps", SIM + ["product_id", "index", "kind", "volume", "time", "length", "energy_g4",
                                 "momentum_x", "momentum_y", "momentum_z", "post_momentum_x", "post_momentum_y", "post_momentum_z"])
    return ledger(roots, s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True)
    ap.add_argument("--events", required=True, help="saved ceupstream events.csv.gz on the same births")
    ap.add_argument("--expected-events", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    material, counts = load(a.export)
    if len(material) != a.expected_events:
        raise ValueError("material ledger differs from externally audited event count")
    d = attach(pd.read_csv(a.events), material)
    result = report(d, counts)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cematerial.json").write_text(json.dumps(result, indent=2) + "\n")
    d.to_csv(out / "events.csv.gz", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
