"""Report-only exact dumper-to-injector genealogy; native ART branch inventory, never raw decoding."""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from mu2efastsim import tables
from trkfastsim import geometry
from trkfastsim.validation import fieldcompare as compare, fieldgasvoids as voids
from trkfastsim.validation import fieldpathaudit as pathaudit

KEY = tables.KEY
TRACK = KEY + ["sp"]
GROUP = TRACK + ["sid"]
EXPORT = KEY + ["nsgs", "sid", "plane", "panel", "straw", "upanel", "shape", "ion", "sp",
                "eion", "len", "width", "t", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz"]
PRODUCTS = ("StepPointMC", "StrawGasStep", "Assns", "SimParticle", "SurfaceStep", "MCTrajectories", "StatusG4")
SCOPE = ("Exact original dumper row/key propagation only; not native ART exact-link certification: "
         "the dumper discarded ProductID. Branch presence does not establish nonempty products, "
         "decoded raw payloads, associations or constituent ordering. No transport, reco or G4 executed.")


def exact(left, right):
    """Ordered payload equality with original constructor dtypes; no tolerance or I/O normalization."""
    pd.testing.assert_frame_equal(left, right, check_exact=True)


def identities(frame, columns, label):
    voids.fg._identities(frame, columns, label)
    return frame[columns].astype(np.uint64)


def reconstruct(part, steps, vd, surf, origin, vds, source_rows):
    """Use the frozen first-entry and write_tables constructors without changing their order/dtypes."""
    cols = KEY + ["sp", "pdg", "x", "y", "z", "px", "py", "pz", "t", "vdid"]
    cols += ["x0", "y0", "z0", "x1", "y1", "z1"]
    ent = pd.concat([compare.entry_states(vd, vds).reindex(columns=cols),
                     compare.entry_states_surface(surf, origin).reindex(columns=cols)]).sort_values("t")
    ent = ent.groupby(TRACK, sort=False).first().reset_index()
    identities(ent, TRACK, "original entry")
    if ent.empty or ent.duplicated(TRACK).any():
        raise ValueError("empty or duplicate original entries")
    rk = ent[TRACK + ["pdg"]].sort_values(TRACK).copy()
    rk["key"] = rk.groupby(KEY).cumcount() + 1
    parts = rk[KEY + ["key", "pdg"]].assign(parent=-1, isprim=1, ismcprim=1)
    rmap = rk[TRACK + ["key"]]
    exact(steps.reset_index(drop=True), source_rows.drop(columns="sourceoriginalrowindex").reset_index(drop=True))
    tagged = steps.assign(sourceoriginalrowindex=source_rows.sourceoriginalrowindex.to_numpy())
    g4 = tagged.merge(ent[TRACK + ["t"]].rename(columns={"t": "t_entry"}), on=TRACK)
    g4 = g4[g4.t >= g4.t_entry - 1e-6]
    postentry = g4.drop(columns="sourceoriginalrowindex").copy()
    g4 = g4.assign(nsgs=g4.groupby(KEY).sid.transform("size"))
    exported = g4.merge(rmap, on=TRACK).drop(columns="sp").rename(columns={"key": "sp"})
    source = exported[KEY + ["sp", "sid", "sourceoriginalrowindex"]].reset_index(drop=True)
    inverse = rmap.rename(columns={"sp": "originalsp", "key": "sp"})
    source = source.merge(inverse, on=TRACK, how="left", validate="many_to_one", sort=False)
    if source.originalsp.isna().any():
        raise ValueError("missing inverse injector genealogy")
    genealogy = rk.rename(columns={"sp": "originalsp", "key": "injectorkey"}).reset_index(drop=True)
    entry = ent[TRACK + ["vdid", "t", "px", "py", "pz"]].rename(columns={
        "sp": "originalsp", "vdid": "entryvd", "t": "entrytime", "px": "entrypx", "py": "entrypy", "pz": "entrypz"})
    entry["entryp"] = np.sqrt(entry.entrypx ** 2 + entry.entrypy ** 2 + entry.entrypz ** 2)
    genealogy = genealogy.merge(entry, on=KEY + ["originalsp"], validate="one_to_one")
    for frame, names in ((genealogy, KEY + ["originalsp"]), (source, KEY + ["originalsp"])):
        frame[names] = identities(frame, names, "source genealogy")
    return dict(particles=parts.reset_index(drop=True), sgsteps=exported[EXPORT].reset_index(drop=True),
                source=source, genealogy=genealogy, entry=ent, g4postentry=postentry)


def clock_index(all_steps, clock):
    """Exact unique KEY/sp/sid index association, not a time/proximity match or guessed offset."""
    for label, frame in (("G4all", all_steps), ("clock", clock)):
        identities(frame, GROUP, label)
        if frame.duplicated(GROUP).any():
            raise ValueError(f"duplicate {label} KEY/sp/sid")
    if list(clock.columns) != list(all_steps.columns):
        raise ValueError("clock column/order differs from G4all")
    positions = pd.MultiIndex.from_frame(all_steps[GROUP]).get_indexer(pd.MultiIndex.from_frame(clock[GROUP]))
    if (positions < 0).any():
        raise ValueError("foreign clock KEY/sp/sid")
    payload = [c for c in all_steps.columns if c != "nsgs"]
    exact(clock[payload].reset_index(drop=True), all_steps.iloc[positions][payload].reset_index(drop=True))
    counts = clock.groupby(KEY).sid.transform("size").rename("nsgs")
    pd.testing.assert_series_equal(clock.nsgs.reset_index(drop=True), counts.reset_index(drop=True), check_exact=True)
    return positions


def csv_representation(frame):
    """Serialize once exactly as fieldgasvoids, then parse with round-trip floats (including ROOT float32)."""
    return pd.read_csv(io.StringIO(frame.to_csv(index=False)), float_precision="round_trip",
                       dtype={k: "uint64" for k in GROUP})


def validate_outliers(path, steps, denominator, geom, origin, label):
    """Recompute the complete unbounded selection, copied payload and residuals before trusting any CSV row."""
    _, recomputed, report = voids.audit_arm(steps, denominator, geom, origin, label)
    saved = pd.read_csv(path, float_precision="round_trip", dtype={k: "uint64" for k in GROUP})
    expected = csv_representation(recomputed)
    exact(saved, expected)
    floats = expected.select_dtypes(include="floating").columns
    if not np.array_equal(saved[floats].to_numpy(dtype="float64").view("uint64"),
                          expected[floats].to_numpy(dtype="float64").view("uint64")):
        raise ValueError("outlier CSV floating bit values differ from original serialization")
    identities(saved, GROUP, "outlier CSV")
    if not saved.empty:
        identities(saved, ["originalrowindex"], "outlier CSV")
    if saved.originalrowindex.duplicated().any():
        raise ValueError("duplicate outlier originalrowindex")
    return saved, report


def art_inventory(path):
    """Inspect highest Events cycle and typed top-level branch names only; never request arrays."""
    with uproot.open(path) as root:
        top = root.classnames(recursive=False, cycle=True)
        events = [k for k in top if k.rsplit(";", 1)[0] == "Events"]
        if not events:
            raise ValueError("native ART has no Events tree")
        selected = max(events, key=lambda k: int(k.rsplit(";", 1)[1]))
        tree = root[selected]
        branches = [{"name": k, "typename": str(tree[k].typename)} for k in tree.keys(recursive=False)]
        metadata = []
        for name, classname in top.items():
            if name.rsplit(";", 1)[0] != "Events" and ("TTree" in classname or "RNTuple" in classname):
                metadata.append({"name": name, "branches": list(root[name].keys(recursive=False))})
        candidates = {}
        for product in PRODUCTS:
            tokens = ("mctrajector",) if product == "MCTrajectories" else (product.lower(),)
            candidates[product] = [b for b in branches if any(t in (b["name"] + " " + b["typename"]).lower() for t in tokens)]
        registry = [name for name in top if "productregistry" in name.lower()]
        registry += [m["name"] + "/" + b for m in metadata for b in m["branches"] if "productregistry" in b.lower()]
        return dict(path=str(Path(path).resolve()), events_cycle=selected, num_entries=int(tree.num_entries),
                    top_level_objects=top, branches=branches, candidate_branches=candidates,
                    absent_candidates=[p for p in PRODUCTS if not candidates[p]], metadata_inventory=metadata,
                    productRegistry_metadata_present=bool(registry), productRegistry_metadata_locations=registry,
                    raw_payload_read=False, raw_decoded=False, native_art_exact_link_certified=False,
                    presence_scope="branch presence only; neither nonempty nor decoded nor linked", scope=SCOPE)


def prior_receipts(report, field_dir, outlier_dir):
    """Verify every prior input/output hash, and bind the selected saved arms to that exact prior audit."""
    paths = {}
    for kind in ("inputs", "outputs"):
        if not report.get(kind):
            raise ValueError(f"missing prior void-audit {kind}")
        for name, receipt in report[kind].items():
            path = Path(receipt["path"]).resolve()
            expected = receipt["sha256_before"] if kind == "inputs" else receipt["sha256"]
            if kind == "inputs" and receipt["sha256_after"] != expected:
                raise ValueError("prior void-audit input changed")
            if voids.fg._sha256(path) != expected:
                raise ValueError(f"prior void-audit hash mismatch: {kind}/{name}")
            paths[f"voids/{kind}/{name}"] = path
    expected = {"fieldcompare": field_dir / "fieldcompare.json",
                **{label: field_dir / filename for label, filename in voids.FILES.items()}}
    for name, path in expected.items():
        if paths.get(f"voids/inputs/{name}") != path.resolve():
            raise ValueError(f"prior void-audit input path mismatch: {name}")
    for label in voids.FILES:
        path = outlier_dir / f"fieldgasvoids_{label}_outliers.csv.gz"
        if paths.get(f"voids/outputs/{label}_outliers") != path.resolve():
            raise ValueError(f"prior void-audit outlier path mismatch: {label}")
    return paths


def run(input_table, field_dir, outlier_dir, source_art, outdir):
    """Fail closed on any payload/provenance mismatch and publish exclusively, with before/after receipts."""
    input_table, field_dir, outlier_dir, source_art, outdir = map(
        lambda p: Path(p).resolve(), (input_table, field_dir, outlier_dir, source_art, outdir))
    if outdir.exists():
        raise FileExistsError(outdir)
    if any(p == outdir or p in outdir.parents for p in (field_dir, outlier_dir)):
        raise ValueError("output must be separate from frozen inputs")
    prior_path = outlier_dir / "fieldgasvoids.json"
    prior_hash = voids.fg._sha256(prior_path)
    prior = json.loads(prior_path.read_text())
    paths = prior_receipts(prior, field_dir, outlier_dir)
    from trkfastsim.validation import cecontract, fieldmultiplicity, fieldreco
    from mu2efastsim import world
    modules = dict(fieldcompare=compare, fieldpathaudit=pathaudit, fieldgasvoids=voids,
                   fieldgasledger=voids.fg, fieldmaterial=voids.fm, fieldcrossing=voids.fc,
                   tables=tables, tracker_tables=compare.T, geometry=geometry, cecontract=cecontract,
                   fieldmultiplicity=fieldmultiplicity, fieldreco=fieldreco, world=world)
    paths.update(input_table=input_table, source_art=source_art, voids_report=prior_path, code=Path(__file__))
    paths.update({f"code/{name}": Path(mod.__file__) for name, mod in modules.items()})
    before = {k: voids.fg._sha256(p) for k, p in paths.items()}
    if before["voids_report"] != prior_hash:
        raise ValueError("prior void-audit report changed during receipt verification")
    prior_receipts(prior, field_dir, outlier_dir)
    old = json.loads((field_dir / "fieldcompare.json").read_text())
    settings = pathaudit.frozen_settings(old, "first")
    if any(old.get(k) for k in ("max_events", "unbiased_only", "self_pass_only")):
        raise ValueError("truncated/self/unbiased filters unsupported")
    part, steps, vd, surf, _ = compare.primaries([str(input_table)], 13)
    raw = tables.read(input_table, "sgsteps").reset_index(drop=True)
    source_rows = raw.assign(sourceoriginalrowindex=np.arange(len(raw), dtype=np.int64)).merge(part, on=TRACK, how="inner")
    rebuilt = reconstruct(part, steps, vd, surf, np.asarray(settings["origin"]), settings["entry_vds"], source_rows)
    if old.get("entries", len(rebuilt["entry"])) != len(rebuilt["entry"]):
        raise ValueError("frozen entry count mismatch")
    particles, arms = {}, {}
    for label, filename in voids.FILES.items():
        particles[label] = tables.read(field_dir / filename, "particles")
        arms[label] = tables.read(field_dir / filename, "sgsteps")
    for label in ("g4_all", "g4_clock"):
        exact(particles[label], rebuilt["particles"])
    exact(arms["g4_all"], rebuilt["sgsteps"])
    denominator = voids.fc.verify_particles({k: v.rename(columns={"key": "sp"}) for k, v in particles.items()})
    positions = clock_index(arms["g4_all"], arms["g4_clock"])
    native = art_inventory(source_art)
    geom = geometry.from_gdml(str(paths["voids/inputs/gdml"]))
    original_parts = tables.read(input_table, "particles", KEY + ["key", "pdg", "parent", "isprim", "ismcprim"])
    original_parts = original_parts.rename(columns={"key": "originalsp", "pdg": "originalpdg",
        "parent": "originalparent", "isprim": "originalisprim", "ismcprim": "originalismcprim"})
    identities(original_parts, KEY + ["originalsp"], "original particles")
    genealogy = rebuilt["genealogy"].merge(original_parts, on=KEY + ["originalsp"], validate="one_to_one")
    if len(genealogy) != len(rebuilt["genealogy"]) or not np.array_equal(genealogy.pdg, genealogy.originalpdg):
        raise ValueError("original particle genealogy mismatch")
    combined, audits, validated = [], {}, {}
    for label in voids.FILES:
        validated[label], audits[label] = validate_outliers(
            paths[f"voids/outputs/{label}_outliers"], arms[label], denominator, geom, settings["origin"], label)
        if prior["arms"][label]["all_outlier_rows"] != len(validated[label]):
            raise ValueError("prior outlier count mismatch")
    for label in ("g4_all", "g4_clock"):
        outliers = validated[label]
        rows = outliers.originalrowindex.to_numpy(dtype=np.int64)
        all_rows = rows if label == "g4_all" else positions[rows]
        association = rebuilt["source"].iloc[all_rows].reset_index(drop=True)
        exact(outliers[GROUP].reset_index(drop=True), association[GROUP].astype(outliers[GROUP].dtypes))
        outliers["arm"] = label
        outliers["g4allrowindex"] = all_rows
        outliers["sourceoriginalrowindex"] = association.sourceoriginalrowindex.to_numpy(dtype=np.int64)
        outliers["originalsp"] = association.originalsp.to_numpy(dtype=np.uint64)
        outliers["injectorkey"] = outliers.sp
        outliers = outliers.merge(genealogy.drop(columns="pdg"), on=KEY + ["originalsp", "injectorkey"],
                                  how="left", validate="many_to_one", sort=False)
        outliers["source_art"] = str(source_art)
        outliers["art_events_cycle"] = native["events_cycle"]
        outliers["art_branch_inventory"] = "fieldsourceaudit_art_inventory.json"
        outliers["art_candidate_branches"] = json.dumps(
            {k: [b["name"] for b in v] for k, v in native["candidate_branches"].items()}, sort_keys=True)
        outliers["art_branch_scope"] = "presence only; no decoded product or row association"
        outliers["native_art_exact_link_certified"] = False
        combined.append(outliers)
    alloutliers = pd.concat(combined, ignore_index=True)
    voids.fg._check_hashes(paths, before)
    outdir.mkdir(parents=True, exist_ok=False)
    outputs = {"renumber_map": outdir / "fieldsourceaudit_renumber.csv.gz",
               "all_outliers": outdir / "fieldsourceaudit_outliers.csv.gz",
               "art_inventory": outdir / "fieldsourceaudit_art_inventory.json"}
    for name, frame in (("renumber_map", genealogy), ("all_outliers", alloutliers)):
        frame.to_csv(outputs[name], index=False, compression={"method": "gzip", "mtime": 0})
    with outputs["art_inventory"].open("x") as stream:
        json.dump(native, stream, indent=2, allow_nan=False)
        stream.write("\n")
    after = voids.fg._check_hashes(paths, before)
    result = dict(passed=True, scope=SCOPE, species=13, entry="first", origin_mm=settings["origin"],
                  entry_particles=len(genealogy), original_selected_sgs=len(steps), g4postentry_rows=len(rebuilt["g4postentry"]),
                  exact_g4all_rows=len(arms["g4_all"]), unique_g4all_keys=len(arms["g4_all"]),
                  exact_clock_subset_rows=len(positions), outlier_rows={k: v["all_outlier_rows"] for k, v in audits.items()},
                  g4all_ordered_payload_exact=True, clock_payload_exact_except_event_nsgs=True,
                  all_outlier_csv_payloads_and_recomputed_selection_exact=True,
                  environment_scope="local code/input receipts, not historical code/environment attestation",
                  source_row_scope="zero-based original dumper dump/sgsteps row before primary selection/merges",
                  genealogy_key_scope="run/subrun/event are unchanged original uint64 KEY; originalsp is original uint64 SimParticle key",
                  clock_association="exact unique KEY+sp+sid G4all index; no time/proximity or invented offsets",
                  art_inventory=native, inputs_unchanged=before == after,
                  inputs={k: dict(path=str(p), sha256_before=before[k], sha256_after=after[k]) for k, p in paths.items()},
                  outputs={k: dict(path=str(p), sha256=voids.fg._sha256(p)) for k, p in outputs.items()})
    with (outdir / "fieldsourceaudit.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-table", "field-dir", "outlier-dir", "source-art", "outdir"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(run(**vars(parser.parse_args(argv))), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
