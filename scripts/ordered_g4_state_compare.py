"""Saved-only same-generated-particle/straw state comparison; never a range certificate."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mu2efastsim import eloss
from mu2efastsim.tables import all_primary_muons
from scripts.transport_capture_audit import digest
from trkfastsim import ledger, tables

KEY = tables.KEY
IDENTITY = KEY + ["sp_g4"]
STRAW_KEY = IDENTITY + ["sid"]


def summary(values, events):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite comparison observable")
    if not len(values):
        return {"n": 0, "event_clusters": 0, "mean": None, "cluster_mean_se": None, "quantiles": None}
    grouped = events[KEY].copy().assign(value=values).groupby(KEY).value.agg(["sum", "count"])
    mean = float(values.mean())
    clusters = len(grouped)
    se = (float(np.sqrt(clusters / (clusters - 1) * np.square(grouped["sum"] - mean * grouped["count"]).sum())
                / len(values)) if clusters > 1 else None)
    return {"n": len(values), "event_clusters": clusters, "mean": mean, "cluster_mean_se": se,
            "quantiles": np.quantile(values, [0, .1, .5, .9, 1]).tolist()}


def check_generation(gen, reference_gen):
    key = KEY + ["idx"]
    selected = reference_gen.merge(gen[KEY].drop_duplicates(), on=KEY, validate="many_to_one")
    if gen.duplicated(key).any() or selected.duplicated(key).any():
        raise ValueError("duplicate generated-particle identity")
    joined = selected.merge(gen, on=key, how="left", suffixes=("_g4", "_fast"), validate="one_to_one", indicator=True)
    if not (joined._merge == "both").all():
        raise ValueError("reference generated particle is absent from fast input")
    if not (joined.pdg_g4 == joined.pdg_fast).all():
        raise ValueError("reference generator particle PDG differs")
    deviations = {}
    for name in ("x", "y", "z", "px", "py", "pz", "t"):
        a, b = joined[f"{name}_g4"].to_numpy(float), joined[f"{name}_fast"].to_numpy(float)
        if not np.allclose(a, b, rtol=2e-6, atol=1e-4):
            raise ValueError(f"reference generator {name} differs")
        deviations[name] = float(np.max(np.abs(a - b), initial=0.))
    expected = gen.merge(selected[KEY].drop_duplicates(), on=KEY, validate="many_to_one")
    if len(expected) != len(selected):
        raise ValueError("reference generated-particle coverage differs in retained events")
    return dict(reference_events=len(selected[KEY].drop_duplicates()), reference_particles=len(selected),
                max_abs_deviation=deviations)


def compare(gen, fast, g4, particles):
    muons = all_primary_muons(gen)
    if "idx" not in muons:
        raise ValueError("generator index is required for particle identity")
    mapping = muons[KEY + ["mukey", "idx", "pdg"]].rename(columns={"mukey": "sp", "pdg": "pdg_gen"})
    if mapping.duplicated(KEY + ["sp"]).any():
        raise ValueError("duplicate generator muon identity")
    mapping["sp_g4"] = mapping.idx + 1
    f = fast.merge(mapping, on=KEY + ["sp"], validate="many_to_one")
    if len(f) != len(fast):
        raise ValueError("unmapped fast primary identity")
    primaries = particles[(particles.ismcprim == 1) & (particles.pdg.abs() == 13)].rename(columns={"key": "sp_g4"})
    primaries = primaries.merge(mapping[IDENTITY + ["pdg_gen"]], on=IDENTITY, validate="one_to_one")
    if not (primaries.pdg == primaries.pdg_gen).all():
        raise ValueError("mapped G4/generator PDG mismatch")
    g = g4.rename(columns={"sp": "sp_g4"}).merge(primaries[IDENTITY], on=IDENTITY, validate="many_to_one")
    f_ambiguous = f.duplicated(STRAW_KEY, keep=False)
    g_ambiguous = g.duplicated(STRAW_KEY, keep=False)
    paired = f.loc[~f_ambiguous].merge(g.loc[~g_ambiguous], on=STRAW_KEY, suffixes=("_fast", "_g4"),
                                      validate="one_to_one")
    result = {"matching": "event + generated all-particle index + 1 + straw; no time/position matching",
              "generated_events": len(gen[KEY].drop_duplicates()), "generated_muons": len(muons),
              "fast_primary_steps": len(f), "g4_primary_steps_in_generated_events": len(g),
              "ambiguous_fast_rows_excluded": int(f_ambiguous.sum()),
              "ambiguous_g4_rows_excluded": int(g_ambiguous.sum()),
              "matched_straws": len(paired), "matched_muons": len(paired[IDENTITY].drop_duplicates()),
              "matched_events": len(paired[KEY].drop_duplicates()), "observables": {}}
    first_fast = f.sort_values("t", kind="stable").drop_duplicates(IDENTITY)
    first_g4 = g.sort_values("t", kind="stable").drop_duplicates(IDENTITY)
    first = first_fast.merge(first_g4, on=IDENTITY, suffixes=("_fast", "_g4"), validate="one_to_one")
    first_p = {label: np.linalg.norm(first[[f"{axis}_{label}" for axis in ("px", "py", "pz")]].to_numpy(float), axis=1)
               for label in ("fast", "g4")}
    result["same_particle_first_steps"] = {
        "muons": len(first), "events": len(first[KEY].drop_duplicates()),
        "same_straw": int((first.sid_fast == first.sid_g4).sum()),
        "time_fast_minus_g4_ns": summary(first.t_fast - first.t_g4, first),
        "ke_fast_minus_g4_MeV": summary(ledger.kinetic(first_p["fast"]) - ledger.kinetic(first_p["g4"]), first),
        "interpretation": "particle identity only; different first straws/paths include world delivery and acceptance differences"}
    if not len(paired):
        return result, paired
    pf = np.linalg.norm(paired[[f"{a}_fast" for a in ("px", "py", "pz")]].to_numpy(float), axis=1)
    pg = np.linalg.norm(paired[[f"{a}_g4" for a in ("px", "py", "pz")]].to_numpy(float), axis=1)
    paired["delta_time_ns"] = paired.t_fast - paired.t_g4
    paired["delta_entry_ke_MeV"] = ledger.kinetic(pf) - ledger.kinetic(pg)
    metrics = {"entry_time_fast_minus_g4_ns": paired.delta_time_ns,
               "entry_ke_fast_minus_g4_MeV": paired.delta_entry_ke_MeV,
               "gas_arc_length_fast_minus_g4_mm": paired.len_fast - paired.len_g4,
               "gas_deposit_fast_minus_g4_MeV": paired.eion_fast - paired.eion_g4}
    for axis in "xyz":
        metrics[f"entry_{axis}_fast_minus_g4_mm"] = paired[f"{axis}0_fast"] - paired[f"{axis}0_g4"]
    result["observables"] = {name: summary(value, paired) for name, value in metrics.items()}
    ordered = paired.sort_values(IDENTITY + ["t_g4"], kind="stable")
    grouped = ordered.groupby(IDENTITY, sort=False)
    first, last = grouped.first().reset_index(), grouped.last().reset_index()
    multiple = grouped.size().to_numpy() > 1
    endpoints = first.loc[multiple, IDENTITY].copy()
    interval_dt = (last.t_fast - first.t_fast) - (last.t_g4 - first.t_g4)
    interval_de = last.delta_entry_ke_MeV - first.delta_entry_ke_MeV
    reversed_order = (last.t_fast < first.t_fast) & multiple
    result["same_particle_shared_straw_interval"] = {
        "muons_with_multiple_shared_straws": int(multiple.sum()),
        "reversed_fast_endpoint_order": int(reversed_order.sum()),
        "flight_time_fast_minus_g4_ns": summary(interval_dt[multiple], endpoints),
        "ke_change_fast_minus_g4_MeV": summary(interval_de[multiple], endpoints),
        "interpretation": "first/last shared straw by G4 time; world entry offsets cancel, paths and random histories differ"}
    result["reference_primary_endpoints"] = {
        "matched_reference_muons": len(primaries),
        "below_0p5MeV": int(((primaries.ee - eloss.MMU) <= .5).sum()),
        "range_comparable": False,
        "reason": "G4 particle end and finite requested fast path are not common terminal boundaries"}
    return result, paired


def run(directory, g4_path, outdir):
    directory, g4_path, outdir = Path(directory), Path(g4_path), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    paths = [directory / "complete.json", directory / "ordered_steps.root", g4_path]
    before = {str(path): digest(path) for path in paths}
    report = json.loads(paths[0].read_text())
    if report["status"] != "bounded_full_chain_complete":
        raise ValueError("completed ordered producer output required")
    gen = tables.read(paths[1], "gen")
    if len(gen[KEY].drop_duplicates()) != report["generated_events"]:
        raise ValueError("generated-event coverage differs from completion receipt")
    generation = check_generation(gen, tables.read(g4_path, "gen"))
    result, paired = compare(gen, tables.read(paths[1], "sgsteps"), tables.read(g4_path, "sgsteps"),
                             tables.read(g4_path, "particles"))
    result.update(generation_check=generation, scope="bounded saved conditional shared-straw diagnostic; not inclusive rate/range closure",
                  navigation=report["navigation"], input_sha256=before,
                  source_sha256={str(Path(__file__)): digest(__file__)}, admitted_for_production=False)
    if before != {str(path): digest(path) for path in paths}:
        raise ValueError("saved input changed")
    paired.to_csv(outdir / "paired_straws.csv", index=False)
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--g4", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run(args.directory, args.g4, args.outdir)
