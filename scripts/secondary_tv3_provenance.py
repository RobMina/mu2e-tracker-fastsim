"""Audit saved path-A generator identity, including every particle in retained G4 events.

Compare each fast table with its merged input and unmerged all-event gen tables,
then compare the G4-kept event subset without dropping non-muon particles.
No art processing, reconstruction, transport, or candidate selection is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from trkfastsim import tables as T
from trkfastsim.analyses.seccensus import _paired_files

GKEY = T.KEY + ["idx"]
GCOLS = GKEY + ["pdg", "genid", "x", "y", "z", "px", "py", "pz", "e", "t"]
ICOLS = GKEY + ["pdg", "genid"]


def canonical(frame):
    """Return complete, finite generator rows ordered by full particle identity."""
    if set(frame.columns) != set(GCOLS) or frame.empty:
        raise ValueError("missing, unexpected or empty generator columns")
    out = frame.loc[:, GCOLS].sort_values(GKEY, kind="stable").reset_index(drop=True)
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("nonfinite generator payload")
    for col in ICOLS:
        v = out[col].to_numpy()
        if np.any(v != np.floor(v)) or np.any(np.abs(v) >= 2**53):
            raise ValueError(f"invalid integral generator identity: {col}")
        out[col] = v.astype(np.int64)
    if (out[T.KEY] < 0).any().any() or (out.idx < 0).any():
        raise ValueError("negative generator identity")
    if out.duplicated(GKEY).any():
        raise ValueError("duplicate generator particle identity")
    if not np.array_equal(out.idx, out.groupby(T.KEY, sort=False).cumcount()):
        raise ValueError("generator indices must be complete and contiguous within each event")
    for col in set(GCOLS) - set(ICOLS):
        out[col] = out[col].astype(np.float64)
    return out


def payload_hash(frame):
    """Hash canonical numeric columns, including schema, row count and normalized dtypes."""
    digest = hashlib.sha256()
    digest.update(str(frame.shape).encode())
    for col in GCOLS:
        dtype = "<i8" if col in ICOLS else "<f8"
        digest.update(f"{col}:{dtype}".encode())
        digest.update(frame[col].to_numpy(dtype=dtype).tobytes())
    return digest.hexdigest()


def require_equal(a, b, label):
    if a.shape != b.shape:
        raise ValueError(f"{label}: generator row count mismatch")
    bad = [c for c in GCOLS if not np.array_equal(a[c].to_numpy(), b[c].to_numpy())]
    if bad:
        raise ValueError(f"{label}: generator payload differs in {bad}")


def audit_group(fast, merged, original, g4, expected_events=None):
    f, m, o, r = [canonical(v) for v in (fast, merged, original, g4)]
    require_equal(f, m, "fast vs merged")
    require_equal(f, o, "fast vs original all-event gen")
    events = f[T.KEY].drop_duplicates()
    if expected_events is not None:
        sizes = events.groupby(["run", "subrun"]).size()
        if (sizes != expected_events).any():
            raise ValueError("unexpected events per subrun in all-event generator tables")
    rkeys = r[T.KEY].drop_duplicates()
    matched = f.merge(rkeys, on=T.KEY, how="inner", validate="many_to_one")
    require_equal(canonical(matched), r, "G4 retained event subset")
    return {"all_event_count": len(events), "generator_particles": len(f),
            "subruns": len(events[["run", "subrun"]].drop_duplicates()),
            "g4_kept_events": len(rkeys), "g4_kept_generator_particles": len(r),
            "all_event_multimuon_events": int((f.loc[f.pdg.abs() == 13].groupby(T.KEY).size() > 1).sum()),
            "all_event_payload_sha256": payload_hash(f), "g4_kept_payload_sha256": payload_hash(r),
            "fast_merged_original_exact": True, "g4_retained_all_particles_exact": True}


def run(comparison, gen_dir, merged_dir, outdir, expected_events=50000, max_groups=None):
    start = time.perf_counter()
    outdir.mkdir(parents=True, exist_ok=False)
    data = json.loads(comparison.read_text())
    if not data.get("paired"):
        raise ValueError("requires an explicitly paired comparison manifest")
    paths = data["input_files"]
    groups = _paired_files(paths["steps_fast"], paths["steps_g4"])
    if max_groups is not None:
        groups = groups[:max_groups]
    if not groups:
        raise ValueError("no groups to audit")
    result = {"comparison": str(comparison), "comparison_sha256": hashlib.sha256(comparison.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "groups": [],
              "expected_events_per_subrun": expected_events,
              "scope": "exact saved table content; all original gen rows and every particle in G4-kept events",
              "limitation": "Does not independently re-read art products or verify generator payload in G4-rejected events.",
              "full_manifest": len(groups) == len(paths["steps_fast"])}
    print(f"matched {len(groups)} groups; manifest scan {time.perf_counter() - start:.2f} s", flush=True)
    try:
        for fast, refs, _ in groups:
            t0 = time.perf_counter()
            sub = T.read(fast, "subruns").sort_values(["run", "subrun"]).reset_index(drop=True)
            pairs = sub[["run", "subrun"]].to_numpy(dtype=np.int64)
            originals = [gen_dir / f"gen.{run:06d}_{sr:08d}.root" for run, sr in pairs]
            merged = merged_dir / ("gen.grp." + Path(fast).name.split("grp.", 1)[1])
            for label, inputs in (("merged", [merged]), ("original", originals), ("G4", refs)):
                other = T.read_many([str(p) for p in inputs], "subruns").sort_values(["run", "subrun"]).reset_index(drop=True)
                pd.testing.assert_frame_equal(sub, other, check_dtype=False, check_exact=True, obj=label + " exposure")
            row = audit_group(T.read(fast, "gen"), T.read(merged, "gen"),
                              T.read_many([str(p) for p in originals], "gen"), T.read_many(refs, "gen"), expected_events)
            row.update({"fast": fast, "merged": str(merged), "originals": [str(p) for p in originals], "g4": refs,
                        "elapsed_s": time.perf_counter() - t0})
            result["groups"].append(row)
            print(f'{Path(fast).name}: {row["all_event_count"]} events; {row["g4_kept_events"]} G4-kept exact; {row["elapsed_s"]:.2f} s', flush=True)
        result["status"] = "passed"
        result["totals"] = {k: sum(row[k] for row in result["groups"]) for k in
                            ("all_event_count", "generator_particles", "subruns", "g4_kept_events",
                             "g4_kept_generator_particles", "all_event_multimuon_events")}
        result["elapsed_s"] = time.perf_counter() - start
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_s=time.perf_counter() - start)
        raise
    finally:
        (outdir / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "groups"}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--gen-dir", type=Path, required=True)
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--expected-events", type=int, default=50000)
    parser.add_argument("--max-groups", type=int)
    a = parser.parse_args()
    if a.expected_events < 1 or (a.max_groups is not None and a.max_groups < 1):
        parser.error("event and group counts must be positive")
    run(a.comparison, a.gen_dir, a.merged_dir, a.outdir, a.expected_events, a.max_groups)


if __name__ == "__main__":
    main()
