"""Bounded saved-bank replica response/F2 replay on fixed baseline-selected recipients."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scripts import secondary_replay as replay
from scripts.secondary_bank_replicas import variance_components
from scripts.transport_capture_audit import digest
from trkfastsim import bankreplica, digicond, geometry, library, tables
from trkfastsim.analyses import fastdigi, digicmp
from trkfastsim.reco import hitreco


def digi_view(keys, digis):
    return dict(evkey=keys, ev=digis["ev"], sid=digis["sid"], pbt=np.zeros(len(keys)),
        tdc0=digis["tdc"][:, 0], tdc1=digis["tdc"][:, 1], tot0=digis["tot"][:, 0], tot1=digis["tot"][:, 1],
        nadc=np.full(len(digis["sid"]), digis["adc"].shape[1], np.int64), adc=digis["adc"].reshape(-1))


def measure(digis, fits, events):
    counts = np.bincount(digis["ev"], minlength=events)
    if len(counts) != events or len(fits) != events:
        raise ValueError("response/reco event coverage mismatch")
    return dict(digis_per_event=float(counts.mean()), good_fits_per_event=float(np.mean(fits)),
                events_with_fit_fraction=float(np.mean(np.asarray(fits) > 0)))


def validate_campaign_inputs(saved, hashes, supplied):
    for path in supplied:
        name = str(Path(path))
        if saved["input_sha256"].get(name) != hashes[name]:
            raise ValueError("bank ancestry campaign input identity differs from supplied response input")


def run(campaign, baseline, bank_path, gdml, conditions, hit_conditions, drift_source, outdir,
        *, max_events=256, banks=4, draws=2):
    if not 2 <= max_events <= 256 or not 2 <= banks <= 16 or not 2 <= draws <= 4:
        raise ValueError("bounded response needs2..256 events,2..16 banks,2..4 draws")
    campaign, outdir = Path(campaign), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    paths = list(map(Path, (campaign / "complete.json", baseline, bank_path, gdml, conditions, hit_conditions, drift_source)))
    paths += [campaign / f"events.{index:03d}.parquet" for index in range(banks)]
    hashes = {str(p): digest(p) for p in paths}
    saved = json.loads(paths[0].read_text())
    if len(saved["variance"]) == 0:
        raise ValueError("completed bank replay required")
    validate_campaign_inputs(saved, hashes, (baseline, bank_path, gdml))
    products = replay.read_baseline(baseline)
    bank = library.read(str(bank_path))
    geom = geometry.from_gdml(gdml)
    c = digicond.Conditions.load(conditions)
    hc = hitreco.load_conditions(hit_conditions)
    decisions = tables.trk_filter(products["sgsteps"])
    keys = decisions.loc[decisions.pass_trk].reset_index()[tables.KEY].sort_values(tables.KEY).iloc[:max_events].to_numpy(np.int64)
    if len(keys) < 2:
        raise ValueError("insufficient fixed selected recipients")
    np.save(outdir / "recipient_keys.npy", keys)
    recipients = pd.DataFrame(keys, columns=tables.KEY)
    bounded = {name: (frame.merge(recipients, on=tables.KEY, validate="many_to_one")
                     if frame is not None and set(tables.KEY).issubset(frame.columns) else frame)
               for name, frame in products.items()}
    rows = []
    started = time.perf_counter()
    for bank_index in range(banks):
        ancestry = pd.read_parquet(campaign / f"events.{bank_index:03d}.parquet")
        replica, _ = bankreplica.replica(bank, None, selection=ancestry.source_event_row.to_numpy())
        for draw_index in range(draws):
            record = next(r for r in saved["replicas"] if r["bank"] == bank_index and r["draw"] == draw_index)
            # Replaying bounded recipients changes RNG addressing; this is a new explicitly bounded response study.
            pasted, _ = replay.repaste(bounded, replica, geom, record["seed"], 0)
            step_path = outdir / f"steps.{bank_index}.{draw_index}.root"
            replay.core_tables.write(str(step_path), pasted, as_ttree=True)
            stats = {}
            root_seed = int(np.random.SeedSequence([2718, bank_index, draw_index]).generate_state(1)[0])
            event_keys, digis, _ = fastdigi.digitize_table(str(step_path), c, np.random.default_rng(root_seed),
                                                         keys=keys, chunk=64, stats=stats)
            if not np.array_equal(event_keys, keys):
                raise ValueError("digitizer dropped/changed fixed recipient keys")
            view = digi_view(keys, digis)
            fits = digicmp.f2_yield(view, hc, drift_source, chunk=64, fixed_shapes=True,
                progress=lambda p: print(json.dumps(dict(bank=bank_index, draw=draw_index, reco=p)), flush=True))
            metrics = measure(digis, fits["fits_per_event"], len(keys))
            np.savez_compressed(outdir / f"response.{bank_index}.{draw_index}.npz", evkey=keys,
                                fits_per_event=fits["fits_per_event"], **digis)
            step_path.unlink()
            row = dict(bank=bank_index, draw=draw_index, metrics=metrics, digi_stats=stats,
                       good_fits=fits["good_fits"], events_with_fit=fits["events_with_fit"], seeds=fits["seeds"])
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = dict(scope="bounded fixed baseline-selected recipient response + fast F2; not stock KKLine or unconditional T-V3 errors",
        recipients=len(keys), banks=banks, draws=draws, rows=rows,
        variance={name: variance_components([[r["metrics"][name] for r in rows if r["bank"] == index]
                                             for index in range(banks)]) for name in rows[0]["metrics"]},
        selection="first full-key-sorted baseline tracker-selected events; ignores selection changes by construction",
        rng_warning="bounded recipient replay realigns sampler draws; neither historical replay nor same-draw causal comparison",
        elapsed_s=time.perf_counter() - started, input_sha256=hashes, admitted_for_production=False,
        source_sha256={str(p): digest(p) for p in (Path(__file__), Path(bankreplica.__file__), Path(replay.__file__),
            Path(fastdigi.__file__), Path(digicmp.__file__), Path(library.__file__))})
    if hashes != {str(p): digest(p) for p in paths}:
        raise ValueError("saved bank/response input changed")
    (outdir / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--gdml", required=True)
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--hit-conditions", required=True)
    parser.add_argument("--drift-source", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-events", type=int, default=256)
    parser.add_argument("--banks", type=int, default=4)
    parser.add_argument("--draws", type=int, default=2)
    args = parser.parse_args()
    run(args.campaign, args.baseline, args.library, args.gdml, args.conditions, args.hit_conditions, args.drift_source,
        args.outdir, max_events=args.max_events, banks=args.banks, draws=args.draws)
