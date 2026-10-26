"""Pin reviewed envelope outputs, then run a separate deterministic mean support-energy sensitivity."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .. import pairedenergy as energy
from ..analyses.ledgercheck import sha256, source_manifest


def run(pin_path, *, pin_sha256, outdir, tracker_ref, core_ref, max_rows=512, max_interval_rows=1000000,
        max_step_mm=1., fractional_loss=.05, stop_ke_MeV=1., max_steps=10000):
    """Execute both full populations before writing; unique completion includes hashes of every output."""
    output = Path(outdir)
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    bundle = energy.load_pinned(pin_path, pin_sha256, max_rows=max_rows, max_interval_rows=max_interval_rows)
    settings = dict(max_step_mm=max_step_mm, fractional_loss=fractional_loss, stop_ke_MeV=stop_ke_MeV,
                    max_steps=max_steps, timing="unknown", loss_model="mean", tcut_MeV=1e9)
    paired = energy.transport_pair(bundle, tracker_ref=tracker_ref, core_ref=core_ref,
        max_step_mm=max_step_mm, fractional_loss=fractional_loss, stop_ke_MeV=stop_ke_MeV, max_steps=max_steps)
    table = paired["paired"]
    summaries = {}
    for arm in ("inner", "outer"):
        final, history = paired["finals"][arm], paired["results"][arm]["history"]
        summaries[arm] = dict(final_rows=len(final), canonical_interval_rows=len(bundle["intervals"][arm]),
            history_rows=len(history), states={str(k): int(v) for k, v in final.status.value_counts().items()},
            max_primary_row_residual_MeV=float(np.max(np.abs(history.accounting()["primary_residual_MeV"]), initial=0.)),
            unmodeled_partition_rows=int((~history.accounting()["partition_complete"]).sum()))
    summary = dict(schema="trkfastsim.paired-support-energy.v1", mode=energy.MODE, scope=energy.SCOPE,
        input_pin=dict(path=str(Path(pin_path).resolve()), sha256=pin_sha256), pinned_inputs=bundle["pin"],
        seed_rows=len(bundle["seeds"]), seed_events=len(bundle["seeds"][energy.KEY[:3]].drop_duplicates()),
        seed_domain=bundle["summary"]["seed_domain"], geometry_summary_sha256=bundle["pin"]["files"]["summary.json"],
        sources_sha256=source_manifest(), tracker_ref=tracker_ref, core_ref=core_ref, settings=settings,
        arms=summaries, comparison=dict(status_changes=int(table.status_changed.sum()),
            max_absolute_final_ke_difference_MeV=float(table.outer_minus_inner_ke_MeV.abs().max()) if len(table) else 0.,
            max_absolute_path_difference_mm=float(table.outer_minus_inner_path_mm.abs().max()) if len(table) else 0.,
            changed_ke_identities=int((table.outer_minus_inner_ke_MeV != 0).sum()),
            interpretation="signed outer-minus-inner sensitivity, NOT an energy/range enclosure"),
        geometry_admitted_for_transport=False, production_admitted=False,
        elapsed_seconds=time.perf_counter()-start, complete=True)
    output.mkdir(parents=True, exist_ok=False)
    for arm in ("inner", "outer"):
        paired["results"][arm]["history"].save(output / f"{arm}_history.npz")
        paired["finals"][arm].to_csv(output / f"{arm}_final.csv", index=False)
        bundle["intervals"][arm].to_csv(output / f"{arm}_canonical_intervals.csv", index=False)
        bundle["mapping"][arm].to_csv(output / f"{arm}_interval_membership.csv", index=False)
    table.to_csv(output / "paired_final.csv", index=False)
    summary["outputs_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return {key: summary[key] for key in ("mode", "seed_rows", "arms", "comparison", "elapsed_seconds", "complete")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pin = commands.add_parser("pin", help="explicitly record reviewed geometry output hashes; no transport")
    pin.add_argument("directory")
    pin.add_argument("--pin-path", required=True)
    pin.add_argument("--endpoint-budget-mm", type=float, required=True)
    execute = commands.add_parser("run", help="run only with an independently supplied approved pin SHA256")
    execute.add_argument("pin_path")
    execute.add_argument("--pin-sha256", required=True)
    execute.add_argument("--outdir", required=True)
    execute.add_argument("--tracker-ref", required=True)
    execute.add_argument("--core-ref", required=True)
    execute.add_argument("--max-rows", type=int, default=512)
    execute.add_argument("--max-interval-rows", type=int, default=1000000)
    execute.add_argument("--max-step-mm", type=float, default=1.)
    execute.add_argument("--fractional-loss", type=float, default=.05)
    execute.add_argument("--stop-ke-MeV", dest="stop_ke_MeV", type=float, default=1.)
    execute.add_argument("--max-steps", type=int, default=10000)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    print(json.dumps(energy.pin_inputs(**args) if command == "pin" else run(**args), indent=2))


if __name__ == "__main__":
    main()
