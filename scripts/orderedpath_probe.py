"""Bounded stored-geometry candidate probe on saved capture entrance states.

No generation, primary acceptance denominator or production repair is claimed.
Strict navigation errors are retained as refusals, not dropped or converted into
vacuum. A successful bounded probe does not establish GDML completeness.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.transport_capture_audit import digest, flatten, load_capture
from trkfastsim import geometry, orderedpath, steps, transportledger as L


def run(capture_dir, gdml, outdir, tracker_ref, core_ref, max_lanes=16, length_mm=2500.):
    if not 0 < max_lanes <= 64 or not np.isfinite(length_mm) or length_mm <= 0:
        raise ValueError('bounded probe requires 1..64 lanes and positive finite extent')
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    files = [Path(capture_dir) / 'capture.npz', Path(capture_dir) / 'complete.json', Path(gdml)]
    before = {str(p): digest(p) for p in files}
    report, phases = load_capture(capture_dir)
    initial = flatten(phases['interior'][1], 'initial')
    size = min(max_lanes, len(initial['identity']))
    identity = initial['identity'][:size]
    origin, direction = initial['position_mm'][:size], initial['direction'][:size]
    kinetic = initial['model_ke_MeV'][:size]
    geom = geometry.from_gdml(gdml)
    interior = steps.TrackerInterior(gdml, geom)
    registry = {**steps.materials.BY_NAME, steps.STRAW_GAS.name: steps.STRAW_GAS}
    names = sorted({m.name for m in registry.values()})
    by_name = {m.name: m for m in registry.values()}
    metadata = L.provenance(geometry_sha256=before[str(Path(gdml))], tracker_commit=tracker_ref,
        core_commit=core_ref, inputs=[{'name': str(p), 'sha256': before[str(p)]} for p in files[:2]],
        frame='detector', media=[L.medium(), *[L.medium(by_name[n], cut_MeV=1e9,
                                                       loss_basis='unrestricted') for n in names]],
        settings={'probe': 'first captured interior lanes; not an acceptance sample'})
    started = time.perf_counter()
    result = dict(scope='bounded stored-geometry diagnostic; not production transport validation',
        mode=orderedpath.MODE, lanes=size, identities=identity.tolist(), requested_length_mm=length_mm,
        input_sha256=before, capture_arm=report['arm'],
        skipped_calorimeter_solids=[{'solid': str(k), 'count': int(v)}
                                    for k, v in interior.calo_skipped.items()])
    try:
        candidate = orderedpath.transport(interior.shells, geom, identity, origin, direction, kinetic,
            np.full(size, length_mm), metadata, mode=orderedpath.MODE, max_steps=10000)
    except (ValueError, RuntimeError) as error:
        result.update(status='refused', reason=str(error), admitted_for_production=False,
                      refusal_details=getattr(error, 'details', None))
    else:
        candidate['history'].save(outdir / 'history.npz')
        candidate['intervals'].to_csv(outdir / 'intervals.csv', index=False)
        result.update(status='bounded_candidate_complete', admitted_for_production=False,
            history_rows=len(candidate['history']), final_status=candidate['status'].tolist(),
            final_ke_MeV=candidate['final_ke_MeV'].tolist(), traveled_mm=candidate['traveled_mm'].tolist())
    result['elapsed_s'] = time.perf_counter() - started
    result['source_sha256'] = {p.name: digest(p) for p in [Path(__file__), *sorted(
        Path(orderedpath.__file__).parent.glob('*.py'))]}
    if before != {str(p): digest(p) for p in files}:
        raise ValueError('input changed during probe')
    (outdir / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture_dir')
    parser.add_argument('--gdml', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--tracker-ref', required=True)
    parser.add_argument('--core-ref', required=True)
    parser.add_argument('--max-lanes', type=int, default=16)
    args = parser.parse_args()
    run(args.capture_dir, args.gdml, args.outdir, args.tracker_ref, args.core_ref, args.max_lanes)
