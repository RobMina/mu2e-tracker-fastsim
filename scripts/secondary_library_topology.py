"""Build a separate annotated donor library, requiring exact legacy payload identity.

Read the retained library's sidecar for its exact ordered donor basenames. Build
one donor file at a time, offset crossing IDs, and compare every legacy column
before publishing a candidate. No default library is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from trkfastsim import geometry, library
from trkfastsim.placement import validate


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def donor_paths(metadata, source_dir):
    names = metadata['inputs'].split(';')
    if not names or len(names) != metadata['nfiles'] or len(names) != len(set(names)):
        raise ValueError('invalid or duplicated donor manifest')
    if any(Path(n).name != n or not n.endswith('.root') for n in names):
        raise ValueError('donor manifest must contain ROOT basenames only')
    return [Path(source_dir) / name for name in names]


def build_bounded(paths, geom):
    chunks = {k: [] for k in ('instraw', 'instraw_steps', 'offaxis', 'events')}
    offset = 0
    for path in paths:
        start = time.perf_counter()
        part = library.build([str(path)], geom, topology=True)
        for tree in ('instraw', 'instraw_steps'):
            part[tree]['xing'] += offset
        offset += len(part['instraw'])
        for tree, frame in part.items():
            chunks[tree].append(frame)
        print(f'{path.name}: {time.perf_counter() - start:.3f} s; {len(part["offaxis"])} offaxis', flush=True)
    return {k: pd.concat(v, ignore_index=True) for k, v in chunks.items()}


def payload_differences(reference, candidate):
    out = {}
    for tree, frame in reference.items():
        other = candidate[tree]
        if len(other) != len(frame) or not set(frame.columns).issubset(other.columns):
            out[tree] = {'schema_or_count_mismatch': True}
            continue
        diffs = {}
        for c in frame:
            a, b = frame[c].to_numpy(), other[c].to_numpy()
            equal = (a == b) | (pd.isna(a) & pd.isna(b))
            if equal.all():
                continue
            delta = np.abs(a.astype(float) - b.astype(float))
            diffs[c] = {'count': int((~equal).sum()), 'max_abs': float(np.nanmax(delta)),
                        'max_relative': float(np.nanmax(delta / np.maximum(np.abs(a), 1e-30))),
                        'reference_dtype': str(a.dtype), 'rebuilt_dtype': str(b.dtype)}
        if diffs:
            out[tree] = diffs
    return out


def retain_payload(reference, rebuilt):
    allowed = {'instraw': {'p'}, 'events': {'p', 'ux', 'uy', 'uz'}}
    for tree, frame in reference.items():
        other = rebuilt[tree]
        if len(other) != len(frame) or not set(frame.columns).issubset(other.columns):
            raise ValueError(f'{tree}: annotation row alignment failed')
        for c in frame:
            a, b = frame[c].to_numpy(), other[c].to_numpy()
            if np.array_equal(a, b, equal_nan=True):
                continue
            if c not in allowed.get(tree, set()) or a.dtype != b.dtype or a.dtype.kind != 'f':
                raise ValueError(f'{tree}.{c}: annotation payload mismatch')
            if not np.allclose(a, b, rtol=2 * np.finfo(a.dtype).eps, atol=0, equal_nan=True):
                raise ValueError(f'{tree}.{c}: beyond derived-column roundoff allowance')
    result = {k: v.copy() for k, v in reference.items()}
    from trkfastsim.placement import TOPOLOGY
    for c in TOPOLOGY:
        result['offaxis'][c] = rebuilt['offaxis'][c].to_numpy()
    return result


def require_legacy_identity(reference, candidate):
    counts = {}
    for tree, frame in reference.items():
        other = candidate[tree]
        if len(other) != len(frame) or not set(frame.columns).issubset(other.columns):
            raise ValueError(f'{tree}: legacy schema/row-count mismatch')
        bad = [c for c in frame if not np.array_equal(frame[c].to_numpy(), other[c].to_numpy(), equal_nan=True)]
        if bad:
            raise ValueError(f'{tree}: legacy payload differs in {bad}')
        counts[tree] = len(frame)
    validate(candidate['offaxis'])
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--library', required=True, type=Path)
    ap.add_argument('--source-dir', required=True, type=Path)
    ap.add_argument('--gdml', required=True, type=Path)
    ap.add_argument('--outdir', required=True, type=Path)
    a = ap.parse_args(argv)
    a.outdir.mkdir(parents=True, exist_ok=False)
    metadata_path = a.library.with_suffix('.meta.json')
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('format') != 2:
        raise ValueError('this identity experiment requires a retained format-2 library')
    paths = donor_paths(metadata, a.source_dir)
    geom = geometry.from_gdml(str(a.gdml))
    start = time.perf_counter()
    candidate = build_bounded(paths, geom)
    reference = library.read(str(a.library))
    differences = payload_differences(reference, candidate)
    (a.outdir / 'rebuild_differences.json').write_text(json.dumps(differences, indent=2))
    candidate = retain_payload(reference, candidate)
    counts = require_legacy_identity(reference, candidate)
    provenance = {**metadata, 'anchor_topology': True, 'legacy_payload_identical': True,
                  'annotation_method': 'retain all original payload; append verified rebuilt topology only',
                  'rebuild_differences': differences, 'derived_roundoff_rtol_eps': 2,
                  'reference_library': str(a.library), 'reference_sha256': sha256(a.library),
                  'reference_metadata_sha256': sha256(metadata_path), 'gdml_sha256': sha256(a.gdml),
                  'donors': [{'path': str(p), 'sha256': sha256(p)} for p in paths],
                  'source_sha256': {str(p): sha256(p) for p in (Path(__file__), Path(library.__file__))},
                  'rows': counts, 'elapsed_s': time.perf_counter() - start,
                  'source_exterior_counts': {str(k): int(v) for k, v in
                                              candidate['offaxis'].source_exterior.value_counts().items()}}
    output = a.outdir / 'trklib_topology.root'
    library.write(str(output), candidate, provenance)
    print(json.dumps({'output': str(output), 'sha256': sha256(output), **provenance}, indent=2))


if __name__ == '__main__':
    main()
