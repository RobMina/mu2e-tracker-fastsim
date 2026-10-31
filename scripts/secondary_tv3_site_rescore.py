"""Rescore immutable saved products after a manifest-verified site relocation.

A staging root mirrors paths below an explicitly supplied original data root.
No input payload or saved comparison is modified. All payload hashes are recorded
before and after the rescore. This evaluates current analysis of historical
products, not production with current transport or a different secondary bank.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.secondary_tv3_rescore import differences
from trkfastsim.analyses import tv2compare as TV


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def relocate(paths, original_root, staging_root):
    original_root, staging_root = Path(original_root), Path(staging_root)
    result = {}
    for kind, files in paths.items():
        result[kind] = []
        for name in files:
            relative = Path(name).relative_to(original_root)
            candidate = staging_root / relative
            if not candidate.is_file() or not candidate.resolve().is_relative_to(staging_root.resolve()):
                raise ValueError(f'missing or escaped staged input: {candidate}')
            result[kind].append(str(candidate))
    return result


def expected_products(path, original_root, staging_root):
    """Relocate an existing sha256sum receipt without rewriting its source paths."""
    result = {}
    for line in Path(path).read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        if len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ValueError('invalid source checksum receipt')
        relocated = relocate({'payload': [name.lstrip('*')]}, original_root, staging_root)['payload'][0]
        if relocated in result and result[relocated] != checksum:
            raise ValueError('conflicting source checksum receipt')
        result[relocated] = checksum
    return result


def run(comparison, original_root, staging_root, outdir, product_hashes=None):
    comparison, outdir = Path(comparison), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    saved_hash = digest(comparison)
    saved = json.loads(comparison.read_text())
    paths = relocate(saved['input_files'], original_root, staging_root)
    inputs = sorted({p for files in paths.values() for p in files})
    hashes = {p: digest(p) for p in inputs}
    receipt_hash = digest(product_hashes) if product_hashes else None
    expected = expected_products(product_hashes, original_root, staging_root) if product_hashes else {}
    if any(hashes.get(p) != checksum for p, checksum in expected.items()):
        raise ValueError('staged payload differs from source checksum receipt')
    TV.run(paths['ntuple_fast'], str(outdir / 'comparison'), compare=paths['ntuple_g4'],
           steps=paths['steps_fast'], steps_ref=paths['steps_g4'], hits=paths['hits_fast'],
           hits_ref=paths['hits_g4'], paired=saved['paired'])
    result = json.loads((outdir / 'comparison/tv2compare.json').read_text())
    comparable = dict(result, input_files=saved['input_files'])
    if hashes != {p: digest(p) for p in inputs} or digest(comparison) != saved_hash:
        raise ValueError('saved input changed during rescore')
    audit = dict(scope='current analysis on historical saved products; fixed-library conditional errors',
        comparison_sha256=saved_hash, inputs=hashes,
        source_receipt_sha256=receipt_hash, source_receipt_payloads_checked=len(expected),
        changed_fields_excluding_relocation=differences(saved, comparable),
        driver_sha256=digest(__file__), analysis_sha256=digest(TV.__file__),
        result_sha256=digest(outdir / 'comparison/tv2compare.json'))
    (outdir / 'complete.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison')
    parser.add_argument('--original-root', required=True)
    parser.add_argument('--staging-root', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--product-hashes', help='existing source sha256sum receipt for saved products')
    args = parser.parse_args()
    run(args.comparison, args.original_root, args.staging_root, args.outdir, args.product_hashes)
