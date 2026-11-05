"""Inventory saved stock cosmic MC identities without inferring unavailable particle links."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from trkfastsim.validation.fieldcrossing import _sha256


def inventory(path):
    with uproot.open(path) as root:
        tree = root['EventNtuple/ntuple']
        report = {'path': str(Path(path).resolve()), 'sha256': _sha256(path),
                  'tree': tree.object_path, 'entries': tree.num_entries, 'branches': {}}
        for name in ('trk', 'trkmc', 'trkmcsim', 'trksegs', 'trksegsmc', 'trkhitsmc'):
            if name not in tree:
                report['branches'][name] = {'present': False}
                continue
            data = tree[name].array(library='ak')
            fields = ak.fields(data)
            leaf = data[fields[0]] if fields else data
            counts = np.asarray(ak.to_numpy(ak.num(leaf, axis=1)))
            record = {'present': True, 'type': str(ak.type(data)), 'fields': fields,
                      'track_records': int(counts.sum()), 'events_with_records': int((counts > 0).sum())}
            if name in ('trkmcsim', 'trksegsmc', 'trkhitsmc'):
                per_track = ak.to_numpy(ak.flatten(ak.num(leaf, axis=2)))
                record['nested_records'] = int(per_track.sum())
                record['tracks_with_nested_records'] = int((per_track > 0).sum())
            report['branches'][name] = record
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', help='ARM=closed-stock-ntuple')
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args(argv)
    reports = {}
    for item in args.inputs:
        name, path = item.split('=', 1)
        if not name or name in reports:
            raise ValueError('nonempty unique arm labels required')
        reports[name] = inventory(path)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    result = {'arms': reports, 'scope': 'actual branch/record inventory only; no particle association or physics closure'}
    (out / 'fieldmcinventory.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
