"""Inventory explicit detector/stock files without selecting events by fitted-track content."""
import argparse
import json

import awkward as ak
import numpy as np
import uproot

from trkfastsim.reco.autonomous import digest


def inventory(path):
    with uproot.open(path) as root:
        out = {'path': path, 'sha256': digest(path), 'trees': {}}
        for name in ('trkdump/events', 'trkdump/conditions', 'EventNtuple/ntuple'):
            if name not in root:
                continue
            tree = root[name]
            record = {'entries': int(tree.num_entries), 'cycle': tree.object_path,
                      'has_digis': 'd_sid' in tree.keys(), 'has_stock_fits': 'kl_status' in tree.keys()}
            if name == 'trkdump/events':
                ids = tree.arrays(['run', 'subrun', 'event'], library='np')
                key = np.column_stack([ids[k] for k in ('run', 'subrun', 'event')])
                record['unique_events'] = len(np.unique(key, axis=0))
                if 'd_sid' in tree.keys():
                    sid = tree['d_sid'].array()
                    n = ak.to_numpy(ak.num(sid))
                    flat = ak.to_numpy(ak.flatten(sid)).astype(np.int64)
                    planes = flat >> 10
                    panels = planes * 6 + ((flat & 0x380) >> 7)
                    record.update(digis=int(n.sum()), empty_events=int((n == 0).sum()),
                                  readout_planes=np.unique(planes).tolist(),
                                  readout_unique_panels=np.unique(panels).tolist())
                if 'd_nadc' in tree.keys():
                    nadc = ak.to_numpy(ak.flatten(tree['d_nadc'].array()))
                    record['digis_without_adc'] = int((nadc == 0).sum())
                    record['adc_samples'] = int(nadc.sum())
                for branch, label in (('tc_n', 'clusters'), ('sd_n', 'seeds'), ('kl_status', 'stock_fits')):
                    if branch in tree.keys():
                        counts = ak.to_numpy(ak.num(tree[branch].array()))
                        record[label] = {'total': int(counts.sum()), 'zero_events': int((counts == 0).sum()),
                                         'multiple_events': int((counts > 1).sum()),
                                         'max_per_event': int(counts.max(initial=0))}
            out['trees'][name] = record
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    print(json.dumps([inventory(path) for path in args.files], indent=2))
