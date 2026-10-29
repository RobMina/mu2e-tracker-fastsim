"""All-event detector census from the latest saved ROOT tree cycle."""
import importlib.util
from pathlib import Path

import awkward as ak
import uproot


spec = importlib.util.spec_from_file_location('autonomous_inventory',
    Path(__file__).resolve().parents[1] / 'scripts/autonomous_inventory.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_census_includes_empty_events_and_repeated_straws(tmp_path):
    path = tmp_path / 'dump.root'
    sid = (3 << 10) | (2 << 7) | 5
    with uproot.recreate(path) as f:
        f.mktree('trkdump/events', {'run': 'int64', 'subrun': 'int64', 'event': 'int64',
            **{k: 'var * int64' for k in ('d_sid', 'd_nadc', 'tc_n', 'sd_n', 'kl_status')}})
        f['trkdump/events'].extend({'run': [1, 1, 1], 'subrun': [2, 2, 2], 'event': [3, 4, 5],
            'd_sid': ak.Array([[], [sid, sid], [0]]), 'd_nadc': ak.Array([[], [15, 0], [15]]),
            'tc_n': ak.Array([[], [2, 3], [1]]), 'sd_n': ak.Array([[], [2, 3], []]),
            'kl_status': ak.Array([[], [1], []])})
    out = module.inventory(str(path))['trees']['trkdump/events']
    assert out['entries'] == out['unique_events'] == 3
    assert out['digis'] == 3 and out['empty_events'] == 1
    assert out['readout_planes'] == [0, 3]
    assert out['readout_unique_panels'] == [0, 20]
    assert out['digis_without_adc'] == 1 and out['adc_samples'] == 30
    assert out['seeds'] == dict(total=2, zero_events=2, multiple_events=1, max_per_event=2)
    assert out['stock_fits']['total'] == 1
