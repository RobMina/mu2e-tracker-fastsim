"""The source population audit reproduces the builder's event-anchor join."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('source_audit', Path(__file__).parents[1] / 'scripts/secondary_source_audit.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def test_multi_anchor_primary_template_contamination():
    steps = pd.DataFrame({'run': 1, 'subrun': 1, 'event': 1, 'sp': [1, 2, 3], 'sid': [10, 20, 30], 'eion': [1., 2., 3.]})
    parts = pd.DataFrame({'run': 1, 'subrun': 1, 'event': 1, 'key': [1, 2, 3], 'pdg': [13, -13, 11],
                         'isprim': [1, 1, 0], 'ismcprim': [1, 1, 0]})
    r = A.audit(steps, parts)
    assert r['multi_anchor_events'] == 1 and r['anchor_particles'] == 2
    assert r['template_unique_source_steps'] == 3 and r['template_steps'] == 4
    assert r['primary_muon_steps_in_template'] == 2 and r['primary_muon_eion_in_template'] == 3.
    assert r['all_primary_offaxis_steps'] == 0 and r['all_template_offaxis_steps'] == 2


def test_source_population_without_anchor_crossing_is_not_library_eligible():
    s = pd.DataFrame({'run': 1, 'subrun': 1, 'event': [1, 1, 2], 'sp': [1, 2, 2],
                      'sid': [10, 20, 30], 'eion': [1., 2., 3.]})
    p = pd.DataFrame({'run': 1, 'subrun': 1, 'event': [1, 1, 2, 2], 'key': [1, 2, 1, 2],
                      'pdg': [13, 11, 13, 11], 'isprim': [1, 0, 1, 0], 'ismcprim': [1, 0, 1, 0]})
    r = A.audit(s, p)
    assert r['anchor_events'] == 2 and r['library_eligible_events'] == 1
    assert r['all_template_offaxis_steps'] == 2 and r['all_template_eligible_offaxis_steps'] == 1


def test_audit_rejects_missing_or_duplicate_particle_coverage():
    s = pd.DataFrame({'run': 1, 'subrun': 1, 'event': 1, 'sp': [1, 2], 'sid': [10, 20], 'eion': [1., 2.]})
    p = pd.DataFrame({'run': 1, 'subrun': 1, 'event': 1, 'key': [1], 'pdg': [13], 'isprim': [1], 'ismcprim': [1]})
    with pytest.raises(ValueError, match='metadata'):
        A.audit(s, p)
    with pytest.raises(ValueError, match='duplicate'):
        A.audit(s, pd.concat([p, p]))
