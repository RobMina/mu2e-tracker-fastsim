"""Saved placement-counter audit guards and accounting."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('placement_audit', Path(__file__).parents[1] / 'scripts/secondary_placement_audit.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def _row(**kw):
    return {'file': 'group.root', 'drawn': 10, 'placed': 9, 'in_muon_straw': 1, 'off_lattice': 0,
            'merged_same_particle': 2, 'reflected': 3, 'draw_level_counts': [1], 'distinct_with_steps': 1,
            'top10_step_share': 1., **kw}


def _census(n):
    row = {'bin': 1, 'steps': [n, 8], 'ratio_minus_1': n / 8 - 1, 'err': .1}
    return {'selected_events': {'fast': 1, 'g4': 1, 'both': 1}, 'totals': {'fast': {'mu_steps': 5, 'mu_eion': 4.}},
            'breakdowns': {'location': [row], 'plane_distance': [row]}}


def test_audit_preserves_counter_conservation_and_changes():
    result = A.compare(_census(7), _census(6), [_row()], [_row(merged_same_particle=3, reflected=4)])
    assert result['same_draw_summaries']
    assert result['counters']['edge_minus_base']['output_offaxis_rows'] == -1
    assert result['migration']['location'][0]['edge_minus_base_steps'] == -1


def test_audit_rejects_mismatched_coverage_and_nonconserving_counts():
    with pytest.raises(ValueError, match='coverage'):
        A.compare(_census(7), _census(6), [_row()], [_row(file='other.root')])
    with pytest.raises(ValueError, match='nonconserving'):
        A.compare(_census(7), _census(6), [_row()], [_row(placed=10)])


def test_log_parser_rejects_incomplete_and_duplicate_groups(tmp_path):
    p = tmp_path / 'steps.txt'
    p.write_text("#   off-axis: {'drawn': 1}\n")
    with pytest.raises(ValueError, match='missing'):
        A.read_log(p)
    line = "#   off-axis: {'drawn': 1}\nwrote /data/group.root\n"
    p.write_text(line)
    assert A.read_log(p) == [{'file': 'group.root', 'drawn': 1}]
    p.write_text(line + line)
    with pytest.raises(ValueError, match='duplicate'):
        A.read_log(p)
