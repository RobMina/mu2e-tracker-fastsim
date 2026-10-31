"""Actual omission chronology retains selected, no-gas and stopped identities separately."""
import pandas as pd

from scripts.ordered_geometry_impact import summarize


def test_omission_chronology_and_stop_coverage():
    mu = pd.DataFrame(dict(run=1, subrun=2, event=[3, 4, 5], sp=1, stopped_interior=[0, 1, 1], path_interior_mm=20.))
    seg = pd.DataFrame(dict(run=1, subrun=2, event=[3, 3], sp=1, path_mm=[5., 10.]))
    cells = pd.DataFrame(dict(run=1, subrun=2, event=[3, 3, 4], sp=1, path=0.,
        cell_start_path_mm=[2., 12., 1.], cell_end_path_mm=[4., 15., 2.], cell_length_mm=[2., 3., 1.],
        state="inside", material="steel", loader_aabb_inside_mm=0., vd_cylinder_inside_mm=0.))
    selected = pd.DataFrame(dict(run=[1], subrun=[2], event=[3]))
    joined, totals, identities, stops = summarize(mu, seg, cells, selected)
    assert joined.before_first_gas_mm.tolist() == [2., 0., 0.]
    assert joined.after_last_gas_entry_mm.tolist() == [0., 3., 0.]
    assert joined.selected_event.tolist() == [True, True, False]
    assert joined.has_gas.tolist() == [True, True, False]
    assert len(identities) == 2 and totals.total_mm.sum() == 6.
    assert stops.has_actual_omission.tolist() == [True, False]
