"""Omission witnesses preserve full identity and exact traversed slice offset."""
import pandas as pd

from scripts.ordered_omission_witness import locate


def test_witness_maps_slice_to_primary_and_last_gas_entry():
    v = pd.DataFrame(dict(run=1, subrun=2, event=[3, 3, 3], sp=4, path=[0., 10., 20.],
        x=[0., 10., 20.], y=0., z=0., ux=1., uy=0., uz=0., kinetic_MeV=100., time_ns=7.))
    s = pd.DataFrame(dict(run=[1], subrun=[2], event=[3], sp=[4], path_mm=[4.]))
    c = pd.DataFrame(dict(lane=[1], s0_mm=[1.], s1_mm=[3.], state="inside", material="G4_Cu",
                           source="pipe", ancestors_verified=True, replaces_material="vacuum"))
    result = locate(v, s, c)
    assert result.event.tolist() == [3] and result.sp.tolist() == [4]
    assert result.cell_start_path_mm.tolist() == [11.]
    assert result.after_last_emitted_gas_entry.tolist() == [True]
