"""State diagnostics use full generator identity, exclude recrossings, and cluster by event."""
import numpy as np
import pandas as pd
import pytest

from scripts.ordered_g4_state_compare import check_generation, compare, summary


def fixture():
    gen = pd.DataFrame(dict(run=1, subrun=1, event=[1, 1, 1, 2], idx=[0, 1, 2, 0], pdg=[22, 13, -13, 13]))
    fast = pd.DataFrame(dict(run=1, subrun=1, event=[1, 1, 1, 2], sp=[1, 1, 2, 1], sid=[3, 4, 3, 8],
                             t=[100., 102., 101., 200.], px=0., py=100., pz=0., len=5., eion=.001,
                             x0=0., y0=0., z0=0.))
    g4 = fast.copy().assign(sp=[2, 2, 3, 1], t=[99., 100., 100., 199.])
    particles = pd.DataFrame(dict(run=1, subrun=1, event=[1, 1, 2], key=[2, 3, 1], pdg=[13, -13, 13],
                                  ismcprim=1, ee=106.))
    return gen, fast, g4, particles


def test_generator_identity_and_shared_interval_cancel_world_time():
    result, paired = compare(*fixture())
    assert result["matched_straws"] == 4 and result["matched_muons"] == 3
    assert result["observables"]["entry_time_fast_minus_g4_ns"]["event_clusters"] == 2
    assert paired.sp_g4.tolist() == [2, 2, 3, 1]
    interval = result["same_particle_shared_straw_interval"]
    assert interval["muons_with_multiple_shared_straws"] == 1
    assert interval["flight_time_fast_minus_g4_ns"]["mean"] == 1.
    assert result["reference_primary_endpoints"]["range_comparable"] is False


def test_repeated_particle_straw_is_excluded_not_nearest_matched():
    gen, fast, g4, particles = fixture()
    fast = pd.concat([fast, fast.iloc[[0]]], ignore_index=True)
    result, paired = compare(gen, fast, g4, particles)
    assert result["ambiguous_fast_rows_excluded"] == 2
    assert len(paired) == 3


def test_duplicate_primary_and_pdg_disagreement_refuse():
    gen, fast, g4, particles = fixture()
    with pytest.raises(pd.errors.MergeError):
        compare(gen, fast, g4, pd.concat([particles, particles.iloc[[0]]]))
    particles.loc[0, "pdg"] = -13
    with pytest.raises(ValueError, match="PDG"):
        compare(gen, fast, g4, particles)


def test_cluster_variance_uses_event_sums_and_counts():
    events = pd.DataFrame(dict(run=1, subrun=1, event=[1, 1, 2]))
    result = summary([1., 1., 4.], events)
    assert result["mean"] == 2.
    np.testing.assert_allclose(result["cluster_mean_se"], 4. / 3.)
    with pytest.raises(ValueError, match="nonfinite"):
        summary([1., np.nan, 4.], events)


def test_generator_reference_must_match_before_identity_pairing():
    gen = fixture()[0].assign(x=1., y=2., z=3., px=4., py=5., pz=6., t=7.)
    assert check_generation(gen, gen.iloc[:3])["reference_particles"] == 3
    other = gen.iloc[:3].copy()
    other.loc[1, "px"] += 1.
    with pytest.raises(ValueError, match="generator px differs"):
        check_generation(gen, other)
