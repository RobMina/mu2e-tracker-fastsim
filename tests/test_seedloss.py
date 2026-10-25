"""Saved seed identity and lost-muon census guards."""
import pandas as pd
import pytest
from trkfastsim.analyses.seedloss import select, summarize, compare


def fixture():
    return pd.DataFrame(dict(run=[1, 1], subrun=[2, 2], event=[3, 4], variant="plane", crv_pass=True,
                             g4_pass=[True, False], seed_pass=[False, True], straight_pass=True,
                             p_seed=[200., 400.], n_seed=[11, 12], fast_pmin=[150., 300.]))


def test_loss_summary_and_recovery():
    base = select([fixture()])
    row = summarize(base)[0]
    assert row["lost"] == row["lost_fast_eleven"] == row["lost_straight_recovers"] == 1
    assert row["lost_fast_zero"] == row["lost_pmin_lt75"] == 0
    candidate = fixture()
    candidate["seed_pass"] = True
    assert compare(base, select([candidate]))["baseline_lost_recovered"] == 1


def test_duplicate_and_mismatch_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        select([fixture(), fixture()])
    with pytest.raises(ValueError, match="identities"):
        compare(select([fixture()]), select([fixture().iloc[:1]]))
    candidate = fixture()
    candidate["p_seed"] += 1
    with pytest.raises(ValueError, match="momenta"):
        compare(select([fixture()]), select([candidate]))
