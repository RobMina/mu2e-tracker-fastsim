"""Partial upstream ledger requires exact raw-product ancestry and preserves all birth events."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import cematerial as cm


def data():
    den = pd.DataFrame({"run": np.full(2, 1430, dtype=np.uint64), "subrun": np.full(2, 6000, dtype=np.uint64),
                        "event": np.array([2**54 + 1, 2**54 + 3], dtype=np.uint64)})
    events = den.assign(g4_status=0)
    reg = den.assign(kind=-1, product_id=20, branch_name="mu2e::SimParticle_g4run__CaloLibraryCE.")
    generated = den.assign(pdg=11, creation=56, parent_valid=True, parent_product_id=10, parent_key=7)
    original = den.assign(sim_product_id=10, sim_key=7, parent_valid=False, parent_product_id=0, parent_key=0, pdg=13, creation=1)
    reseated = original.assign(sim_product_id=20)
    electron = den.assign(sim_product_id=20, sim_key=8, parent_valid=True, parent_product_id=20, parent_key=7, pdg=11, creation=56)
    particles = pd.concat([original, reseated, electron], ignore_index=True)
    s = pd.concat([den.iloc[:1]] * 5, ignore_index=True).assign(
        sim_product_id=20, sim_key=[8, 8, 8, 8, 9], product_id=[30, 31, 31, 32, 31], index=[0, 0, 1, 0, 2],
        kind=[0, 3, 3, 4, 3], volume=[13, 2, 2, 0, 3], time=[10., 5., 15., 6., 5.], length=[0., 1., 9., 2., 100.],
        energy_g4=[0., 0.1, 9., 0.2, 100.], momentum_x=100., momentum_y=0., momentum_z=0.,
        post_momentum_x=99., post_momentum_y=0., post_momentum_z=0.)
    return events, reg, particles, generated, s


def test_exact_parent_reseating_rejects_secondary_and_missing_original_or_reseated_parent():
    e, r, p, g, s = data()
    roots = cm.primary_roots(e, r, p, g)
    assert (roots.sim_key == 8).all() and (roots.sim_product_id == 20).all()
    with pytest.raises(ValueError, match="original StageParticle parent"):
        cm.primary_roots(e, r, p[p.sim_product_id != 10], g)
    with pytest.raises(ValueError, match="reseated StageParticle parent"):
        cm.primary_roots(e, r, p[~((p.sim_product_id == 20) & (p.sim_key == 7))], g)
    with pytest.raises(ValueError, match="duplicate persistent"):
        cm.primary_roots(e, r, pd.concat([p, p.iloc[:1]]), g)
    with pytest.raises(ValueError, match="status-zero"):
        cm.primary_roots(e.assign(g4_status=1), r, p, g)


def test_material_prefix_zero_rows_and_direct_primary_identity_no_encounter_claim():
    e, r, p, g, s = data()
    d, counts = cm.ledger(cm.primary_roots(e, r, p, g), s)
    assert len(d) == 2 and d.g4_target_records.tolist() == [1, 0]
    assert d.g4_target_length_mm.tolist() == [1., 0.]
    assert d.g4_absorber_length_mm.tolist() == [2., 0.]
    assert counts == dict(raw_primary_material_rows=3, prefix_material_rows=2, rows_after_or_at_vd13=1,
                          direct_primary_vd13_events=1)
    np.testing.assert_allclose(d.g4_target_primary_delta_MeV.iloc[0],
                               np.hypot(100., cm.ME) - np.hypot(99., cm.ME))
    with pytest.raises(ValueError, match="duplicate raw step"):
        cm.ledger(cm.primary_roots(e, r, p, g), pd.concat([s, s.iloc[:1]]))
    with pytest.raises(ValueError, match="invalid material path"):
        cm.ledger(cm.primary_roots(e, r, p, g), s.assign(length=-1.))


def test_material_attach_requires_identical_births_and_reports_reciprocal_classes():
    e, r, p, g, s = data()
    material, counts = cm.ledger(cm.primary_roots(e, r, p, g), s)
    events = e[cm.KEY].assign(g4_steps=[1, 0], chain_steps=[0, 1], g4_track=[True, False], chain_track=[False, True])
    d = cm.attach(events, material)
    out = cm.report(d, counts)
    assert out["events"] == 2 and out["classes"]["g4_only_steps"]["events"] == 1
    assert out["classes"]["fast_only_steps"]["events"] == 1
    assert "no fast per-material path ledger" in out["limits"]
    with pytest.raises(AssertionError):
        cm.attach(events.iloc[:1], material)


def test_saved_reference_vd13_coverage_and_clock_must_agree():
    e, r, p, g, s = data()
    material, counts = cm.ledger(cm.primary_roots(e, r, p, g), s)
    events = e[cm.KEY].assign(entry_has_vd13=[True, False], entry_t=[10., np.nan])
    assert len(cm.attach(events, material)) == 2
    with pytest.raises(ValueError, match="coverage differs"):
        cm.attach(events.assign(entry_has_vd13=True), material)
    with pytest.raises(ValueError, match="clock differs"):
        cm.attach(events.assign(entry_t=11.), material)


def test_no_vd13_retains_entire_saved_primary_material_history():
    e, r, p, g, s = data()
    d, counts = cm.ledger(cm.primary_roots(e, r, p, g), s[s.kind != 0])
    assert d.g4_vd13_t.isna().all() and d.g4_target_length_mm.tolist() == [10., 0.]
    assert counts["rows_after_or_at_vd13"] == 0


def test_common_entry_classes_separate_no_vd13_whole_history_from_prefixes():
    e, r, p, g, s = data()
    material, counts = cm.ledger(cm.primary_roots(e, r, p, g), s)
    events = e[cm.KEY].assign(g4_steps=[1, 0], chain_steps=[0, 1], g4_track=[True, False], chain_track=[False, True],
                            chain_has_vd13=True)
    result = cm.report(cm.attach(events, material), counts)
    assert result["classes"]["all_births"]["events"] == 2
    assert result["common_vd13_classes"]["all_births"]["events"] == 1
    assert result["common_vd13_classes"]["fast_only_steps"]["events"] == 0
