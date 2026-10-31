"""Saved physical gas slices preserve energy/arc/identity while exposing chord compression."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import ordered_response_probe as P


def fixture(monkeypatch):
    monkeypatch.setattr(P.crossings, "point_straw", lambda pos, geom: np.zeros(len(pos), np.int64))
    monkeypatch.setattr(P.geometry, "to_detector", lambda pos: pos)
    merged = pd.DataFrame(dict(run=[1], subrun=[2], event=[3], sp=[4], sid=[7], plane=[0], panel=[0], straw=[7],
        shape=[0], ion=[0], eion=[.3], len=[2.], t=[10.], width=[0.], nsgs=[1],
        x0=[0.], y0=[0.], z0=[0.], x1=[1.], y1=[1.], z1=[0.], px=[100.], py=[0.], pz=[0.]))
    saved = dict(muon_keys=np.array([[1, 2, 3, 4]]), t_start_ns=np.array([10.]),
        slice_material=np.array([P.steps.STRAW_GAS.name] * 2), slice_it=np.array([0, 0]),
        slice_pos0=np.array([[0., 0., 0.], [1., 0., 0.]]), slice_pos1=np.array([[1., 0., 0.], [1., 1., 0.]]),
        slice_start=np.array([0., 1.]), slice_end=np.array([1., 2.]), slice_t0=np.array([0., .1]),
        slice_ke0=np.array([100., 99.9]), slice_deposited=np.array([.1, .2]),
        slice_dir=np.array([[1., 0., 0.], [0., 1., 0.]]))
    return merged, saved, SimpleNamespace(sid=np.array([7]))


def test_curved_rows_match_writer_but_not_chord(monkeypatch):
    merged, saved, geom = fixture(monkeypatch)
    actual, straight, audit = P.slice_rows(merged, saved, geom)
    assert len(actual) == 2 and actual.nsgs.tolist() == [2, 2]
    np.testing.assert_allclose(actual.eion, [.1, .2])
    np.testing.assert_allclose(actual.t, [10., 10.1])
    np.testing.assert_allclose(straight[["x1", "y1"]], [[.5, .5], [1., 1.]])
    assert audit["multiple_slice_rows"] == 1 and audit["max_slices_per_row"] == 2
    assert audit["max_arc_minus_chord_mm"] == pytest.approx(2 - np.sqrt(2))
    assert audit["max_slice_start_distance_from_merged_chord_mm"] == pytest.approx(np.sqrt(.5))
    assert audit["non_gas_path_inside_merged_span_mm"] == 0
    np.testing.assert_array_equal(merged.nsgs, [1])


@pytest.mark.parametrize("column,value", [("eion", .4), ("len", 3.), ("t", 11.), ("sp", 8), ("x1", 4.)])
def test_saved_writer_mismatch_refuses(monkeypatch, column, value):
    merged, saved, geom = fixture(monkeypatch)
    merged[column] = value
    with pytest.raises(ValueError, match="differ"):
        P.slice_rows(merged, saved, geom)


def test_complete_denominator_retains_empty_and_multimuon_events():
    gen = pd.DataFrame(dict(run=[1, 1, 1], subrun=[2, 2, 2], event=[3, 3, 4]))
    np.testing.assert_array_equal(P.complete_keys(gen), [[1, 2, 3], [1, 2, 4]])
    with pytest.raises(ValueError, match="complete"):
        P.complete_keys(gen.iloc[:0])


def test_gas_gap_not_mislabelled_continuous(monkeypatch):
    merged, saved, geom = fixture(monkeypatch)
    saved["slice_start"][1], saved["slice_end"][1] = 2., 3.
    _, _, audit = P.slice_rows(merged, saved, geom)
    assert audit["non_gas_path_inside_merged_span_mm"] == 1.


@pytest.mark.parametrize("field,value", [("slice_it", [-1, 0]), ("slice_it", [0, 1]),
    ("slice_end", [3., 0.]), ("slice_deposited", [-.1, .4]),
    ("slice_dir", [[2., 0., 0.], [0., 1., 0.]]), ("slice_pos1", [[2., 0., 0.], [1., 1., 0.]])])
def test_malformed_individual_slices_refuse(monkeypatch, field, value):
    merged, saved, geom = fixture(monkeypatch)
    saved[field] = np.asarray(value)
    with pytest.raises(ValueError, match="lane|nonphysical"):
        P.slice_rows(merged, saved, geom)


def test_duplicate_primary_lanes_refuse(monkeypatch):
    merged, saved, geom = fixture(monkeypatch)
    saved["muon_keys"] = np.repeat(saved["muon_keys"], 2, axis=0)
    with pytest.raises(ValueError, match="unique"):
        P.slice_rows(merged, saved, geom)
