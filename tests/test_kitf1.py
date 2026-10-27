"""Stock-conditioned companion source guard and writer; no F1 physics closure claim."""
import json
from pathlib import Path

import numpy as np
import pytest

from trkfastsim import kitf1
from test_ntuple_writer import _table, VERT


def saved(path, source, own=True):
    np.savez(path, meta=json.dumps(dict(input=str(source), own_states=own, max_tracks=2)))


def test_saved_source_guard(tmp_path):
    dump, tracks = tmp_path / "stock.root", tmp_path / "f1.npz"
    saved(tracks, dump)
    assert kitf1.check_saved_source(tracks, dump)["own_states"]
    with pytest.raises(ValueError, match="source path"):
        kitf1.check_saved_source(tracks, tmp_path / "other.root")
    saved(tracks, dump, False)
    with pytest.raises(ValueError, match="own-state"):
        kitf1.check_saved_source(tracks, dump)


def test_saved_companion_never_runs_fitter(tmp_path, monkeypatch):
    dump, cond, tracks = tmp_path / "stock.root", tmp_path / "cond.json", tmp_path / "f1.npz"
    dump.write_text("fixture")
    cond.write_text("{}")
    saved(tracks, dump)
    tb = _table([VERT], n_events=3)
    tb["meta"].update(fit_type="F1", stock_conditioned=True, conditioning="stock seeds, hits and crossings")
    monkeypatch.setattr(kitf1.hitreco, "load_conditions", lambda path: {})
    monkeypatch.setattr(kitf1.tracktable, "from_f1", lambda *args, **kwargs: tb)
    out = tmp_path / "out"
    evidence = kitf1.run(dump, cond, out, tracks=tracks)
    assert evidence["stock_conditioned"] and evidence["not_generated_event_efficiency"]
    assert evidence["persisted"]["written_events"] == 3
    assert evidence["persisted"]["fit_successes"] == 1
    assert not (out / "metrics.json").exists()
    assert (out / "EventNtuple_F1_stock_conditioned.root").is_file()
    with pytest.raises(ValueError, match="empty output"):
        kitf1.run(dump, cond, out, tracks=tracks)
