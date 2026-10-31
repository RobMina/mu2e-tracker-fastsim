"""Highest-cycle stock output coverage uses exact selected keys, not exposure metadata."""
import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from scripts.ordered_stock_output_audit import keys, run
from mu2efastsim import tables as core_tables


def test_exact_output_coverage_and_zero_track_row(tmp_path):
    table, ntuple, hits = [tmp_path / f"{name}.root" for name in ("steps", "ntuple", "hits")]
    steps = pd.DataFrame(dict(run=1, subrun=2, event=3, sp=1, px=np.full(12, 100.), py=0., pz=0., eion=.001))
    gen = pd.DataFrame(dict(run=1, subrun=2, event=[3, 4]))
    core_tables.write(str(table), dict(sgsteps=steps, gen=gen), as_ttree=True)
    with uproot.recreate(ntuple) as root:
        root.mktree("EventNtuple/ntuple", {"evtinfo/run": "int64", "evtinfo/subrun": "int64", "evtinfo/event": "int64", "trk/trk.status": "var * int64"})
        root["EventNtuple/ntuple"].extend({"evtinfo/run": np.array([1]), "evtinfo/subrun": np.array([2]),
                                         "evtinfo/event": np.array([3]), "trk/trk.status": ak.Array([np.array([], np.int64)])})
    with uproot.recreate(hits) as root:
        root.mktree("trkdump/events", {name: "int64" for name in core_tables.KEY})
        root["trkdump/events"].extend(dict(run=np.array([1]), subrun=np.array([2]), event=np.array([3])))
    result = run(table, ntuple, hits, tmp_path / "result.json")
    assert result["generated_events"] == 2 and result["selected_events"] == 1
    assert result["output"]["ntuple"]["tracks"] == 0
    assert result["highest_cycle_exact_event_coverage"]
    with pytest.raises(FileExistsError):
        run(table, ntuple, hits, tmp_path / "result.json")
    with uproot.recreate(hits) as root:
        root.mktree("trkdump/events", {name: "int64" for name in core_tables.KEY})
        root["trkdump/events"].extend(dict(run=np.array([1]), subrun=np.array([2]), event=np.array([4])))
    with pytest.raises(ValueError, match="coverage"):
        run(table, ntuple, hits, tmp_path / "bad.json")


def test_duplicate_event_identity_refuses():
    with pytest.raises(ValueError, match="duplicate"):
        keys(pd.DataFrame(dict(run=[1, 1], subrun=[2, 2], event=[3, 3])))
