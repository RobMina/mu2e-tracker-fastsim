"""Overlap witnesses preserve same/different source pairs and full primary identity."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts.ordered_overlap_witness import run


def test_witness_localizes_both_kinds_without_admission(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    pd.DataFrame(dict(lane=[0, 0, 0], s0_mm=[0., 1., 1.5], s1_mm=[2., 3., 2.5],
        source=["a", "a", "b"], material=["steel"]*3, state=["inside", "unresolved", "inside"])).to_csv(source / "cells.csv", index=False)
    np.savez(source / "rays.npz", primary_keys=np.array([[1, 2, 3, 4]]))
    for name in ("placements.json", "complete.json"):
        (source / name).write_text("{}")
    result = run(source, out)
    assert result["pairs"] == 3 and result["same_source_pairs"] == 1 and result["lanes"] == 1
    assert not result["admitted_for_physics"]
    rows = pd.read_csv(out / "overlaps.csv")
    assert rows.event.eq(3).all() and rows.sp.eq(4).all()
    with pytest.raises(ValueError, match="cap"):
        run(source, tmp_path / "capped", max_pairs=1)
