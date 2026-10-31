"""Bound continuation inputs and complete partial-refusal receipts."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import ordered_torus_evolution as E


def test_pair_provenance_binds_every_consumed_capture_geometry_and_algorithm(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    files = [tmp_path / n for n in ("traversed.npz", "segments.csv", "complete.json", "fixture.gdml")]
    files += [source / f"{n}.py" for n in ("navigationcensus", "torusprobe", "omittedpath", "supportnavigation", "supportenvelope")]
    for path in files:
        path.write_text(str(path))
    receipt = dict(input_sha256={str(p): E.digest(p) for p in files})
    E.bind_pair_inputs(receipt, tmp_path, files[3], source)
    for path in files:
        old = path.read_text()
        path.write_text("substitute")
        with pytest.raises(ValueError, match="not bound"):
            E.bind_pair_inputs(receipt, tmp_path, files[3], source)
        path.write_text(old)


def test_staged_refusal_saves_nested_control_and_effective_policies(tmp_path):
    frame = pd.DataFrame(dict(lane=[0], reason=["original-refusal"]))
    call = dict(origin=np.zeros((1, 3)), direction=np.array([[1., 0., 0.]]), lengths=np.array([3.]),
        stage="paired-straw-refinement", status="refused", reason="overlap", replacements_strictly_below_plane=True,
        cells=frame, reports=[], endpoints=dict(endpoints=frame, summary={"actual_max_shift_mm": 1e-12}),
        result=dict(control=dict(decisions=frame, refusals=frame, adjusted_hits=frame), summary={"endpoint_budget_mm": 0.}))
    provider = SimpleNamespace(calls=[call], policy="fixture:inner", plane=4., endpoint_budget=1e-10,
        min_cell=1e-7, component_budget=1e-6, ray_budget=1e-5, max_cells=100, max_pairs=200)
    E.save_navigation(provider, tmp_path)
    target = tmp_path / "navigation_0000"
    assert (target / "result.control.refusals.csv").is_file()
    assert (target / "result.control.adjusted_hits.csv").is_file()
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["status"] == "refused" and receipt["replacements_strictly_below_plane"]
    assert receipt["endpoint_budget_mm"] == 1e-10
    assert receipt["composition_additional_endpoint_shift_mm"] == 0.
    assert receipt["endpoint_summary"]["actual_max_shift_mm"] == 1e-12
    with pytest.raises(FileExistsError):
        E.save_navigation(provider, tmp_path)


def test_new_evolution_payload_needs_no_object_pickle(tmp_path):
    payload = E.safe_payload(dict(slice_kind=np.array(["support", "thin"], object), numeric=np.array([1., 2.])))
    path = tmp_path / "safe.npz"
    np.savez_compressed(path, **payload)
    with np.load(path, allow_pickle=False) as saved:
        assert saved["slice_kind"].tolist() == ["support", "thin"]
        assert saved["numeric"].tolist() == [1., 2.]
    with pytest.raises(ValueError, match="only string"):
        E.safe_payload(dict(invalid=np.array([object()], object)))
