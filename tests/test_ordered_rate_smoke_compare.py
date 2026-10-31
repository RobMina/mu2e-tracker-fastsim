"""Primary-filter smoke census excludes nonmuons and does not sum muon counts."""
import numpy as np
import pandas as pd

from scripts.ordered_rate_smoke_compare import primary_flags


def test_primary_flags_keep_any_muon_not_sum_or_secondary():
    particles = pd.DataFrame(dict(run=[1] * 4, subrun=[1] * 4, event=[1, 1, 2, 3],
        key=[1, 2, 1, 1], isprim=[1, 1, 0, 1], pdg=[13, -13, 13, 11]))
    steps = pd.DataFrame(dict(run=[1] * 36, subrun=[1] * 36,
        event=np.repeat([1, 1, 2, 3], [6, 6, 12, 12]), sp=np.repeat([1, 2, 1, 1], [6, 6, 12, 12]),
        px=0., py=-100., pz=0., eion=.01))
    assert primary_flags(steps, particles).empty
    extra = steps.iloc[:6].copy().assign(sp=2)
    result = primary_flags(pd.concat([steps, extra], ignore_index=True), particles)
    assert result.event.tolist() == [1]


def test_rate_probe_rejects_missing_generated_reference(tmp_path, monkeypatch):
    import json
    import pytest
    from scripts import ordered_rate_smoke_compare as probe
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "complete.json").write_text(json.dumps(dict(status="bounded_full_chain_complete", generated_events=1)))
    (directory / "ordered_steps.root").write_bytes(b"fixture")
    g4 = tmp_path / "g4.root"
    g4.write_bytes(b"fixture")
    generated = pd.DataFrame(dict(run=[1], subrun=[2], event=[3], idx=[0]))
    monkeypatch.setattr(probe.tables, "read", lambda path, tree: generated)
    monkeypatch.setattr(probe, "check_generation", lambda a, b: dict(reference_events=0))
    with pytest.raises(ValueError, match="no matching"):
        probe.run(directory, g4, tmp_path / "out.json")
