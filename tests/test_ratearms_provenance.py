"""Completion and identity gates for rate-ledger comparisons and resumable chunks."""
import json

import pandas as pd
import pytest

from trkfastsim.analyses import ratearms as ra


def _chunk(directory, tag="0000", index=0, event=1, settings=None, passes=1):
    directory.mkdir(exist_ok=True)
    config = settings or {"arm": "legacy", "source": {"steps.py": "test"}}
    rows = pd.DataFrame({"run": [1], "subrun": [1], "event": [event], "sp": [1],
                         "file_index": [index], "ngood": [12]})
    rows.to_parquet(directory / f"muons.{tag}.parquet", index=False)
    records = [{"settings": config}, {"file": f"gen.{index}.root", "index": index,
               "sha256": f"hash{index}", "gen_events": 50000, "events_kept": 1, "fast_pass_any": passes}]
    (directory / f"files.{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return config


def test_complete_chunk_and_empty_kept_events(tmp_path):
    _chunk(tmp_path)
    rows, files, settings = ra.load_arm(tmp_path)
    assert len(rows) == len(files) == len(settings) == 1
    assert files[0]["gen_events"] == 50000


def test_partial_chunk_manifest_is_not_silently_skipped(tmp_path):
    _chunk(tmp_path)
    (tmp_path / "files.0055.jsonl").write_text('{"settings": {}}\n')
    with pytest.raises(ValueError, match="incomplete chunk"):
        ra.load_arm(tmp_path)


def test_wrong_passing_event_count_is_rejected(tmp_path):
    _chunk(tmp_path, passes=2)
    with pytest.raises(ValueError, match="pass count"):
        ra.load_arm(tmp_path)


def test_mixed_source_chunks_are_rejected(tmp_path):
    _chunk(tmp_path)
    _chunk(tmp_path, tag="0001", index=1, event=2, settings={"arm": "legacy", "source": {"steps.py": "other"}})
    with pytest.raises(ValueError, match="mixed settings/source"):
        ra.load_arm(tmp_path)


def test_duplicate_event_muon_identity_is_rejected(tmp_path):
    _chunk(tmp_path)
    _chunk(tmp_path, tag="0001", index=1)
    with pytest.raises(ValueError, match="duplicate muon identities"):
        ra.load_arm(tmp_path)


def test_missing_requested_history_is_rejected(tmp_path):
    _chunk(tmp_path, settings={"history_steps": True})
    with pytest.raises(ValueError, match="segment history is missing"):
        ra.load_arm(tmp_path)


def test_chunk_check_requires_exact_inputs_and_physics(tmp_path, monkeypatch):
    config = {"arm": "legacy", "seed": 1, "gdml_sha256": "geometry", "g4": "g4.parquet",
              **ra.ledger.preset("legacy").as_dict(), **ra.ledger.core_world_convention(),
              "source": {"steps.py": "transport"}}
    _chunk(tmp_path, settings=config)
    monkeypatch.setattr(ra, "_sha", lambda path: {"gen0.root": "hash0", "gen1.root": "hash1", "geom.gdml": "geometry"}[path])
    monkeypatch.setattr(ra.T, "expand", lambda paths: paths)
    monkeypatch.setattr(ra, "_src_identity", lambda: {"steps.py": "transport", "analyses/ratearms.py": "reporter"})
    ra.check_arm(tmp_path, ["gen0.root"], 0, "legacy", 1, "geom.gdml", "g4.parquet", tag="0000")
    with pytest.raises(ValueError, match="input file coverage"):
        ra.check_arm(tmp_path, ["gen0.root", "gen1.root"], 0, "legacy", 1, "geom.gdml", "g4.parquet")
    with pytest.raises(ValueError, match="requested settings"):
        ra.check_arm(tmp_path, ["gen0.root"], 0, "legacy", 2, "geom.gdml", "g4.parquet")
    monkeypatch.setattr(ra, "_src_identity", lambda: {"steps.py": "changed"})
    with pytest.raises(ValueError, match="transport source"):
        ra.check_arm(tmp_path, ["gen0.root"], 0, "legacy", 1, "geom.gdml", "g4.parquet")


def test_any_muon_rate_counts_event_once_without_summing_tracks():
    rows = pd.DataFrame({"run": 1, "subrun": 1, "event": [1, 1, 2, 2, 3, 3],
                         "sp": [1, 2, 1, 2, 1, 2], "ngood": [0, 12, 6, 6, 12, 13]})
    ev = ra._events(rows, "fast")
    assert ev.fast_any.tolist() == [True, False, True]
    assert ev.fast_mu1.tolist() == [False, False, True]
    assert ev.fast_nmu_pass.tolist() == [1, 0, 2]
