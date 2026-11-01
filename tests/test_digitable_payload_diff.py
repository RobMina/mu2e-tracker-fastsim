"""Byte localization is diagnostic only and retains the full unique event population."""

from test_digitable_payload import fixture, payload

from trkfastsim.validation import digitable_payload_diff as mod


def test_one_byte_localizes_event_and_offset(tmp_path):
    fixture(tmp_path)
    payload(tmp_path / "original_reco.root")
    payload(tmp_path / "replay_reco.root", change=True)
    result = mod.differences(tmp_path / "original_reco.root", tmp_path / "replay_reco.root")
    changed = result["kal_seeds"]
    assert changed["different_events"] == changed["different_bytes"] == 1
    assert changed["event_ids"] == [[2, 2, 2]]
    assert changed["sample_rows"] == [1] and changed["sample_offsets"] == [0]
    assert changed["sample_reference"] == [2] and changed["sample_candidate"] == [3]
    assert result["line_seeds"]["different_bytes"] == 0
