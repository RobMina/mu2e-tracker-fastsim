"""Complete persistent-payload audits fail on any byte or missing product."""
import importlib.util
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot

from test_digitable_report import fixture

SPEC = importlib.util.spec_from_file_location("digitable_report", Path(__file__).parents[1] / "scripts/digitable_report.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def payload(path, *, change=False, missing=False, active=True, empty_active=False):
    cols = {key: np.array([1, 2], np.uint32) for key in ("run", "subrun", "event")}
    cols.update({key: np.full(2, active, np.uint8) for key in ("flag_ph_active", "backgrounds_active")})
    cols.update({key: ak.unflatten(np.array([1, 2, 255], np.uint8), [1, 2])
                 for key in mod.PAYLOAD_BRANCHES - cols.keys()})
    inactive = {"flagged_hits", "cluster_intensity", "background_clusters", "background_hits"}
    if not active or empty_active:
        cols.update({key: ak.unflatten(np.array([], np.uint8), [0, 0]) for key in inactive})
    if change:
        cols["kal_seeds"] = ak.unflatten(np.array([1, 3, 255], np.uint8), [1, 2])
    if missing:
        del cols["background_hits"]
    with uproot.update(path) as root:
        root.mktree("payload/events", {key: (value.type.content if isinstance(value, ak.Array) else value.dtype)
                                     for key, value in cols.items()}).extend(cols)


def test_equal_payload_covers_complete_population(tmp_path):
    fixture(tmp_path)
    payload(tmp_path / "original_reco.root")
    payload(tmp_path / "replay_reco.root")
    result = mod.report(tmp_path, 2, payload=True)
    assert result["equal"] and result["full_persistent_reco_product_equality"] is True


def test_one_changed_byte_fails(tmp_path):
    fixture(tmp_path)
    payload(tmp_path / "original_reco.root")
    payload(tmp_path / "replay_reco.root", change=True)
    result = mod.report(tmp_path, 2, payload=True)
    assert not result["equal"]
    assert result["exported_reco"]["trees"]["payload/events"]["different"] == ["kal_seeds"]


def test_missing_persistent_product_fails(tmp_path):
    fixture(tmp_path)
    payload(tmp_path / "original_reco.root")
    payload(tmp_path / "replay_reco.root", missing=True)
    with pytest.raises(ValueError, match="incomplete persistent"):
        mod.report(tmp_path, 2, payload=True)


def test_null_producer_is_explicitly_inactive(tmp_path):
    fixture(tmp_path)
    payload(tmp_path / "original_reco.root", active=False)
    payload(tmp_path / "replay_reco.root", active=False)
    assert mod.report(tmp_path, 2, payload=True)["equal"]


def test_empty_active_product_fails(tmp_path):
    fixture(tmp_path)
    payload(tmp_path / "original_reco.root", empty_active=True)
    payload(tmp_path / "replay_reco.root", empty_active=True)
    with pytest.raises(ValueError, match="missing active"):
        mod.report(tmp_path, 2, payload=True)


def test_payload_must_match_complete_reco_event_identities(tmp_path):
    fixture(tmp_path)
    payload(tmp_path / "original_reco.root")
    payload(tmp_path / "replay_reco.root")
    for name in ("original_reco.root", "replay_reco.root"):
        with uproot.open(tmp_path / name) as root:
            arrays = root["payload/events"].arrays(library="ak", how=dict)
        arrays["event"] = np.array([1, 3], np.uint32)
        with uproot.update(tmp_path / name) as root:
            root.mktree("payload/events", {key: value.type.content if isinstance(value, ak.Array) else value.dtype
                                          for key, value in arrays.items()}).extend(arrays)
    with pytest.raises(ValueError, match="event population"):
        mod.report(tmp_path, 2, payload=True)
