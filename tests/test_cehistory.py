"""Synthetic finished-stage identity/filter history; no remote art or physics run."""
import json

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from trkfastsim.validation import cehistory as C


def keys(events):
    return pd.DataFrame({"run": np.ones(len(events), np.uint64),
                         "subrun": np.zeros(len(events), np.uint64),
                         "event": np.asarray(events, np.uint64)})


def frames():
    return [keys([1, 2, 3, 4]), keys([2, 3, 4]).assign(key=np.uint64(1)),
            keys([3, 3, 4]).assign(sp=np.uint64(1)), keys([3, 4]), keys([3, 4]),
            keys([4]), keys([4]).assign(ntrack=np.uint64(2))]


def test_full_birth_ledger_partition_skips_and_good_tracks():
    args = frames()
    before = [f.copy(deep=True) for f in args]
    d = C.reconcile(*args)
    assert d.event.tolist() == [1, 2, 3, 4]
    assert d.hasentry.tolist() == [False, True, True, True]
    assert d.hassteps.tolist() == [False, False, True, True]
    assert d.injected.equals(d.digitized)
    assert d.reco_selected.equals(d.ntuple)
    assert d.ntrack.dtype == "UInt64" and d.ntrack.iloc[-1] == 2
    assert d.ntrack.iloc[:3].isna().all()
    r = C.summarize(d)
    assert r["count_partitions"] == dict(no_entry=1, entry_without_steps=1,
                                          digitized_not_reco_selected=1, reco_selected_and_ntuplized=1)
    assert r["partition_total"] == r["births"] == 4
    assert set(r["expected_skipped"].values()) == {1}
    assert r["identity_sha256"]["hassteps"] == r["identity_sha256"]["injected"]
    assert "not physics closure" in r["scope"]
    assert "no zero-event validity certification" in r["scope"]
    for old, new in zip(before, args):
        pd.testing.assert_frame_equal(old, new)


@pytest.mark.parametrize("index", [0, 1, 3, 4, 5, 6])
def test_duplicate_event_records_fail(index):
    args = frames()
    args[index] = pd.concat([args[index], args[index].iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        C.reconcile(*args)


@pytest.mark.parametrize("index", range(1, 7))
def test_foreign_event_records_fail(index):
    args = frames()
    args[index].loc[args[index].index[-1], "event"] = np.uint64(99)
    with pytest.raises(ValueError, match="outside birth"):
        C.reconcile(*args)


@pytest.mark.parametrize("index,events,match", [(3, [3], "injected"), (3, [2, 3, 4], "injected"),
                                              (4, [3], "digitized"), (4, [2, 3, 4], "digitized"),
                                              (5, [2], "reco"), (6, [3], "ntuple"),
                                              (6, [], "ntuple")])
def test_missing_injection_digi_loss_foreign_reco_and_ntuple_mismatch(index, events, match):
    args = frames()
    args[index] = keys(events)
    with pytest.raises(ValueError, match=match):
        C.reconcile(*args)


@pytest.mark.parametrize("column,index,value,match", [("key", 1, 2, "key-1"), ("sp", 2, 2, "foreign steps"),
                                                      ("ntrack", 6, -1, "ntrack")])
def test_particle_and_track_contract(column, index, value, match):
    args = frames()
    args[index] = args[index].assign(**{column: value})
    with pytest.raises(ValueError, match=match):
        C.reconcile(*args)


@pytest.mark.parametrize("index", range(7))
def test_float_ids_fail_before_join(index):
    args = frames()
    args[index]["event"] = args[index].event.astype(float)
    with pytest.raises(ValueError, match="invalid event identities"):
        C.reconcile(*args)


def test_steps_cannot_reference_birth_without_table_entry():
    args = frames()
    args[2] = keys([1, 3, 4]).assign(sp=np.uint64(1))
    with pytest.raises(ValueError, match="foreign steps"):
        C.reconcile(*args)


def test_uint64_keys_above_float_precision_and_signed_range_stay_exact():
    args = frames()
    for i, frame in enumerate(args):
        args[i] = frame.assign(event=frame.event + np.uint64(2**63))
    d = C.reconcile(*args)
    assert all(d[k].dtype == np.uint64 for k in C.KEY)
    assert d.event.tolist() == [2**63 + i for i in (1, 2, 3, 4)]
    assert C.summarize(d)["partition_total"] == 4


@pytest.mark.parametrize("entry", [False, True])
def test_empty_step_tables_and_zero_entries_are_explicit(entry):
    empty = keys([])
    d = C.reconcile(keys([1, 2]), (keys([1]) if entry else empty).assign(key=np.uint64(1)),
                    empty, empty, empty, empty, empty)
    assert len(d) == 2 and not d.hassteps.any() and not d.injected.any()
    assert d.expected_skipped_noentry.sum() == (1 if entry else 2)
    assert d.expected_skipped_zero_steps.sum() == int(entry)
    assert C.summarize(d)["partition_total"] == 2


def test_explicit_empty_birth_denominator_remains_empty_not_a_validity_claim():
    empty = keys([])
    d = C.reconcile(empty, empty.assign(key=np.uint64(1)), empty, empty, empty, empty, empty)
    assert d.empty and C.summarize(d)["partition_total"] == 0
    assert "no zero-event validity certification" in C.summarize(d)["scope"]


def write_tree(root, name, frame):
    tree = root.mktree(name, {c: v.dtype for c, v in frame.items()})
    if len(frame):
        tree.extend({c: v.to_numpy() for c, v in frame.items()})


def write_art(path, events, byteswap=False):
    # Two Events/FileIndex cycles: only the latest closed tree is authoritative.
    frame = keys(events)
    run = np.r_[np.uint32(1), frame.run.to_numpy(np.uint32)]
    subrun = np.zeros(len(events) + 1, np.uint32)
    event = np.r_[np.uint32(0), frame.event.to_numpy(np.uint32)]
    entry = np.r_[np.int64(0), np.arange(len(events), dtype=np.int64)]
    if byteswap:
        run, subrun, event, entry = [v.byteswap() for v in (run, subrun, event, entry)]
    record = ak.Array(dict(eventID=dict(subRun_=dict(run_=dict(run_=run), subRun_=subrun), event_=event),
                           entry=entry))
    with uproot.recreate(path) as root:
        root["FileIndex"] = {"Element": record}
        root["FileIndex"] = {"Element": record}
        write_tree(root, "Events", keys([99]))
        write_tree(root, "Events", frame)


def write_ntuple(path, events):
    with uproot.recreate(path) as root:
        for current in ([99], events):
            frame = keys(current)
            columns = {f"evtinfo/{k}": frame[k].to_numpy() for k in C.KEY}
            columns["trk/trk.status"] = ak.unflatten(np.tile([1, -1, 2], len(frame)),
                                                     np.full(len(frame), 3, np.int64))
            root[C.TREE] = columns


@pytest.mark.parametrize("events", [[3, 4], []])
@pytest.mark.parametrize("byteswap", [False, True])
def test_highest_closed_art_count_zero_and_explicit_byteswap(tmp_path, events, byteswap):
    path = tmp_path / "stage.art"
    write_art(path, events, byteswap)
    d, info = C.load_art(path, byteswap)
    assert d.event.tolist() == events
    assert info["Events"] == dict(cycle=2, entries=len(events))
    assert info["FileIndex"]["cycle"] == 2


@pytest.mark.parametrize("events", [[4], []])
def test_ntuple_direct_highest_cycle_and_good_track_count(tmp_path, events):
    path = tmp_path / "nts.root"
    write_ntuple(path, events)
    d, info = C.load_ntuple(path)
    assert d.event.tolist() == events
    assert d.ntrack.tolist() == [2] * len(events)
    assert info[C.TREE] == dict(cycle=2, entries=len(events))


@pytest.mark.parametrize("empty_tables", [False, True])
def test_cli_read_only_artifacts_receipts_fresh_outputs_and_empty_writer_convention(tmp_path, empty_tables):
    args = frames()
    with uproot.recreate(tmp_path / "states.root") as root:
        write_tree(root, "dump/gen", args[0])
    with uproot.recreate(tmp_path / "table.root") as root:
        if not empty_tables:
            write_tree(root, "dump/particles", args[1])
            write_tree(root, "dump/sgsteps", args[2])
    for name, events in (("inj.art", [] if empty_tables else [3, 4]),
                         ("dig.art", [] if empty_tables else [3, 4]),
                         ("rec.art", [] if empty_tables else [4])):
        write_art(tmp_path / name, events, byteswap=True)
    write_ntuple(tmp_path / "nts.root", [] if empty_tables else [4])
    original = {p: C.cestopwindow.checksum(p) for p in tmp_path.iterdir()}
    argv = ["--births", str(tmp_path / "states.root"), "--table", str(tmp_path / "table.root"),
            "--inject-art", str(tmp_path / "inj.art"), "--digi-art", str(tmp_path / "dig.art"),
            "--reco-art", str(tmp_path / "rec.art"), "--ntuple", str(tmp_path / "nts.root"),
            "--fileindex-byteswap", "--outdir", str(tmp_path / "out")]
    C.main(argv)
    r = json.loads((tmp_path / "out/cehistory.json").read_text())
    d = pd.read_csv(tmp_path / "out/events.csv.gz", dtype={k: "uint64" for k in C.KEY})
    assert r["births"] == r["partition_total"] == len(d) == 4
    assert r["explicit_uproot_metadata_byteswap"]
    assert r["inputs"]["injected"]["trees"]["Events"]["cycle"] == 2
    assert r["zero_event_stages"] == (["injected", "digitized", "reco", "ntuple"] if empty_tables else [])
    if empty_tables:
        assert r["inputs"]["table"]["trees"]["dump/particles"]["absent_empty_writer_convention"]
        assert r["expected_skipped"]["expected_skipped_noentry"] == 4
    assert all(C.cestopwindow.checksum(p) == digest for p, digest in original.items())
    with pytest.raises(FileExistsError, match="fresh"):
        C.main(argv)


def test_equal_counts_do_not_replace_exact_identity_sets_and_order_is_irrelevant():
    args = frames()
    d = C.reconcile(*[f.iloc[::-1].reset_index(drop=True) for f in args])
    pd.testing.assert_frame_equal(d, C.reconcile(*args))
    for index, events, match in ((3, [2, 4], "injected"), (4, [2, 4], "digitized")):
        changed = frames()
        changed[index] = keys(events)
        with pytest.raises(ValueError, match=match):
            C.reconcile(*changed)


def test_failed_cli_reconciliation_creates_no_output_and_preserves_inputs(tmp_path):
    args = frames()
    with uproot.recreate(tmp_path / "states.root") as root:
        write_tree(root, "dump/gen", args[0])
    with uproot.recreate(tmp_path / "table.root") as root:
        write_tree(root, "dump/particles", args[1])
        write_tree(root, "dump/sgsteps", args[2])
    for name, events in (("inj.art", [3, 4]), ("dig.art", [3, 4]), ("rec.art", [4])):
        write_art(tmp_path / name, events)
    write_ntuple(tmp_path / "nts.root", [3])
    before = {p: C.cestopwindow.checksum(p) for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="ntuple"):
        C.main(["--births", str(tmp_path / "states.root"), "--table", str(tmp_path / "table.root"),
                "--inject-art", str(tmp_path / "inj.art"), "--digi-art", str(tmp_path / "dig.art"),
                "--reco-art", str(tmp_path / "rec.art"), "--ntuple", str(tmp_path / "nts.root"),
                "--outdir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
    assert all(C.cestopwindow.checksum(p) == digest for p, digest in before.items())


def test_missing_birth_tree_is_not_an_empty_denominator(tmp_path):
    path = tmp_path / "empty.root"
    with uproot.recreate(path):
        pass
    with pytest.raises(ValueError, match="required birth tree"):
        C.load_table(path, {"gen": C.KEY})


def test_verify_closed_report_rehashes_inputs_and_detects_changed_or_missing(tmp_path):
    inputs = {}
    for name in ("births", "table", "injected", "digitized", "reco", "ntuple"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        inputs[name] = {"path": str(path), "sha256": C.cestopwindow.checksum(path)}
    report = tmp_path / "history.json"
    report.write_text(json.dumps({"inputs": inputs}))
    result = C.verify_report(report)
    assert result["verified"] and len(result["verified_inputs"]) == 6
    (tmp_path / "table").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed: table"):
        C.verify_report(report)
    del inputs["table"]
    report.write_text(json.dumps({"inputs": inputs}))
    with pytest.raises(ValueError, match="every exact"):
        C.verify_report(report)
