"""Stock injection must retain all source events and exact generator/primary identities."""
import pandas as pd
import pytest

from scripts.ordered_stock_preflight import validate


def frames():
    gen = pd.DataFrame(dict(run=1, subrun=2, event=[3, 3, 4], idx=[0, 1, 0], pdg=[22, 13, -13],
        x=0., y=0., z=0., px=0., py=-100., pz=0., e=150., t=0.))
    parts = pd.DataFrame(dict(run=1, subrun=2, event=[3, 4], key=[1, 1], pdg=[13, -13], parent=-1, isprim=1))
    straw = pd.DataFrame(dict(run=[1], subrun=[2], event=[3], sp=[1], px=[0.], py=[100.], pz=[0.], eion=[.001]))
    return gen, parts, straw


def test_exact_source_and_fast_muon_primary_ids():
    gen, parts, straw = frames()
    result = validate(gen, gen.copy(), parts, straw, straw.iloc[:0], expected_events=2)
    assert result["generated_events"] == 2 and result["generated_particles"] == 3
    assert result["primary_muons"] == 2 and result["tracker_filter_passes"] == 0


def test_wrong_source_or_missing_primary_refuses():
    gen, parts, straw = frames()
    with pytest.raises(ValueError, match="event count"):
        validate(gen, gen.iloc[:2], parts, straw, straw, expected_events=2)
    with pytest.raises(AssertionError):
        validate(gen, gen.assign(t=1.), parts, straw, straw, expected_events=2)
    with pytest.raises(ValueError, match="parentless"):
        validate(gen, gen, parts.assign(parent=0), straw, straw, expected_events=2)
    with pytest.raises(ValueError, match="unlinked"):
        validate(gen, gen, parts, straw.assign(sp=8), straw, expected_events=2)


def test_identically_truncated_source_and_table_refuse_requested_prefix():
    gen, parts, straw = frames()
    with pytest.raises(ValueError, match="event count"):
        validate(gen.iloc[:2], gen.iloc[:2], parts.iloc[:1], straw, straw, expected_events=2)


def test_source_acquisition_hash_and_completed_prefix_log_required(tmp_path):
    from scripts.ordered_stock_preflight import run
    from scripts.transport_capture_audit import digest
    source, log, table, dump = [tmp_path / n for n in ("source.art", "source.log", "table.root", "dump.root")]
    for path in (source, table, dump):
        path.write_bytes(b"fixture")
    log.write_text("TrigReport Events total = 1 passed = 1 failed = 0\nArt has completed and will exit with status 0.\n")
    args = dict(expected_events=2, source_log=log, source_art=source, trusted_source_sha256=digest(source))
    with pytest.raises(ValueError, match="requested complete prefix"):
        run(table, dump, tmp_path / "out.json", **args)
    args["trusted_source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="trusted staged"):
        run(table, dump, tmp_path / "out.json", **args)
