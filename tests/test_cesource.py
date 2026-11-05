"""Source identities are exact and declared order cannot be repaired by a chosen mixer shift."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import cesource as C


def keys():
    return pd.DataFrame(dict(run=np.full(3, 1430, np.uint64), subrun=np.full(3, 0, np.uint64),
                             event=np.array([2**54 + 1, 2**54 + 2, 2**54 + 3], np.uint64)))


def test_exact_identity_order_and_complete_source_window():
    source = keys()
    generated = source.assign(subrun=np.uint64(6200), event=np.arange(1, 4, dtype=np.uint64))
    r = C.compare(generated, source.astype(np.int64), source)
    assert r["events"] == 3 and r["exact_sequential_order"]
    assert r["source_keys_sha256"] != r["generated_keys_sha256"]


def test_stock_source_printer_exact_serial_association():
    def row(seq, event, source):
        return f"Event: {seq} run: 1430 subRun: 6200 event: {event}\n" + (
            "EventIDSequence InputTag: label = 'TargetStopResampler', instance = '', process = 'CaloLibraryCE' : "
            f" ( run: 1430 subRun: 0 event: {source} ) \n")
    text = row(1, 1, 2001) + row(2, 2, 2002)
    gen, src = C.read_log(text)
    assert gen.event.tolist() == [1, 2] and src.event.tolist() == [2001, 2002]
    for bad in (text.replace("event: 2002", "event: 2001"), text.replace("Event: 2", "Event: 3"),
                text.replace("event: 2002 )", "event: 2002 ) ( run: 1430 subRun: 0 event: 2003 )"),
                text.replace("CaloLibraryCE", "Other")):
        with pytest.raises(ValueError):
            C.read_log(bad)


def test_overlap_shift_missing_duplicate_and_float_refused():
    d = keys()
    with pytest.raises(AssertionError):
        C.compare(d, d.iloc[::-1], d)
    with pytest.raises(ValueError, match="counts differ"):
        C.compare(d, d.iloc[:2], d)
    with pytest.raises(ValueError, match="duplicate"):
        C.compare(d, pd.concat([d.iloc[:1], d.iloc[:2]]), d)
    with pytest.raises(ValueError, match="invalid event"):
        C.compare(d, d.astype(float), d)
