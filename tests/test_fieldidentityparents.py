"""Saved-ledger admission, exact identities and singleton parent-core diagnostics."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import fieldidentityparents as fip, fieldparents as fp


def frame(subrun, shift=0.0, scale=1.0, n=24):
    p = 2500.0
    x = np.linspace(-0.02, 0.02, n)
    d = pd.DataFrame({"run": np.uint64(1430), "subrun": np.uint64(subrun),
                      "event": np.arange(2**63 + 7, 2**63 + 7 + n, dtype=np.uint64),
                      "p_first": p, "primary_count": np.uint64(1),
                      "fast_ntrk": 1, "g4_ntrk": 1, "fast_sid": 0, "g4_sid": 0,
                      "fast_nactive": 40.0, "g4_nactive": 40.0,
                      "fast_p_first_own": p, "g4_p_first_own": p,
                      "fast_p_reco": p * (1 + shift + scale * x),
                      "g4_p_reco": p * (1 + x)})
    for arm in ("fast", "g4"):
        d[f"{arm}_mc_id"] = pd.array([pd.NA] * n, dtype="UInt64")
        d[f"{arm}_mc_raw_rank0_id"] = pd.array([0] * n, dtype="UInt64")
    return d


def parents():
    return {"pilot": frame(0, 0.8, 10), "a": frame(1, 0.01, 1.5), "b": frame(2, 0.03, 2)}


def save(tmp_path, values=None, unchanged=True):
    paths = {}
    for label, d in (values or parents()).items():
        path = tmp_path / label
        path.mkdir()
        d.to_csv(path / "events.csv.gz", index=False)
        (path / "fieldidentity.json").write_text(json.dumps({
            "events": len(d), "baseline_check": {"unchanged": unchanged}}))
        paths[label] = path
    return paths


def test_no_multi_equal_views_pilot_separate_and_original_core_exact():
    ps = parents()
    r = fip.report(ps, "pilot")
    c = r["confirmation_only"]
    assert c["original_event_proxy"] == c["single_entry_particle"]
    assert c["original_event_proxy"]["parents"] == ["a", "b"]
    assert c["original_event_proxy"]["events"] == 48
    assert r["pilot_report"]["original_event_proxy"]["events"] == 24
    assert r["actual_parent_count"] == 3
    assert r["actual_confirmation_parent_count"] == 2
    original = fp.summary(fp.prepare(ps).query("parent != 'pilot'"))
    expected = fip._finite({k: v for k, v in original["estimates"].items() if k.endswith(fip.SUFFIXES)})
    assert c["original_event_proxy"]["estimates"] == expected
    assert all(k.endswith(fip.SUFFIXES) for k in expected)
    assert c["original_event_proxy"]["estimates"]["bin2_relative_median_difference"]["pooled"] < .03
    assert r["pilot_report"]["original_event_proxy"]["estimates"]["bin2_relative_median_difference"]["pooled"] == pytest.approx(.8)
    assert all(v["same_id_link_known"] == 0 for v in r["per_parent"].values())
    assert "yield_difference" not in json.dumps(r)
    assert "mc_raw_rank0_id" not in json.dumps(r)
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("bad", [False, None, 1, "true"])
def test_baseline_not_literal_true_rejected_without_output(tmp_path, bad):
    paths = save(tmp_path, unchanged=bad)
    with pytest.raises(ValueError, match="baseline_check"):
        fip.run(paths, "pilot", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_receipt_count_mismatch_and_overlap_rejected(tmp_path):
    paths = save(tmp_path)
    receipt = paths["a"] / "fieldidentity.json"
    receipt.write_text('{}')
    with pytest.raises(ValueError, match="baseline_check"):
        fip.run(paths, "pilot", tmp_path / "out")
    receipt.write_text(json.dumps({"events": 25, "baseline_check": {"unchanged": True}}))
    with pytest.raises(ValueError, match="count"):
        fip.run(paths, "pilot", tmp_path / "out")
    ps = parents()
    ps["b"] = ps["a"].copy()
    with pytest.raises(ValueError, match="duplicate event"):
        fip.report(ps, "pilot")
    with pytest.raises(ValueError, match="pilot"):
        fip.report(parents(), "absent")


def test_exclusions_counts_and_delete_parent_errors():
    ps = parents()
    for label in ("a", "b"):
        ps[label].loc[:5, "primary_count"] = 2
    ps["a"].loc[0, "fast_ntrk"] = 0  # multi, but not an original pair
    ps["b"].loc[23, "fast_sid"] = 1  # singleton, but mismatched reco surface
    r = fip.report(ps, "pilot")
    a = r["per_parent"]["a"]
    assert a["original_reference_events"] == 24 and a["multiprimary_events"] == 6
    assert a["original_proxy_pairs"] == 23 and a["singleton_proxy_pairs"] == 18
    assert a["two_to_five_GeV_pairs_removed"] == 5
    assert r["per_parent"]["b"]["singleton_proxy_pairs"] == 17
    c = r["confirmation_only"]
    assert c["changes"]["reference_events"] == -12
    assert c["changes"]["proxy_pairs"] == -11
    d = fp.prepare(ps).query("parent != 'pilot'")
    assert c["original_event_proxy"]["estimates"] == fip._finite({
        k: v for k, v in fp.summary(d)["estimates"].items() if k.endswith(fip.SUFFIXES)})
    for name, selected in (("original_event_proxy", d), ("single_entry_particle", d[d.primary_count == 1])):
        for key, entry in c[name]["estimates"].items():
            if entry["pooled"] is None:
                continue
            values = np.array([fp.metrics(selected[selected.parent != label])[key] for label in ("a", "b")])
            error = np.sqrt(.5 * ((values - values.mean()) ** 2).sum())
            assert entry["delete_one_parent_estimates"] == pytest.approx(values)
            assert entry["delete_one_parent_error"] == pytest.approx(error)
    for key, delta in c["changes"]["relative_core_pooled"].items():
        if delta is not None:
            assert delta == c["single_entry_particle"]["estimates"][key]["pooled"] - c["original_event_proxy"]["estimates"][key]["pooled"]


def test_lost_parent_remains_deletion_unit_and_empty_singleton_explicit():
    ps = parents()
    ps["a"]["primary_count"] = np.uint64(2)
    c = fip.report(ps, "pilot")["confirmation_only"]["single_entry_particle"]
    assert c["parents"] == ["a", "b"]
    assert c["estimates"]["bin2_relative_width_difference"]["delete_one_parent_estimates"][1] is None
    ps["b"]["primary_count"] = np.uint64(2)
    c = fip.report(ps, "pilot")["confirmation_only"]["single_entry_particle"]
    assert c["events"] == 0
    assert c["estimates"]["bin2_relative_width_difference"]["pooled"] is None


def test_cli_exact_integer_reads_hashes_fresh_outdir_and_no_event_ids(tmp_path, capsys):
    ps = parents()
    # A nullable integer larger than the signed range must not take a float detour.
    ps["a"].loc[0, "fast_mc_id"] = 2**63 + 9
    ps["a"]["fast_p_own_id"] = 2500.125
    ps["a"]["fast_mc_invalid_id"] = True
    ps["a"]["fast_foreign_id"] = False
    paths = save(tmp_path, ps)
    d = fip.read_ledger(paths["a"] / "events.csv.gz")
    assert d.fast_p_own_id.iloc[0] == 2500.125
    assert d.fast_mc_invalid_id.all() and not d.fast_foreign_id.any()
    assert d.event.dtype == np.uint64
    assert int(d.event.iloc[0]) == 2**63 + 7
    assert str(d.fast_mc_id.dtype) == "UInt64"
    assert int(d.fast_mc_id.iloc[0]) == 2**63 + 9 and pd.isna(d.fast_mc_id.iloc[1])
    out = tmp_path / "out"
    fip.main([*(f"{k}={v}" for k, v in paths.items()), "--pilot", "pilot", "--outdir", str(out)])
    result = json.loads((out / "fieldidentityparents.json").read_text())
    assert json.loads(capsys.readouterr().out) == result
    for label, inputs in result["inputs"].items():
        for kind, filename in (("report", "fieldidentity.json"), ("ledger", "events.csv.gz")):
            digest = hashlib.sha256((paths[label] / filename).read_bytes()).hexdigest()
            assert inputs[kind]["sha256_before"] == inputs[kind]["sha256_after"] == digest
    assert str(2**63 + 7) not in json.dumps(result)
    with pytest.raises(FileExistsError):
        fip.run(paths, "pilot", out)


def test_mutated_input_rejected_before_output(tmp_path, monkeypatch):
    paths = save(tmp_path)
    original = fip.report
    def mutate(*args):
        result = original(*args)
        p = paths["b"] / "fieldidentity.json"
        p.write_bytes(p.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(fip, "report", mutate)
    with pytest.raises(ValueError, match="closed input changed"):
        fip.run(paths, "pilot", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_saved_csv_original_estimates_match_original_csv_pipeline(tmp_path):
    paths = save(tmp_path)
    loaded = {label: fip.read_ledger(path / "events.csv.gz") for label, path in paths.items()}
    old = fp.summary(fp.prepare({label: pd.read_csv(path / "events.csv.gz")
                                 for label, path in paths.items()}).query("parent != 'pilot'"))
    new = fip.report(loaded, "pilot")["confirmation_only"]["original_event_proxy"]
    assert new["estimates"] == fip._finite({
        k: v for k, v in old["estimates"].items() if k.endswith(fip.SUFFIXES)})
