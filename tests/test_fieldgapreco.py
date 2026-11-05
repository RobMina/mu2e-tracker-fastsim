"""Saved-only field-gap/reco diagnostic contracts."""
import json

import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import fieldgapreco as fr, fieldparents, fieldquantiles


def baseline():
    n = 14
    d = pd.DataFrame({"run": np.ones(n, dtype=np.uint64), "subrun": np.zeros(n, dtype=np.uint64),
                      "event": np.arange(2**53 + 1, 2**53 + 1 + n, dtype=np.uint64),
                      "p_first": [1999.999, 2000., 4999.999, 5000., 300., 1000., 2500., 2600., 2700., 2800., 2900., 3000., 3100., 3200.]})
    for arm in ("fast", "g4"):
        d[f"{arm}_ntrk"] = 1
        d[f"{arm}_sid"] = np.uint64(7)
        d[f"{arm}_nactive"] = 20 if arm == "fast" else 19
        d[f"{arm}_p_first_own"] = 2500.
        d[f"{arm}_p_reco"] = 2500. + np.arange(n) * (2 if arm == "fast" else 1)
    d.loc[6, "fast_sid"] = np.uint64(8)
    d.loc[7, "fast_p_reco"] = np.nan
    d.loc[8, "fast_ntrk"] = 0
    d.loc[9, "g4_ntrk"] = 0
    return d


def outliers(indices=(), foreign=False):
    d = baseline().iloc[list(indices)][fr.KEY].reset_index(drop=True)
    if foreign and len(d):
        d["event"] = d.event + np.uint64(10000)
    d["sp"], d["sid"] = np.uint64(2**63 + 7), np.uint64(7)
    d["originalrowindex"] = np.arange(len(d), dtype=np.uint64) + np.uint64(2**53 + 3)
    d["raw_shortfall_mm"], d["residual_mm"] = 2., .5
    return d


def arms(clock=(), all_=None):
    return {"g4_clock": outliers(clock), "g4_all": outliers(clock if all_ is None else all_)}


def saved(tmp_path, selected=None):
    selected = arms((1, 1, 8, 9), (1, 2)) if selected is None else selected
    base = tmp_path / "events.csv.gz"
    baseline().to_csv(base, index=False)
    original = tmp_path / "original-input"
    original.write_bytes(b"unaltered saved input")
    result = {"denominator_particles": 43156, "inputs": {"saved": {"path": str(original.resolve()), "sha256_before": fr._sha(original),
                                    "sha256_after": fr._sha(original)}}, "outputs": {}, "arms": {}}
    for arm in fr.ARMS:
        path = tmp_path / f"{arm}.csv.gz"
        selected[arm].to_csv(path, index=False)
        result["outputs"][f"{arm}_outliers"] = {"path": str(path.resolve()), "sha256": fr._sha(path)}
        result["arms"][arm] = {"all_outlier_rows": len(selected[arm]),
                              "raw_short_by1_rows": int((selected[arm].raw_shortfall_mm > 1).sum())}
    report = tmp_path / "fieldgasvoids.json"
    report.write_text(json.dumps(result))
    return report, base, result


def test_exact_uint64_default_numeric_parser_and_cli_receipts(tmp_path, capsys):
    report, base, previous = saved(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    output = tmp_path / "diagnostic"
    fr.main(["--void-report", str(report), "--baseline-events", str(base), "--outdir", str(output)])
    result = json.loads((output / "fieldgapreco.json").read_text(),
                        parse_constant=lambda x: pytest.fail(f"not strict JSON: {x}"))
    flagged = pd.read_csv(output / "fieldgapreco_events.csv.gz", dtype={k: "uint64" for k in fr.KEY})
    assert flagged.event.tolist() == baseline().event.tolist()
    assert flagged.g4_clock_largegap.sum() == 3
    assert flagged.loc[1, "g4_clock_outlier_rows"] == 2
    assert flagged.loc[1, "g4_clock_raw_positive_shortfall_sum_mm"] == 4.
    expected = fr._strict(fieldparents.metrics(pd.read_csv(base)))
    assert result["original_baseline"]["metrics"] == expected
    assert result["original_baseline"]["independently_verified"] is False
    assert result["void_denominator_particles_not_reco_events"] == 43156
    assert "yield_difference" not in result["clock_complement_diagnostic_not_efficiency"]["non_yield_metrics"]
    for receipt in result["inputs"].values():
        assert receipt["sha256_before"] == receipt["sha256_after"] == fr._sha(receipt["path"])
    assert result["outputs"]["events_with_flags"]["sha256"] == fr._sha(output / "fieldgapreco_events.csv.gz")
    assert all(p.read_bytes() == value for p, value in before.items())
    exact = fr._read(previous["outputs"]["g4_clock_outliers"]["path"], fr.IDENTITY)
    assert exact.sp.iloc[0] == np.uint64(2**63 + 7)
    assert exact.originalrowindex.iloc[0] == np.uint64(2**53 + 3)
    with pytest.raises(FileExistsError):
        fr.run(report, base, output)
    capsys.readouterr()


def test_precise_bin_pairing_multioutliers_and_gains_losses():
    d = baseline()
    original = d.copy(deep=True)
    flagged, r = fr.report(d, arms((0, 1, 1, 3, 6, 7, 8, 9), (2,)))
    b = r["overlap"]["g4_clock"]["pilot_bin2_2_to_5_GeV"]
    assert b["events"] == 10
    assert b["paired_own_proxy"] == 6
    assert b["largegap_events"] == 5
    assert b["intersection"] == 1
    assert b["union"] == b["inclusion_exclusion"] == 10
    assert b["gained"] == {"all": 1, "flagged": 1, "unflagged": 0}
    assert b["lost"] == {"all": 1, "flagged": 1, "unflagged": 0}
    assert r["flag_agreement_in_baseline"] == {"both": 0, "clock_only": 7, "all_only": 1, "neither": 6}
    assert r["overlap"]["g4_all"]["pilot_bin2_2_to_5_GeV"]["intersection"] == 1
    pd.testing.assert_frame_equal(original, d)
    pd.testing.assert_frame_equal(flagged[original.columns], original)
    expected = fr._strict(fieldparents.metrics(original.loc[~flagged.g4_clock_largegap]))
    assert r["clock_complement_diagnostic_not_efficiency"]["non_yield_metrics"] == {k: v for k, v in expected.items() if "yield" not in k}


def test_zero_outliers_bit_equal_baseline_and_fixed_bootstrap():
    d = baseline()
    _, r = fr.report(d, arms())
    assert r["original_baseline"]["metrics"] == fr._strict(fieldparents.metrics(d))
    assert r["original_baseline"]["relative_core_event_bootstrap"] == fr._strict(fieldquantiles.report(d))
    assert r["clock_complement_diagnostic_not_efficiency"]["relative_core_event_bootstrap"] == r["original_baseline"]["relative_core_event_bootstrap"]
    assert r["clock_complement_diagnostic_not_efficiency"]["events"] == len(d)
    assert r["original_baseline"]["relative_core_event_bootstrap"]["bins"][2]["bootstrap_draws"] == 500
    json.dumps(r, allow_nan=False)


def test_same_count_foreign_events_explicit_not_silently_lost():
    d = baseline()
    foreign = outliers(range(len(d)), foreign=True)
    _, r = fr.report(d, {a: foreign for a in fr.ARMS})
    for arm in fr.ARMS:
        assert r["outside_baseline"][arm] == {"outlier_events_total": len(d), "outlier_events_outside_baseline": len(d), "outlier_rows_outside_baseline": len(d)}
        assert r["overlap"][arm]["events"]["flagged"] == 0
    assert r["original_baseline"]["events"] == len(d)


def test_empty_complement_strict_nulls():
    _, r = fr.report(baseline(), arms(range(len(baseline()))))
    c = r["clock_complement_diagnostic_not_efficiency"]
    assert c["events"] == 0
    assert c["non_yield_metrics"]["bin2_relative_width_difference"] is None
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("kind", ["output", "input", "beforeafter", "canonical"])
def test_corrupt_previous_receipts_fail_closed(tmp_path, kind):
    report, base, previous = saved(tmp_path)
    if kind == "output":
        previous["outputs"]["g4_clock_outliers"]["sha256"] = "0" * 64
    elif kind == "input":
        (tmp_path / "original-input").write_bytes(b"changed")
    elif kind == "beforeafter":
        previous["inputs"]["saved"]["sha256_after"] = "0" * 64
    else:
        previous["inputs"]["saved"]["path"] = str(tmp_path) + "/./original-input"
    report.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="hash|canonical"):
        fr.run(report, base, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("kind", ["duplicate", "nan", "inf", "threshold", "fraction", "negative", "overflow", "truncated"])
def test_invalid_all_outlier_csv_rejected(tmp_path, kind):
    d = outliers((1, 2))
    if kind == "duplicate":
        d = pd.concat([d.iloc[[0]], d.iloc[[0]]], ignore_index=True)
    elif kind in ("nan", "inf"):
        d.loc[0, "residual_mm"] = float(kind)
    elif kind == "threshold":
        d.loc[0, ["raw_shortfall_mm", "residual_mm"]] = 1.
    elif kind in ("fraction", "negative", "overflow"):
        d["sid"] = ["1.5" if kind == "fraction" else "-1" if kind == "negative" else str(2**64), "7"]
    path = tmp_path / "bad.csv"
    d.to_csv(path, index=False)
    with pytest.raises(ValueError):
        fr.read_outliers(path, {"all_outlier_rows": 3 if kind == "truncated" else 2})


@pytest.mark.parametrize("when", ["report", "output"])
@pytest.mark.parametrize("target", ["baseline", "original_input", "previous_csv"])
def test_hashes_rechecked_before_and_after_output(tmp_path, monkeypatch, when, target):
    report, base, _ = saved(tmp_path)
    original = fr.report if when == "report" else pd.DataFrame.to_csv
    changed = base if target == "baseline" else tmp_path / ("original-input" if target == "original_input" else "g4_clock.csv.gz")
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with changed.open("ab") as stream:
            stream.write(b"mutation")
        return result
    if when == "report":
        monkeypatch.setattr(fr, "report", mutate)
    else:
        monkeypatch.setattr(pd.DataFrame, "to_csv", mutate)
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="changed during"):
        fr.run(report, base, output)
    assert not (output / "fieldgapreco.json").exists()


def test_duplicate_baseline_rejected():
    d = pd.concat([baseline(), baseline().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate event"):
        fr.report(d, arms())


def test_residual_only_threshold_union_and_empty_baseline():
    selected = arms((1, 2))
    for d in selected.values():
        d.loc[0, ["raw_shortfall_mm", "residual_mm"]] = [1., 1.001]
    flagged, r = fr.report(baseline(), selected)
    assert flagged.g4_clock_largegap.sum() == 2
    assert flagged.loc[1, "g4_clock_raw_by1_rows"] == 0
    assert flagged.loc[1, "g4_clock_residual_by1_rows"] == 1
    empty, r = fr.report(baseline().iloc[:0].copy(), arms())
    assert empty.empty
    assert r["original_baseline"]["events"] == 0
    json.dumps(r, allow_nan=False)


def test_first_good_same_sid_finite_own_residual_gate_matches_existing():
    d = baseline()
    d.loc[10, "fast_p_first_own"] = 0.
    d.loc[11, "g4_p_reco"] = np.inf
    d.loc[12, "fast_ntrk"] = np.nan
    _, r = fr.report(d, arms((1, 10, 11, 12)))
    q = fieldquantiles.report(d)
    assert r["overlap"]["g4_clock"]["paired_own_proxy"]["all"] == q["common_reco_surface_events"]
    assert r["overlap"]["g4_clock"]["pilot_bin2_2_to_5_GeV"]["paired_own_proxy"] == q["bins"][2]["events"]
    assert r["overlap"]["g4_clock"]["pilot_bin2_2_to_5_GeV"]["intersection"] == 1
