"""Synthetic Run1A hit comparison tests; no saved detector data required."""
import copy
import json
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

from trkfastsim.reco import hitreco, combo
from trkfastsim.validation import run1a_reco as reco


def conditions():
    st = {k: [0.] * 3 for k in ("midx", "midy", "midz", "dirx", "diry", "dirz", "hlen", "pmpscale",
                                "toff_hv", "toff_cal", "dead", "noisy", "halfvp")}
    st.update(midx=[500.] * 3, diry=[1.] * 3, hlen=[400.] * 3, halfvp=[100.] * 3)
    return {"scalar": dict(tdcLSB=1., adcLSB=1., totLSB=1., strawGain=1., eionPerQ=1e-5,
                           pmpScaleAvg=1., strawOuterRadius=2.5, nADCPreSamples=2),
            "straw": st, "halfvpscale": [1.] * 59, "centres": [30.] * 59, "resslope": [0.] * 59,
            "totdtime": [1.] * 80, "totderror": [1.] * 80, "tot_bins": [16, 5., 5, .0005]}


def jagged(values, dtype=None):
    counts = np.array([len(a) for a in values], dtype=int)
    return ak.unflatten(np.array([v for a in values for v in a], dtype=dtype), counts)


def rows(empty=False):
    n = 0 if empty else 2
    raw = {"run": np.array([1430, 1430, 1430], dtype=np.uint32),
           "subrun": np.array([0, 0, 1], dtype=np.uint32), "event": np.array([1, 2, 1], dtype=np.uint32),
           "spill": np.zeros(3, dtype=int), "pbt": np.zeros(3)}
    d = dict(sid=np.array([0, 1][:n]), tdc0=np.full(n, 50), tdc1=np.full(n, 50),
             tot0=np.full(n, 5), tot1=np.full(n, 5), pmp=np.full(n, 10), nadc=np.full(n, 3),
             adc=np.tile([10, 10, 20], n))
    d["sid"] = d["sid"].astype(int)
    for k in reco.DIGI_COLS:
        raw["d_" + k] = jagged([d[k], [], []], int)
    d.update(ev=np.zeros(n, dtype=int), pbt=np.zeros(3))
    c = conditions()
    c["straw"] = {k: np.array(v) for k, v in c["straw"].items()}
    for k in ("halfvpscale", "centres", "resslope", "totdtime", "totderror"):
        c[k] = np.array(c[k])
    sh = hitreco.straw_hits(d, c)
    ph = combo.panel_hits(sh)
    for pre, tab, cols in (("s_", sh, reco.SH_COLS), ("p_", ph, reco.PH_COLS)):
        for k in ("sid",) + cols + (("ncombo", "ind") if pre == "p_" else ()):
            raw[pre + k] = jagged([tab[k], [], []], tab[k].dtype)
    return ak.Array(raw)


def replace(a, key, values, dtype=None):
    return ak.with_field(a, jagged(values, dtype), key)


def test_full_census_and_full_keys_retained():
    d, (sh, ph) = reco.decode_reference(rows(), 2)
    assert d["evkey"].tolist() == [[1430, 0, 1], [1430, 0, 2], [1430, 1, 1]]
    assert d["pbt"].tolist() == [0, 0, 0]
    assert d["ev"].tolist() == [0, 0]
    assert ph["ind"].tolist() == [0, 1]
    assert reco.compare_hits(sh, sh)["aligned"]
    assert reco.compare_hits(ph, ph, panel=True)["aligned"]


@pytest.mark.parametrize("field,value,message", [
    ("event", [1, 1, 1], "duplicate"), ("spill", [1, 0, 0], "offspill"),
    ("pbt", [np.nan, 0, 0], "finite pbt"), ("pbt", [0, 0, 1], "finite pbt"),
    ("run", [-1, 1430, 1430], "nonnegative"), ("run", [2**32, 1430, 1430], "invalid"),
])
def test_malformed_census(field, value, message):
    with pytest.raises(ValueError, match=message):
        reco.decode_reference(ak.with_field(rows(), np.array(value), field), 2)


@pytest.mark.parametrize("field,value,message", [
    ("d_nadc", [[2, 4], [], []], "greater than npre"),
    ("d_nadc", [[3, 4], [], []], "every ADC"),
    ("d_adc", [[10, 10, 20], [10, 10, 20], []], "every ADC"),
    ("d_adc", [[10, 10, -1, 10, 10, 20], [], []], "nonnegative"),
    ("d_adc", [[10., 10., 20., 10., 10., 20.], [], []], "integers"),
    ("d_tdc0", [[50], [], []], "jagged"),
    ("s_time", [[np.inf, 20], [], []], "nonfinite"),
    ("p_ncombo", [[1], [], []], "constituents"),
    ("p_ind", [[0, 2], [], []], "outside"),
])
def test_malformed_payload(field, value, message):
    with pytest.raises(ValueError, match=message):
        reco.decode_reference(replace(rows(), field, value), 2)


@pytest.mark.parametrize("npre", [0, -1, 2.5, True])
def test_bad_npre(npre):
    with pytest.raises(ValueError, match="positive integer"):
        reco.decode_reference(rows(), npre)


@pytest.mark.parametrize("panel,field", [(False, "ev"), (False, "sid"), (True, "ev"),
                                         (True, "sid"), (True, "ncombo"), (True, "ind")])
def test_alignment_gates_numerical_comparison(panel, field):
    _, refs = reco.decode_reference(rows(), 2)
    ref = refs[int(panel)]
    candidate = copy.deepcopy(ref)
    candidate[field][0] += 1
    result = reco.compare_hits(ref, candidate, panel)
    assert not result["aligned"]
    assert not result["identity"][field]
    assert result["columns"] == {}
    assert result["numerical_skip_reason"]


def test_integer_flag_mismatch_is_exact_not_toleranced():
    _, (ref, _) = reco.decode_reference(rows(), 2)
    candidate = copy.deepcopy(ref)
    candidate["flag"][0] ^= np.uint32(1)
    s = reco.compare_hits(ref, candidate)["columns"]["flag"]
    assert s["integer"] and not s["all_exact"]
    assert s["exact_count"] == 1 and s["exact_fraction"] == .5
    assert s["maxabs"] == 1
    # Preserve exactness for integer values beyond float64's exact integer range.
    s = reco.column_stats(np.array([2**63], np.uint64), np.array([2**63 + 1], np.uint64), True)
    assert not s["all_exact"] and s["maxabs"] == 1
    s = reco.column_stats(np.array([2**63 - 1], np.int64), np.array([2**63], np.uint64), True)
    assert not s["all_exact"] and s["maxabs"] == 1


def test_stats_finite_pairs_and_empty_are_json_safe():
    s = reco.column_stats([1., np.nan, np.inf, 4.], [2., 3., np.inf, 4.])
    assert s["finite_pairs"] == 2 and not s["finite"]
    assert s["exact_count"] == 1 and s["exact_fraction"] == .25
    assert s["maxabs"] == 1 and s["rms"] == pytest.approx(np.sqrt(.5))
    assert reco.column_stats([np.nan], [0.])["rms"] is None
    assert reco.column_stats([], [])["exact_fraction"] is None
    json.dumps(s, allow_nan=False)
    with pytest.raises(ValueError, match="shapes"):
        reco.column_stats([1], [])


def test_nonfinite_candidate_reported_even_when_unaligned():
    _, (ref, _) = reco.decode_reference(rows(), 2)
    cand = copy.deepcopy(ref)
    cand["time"][0] = np.nan
    result = reco.compare_hits(ref, cand)
    assert not result["candidate_finite"]
    assert result["columns"]["time"]["candidate_nonfinite"] == 1
    cand["sid"][0] += 1
    assert not reco.compare_hits(ref, cand)["candidate_finite"]


def test_all_empty():
    d, refs = reco.decode_reference(rows(empty=True), 2)
    assert len(d["evkey"]) == 3 and len(d["ev"]) == 0
    for i, ref in enumerate(refs):
        report = reco.compare_hits(ref, ref, bool(i))
        assert report["aligned"] and report["reference_finite"]
        assert all(s["count"] == 0 and s["maxabs"] is None for s in report["columns"].values())
        json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("empty", [False, True])
def test_cli_synthetic_root(tmp_path, monkeypatch, empty):
    path = tmp_path / "hits.root"
    a = rows(empty)
    with uproot.recreate(path) as root:
        root.mktree("trkdump/events", {k: a[k] for k in a.fields})
    monkeypatch.setattr(hitreco, "extract_conditions", lambda ref, out: conditions())
    out = tmp_path / "new"
    report = reco.main(["--reference", str(path), "--outdir", str(out)])
    assert report["events"] == 3 and report["zero_digi_events"] == (3 if empty else 2)
    assert json.loads((out / "report.json").read_text()) == report
    assert hitreco.load_conditions(out / "conditions.json")["scalar"]["nADCPreSamples"] == 2
    for phase in ("root_read_s", "condition_extraction_s", "numpy_sh_first_s", "numpy_sh_repeat_s",
                  "numpy_ph_first_s", "numpy_ph_repeat_s", "numpy_write_s", "report_write_s"):
        assert report["timings_s"][phase] >= 0
    for stage in ("sh", "ph"):
        result = report["backends"]["numpy"]["stock_comparison"][stage]
        assert result["aligned"] and all(s["all_exact"] for s in result["columns"].values())
    with np.load(out / "numpy.npz", allow_pickle=False) as f:
        assert f["evkey"].shape == (3, 3)
    with pytest.raises(FileExistsError):
        reco.main(["--reference", str(path), "--outdir", str(out)])


def test_require_gpu_without_jax_fails_before_io(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        reco.main(["--reference", "missing.root", "--outdir", str(tmp_path / "out"), "--require-gpu"])
    assert not (tmp_path / "out").exists()


def test_require_gpu_checks_active_backend_before_io(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(default_backend=lambda: "cpu"))
    with pytest.raises(RuntimeError, match="active JAX backend"):
        reco.main(["--reference", "missing.root", "--outdir", str(tmp_path / "out"), "--jax", "--require-gpu"])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("empty", [False, True])
def test_jax_cli_optional(tmp_path, monkeypatch, empty):
    pytest.importorskip("jax")
    path = tmp_path / "jax_input.root"
    a = rows(empty=empty)
    with uproot.recreate(path) as root:
        root.mktree("trkdump/events", {k: a[k] for k in a.fields})
    monkeypatch.setattr(hitreco, "extract_conditions", lambda ref, out: conditions())
    report = reco.main(["--reference", str(path), "--outdir", str(tmp_path / "jax"), "--jax"])
    j = report["backends"]["jax"]
    assert j["stock_comparison"]["sh"]["aligned"] and j["stock_comparison"]["ph"]["aligned"]
    assert j["numpy_comparison"]["ph"]["aligned"]


def test_all_exported_hit_columns_checked_for_finiteness():
    a = replace(rows(), "s_qual", [[0., np.nan], [], []])
    with pytest.raises(ValueError, match="s_qual contains nonfinite"):
        reco.decode_reference(a, 2)


def test_negative_energy_is_counted_not_suppressed(tmp_path, monkeypatch):
    a = replace(rows(), "d_adc", [[20, 20, 10, 20, 20, 10], [], []], int)
    path = tmp_path / "negative.root"
    with uproot.recreate(path) as root:
        root.mktree("trkdump/events", {k: a[k] for k in a.fields})
    monkeypatch.setattr(hitreco, "extract_conditions", lambda ref, out: conditions())
    report = reco.main(["--reference", str(path), "--outdir", str(tmp_path / "negative")])
    assert report["backends"]["numpy"]["negative_energy"]["sh"] == 2
    assert not report["backends"]["numpy"]["stock_comparison"]["sh"]["columns"]["edep"]["all_exact"]
