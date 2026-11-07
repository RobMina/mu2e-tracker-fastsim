"""Run1A driver contracts with synthetic ROOT input and mocked digitisation kernels."""
from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from trkfastsim.validation import run1a_digi as R


def keys():
    # Full event identities can collide under a packed-bit key.
    return np.array([[1, 0, 2**20 + 5], [1, 1, 5], [1, 0, 8], [1, 0, 9]], np.int64)


def reference(path, key=None, **changes):
    key = keys() if key is None else key
    counts = np.array([2, 0, 1, 0])
    columns = dict(zip(R.KEYS, key.T))
    columns.update({f"d_{name}": ak.unflatten(np.array([1, 2, 3], np.int32), counts)
                    for name in R.FIELDS if name not in ("nadc", "adc")})
    columns["d_nadc"] = ak.unflatten(np.array([1, 2, 0], np.int32), counts)
    columns["d_adc"] = ak.unflatten(np.array([5, 6, 7], np.int32), [3, 0, 0, 0])
    columns.update(changes)
    with uproot.recreate(path) as root:
        root.mktree("trkdigi/events", columns)
    return path


def frame():
    result = pd.DataFrame(np.zeros((4, len(R.fastdigi.STEP_COLS)), np.int64), columns=R.fastdigi.STEP_COLS)
    result[list(R.KEYS)] = np.vstack([keys()[[0, 0, 1]], [1, 1, 2**20 + 5]])
    result["eion"] = [1., 1., 0., 1.]
    return result


def test_reference_census_and_explicit_step_membership(tmp_path):
    k, d = R.read_reference(reference(tmp_path / "reference.root"))
    np.testing.assert_array_equal(k, keys())
    np.testing.assert_array_equal(d["ev"], [0, 0, 2])
    np.testing.assert_array_equal(d["nadc"], [1, 2, 0])
    selected, counts = R.restrict_steps(frame(), k)
    assert len(selected) == 3
    assert counts == {"source_steps": 4, "source_step_events": 3, "selected_steps": 3,
                      "excluded_steps": 1, "empty_input_events": 2}
    empty, counts = R.restrict_steps(frame().iloc[:0], k)
    assert empty.empty and counts["empty_input_events"] == 4
    empty, counts = R.restrict_steps(frame(), np.empty((0, 3), np.int64))
    assert empty.empty and counts["excluded_steps"] == 4


def test_duplicate_and_invalid_keys(tmp_path):
    duplicate = keys()
    duplicate[1] = duplicate[0]
    with pytest.raises(ValueError, match="duplicate"):
        R.read_reference(reference(tmp_path / "dup.root", duplicate))
    with pytest.raises(ValueError, match="duplicate"):
        R.restrict_steps(frame(), duplicate)


@pytest.mark.parametrize("bad", [np.ones((2, 3)), np.ones((2, 2), int),
                                 [[-1, 0, 0]], [[2**32, 0, 0]], [[2**64 - 1, 0, 0]]])
def test_key_domain(bad):
    with pytest.raises(ValueError, match="event keys"):
        R.key_ids(bad)


@pytest.mark.parametrize("changes,match", [
    ({"spill": np.ones(4, int)}, "off-spill"),
    ({"pbtime": np.ones(4)}, "off-spill"),
    ({"d_adc": ak.Array([[5, 6], [], [], []])}, "ADC sample"),
    ({"d_tot0": ak.Array([[1], [], [2], []])}, "jagged lengths"),
    ({"d_flags": ak.Array([[1., 2.], [], [3.], []])}, "integers"),
])
def test_bad_reference(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        R.read_reference(reference(tmp_path / "bad.root", **changes))


def test_all_empty_reference_and_zero_event_census(tmp_path):
    blank = ak.unflatten(np.empty(0, np.int32), np.zeros(4, int))
    changes = {f"d_{name}": blank for name in R.FIELDS}
    k, d = R.read_reference(reference(tmp_path / "empty.root", **changes))
    assert len(k) == 4 and not len(d["ev"])
    columns = dict(zip(R.KEYS, np.empty((0, 3), np.int64).T))
    with uproot.recreate(tmp_path / "zero.root") as root:
        root.mktree("trkdigi/events", columns)
    with pytest.raises(ValueError, match="no events"):
        R.read_reference(tmp_path / "zero.root")


def test_field_exact_origin_and_print_precision(tmp_path):
    path = tmp_path / "field.csv"
    path.write_text("# x,y,z,Bx,By,Bz,B\n-3904,0,10170,0,0,1.1,1.1\n-3904,0,10171,0,0,1.2345,1.2345\n")
    assert R.field_at_origin(path, [-3904, 0, 10171]) == 1.2345
    # Never match the neighboring rounded coordinate or silently choose nearest.
    with pytest.raises(ValueError, match="EXACT"):
        R.field_at_origin(path, [-3904, 0, 10171.000001])


@pytest.mark.parametrize("text", ["0,0,0,0,0,1\n", "-3904,0,10171,0,0,nan\n",
                                  "-3904,0,10171,0,0,1\n-3904,0,10171,0,0,1\n",
                                  "-3904,0,10171,0,1\n", "x,y,z,bx,by,bz\n"])
def test_invalid_field_csv(tmp_path, text):
    path = tmp_path / "field.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        R.field_at_origin(path, [-3904, 0, 10171])


def digis(ev, pmp=None, *, flows=True):
    n = len(ev)
    return {"ev": np.asarray(ev, np.int64), "sid": np.full(n, 1024, np.int32),
            "tdc": np.tile([-1 if flows else 1, 5120052], (n, 1)), "tot": np.tile([0, 16], (n, 1)),
            "pmp": np.arange(n) if pmp is None else pmp,
            "adc": np.tile([-1 if flows else 1, 4097], (n, 1)), "step": np.zeros((n, 2), np.int64)}


def test_paired_yield_variance_and_full_histogram_coverage(tmp_path):
    _, stock = R.read_reference(reference(tmp_path / "ref.root"))
    fast = digis([0, 1, 1, 2])
    result = R.comparison(stock, fast, 4)
    s, f = np.array([2, 0, 1, 0]), np.array([1, 2, 1, 0])
    assert result["yield"]["ratio"] == pytest.approx(4 / 3)
    assert result["yield"]["ratio_se"] == pytest.approx(np.std(f - (4 / 3) * s, ddof=1) / 2 / s.mean())
    for name in ("plane", "tdc0", "tdc1", "tot0", "tot1", "pmp", "adc"):
        h = result[name]
        assert sum(h["fast_total"]) == (8 if name == "adc" else 4)
        assert sum(h["stock_total"]) == 3
        assert h["includes_flow_bins"]
    assert result["tdc0"]["fast_total"][0] == 4
    assert result["tdc1"]["fast_total"][-1] == 4
    assert result["adc"]["fast_total"][0] == result["adc"]["fast_total"][-1] == 4
    # Fast plane 1 has per-event counts [1,2,1,0]; stock has none there.
    assert result["plane"]["paired_sum_difference_sq"][2] == 6
    assert result["plane"]["paired_total_variance"][2] == pytest.approx(4 * f.var(ddof=1))
    empty = {k: v[:0] for k, v in stock.items()}
    assert R.comparison(empty, digis([]), 4)["yield"]["ratio"] is None
    one = R.comparison(empty, digis([]), 1)
    assert one["yield"]["ratio_se"] is None and one["plane"]["paired_total_variance"] is None
    json.dumps(one, allow_nan=False)


@pytest.fixture
def cli(tmp_path, monkeypatch):
    ref = reference(tmp_path / "reference.root")
    config = tmp_path / "expanded.fcl"
    config.write_text("physics: { producers: { EWMProducer: { SpillType: 0 } } }\n")
    field = tmp_path / "field.csv"
    field.write_text("-3904,0,10171,0,0,1.2345\n")
    step_path = tmp_path / "steps.root"
    with uproot.recreate(step_path) as root:
        root.mktree("dump/sgsteps", {k: v.to_numpy() for k, v in frame().items()})
    cond = SimpleNamespace(onspill=False, nADCSamples=2, eventWindowLength=100000., TDCLSB=R.LSB_TDC,
                           save=lambda p: np.savez(p, __scalars__=json.dumps(
                               {"nADCSamples": 2, "eventWindowLength": 100000., "TDCLSB": R.LSB_TDC})))
    builds, draws = [], []
    def build(*args, **kw):
        builds.append((args, kw))
        return cond
    def kernel(st, nev, c, rng, stats):
        assert c is cond
        values = rng.integers(0, 100, len(st["ev"]))
        draws.append(values)
        keep = st["eion"] > 0
        out = digis(st["ev"][keep], values[keep], flows=False)
        out["step"] = np.repeat(np.flatnonzero(keep)[:, None], 2, axis=1)
        return out
    monkeypatch.setattr(R.digicond, "build", build)
    monkeypatch.setattr(R.fastdigi.D, "digitize", kernel)
    argv = ["--steps", str(step_path), "--reference", str(ref), "--config", str(config),
            "--field-csv", str(field), "--detector-origin", "-3904", "0", "10171", "--outdir", str(tmp_path / "out")]
    for name in ("preamp", "align", "gdml", "global-constants", "panel.csv", "straw.csv"):
        (tmp_path / name).write_text(f"synthetic {name} input\n")
    for name in ("preamp", "align", "gdml", "global-constants"):
        argv += [f"--{name}", str(tmp_path / name)]
    return SimpleNamespace(argv=argv, builds=builds, draws=draws, tmp=tmp_path, cond=cond, config=config)


def test_main_repeats_census_conditions_and_writer(cli):
    report = R.main(cli.argv + ["--repeats", "2", "--seed", "17", "--delay-panel", str(cli.tmp / "panel.csv"),
                               "--delay-straw", str(cli.tmp / "straw.csv")])
    assert report["reference_events"] == 4 and report["empty_input_events"] == 2
    assert report["selected_steps"] == 3 and report["excluded_steps"] == 1
    assert report["repeat_identity"] is True and len(report["passes"]) == 2
    np.testing.assert_array_equal(cli.draws[0], cli.draws[1])
    np.testing.assert_array_equal(cli.draws[0], np.random.default_rng(17).integers(0, 100, 3))
    kw = cli.builds[0][1]
    assert kw["field_bz_T"] == 1.2345 and kw["detector_origin"] == [-3904, 0, 10171]
    assert tuple(kw["delay_csvs"]) == (cli.tmp / "panel.csv", cli.tmp / "straw.csv")
    assert report["passes"][0]["stats"]["numpy_fallback_chunks"] == 0
    assert report["setup_s"] >= 0 and report["root_write_s"] >= 0
    assert report["paired"]["yield"]["empty_fast_events"] == 3  # includes no-response input
    with uproot.open(cli.tmp / "out/fast.root") as root:
        rows = root["trkdump/events"].arrays(library="ak")
    np.testing.assert_array_equal(np.column_stack([ak.to_numpy(rows[k]) for k in R.KEYS]), keys())
    assert ak.to_list(rows.spill) == ak.to_list(rows.pbt) == [0, 0, 0, 0]
    assert json.loads((cli.tmp / "out/report.json").read_text())["repeat_identity"] is True
    with pytest.raises(FileExistsError):
        R.main(cli.argv)


@pytest.mark.parametrize("extra", [["--chunk", "0"], ["--repeats", "0"], ["--seed", "-1"],
                                  ["--delay-panel", "panel.csv"], ["--delay-straw", "straw.csv"], ["--require-gpu"]])
def test_cli_guards(cli, extra):
    with pytest.raises(SystemExit):
        R.main(cli.argv + extra)
    assert not cli.builds


def test_onspill_refusal(cli):
    cli.config.write_text("physics: { producers: { EWMProducer: { SpillType: 1 } } }\n")
    with pytest.raises(ValueError, match="off-spill"):
        R.main(cli.argv)
    assert not cli.builds


@pytest.mark.parametrize("change", ["keys", "steps", "repeat"])
def test_digitizer_population_and_repeat_guards(cli, monkeypatch, change):
    calls = []
    def fake(steps, cond, rng, *, keys, **kw):
        calls.append(1)
        d = digis([0], np.array([len(calls) if change == "repeat" else 1]), flows=False)
        return (keys[::-1] if change == "keys" else keys), d, len(steps) + (change == "steps")
    monkeypatch.setattr(R.fastdigi, "digitize_steps", fake)
    with pytest.raises(ValueError, match="changed"):
        R.main(cli.argv + ["--repeats", "2"])


def test_jax_gpu_guard_without_importing_jax(cli, monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(devices=lambda: [SimpleNamespace(platform="cpu")],
                                                                    default_backend=lambda: "cpu"))
    with pytest.raises(RuntimeError, match="default backend is not GPU"):
        R.main(cli.argv + ["--backend", "jax", "--require-gpu"])
    assert not (cli.tmp / "out").exists()


def test_mock_jax_seed_repeats_and_fallback_stats(cli, monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(devices=lambda: [SimpleNamespace(platform="gpu")],
                                                            default_backend=lambda: "gpu",
                                                            random=SimpleNamespace(key=lambda seed: (seed,))))
    calls = []
    def fake(steps, cond, rng, *, keys, chunk, stats, backend):
        calls.append(rng)
        assert backend == "jax" and chunk == 32
        stats["numpy_fallback_chunks"] += 1
        return keys, digis([0], flows=False), len(steps)
    monkeypatch.setattr(R.fastdigi, "digitize_steps", fake)
    report = R.main(cli.argv + ["--backend", "jax", "--require-gpu", "--repeats", "2", "--seed", "19"])
    assert calls == [(19,), (19,)]
    assert report["repeat_identity"] and report["passes"][1]["stats"]["numpy_fallback_chunks"] == 1


def test_main_all_inputs_empty_retains_full_census(cli, monkeypatch):
    monkeypatch.setattr(R.fastdigi, "load_steps", lambda path: frame().iloc[:0])
    blank = ak.unflatten(np.empty(0, np.int32), np.zeros(4, int))
    reference(cli.tmp / "reference.root", **{f"d_{name}": blank for name in R.FIELDS})
    report = R.main(cli.argv)
    assert report["empty_input_events"] == 4 and report["selected_steps"] == 0
    assert report["repeat_identity"] is None
    assert report["paired"]["yield"]["ratio"] is None
    assert report["paired"]["yield"]["empty_fast_events"] == 4
    with uproot.open(cli.tmp / "out/fast.root") as root:
        assert root["trkdump/events"].num_entries == 4


def test_tdc_bins_cover_full_window_in_ns(tmp_path):
    _, stock = R.read_reference(reference(tmp_path / "ref.root"))
    d = digis([0])
    d["tdc"][:] = int(90000 / R.LSB_TDC)
    result = R.comparison(stock, d, 4, window_ns=100000.)
    assert result["tdc0"]["units"] == "ns"
    assert result["tdc0"]["edges"][-1] == 100000.
    assert result["tdc0"]["fast_total"][91] == 1
    assert result["tdc0"]["fast_total"][-1] == 0


def test_offspill_conditions_guard(cli):
    cli.cond.onspill = True
    with pytest.raises(ValueError, match="off-spill conditions"):
        R.main(cli.argv)


@pytest.mark.parametrize("origin", [[-3904, 0], [-3904, np.nan, 10171]])
def test_invalid_origin_shape_or_values(tmp_path, origin):
    path = tmp_path / "field.csv"
    path.write_text("-3904,0,10171,0,0,1.0\n")
    with pytest.raises(ValueError, match="finite"):
        R.field_at_origin(path, origin)


def test_reference_requires_lossless_pmp_and_flags(tmp_path):
    path = reference(tmp_path / "full.root")
    with uproot.open(path) as root:
        columns = root["trkdigi/events"].arrays(library="ak", how=dict)
    columns.pop("d_flags")
    with uproot.recreate(tmp_path / "missing.root") as root:
        root.mktree("trkdigi/events", columns)
    with pytest.raises(ak.errors.FieldNotFoundError):
        R.read_reference(tmp_path / "missing.root")


def test_comparison_uses_condition_tdc_lsb(tmp_path):
    _, stock = R.read_reference(reference(tmp_path / "ref.root"))
    fast = digis([0])
    fast["tdc"][:] = 180000
    report = R.comparison(stock, fast, 4, window_ns=100000., tdc_lsb_ns=0.5)
    assert report["tdc0"]["fast_total"][91] == 1


def test_stock_control_requires_disjoint_blocks(tmp_path):
    ref = reference(tmp_path / "ref.root")
    other = reference(tmp_path / "other.root")
    hold = reference(tmp_path / "hold.root", keys() + [0, 0, 100])
    argv = ["--stock-control", str(other), "--reference", str(ref), "--heldout", str(hold),
            "--output", str(tmp_path / "control.json")]
    with pytest.raises(ValueError, match="constant waveform"):
        R.main(argv)
    reference(hold)
    with pytest.raises(ValueError, match="overlap"):
        R.main(argv)


def test_stock_control_equal_seed_and_disjoint_success(tmp_path):
    changes = {"d_nadc": ak.Array([[1, 1], [], [1], []]),
               "d_adc": ak.Array([[5, 6], [], [7], []])}
    ref = reference(tmp_path / "ref.root", **changes)
    other = reference(tmp_path / "other.root", **changes)
    hold = reference(tmp_path / "hold.root", keys() + [0, 0, 100], **changes)
    report = R.main(["--stock-control", str(other), "--reference", str(ref), "--heldout", str(hold),
                     "--output", str(tmp_path / "control.json")])
    assert report["disjoint"] and report["heldout_events"] == report["development_events"] == 4
    assert report["paired"]["yield"]["ratio"] == 1
    assert report["paired"]["yield"]["ratio_se"] == 0
    assert all(h["chi2_diagonal"] == 0 for name, h in report["paired"].items() if name != "yield")


@pytest.mark.parametrize("masked", ["keys", "pmp"])
def test_masked_identity_and_payload_rejected(tmp_path, masked):
    if masked == "keys":
        values = np.ma.array(keys(), mask=False)
        values.mask[0, 0] = True
        with pytest.raises(ValueError, match="missing"):
            R.key_ids(values)
    else:
        _, stock = R.read_reference(reference(tmp_path / "reference.root"))
        fast = digis([0])
        fast["pmp"] = np.ma.array([1], mask=[True])
        with pytest.raises(ValueError, match="missing"):
            R.comparison(stock, fast, 4)


@pytest.mark.parametrize("side,field,value", [
    ("stock", "ev", np.array([-1, 0, 2])),
    ("stock", "ev", np.array([0, 0, 4])),
    ("fast", "ev", np.array([-1])),
    ("fast", "ev", np.array([4])),
    ("fast", "ev", np.array([[0]])),
    ("fast", "tdc", np.zeros((1, 3), int)),
    ("fast", "adc", np.zeros(2, int)),
    ("fast", "adc", np.zeros((2, 2), int)),
    ("stock", "tdc0", np.zeros((3, 1), int)),
    ("stock", "nadc", np.array([1, 1, 0])),
    ("fast", "sid", np.array([1024.])),
    ("fast", "pmp", np.array([np.nan])),
])
def test_comparison_rejects_invalid_arrays(tmp_path, side, field, value):
    _, stock = R.read_reference(reference(tmp_path / "reference.root"))
    fast = digis([0])
    (stock if side == "stock" else fast)[field] = value
    with pytest.raises(ValueError):
        R.comparison(stock, fast, 4)


@pytest.mark.parametrize("changes,match", [
    ({"d_tdc0": ak.Array([[-1, 2], [], [3], []])}, "negative"),
    ({"d_adc": ak.Array([[-1, 6, 7], [], [], []])}, "negative"),
    ({"d_adc": ak.Array([[5., 6., 7.], [], [], []])}, "integers"),
    ({"spill": np.array([0., np.nan, 0., 0.])}, "nonfinite"),
    ({"pbt": np.array([0., np.inf, 0., 0.])}, "nonfinite"),
])
def test_serialized_payload_is_stricter_than_flow_comparison(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        R.read_reference(reference(tmp_path / "invalid.root", **changes))


@pytest.mark.parametrize("values", [
    [[1, None], [], [3], []],
    [[1, 2], None, [3], []],
])
def test_missing_jagged_cells_rejected(tmp_path, monkeypatch, values):
    path = reference(tmp_path / "reference.root")
    with uproot.open(path) as root:
        rows = root["trkdigi/events"].arrays(library="ak")
    rows = ak.with_field(rows, ak.Array(values), "d_tot0")
    # ROOT's writer does not support option arrays; mock only the decoded table.
    tree = SimpleNamespace(arrays=lambda **kw: rows)
    monkeypatch.setattr(R.io.uproot, "open", lambda path: nullcontext({"trkdigi/events": tree}))
    with pytest.raises(ValueError):
        R.read_reference(path)


@pytest.fixture
def stock_control(tmp_path):
    changes = {"d_nadc": ak.Array([[1, 1], [], [1], []]),
               "d_adc": ak.Array([[5, 6], [], [7], []])}
    ref = reference(tmp_path / "ref.root", **changes)
    other = reference(tmp_path / "other.root", **changes)
    hold = reference(tmp_path / "hold.root", keys() + [0, 0, 100], **changes)
    return ["--stock-control", str(other), "--reference", str(ref), "--heldout", str(hold),
            "--output", str(tmp_path / "control.json")]


def test_stock_control_never_overwrites_existing_output(tmp_path, stock_control):
    output = tmp_path / "control.json"
    output.write_bytes(b"original control evidence\n")
    with pytest.raises(FileExistsError):
        R.main(stock_control)
    assert output.read_bytes() == b"original control evidence\n"


def test_all_empty_stock_control_retains_census(tmp_path, stock_control):
    blank = ak.unflatten(np.empty(0, np.int32), np.zeros(4, int))
    for name, key in (("ref.root", keys()), ("other.root", keys()), ("hold.root", keys() + [0, 0, 100])):
        reference(tmp_path / name, key, **{f"d_{field}": blank for field in R.FIELDS})
    report = R.main(stock_control)
    assert report["disjoint"] and report["development_events"] == report["heldout_events"] == 4
    counts = report["paired"]["yield"]
    assert counts["stock_total"] == counts["fast_total"] == 0
    assert counts["empty_stock_events"] == counts["empty_fast_events"] == 4
    assert counts["ratio"] is counts["ratio_se"] is None
    json.dumps(report, allow_nan=False)


def test_stock_control_explicit_lsb_and_window(tmp_path, stock_control):
    changes = {"d_nadc": ak.Array([[1, 1], [], [1], []]),
               "d_adc": ak.Array([[5, 6], [], [7], []]),
               "d_tdc0": ak.Array([[180000, 180000], [], [180000], []])}
    reference(tmp_path / "other.root", **changes)
    report = R.main(stock_control + ["--window-ns", "200000", "--tdc-lsb-ns", "1"])
    assert report["window_ns"] == 200000 and report["tdc_lsb_ns"] == 1
    hist = report["paired"]["tdc0"]
    assert hist["units"] == "ns" and hist["edges"][-1] == 200000
    assert hist["fast_total"][91] == 3 and hist["fast_total"][-1] == 0


def test_gpu_available_but_cpu_selected_refused(cli, monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(
        devices=lambda: [SimpleNamespace(platform="gpu")], default_backend=lambda: "gpu",
        config=SimpleNamespace(jax_default_device=SimpleNamespace(platform="cpu"))))
    with pytest.raises(RuntimeError, match="default backend is not GPU"):
        R.main(cli.argv + ["--backend", "jax", "--require-gpu"])
    assert not (cli.tmp / "out").exists()
