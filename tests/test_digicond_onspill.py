"""Explicit origin-field and on-spill conditions, independent of digitiser event timing."""
import inspect
import math

import numpy as np
import pytest

from trkfastsim.digi import digicond as DC
from test_digi import FIX, _geom, cond


@pytest.fixture
def inputs():
    d = DC.parse_fhicl_dump(FIX.read_text())
    pr = np.zeros((DC.NUSTRAWS, 6))
    pr[:, 0] = np.arange(DC.NUSTRAWS)
    pr[:, 3], pr[:, 4], pr[:, 5] = 11.996, 11.632, 1.5e6
    return d, pr, _geom()


@pytest.fixture
def cached_electronics(monkeypatch, cond):
    keys = ("wire_points", "vthresh", "adc_ped", "dvdi_thresh", "dvdi_adc", "time_offsets", "resp_thresh",
            "resp_adc", "resp_sat", "resp_imp", "current_pulse", "tmax", "linmax", "digitizationStartTDC",
            "digitizationEndTDC", "timeFromProtonsToDRMarker", "maxTDC")
    monkeypatch.setattr(DC, "straw_electronics", lambda *args: {k: getattr(cond, k) for k in keys})


def test_field_on_requires_explicit_origin_sample(inputs):
    d, pr, g = inputs
    d["services"]["GeometryService"]["bFieldFile"] = "Offline/Mu2eG4/geom/bfgeom_nominal1T.txt"
    with pytest.raises(ValueError, match="explicit field_bz_T.*detector origin"):
        DC.assemble(d, pr, g)


@pytest.mark.parametrize("field", [np.nan, np.inf, -np.inf, 1e100, [1.0], "invalid"])
@pytest.mark.parametrize("field_on", [False, True])
def test_field_must_be_finite_scalar(inputs, field, field_on):
    d, pr, g = inputs
    if field_on:
        d["services"]["GeometryService"]["bFieldFile"] = "field-map.txt"
    with pytest.raises(ValueError, match="field_bz_T.*finite"):
        DC.assemble(d, pr, g, field_bz_T=field)


def test_explicit_zero_preserves_offspill_arrays(inputs, cached_electronics, cond):
    zero = DC.assemble(*inputs, field_bz_T=0.0)
    assert zero.eventWindowLength == cond.eventWindowLength == 100000.0
    assert zero.field_bz_T == cond.field_bz_T == 0.0
    assert zero.s == cond.s
    assert zero.a.keys() == cond.a.keys()
    for key in cond.a:
        np.testing.assert_array_equal(zero.a[key], cond.a[key])
    np.testing.assert_array_equal(zero.drift_times, np.broadcast_to(zero.drift_times[:, :1], zero.drift_times.shape))


def test_field_on_azimuth_and_reversal(inputs, cached_electronics, cond, tmp_path):
    d, pr, g = inputs
    d["services"]["GeometryService"]["bFieldFile"] = "field-map.txt"
    d["physics"]["producers"]["EWMProducer"]["SpillType"] = 1
    bz = 1.23456789
    c = DC.assemble(d, pr, g, field_bz_T=bz)
    reverse = DC.assemble(d, pr, g, field_bz_T=-bz)
    np.testing.assert_array_equal(c.drift_times, reverse.drift_times)
    np.testing.assert_array_equal(c.drift_times[:, 0], cond.drift_times[:, 0])
    assert np.all(c.drift_times[1:, -1] > c.drift_times[1:, 0])
    phi = np.array([0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2])
    times = DC.d2t(2.0, phi, c.drift_dist, c.drift_times, c.drift_dD, c.drift_dphi)
    assert times[0] < times[1] < times[2]
    assert times[0] == times[3] and times[2] == times[4]
    for key in c.a:
        if key != "drift_times":
            np.testing.assert_array_equal(c.a[key], cond.a[key])
    assert c.onspill and c.eventWindowLength == 1700.0
    assert c.field_bz_T == c.sources["field_bz_T"] == bz
    c.save(tmp_path / "conditions.npz")
    restored = DC.Conditions.load(tmp_path / "conditions.npz")
    assert restored.s == c.s
    np.testing.assert_array_equal(restored.drift_times, c.drift_times)


def test_nonzero_drift_matches_stock_float_bz_and_cc(inputs, cond):
    d, _, g = inputs
    cfg = d["services"]["ProditionsService"]["strawDrift"]
    bz = 1.23456789
    actual = DC.straw_drift(cfg, g.wire_r, g.wall_rmax, bz=bz)
    factor = math.log(g.wall_rmax / g.wire_r)
    cc = float(np.float32(float(np.float32(bz)) * factor / cfg["wireVoltage"]))
    cc_unrounded = bz * factor / cfg["wireVoltage"]
    r, t = cond.drift_dist[-1], cond.drift_times[-1, 0]
    zeta = cc * (r / t) * 1e6 * r * 0.001
    zeta_unrounded = cc_unrounded * (r / t) * 1e6 * r * 0.001
    for p in range(actual["drift_times"].shape[1]):
        angle = actual["drift_dphi"] * p
        expected = t * (1 + zeta ** 2 / 3.0) / (1 + (zeta * math.cos(angle)) ** 2 / 3.0)
        assert actual["drift_times"][-1, p] == pytest.approx(expected, rel=2e-16)
    assert abs(t * (1 + zeta_unrounded ** 2 / 3.0) - actual["drift_times"][-1, -1]) > 1e-9


@pytest.mark.parametrize("ticks,frequency", [(68, 40.0), (72, 50.0)])
def test_onspill_window_uses_configured_daq_ticks(inputs, cached_electronics, cond, ticks, frequency):
    d, pr, g = inputs
    d["physics"]["producers"]["EWMProducer"]["SpillType"] = 1
    d["services"]["ProditionsService"]["eventTiming"]["OffSpillEventLength"] = 12345
    c = DC.assemble(d, pr, g, gc={"nominalDAQTicks": ticks, "nominalDAQFrequency": frequency})
    assert c.onspill
    assert c.eventWindowLength == ticks * 1000.0 / frequency
    assert c.eventWindowLength != 12345 * 1000.0 / frequency
    for key in c.a:
        np.testing.assert_array_equal(c.a[key], cond.a[key])


def test_onspill_default_window_is_not_offspill(inputs, cached_electronics):
    d, pr, g = inputs
    d["physics"]["producers"]["EWMProducer"]["SpillType"] = 1
    assert DC.assemble(d, pr, g).eventWindowLength == 1700.0


def test_global_constants_reader_includes_onspill_lengths(tmp_path):
    path = tmp_path / "global.txt"
    path.write_text("int physicsParams.nominalDAQTicks = 71;\n"
                    "int physicsParams.nominalDAQOnSpillBins = 4;\n"
                    "double physicsParams.nominalDAQFrequency = 50.;\n")
    gc = DC.read_global_constants(path)
    assert gc["nominalDAQTicks"] == 71 and gc["nominalDAQOnSpillBins"] == 4
    assert gc["nominalDAQFrequency"] == 50.0


def test_build_forwards_explicit_field(inputs, monkeypatch):
    d, pr, g = inputs
    captured = {}

    def assemble(*args, **kwargs):
        captured.update(kwargs)
        return "conditions"

    monkeypatch.setattr(DC, "assemble", assemble)
    monkeypatch.setattr(DC, "read_db_csv", lambda path: pr)
    monkeypatch.setattr(DC.geometry, "from_gdml", lambda path: g)
    assert DC.build(FIX, "preamp.csv", "geometry.gdml", field_bz_T=0.987654321) == "conditions"
    assert captured == {"field_bz_T": 0.987654321, "detector_origin": None}


def test_field_api_is_keyword_only():
    for func in (DC.build, DC.assemble):
        field = inspect.signature(func).parameters["field_bz_T"]
        assert field.kind == inspect.Parameter.KEYWORD_ONLY and field.default is None
