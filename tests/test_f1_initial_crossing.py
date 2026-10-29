"""Full-key initial adapters, measured-input independence and fail-closed guards."""
from copy import deepcopy
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_initial_crossing as I
from test_f1_crossing_constructor import sample


def fixture():
    f = sample()
    f["seeds"], events = [], []
    for r in f["records"]:
        r["category"] = 0
    for i, t in enumerate(f["trajectories"]):
        t["category"] = 0
        t["parameters"] = [0.3+i, .4+.05*i, .1+i, .7+.03*i, 10.+i, 1000.+100*i]
        t["mass"] += i
        t["range"] = [-float(i), 20.+i]
        f["seeds"].append({k: t[k] for k in I.B.KEYS} | dict(valid=True, parameters=t["parameters"], range_begin=t["range"][0], range_end=t["range"][1], mass=t["mass"],
                            sh_shindex=[19], sh_strawid=[17], sh_time=[10.]))
        s = f["straws"][i]
        s.update(wire_cal=[.3+i, 100.+i, 3.2+i], wire_hv=[.3+i, -100.-i, 3.2+i])
        s["tube_cal"] = [.2+i, 100.+i, 3.+i]
        s["tube_hv"] = [.2+i, -100.-i, 3.+i]
        f["tracker"][i]["wire_radius"] += .001*i
        key = {k: t[k] for k in I.B.KEYS[:3]}
        if key not in [dict((k, e[k]) for k in I.B.KEYS[:3]) for e in events]:
            events.append(key | dict(s_sid=[17]*20, s_time=[10.+i]*20, s_et0=[10.1+i]*20, s_et1=[10.2+i]*20, s_edep=[.001*(i+1)]*20, s_eend=[0]*20))
    f["events"] = ak.Array(events)
    for name in ("time", "et0", "et1", "edep"):
        f["events"] = ak.with_field(f["events"], ak.values_astype(f["events"]["s_"+name], np.float32), "s_"+name)
    f["config"] = dict(bin_width=.1, mean_half_velocity=100., scale=[.7, .8, .9])
    return f


def prepare(f):
    c, keys, seeds, captured_time = I.initial_inputs(ak.Array(f["records"]), ak.Array(f["seeds"]))
    q, r, counts, parent, mass = I.construction_inputs(ak.Array(f["trajectories"]), c, seeds)
    g = I.geometry_inputs(c, ak.Array(f["bindings"]), ak.Array(f["tracker"]), ak.Array(f["straws"]))
    m = I.measurements(f["events"], keys, I.column(c, "strawid"))
    return I.generate(q, r, counts, parent, mass, g, m, f["config"]), c, keys, captured_time


def test_full_key_independent_generation():
    f = fixture()
    gen, c, keys, time = prepare(f)
    assert len(c) == 5 and len(np.unique(keys, axis=0)) == 5
    np.testing.assert_array_equal(gen["crossing"]["particle_hint"], gen["hit"]["particle_time"])
    np.testing.assert_array_equal(gen["crossing"]["sensor_hint"], gen["hit"]["sensor_time"])
    assert not np.array_equal(gen["hit_particle_hint"], gen["hit"]["particle_time"])
    assert not np.array_equal(gen["wire_line"]["end"], gen["crossing"]["line"]["end"])
    assert not gen["hit"]["exhausted"].any()


@pytest.mark.parametrize("field", ["particle_poca", "sensor_poca", "line_end", "line_start", "line_direction", "line_length", "line_measurement_time", "search_particle_hint", "search_sensor_hint", "offset", "effect_time", "speed_operand", "radius_operand", "reference_parameters"])
def test_recorded_outcomes_cannot_change_generation(field):
    f = fixture()
    before = prepare(f)[0]
    for row in f["records"]:
        value = row[field]
        row[field] = [999.]*len(value) if isinstance(value, list) else 999.
    after = prepare(f)[0]
    for key in ("offset", "effect_time", "particle_hint", "sensor_hint", "reference_speed"):
        np.testing.assert_array_equal(after["crossing"][key], before["crossing"][key])


def test_seed_hit_time_comparison_only():
    f = fixture()
    before = prepare(f)[0]
    for seed in f["seeds"]:
        seed["sh_time"] = [999.]
    after, _, _, times = prepare(f)
    assert np.all(times == 999.)
    np.testing.assert_array_equal(before["hit"]["particle_time"], after["hit"]["particle_time"])


def test_table_permutations_preserve_full_identity():
    f = fixture()
    before, _, keys, _ = prepare(f)
    for field in ("records", "seeds", "trajectories", "bindings", "tracker", "straws"):
        f[field].reverse()
    f["events"] = f["events"][::-1]
    after, _, after_keys, _ = prepare(f)
    join = I.match_rows(keys, after_keys)
    np.testing.assert_array_equal(before["crossing"]["effect_time"], after["crossing"]["effect_time"][join])


@pytest.mark.parametrize("field", ["seeds", "trajectories", "bindings", "tracker", "straws"])
def test_duplicate_available_identity_fails(field):
    f = fixture()
    f[field].append(deepcopy(f[field][0]))
    with pytest.raises(ValueError):
        prepare(f)


@pytest.mark.parametrize("field", ["seeds", "trajectories", "bindings", "tracker", "straws"])
def test_missing_identity_fails(field):
    f = fixture()
    del f[field][0]
    with pytest.raises(ValueError):
        prepare(f)


@pytest.mark.parametrize("mutation", ["partner", "straw", "piece", "q", "mass", "range", "extra_seed_hit", "bad_seed", "unused_context"])
def test_contract_failures(mutation):
    f = fixture()
    if mutation == "partner":
        f["records"][0]["partner"] += 1
    elif mutation == "straw":
        f["records"][0]["strawid"] += 1
    elif mutation == "piece":
        f["trajectories"][0]["piece_index"] = 1
    elif mutation == "q":
        f["seeds"][0]["parameters"][0] += 1
        f["trajectories"][0]["parameters"] = sample()["trajectories"][0]["parameters"]
    elif mutation == "mass":
        f["seeds"][0]["mass"] += 1
    elif mutation == "range":
        f["seeds"][0]["range_end"] = -1.
    elif mutation == "extra_seed_hit":
        for key, value in (("sh_shindex", 20), ("sh_strawid", 18), ("sh_time", 11.)):
            f["seeds"][0][key].append(value)
    elif mutation == "bad_seed":
        f["seeds"][0]["valid"] = 1
    else:
        f["trajectories"].append(dict(f["trajectories"][0], event=777))
    with pytest.raises((ValueError, TypeError)):
        prepare(f)


@pytest.mark.parametrize("name", ["time", "et0", "et1", "edep"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_measured_fails(name, bad):
    f = fixture()
    f["events"] = ak.with_field(f["events"], f["events"]["s_"+name]*np.float32(bad), "s_"+name)
    with pytest.raises(ValueError):
        prepare(f)


@pytest.mark.parametrize("name", ["time", "et0", "et1", "edep"])
def test_measured_double_rejected(name):
    f = fixture()
    f["events"] = ak.with_field(f["events"], ak.values_astype(f["events"]["s_"+name], np.float64), "s_"+name)
    with pytest.raises(ValueError, match="float32"):
        prepare(f)


def test_tied_endpoint_times_choose_cal():
    f = fixture()
    f["events"] = ak.with_field(f["events"], f["events"]["s_et0"], "s_et1")
    gen = prepare(f)[0]
    np.testing.assert_array_equal(gen["wire_line"]["end"], np.array([s["wire_cal"] for s in f["straws"]]))


def test_hv_earlier_selects_hv():
    f = fixture()
    f["events"] = ak.with_field(f["events"], f["events"]["s_et0"]-np.float32(.1), "s_et1")
    f["events"] = ak.with_field(f["events"], f["events"]["s_eend"]+1, "s_eend")
    gen = prepare(f)[0]
    np.testing.assert_array_equal(gen["wire_line"]["end"], np.array([s["wire_hv"] for s in f["straws"]]))


def test_endpoint_flag_disagreement_rejected():
    f = fixture()
    f["events"] = ak.with_field(f["events"], f["events"]["s_eend"]+1, "s_eend")
    with pytest.raises(ValueError, match="early end"):
        prepare(f)


def test_noncontemporaneous_partner_not_used():
    f = fixture()
    before = prepare(f)[0]
    for row in f["records"]:
        if row["stage"] != 0:
            row["partner"] = 888
    after = prepare(f)[0]
    np.testing.assert_array_equal(before["crossing"]["offset"], after["crossing"]["offset"])


def test_boolean_seed_index_rejected_before_stacking():
    f = fixture()
    for seed in f["seeds"]:
        seed["sh_shindex"] = [True]
    for record in f["records"]:
        record["partner"] = 1
    with pytest.raises(ValueError, match="seed hit index"):
        prepare(f)


def test_distinct_targets_stay_distinct():
    gen = prepare(fixture())[0]
    assert len(np.unique(gen["crossing"]["effect_time"])) == 5
    assert len(np.unique(gen["hit"]["particle_time"])) == 5


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf, True])
@pytest.mark.parametrize("key", ["bin_width", "mean_half_velocity"])
def test_invalid_speed_constants(key, bad):
    config = fixture()["config"]
    config[key] = bad
    with pytest.raises((ValueError, FloatingPointError)):
        I.signal_speed(np.array([.001]), config)


def test_speed_source_grouping_known_values():
    config = dict(bin_width=1., mean_half_velocity=100., scale=[1., 2., 3.])
    np.testing.assert_array_equal(I.signal_speed(np.array([0., .0005, .001, .002, .003]), config), [200., 300., 400., 600., 800.])


def speed_fcl():
    return 'strawResponse: {\nuseDb: false\nstrawHalfPropVelocity: []\neBinWidth: 1\ndefaultHalfPropVelocity: 100\neBins: 3\neHalfPVScale: [1,2,3]\n}\n'


def test_speed_config_literal(tmp_path):
    p = tmp_path/"expanded.fcl"
    p.write_text(speed_fcl())
    assert I.speed_configuration(p) == dict(bin_width=1., mean_half_velocity=100., scale=[1., 2., 3.])


@pytest.mark.parametrize("before,after", [("useDb: false", "useDb: true"), ("strawHalfPropVelocity: []", "strawHalfPropVelocity: [100]"), ("eBins: 3", "eBins: 2"), ("eBinWidth: 1", "eBinWidth: 0"), ("eHalfPVScale: [1,2,3]", "eHalfPVScale: [1,nan,3]"), ("eBins: 3", "eBins: 3\neBins: 3")])
def test_speed_config_rejects_unsupported(tmp_path, before, after):
    p = tmp_path/"expanded.fcl"
    p.write_text(speed_fcl().replace(before, after))
    with pytest.raises(ValueError):
        I.speed_configuration(p)


def test_later_seed_range_does_not_replace_construction_range():
    f = fixture()
    before = prepare(f)[0]
    for seed in f["seeds"]:
        seed["range_begin"], seed["range_end"] = 5., 15.
    after = prepare(f)[0]
    np.testing.assert_array_equal(before["crossing"]["reference_range"], after["crossing"]["reference_range"])
    np.testing.assert_array_equal(before["crossing"]["effect_time"], after["crossing"]["effect_time"])
