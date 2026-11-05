"""Frozen main outputs and default-wall byte parity, with optional in-process main comparison."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import materials
from trkfastsim.sim import fieldsteps as fs
from test_fieldsteps import _geometry, _lines, _start

CASES = ("muon", "tail_off", "radiative_on", "radiative_off", "electron", "support")
GOLDEN = Path(__file__).with_name("data") / "fieldwall_main9e81577.npz"
# Cross-platform float tolerance only; schema, integers and the full RNG state remain exact.
GOLDEN_RTOL, GOLDEN_ATOL = 1e-10, 1e-11


class SlabSupport:
    def _support(self, x, direction, length, moments):
        dz = direction[:, 2]
        safe = np.where(np.abs(dz) > 1e-15, dz, 1.0)
        a, b = (-12 - x[:, 2]) / safe, (-10 - x[:, 2]) / safe
        lo = np.clip(np.minimum(a, b), 0, length)
        hi = np.clip(np.maximum(a, b), 0, length)
        inside = (x[:, 2] >= -12) & (x[:, 2] <= -10)
        lo = np.where(np.abs(dz) > 1e-15, lo, 0.0)
        hi = np.where(np.abs(dz) > 1e-15, hi, np.where(inside, length, 0.0))
        amount = hi - lo
        name = materials.MYLAR.name
        return {name: amount}, {name: (amount, (hi**2 - lo**2) / 2, (hi**3 - lo**3) / 3)}


def unchanged(actual, expected):
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            unchanged(actual[key], expected[key])
    elif hasattr(expected, "__dict__"):
        unchanged(vars(actual), vars(expected))
    else:
        assert actual == expected


def run_case(module, name, **extra):
    geom = _geometry()
    o, d = _lines(16, np.random.default_rng(7100))
    electron = name == "electron"
    start = _start(o, d, 105 if electron else (50000 if name.startswith("radiative") else 150), pdg=11 if electron else 13)
    field = np.array([0., 0., 1.])
    env = module.Envelope(geom, pad_r=10, pad_z=10, axis_xy=(0., 0.))
    interior = SlabSupport() if name == "support" else None
    options = dict(envelope=env, origin=np.zeros(3), chord_mm=4.0, **extra)
    if name == "tail_off" or electron:
        options["tail"] = False
    if name.startswith("radiative"):
        options["radiative"] = name == "radiative_on"
    if electron:
        options["loss"] = module.electron_loss
    inputs = dict(start=start, geom=geom, field=field, env=env, interior=interior)
    before = copy.deepcopy(inputs)
    rng = np.random.default_rng(7101 + CASES.index(name))
    result = module.transport(rng, start, geom, field, interior, **options)
    unchanged(inputs, before)
    return result, rng.bit_generator.state


def compare_bytes(a, b):
    assert a[1] == b[1]
    assert len(a[0]) == len(b[0])
    for left, right in zip(a[0], b[0]):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
        assert left.index.to_numpy().tobytes() == right.index.to_numpy().tobytes()
        for column in left:
            assert left[column].dtype == right[column].dtype
            assert left[column].to_numpy().tobytes() == right[column].to_numpy().tobytes()


def compare_golden(actual, name):
    frames, rng = actual
    assert len(frames) == 2
    with np.load(GOLDEN, allow_pickle=False) as golden:
        assert rng == json.loads(golden[f"{name}_rng"].item())
        for i, frame in enumerate(frames):
            key = f"{name}_{i}"
            records = golden[key]
            assert not records.dtype.hasobject
            metadata = json.loads(golden[f"{key}_schema"].item())
            expected = pd.DataFrame.from_records(records)
            expected.index = pd.RangeIndex(*metadata["index_range"], name=metadata["index_name"])
            expected.columns.name = metadata["columns_name"]
            pd.testing.assert_index_equal(frame.index, expected.index, exact=True)
            pd.testing.assert_index_equal(frame.columns, expected.columns, exact=True)
            for column in expected:
                left, right = frame[column].to_numpy(), expected[column].to_numpy()
                assert left.dtype == right.dtype, column
                if right.dtype.kind == "f":
                    np.testing.assert_allclose(left, right, rtol=GOLDEN_RTOL, atol=GOLDEN_ATOL,
                                               equal_nan=False, err_msg=f"{name}/{i}/{column}")
                else:
                    np.testing.assert_array_equal(left, right, err_msg=f"{name}/{i}/{column}")


@pytest.mark.parametrize("name", CASES)
def test_default_matches_frozen_main(name):
    compare_golden(run_case(fs, name), name)


@pytest.mark.parametrize("change", ("float", "integer", "dtype", "columns", "index", "rng"))
def test_golden_rejects_shared_default_and_false_regression(change):
    a = run_case(fs, "muon")
    b = run_case(fs, "muon", complete_outer_wall=False)
    for frames, rng in (a, b):
        steps = frames[0]
        if change == "float":
            steps.loc[0, "eion"] += 1e-6
        elif change == "integer":
            steps.loc[0, "straw"] += 1
        elif change == "dtype":
            steps["event"] = steps["event"].astype(np.int32)
        elif change == "columns":
            steps.rename(columns={"eion": "wrong"}, inplace=True)
        elif change == "index":
            steps.index = pd.RangeIndex(1, len(steps) + 1)
        else:
            rng["state"]["state"] += 1
    # Same-route parity alone accepts a shared regression; the independent frozen output rejects it.
    compare_bytes(a, b)
    for actual in (a, b):
        with pytest.raises(AssertionError):
            compare_golden(actual, "muon")


@pytest.mark.parametrize("name", CASES)
def test_default_equals_explicit_false_bytes_and_rng(name):
    compare_bytes(run_case(fs, name), run_case(fs, name, complete_outer_wall=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="fieldsteps.py from the current main commit")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("fieldsteps_baseline", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    for name in CASES:
        compare_bytes(run_case(baseline, name), run_case(fs, name))
        print(f"{name}: main/default output bytes and final RNG identical; inputs unchanged")
