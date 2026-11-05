"""Synthetic byte parity for the default wall route, with optional comparison to a main source file."""
import copy
import importlib.util

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import materials
from trkfastsim.sim import fieldsteps as fs
from test_fieldsteps import _geometry, _lines, _start

CASES = ("muon", "tail_off", "radiative_on", "radiative_off", "electron", "support")


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
    for left, right in zip(a[0], b[0]):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
        assert left.index.to_numpy().tobytes() == right.index.to_numpy().tobytes()
        for column in left:
            assert left[column].dtype == right[column].dtype
            assert left[column].to_numpy().tobytes() == right[column].to_numpy().tobytes()


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
