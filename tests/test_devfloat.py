"""Tracker device float (trkfastsim.devfloat): float64 default unchanged; following the core float32 (opt-in,
TRKFASTSIM_DEVICE_FLOAT=core) in subprocesses, where the centred forms keep straw crossings and digitiser counts at float32 rounding of tracker-local quantities."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
from trkfastsim import devfloat  # noqa: E402

TESTS = Path(__file__).resolve().parent


def _run(code, tmp_path, tracker="core", core="float32", draw64=False):
    env = dict(os.environ, MU2EFS_DEVICE_FLOAT=core, TRKFASTSIM_DEVICE_FLOAT=tracker, JAX_PLATFORMS="cpu",
               PYTHONPATH=os.pathsep.join([str(TESTS), str(TESTS.parent), os.environ.get("PYTHONPATH", "")]))
    env.pop("TRKFASTSIM_DRAW64", None)
    if draw64:
        env["TRKFASTSIM_DRAW64"] = "1"
    return subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)


def test_default_is_float64():
    assert os.environ.get("TRKFASTSIM_DEVICE_FLOAT") in (None, "", "float64")
    assert devfloat.NAME == "float64" and not devfloat.SINGLE and devfloat.FLOAT == np.float64
    assert devfloat.TINY == 1e-300 and not devfloat.DRAW64
    assert np.array_equal(devfloat.frame_of(np.full((3, 3), 1e4)), np.zeros(3))
    x = np.arange(3.0)
    assert devfloat.dev(x) is x and devfloat.host(x) is not None


@pytest.mark.slow   # three subprocess starts; per PR: test_default_is_float64
def test_setting_follows_core(tmp_path):
    probe = ("import json; from trkfastsim import devfloat as d; from mu2efastsim import precision as p; "
             "print(json.dumps([d.NAME, d.SINGLE, 'trkfastsim.digi.digi_jax' in p._AWARE or None, d.provenance()]))")
    probe = probe.replace("print(", "import trkfastsim.digi.digi_jax; print(")
    for tracker, core, name in (("core", "float32", "float32"), ("core", "float64", "float64"),
                                ("float64", "float32", "float64")):
        r = _run(probe, tmp_path, tracker=tracker, core=core)
        assert r.returncode == 0, r.stderr[-2000:]
        got, single, aware, prov = json.loads(r.stdout.strip().splitlines()[-1])
        assert got == name and single == (name == "float32") and bool(aware) == single
        assert prov["tracker_device_float"] == name and prov["device_float"] == core and prov["tracker_setting"] == tracker
    r = _run("import trkfastsim.devfloat", tmp_path, tracker="float32")
    assert r.returncode != 0 and "float64 or core" in r.stderr


CROSSINGS = """
import json, numpy as np
from test_steps_jax import _geometry, _lines
from trkfastsim import crossings, devfloat, geometry
from trkfastsim.sim import steps_jax as sj
g = _geometry()
O = geometry.DETECTOR_ORIGIN_EXTRACTED
g = geometry.StrawGeometry(plane=g.plane, panel=g.panel, straw=g.straw, mid=g.mid + O, dir=g.dir, halflen=g.halflen,
                           radius=g.radius, panel_R=g.panel_R, panel_t=g.panel_t, plane_R=g.plane_R, plane_t=g.plane_t,
                           wall_rmax=g.wall_rmax, metal_t=g.metal_t, wire_r=g.wire_r, wire_plate_t=g.wire_plate_t)
o, d = _lines(np.random.default_rng(6), 4000)
o = o + O
ref = crossings.straw_crossings(o, d, g)
got = sj.straw_crossings(g, o, d)
same = ref.shape == got.shape and np.array_equal(ref[:, :2], got[:, :2])
print(json.dumps({"single": devfloat.SINGLE, "n": len(ref), "same": bool(same),
                  "ds": float(np.abs(ref[:, 2:] - got[:, 2:]).max()) if same else None,
                  "frame": devfloat.frame_of(g.mid).tolist()}))
"""


@pytest.mark.slow   # float32 subprocess run; per PR: test_digi_run1a_drift::test_jax_d2t_same_stock_oracle[float32]
def test_float32_straw_crossings_centred(tmp_path):
    """Mu2e-frame straws (z ~ 24 m): same crossing lists; entry/exit distances to float32 rounding of local coordinates
    (at most ~2e-4 mm near tangents), not of the Mu2e frame."""
    r = _run(CROSSINGS, tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["single"] and out["n"] > 1000 and out["same"], out
    assert out["ds"] < 1e-3, out
    assert abs(out["frame"][2] - 24171.0) < 10


DIGI = """
import json, sys, numpy as np, jax
from test_digi import FIX, _geom, _step, _steps
from trkfastsim.digi import digicond as DC, digi_jax as DJ
d = DC.parse_fhicl_dump(FIX.read_text())
pr = np.zeros((DC.NUSTRAWS, 6)); pr[:, 0] = np.arange(DC.NUSTRAWS); pr[:, 3], pr[:, 4], pr[:, 5] = 11.996, 11.632, 1.5e6
c = DC.assemble(d, pr, _geom())
rng = np.random.default_rng(4)
rows, ev = [], []
for i in range(400):
    us = int(rng.integers(0, 20000))
    t = float(rng.choice([700.0, 30000.0, 90000.0])) + rng.uniform(0, 1000)
    rows.append(_step(c, us, b=rng.uniform(0.05, 2.4), zpos=rng.uniform(-400, 400), t=t, eion=rng.uniform(5e-4, 6e-3)))
    ev.append(i)
a = DJ.digitize(_steps(rows, ev), 400, c, jax.random.key(8))
np.savez(sys.argv[1] if len(sys.argv) > 1 else "digi.npz", **a)
"""


@pytest.mark.slow   # float32 subprocess digitiser run (40 s); float32 parity per PR: d2t oracle, grouped charge
def test_float32_digitiser_matches_float64_draws(tmp_path):
    """Same steps and key, float64 draws rounded (TRKFASTSIM_DRAW64=1): float32 electronics give the float64 digis up to
    rounding near thresholds, also at absolute times of 3e4 and 9e4 ns (TDC LSB 0.0195 ns, float32 ulp there 0.008)."""
    (tmp_path / "run.py").write_text(DIGI)
    for name, kw in (("f64", dict(tracker="float64", core="float64")), ("f32", dict(draw64=True))):
        r = _run(f"import runpy, sys; sys.argv = ['run.py', '{name}.npz']; runpy.run_path('run.py')", tmp_path, **kw)
        assert r.returncode == 0, r.stderr[-2000:]
    a, b = np.load(tmp_path / "f64.npz"), np.load(tmp_path / "f32.npz")
    assert len(a["sid"]) > 300
    key = lambda x: list(zip(x["ev"], x["sid"]))
    ka, kb = key(a), key(b)
    common = sorted(set(ka) & set(kb))
    assert len(common) >= 0.99 * len(ka)
    ia, ib = [ka.index(k) for k in common], [kb.index(k) for k in common]
    dt = np.abs(a["tdc"][ia] - b["tdc"][ib])
    assert np.mean(dt == 0) > 0.95 and np.quantile(dt, 0.99) <= 1, np.bincount(dt.ravel())[:5]
    assert np.mean(a["tot"][ia] == b["tot"][ib]) > 0.98
    assert np.mean(np.abs(a["adc"][ia] - b["adc"][ib]) <= 1) > 0.99
    late = a["t_analog"][ia][:, 0] > 9e4
    assert late.sum() > 50 and np.mean(dt[late] == 0) > 0.95
