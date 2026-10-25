"""Single-group momentum convention and end-to-end diagnostic flag propagation."""
import json

import numpy as np
import pytest

from mu2efastsim import eloss, materials
from trkfastsim import cli, steps as st
from trkfastsim.analyses import slabcheck


@pytest.mark.parametrize("mean", [False, True])
def test_slab_core_tail_and_calibration_momenta(monkeypatch, mean):
    p = np.array([100., 300., 1000.])
    loss = np.array([10., 20., 30.])
    expected_exit = st._p_of(np.hypot(p, eloss.MMU) - eloss.MMU - loss)
    expected_scatter = .5 * (p + expected_exit) if mean else expected_exit
    seen = {}
    monkeypatch.setattr(st.tr, "slab_loss", lambda *args: loss.copy())

    def variance(momentum, start, end):
        seen["core"] = momentum.copy()
        return np.full(len(p), .01)

    def calibration(cal, columns, total, momentum):
        seen["calibration"] = momentum.copy()
        return np.ones(len(p))

    def tail(rng, parts, groups, row_p, group_p, width, tail_k):
        seen["tail_row"], seen["tail_group"] = row_p.copy(), group_p.copy()
        return np.zeros((len(p), 2))

    monkeypatch.setattr(st.tr, "highland_var_increment", variance)
    monkeypatch.setattr(st, "interior_msc_var_factor", calibration)
    monkeypatch.setattr(st, "_group_tail", tail)
    ok, pk, _, _ = st.slab_group(np.random.default_rng(4), materials.LEAD, 10., p,
                                msc_cal=st.INTERIOR_MSC_CAL, kink_p_mean=mean)
    assert ok.all()
    np.testing.assert_allclose(pk, expected_exit)
    np.testing.assert_allclose(seen["calibration"], expected_exit)
    for name in ("core", "tail_row", "tail_group"):
        np.testing.assert_allclose(seen[name], expected_scatter)


def test_slab_default_is_explicit_exit_and_mean_preserves_loss():
    p = np.full(2000, 300.)
    args = (materials.LEAD, 10., p)
    default = st.slab_group(np.random.default_rng(4), *args, tail=False)
    explicit = st.slab_group(np.random.default_rng(4), *args, tail=False, kink_p_mean=False)
    mean = st.slab_group(np.random.default_rng(4), *args, tail=False, kink_p_mean=True)
    for a, b in zip(default, explicit):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(default[:2], mean[:2]):
        np.testing.assert_array_equal(a, b)
    assert np.std(mean[2]) < np.std(default[2])


@pytest.mark.parametrize("mean", [False, True])
def test_cli_routes_flag_to_both_slab_arms_and_records_settings(monkeypatch, tmp_path, mean):
    seen = []
    monkeypatch.setattr(slabcheck.T, "expand", lambda inputs: ["G4_Al/slab.t10_p300.root"])
    angles = np.random.default_rng(3).normal(0., .01, (1200, 2))
    monkeypatch.setattr(slabcheck, "_proj_g4", lambda path: (angles, np.full(1200, 280.), 1200))

    def group(rng, mat, length, p, **kwargs):
        seen.append(kwargs)
        return np.ones(len(p), bool), p - 20., angles.copy(), np.zeros_like(angles)

    monkeypatch.setattr(st, "slab_group", group)
    argv = ["slabcheck", "unused.root", "--model", "interior", "--nmu", "1200", "--seed", "7",
            "--outdir", str(tmp_path)] + (["--kink-p-mean"] if mean else [])
    cli.main(argv)
    assert len(seen) == 2
    assert all(call["kink_p_mean"] is mean for call in seen)
    assert seen[0]["msc_cal"] is None
    assert seen[1]["msc_cal"] == st.INTERIOR_MSC_CAL
    output = json.loads((tmp_path / "slabcheck_interior.json").read_text())
    assert len(output["rows"]) == 1
    settings = output["settings"]
    assert settings["kink_p_mean"] is mean
    assert settings["scattering_momentum"] == ("entry_exit_mean" if mean else "exit")
    assert settings["calibration_momentum"] == "exit"
    assert settings["segment_model"] == "single_unsplit_group"
    assert settings["seed"] == 7 and settings["nmu"] == 1200


def test_world_rejects_mean_momentum_before_reading_or_writing(tmp_path):
    outdir = tmp_path / "not_created"
    with pytest.raises(ValueError, match="requires --model interior"):
        cli.main(["slabcheck", "missing.root", "--kink-p-mean", "--outdir", str(outdir)])
    assert not outdir.exists()
