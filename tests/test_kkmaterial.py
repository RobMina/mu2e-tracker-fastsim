import numpy as np
import pytest

pytest.importorskip("jax")

from trkfastsim.reco import kkmaterial as k  # noqa: E402

IR, OR, W = k.STRAW["irad"], k.STRAW["orad"], k.STRAW["wall"]


def test_effective_za_atom_weighted():
    z, a = k.effective_za(k.CO2)
    assert z == pytest.approx((6 + 2 * 8) / 3, rel=1e-12)
    assert a == pytest.approx((12.011 + 2 * 15.9994) / 3, rel=1e-12)
    zw, aw = k.effective_za(k.MATERIALS["straw-wall"]["massfrac"])
    assert 4.5 < zw < 4.8 and 8.8 < aw < 9.1          # Mylar-dominated, a little Au and Al
    assert abs(sum(k.MATERIALS["straw-wall"]["massfrac"].values()) - 1) < 1e-12


@pytest.mark.parametrize("d", [0.0, 0.7, 1.6, 2.3])
def test_path_lengths_narrow_window_is_chord(d):
    w, g = (float(x) for x in k.path_lengths(d, 1e-10, 0.0, 1.0))
    gi, go = 2 * np.sqrt(IR * IR - d * d), 2 * np.sqrt(OR * OR - d * d)
    assert g == pytest.approx(gi, rel=1e-3)
    assert w == pytest.approx(go - gi, rel=2e-2)


def test_path_lengths_average_branch_and_angle_factor():
    w, g = (float(x) for x in k.path_lengths(1.0, 4.0, 0.0, 1.0))      # 2 sigma = 4 mm > irad
    assert g == pytest.approx(0.5 * np.pi * IR) and w == pytest.approx(np.pi * W)
    w6, g6 = (float(x) for x in k.path_lengths(1.0, 4.0, 0.6, 1.0))
    assert g6 == pytest.approx(g / 0.8) and w6 == pytest.approx(w / 0.8)
    wc, gc = (float(x) for x in k.path_lengths(1.0, 4.0, 0.9999, 1.0))
    assert gc == pytest.approx(10 * g)                                   # 1/sin capped at 10


def test_outside_straw_is_clamped_to_inner_radius():
    # stock clamps mindoca = min(|doca|, irad): a track outside the straw still gets the irad..orad wall segment
    w, g = (float(x) for x in k.path_lengths(3.0, 1e-10, 0.0, 1.0))
    seg = lambda d, r: 0.5 * r * r * (2 * np.arccos(d / r) - np.sin(2 * np.arccos(d / r)))
    assert g < 0.02                                                       # window [irad - dr, irad]: tiny gas chord
    assert w == pytest.approx(seg(IR, OR) / (OR - IR), rel=1e-2)


def test_scatter_var_scaling():
    s = lambda n, p, L: float(k.scatter_angle_var(n, p, L))
    assert s("straw-wall", 2000.0, 0.0) == 0.0
    r = s("straw-wall", 2000.0, 0.06) / s("straw-wall", 2000.0, 0.03)
    assert 1.8 < r < 2.2                                                   # about linear in path
    rp = s("straw-gas", 2000.0, 5.0) / s("straw-gas", 4000.0, 5.0)
    assert rp == pytest.approx(4.0 * (4000 / np.hypot(4000, k.MMU)) ** 2 / (2000 / np.hypot(2000, k.MMU)) ** 2, rel=2e-2)


def test_scatter_var_near_highland():
    # one straw at DOCA 0 for a 2 GeV/c muon: wall X0 ~ 28.7 cm (Mylar 28.5), gas X0 ~ 118 m
    w, g = (float(x) for x in k.path_lengths(0.0, 1e-10, 0.0, 1.0))
    x = w / 287.0 + g / 118000.0
    beta = 2000 / np.hypot(2000, k.MMU)
    hl = (13.6 / (beta * 2000) * np.sqrt(x) * (1 + 0.038 * np.log(x))) ** 2
    ld = float(k.straw_scatter_var(0.0, 1e-10, 0.0, 1.0, 2000.0))
    assert 0.5 < ld / hl < 3.0
