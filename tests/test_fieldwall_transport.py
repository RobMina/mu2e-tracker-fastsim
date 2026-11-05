"""Direct opt-in consumer tests, not reconstruction or physics closure."""
import numpy as np
import pandas as pd
import pytest

from mu2efastsim import eloss, materials
from trkfastsim import geometry
from trkfastsim.sim import fieldsteps as fs
from trkfastsim.sim.fieldwall import WallIndex


def geom():
    return geometry.StrawGeometry(
        plane=np.array([0]), panel=np.array([0]), straw=np.array([0]), mid=np.zeros((1, 3)),
        dir=np.array([[0., 0., 1.]]), halflen=np.array([2.]), radius=1.,
        panel_R=np.eye(3)[None], panel_t=np.zeros((1, 3)), plane_R=np.eye(3)[None],
        plane_t=np.zeros((1, 3)), wall_rmax=1.2, metal_t=.02, wire_r=.0125, wire_plate_t=.0002)


def start(x, d=(1., 0., 0.), p=2000., pdg=13):
    return pd.DataFrame(dict(run=[1], subrun=[2], event=np.array([2**54 + 1], dtype=np.uint64), sp=[1],
                             pdg=[pdg], x=[x[0]], y=[x[1]], z=[x[2]], px=[p*d[0]], py=[p*d[1]],
                             pz=[p*d[2]], t=[0.]))


def one_chord(monkeypatch, x, d=(1., 0., 0.), length=4., enabled=True, wire=False, support=None):
    g = geom()
    calls, tails, moments = {}, {}, []

    def loss(rng, mat, p, ln, mass):
        calls[mat.name] = ln.copy()
        return np.zeros_like(p)

    def kink(rng, variance, w1, w2):
        moments.append((w1.copy(), w2.copy()))
        return np.zeros((len(w1), 2)), np.zeros((len(w1), 2)), np.ones(len(w1))

    def tail(rng, parts, idx, p, pm, a, k):
        tails.update({m.name: ln.copy() for m, ln in parts})
        return np.zeros((len(p), 2))

    xcols = []

    def highland(p, before, after, mass):
        xcols.append((before.copy(), after.copy()))
        return np.zeros_like(p)

    monkeypatch.setattr(fs.st, '_kink_core', kink)
    monkeypatch.setattr(fs.st, '_group_tail', tail)
    monkeypatch.setattr(fs.tr, 'highland_var_increment', highland)
    env = fs.Envelope(g, pad_r=20., pad_z=20., axis_xy=(0., 0.))
    steps, final = fs.transport(np.random.default_rng(42), start(x, d), g, None, support,
                                chord_mm=length, max_chords=1, envelope=env, origin=np.zeros(3), loss=loss,
                                wire=wire, complete_outer_wall=enabled)
    return g, steps, final, calls, tails, moments, xcols


@pytest.mark.parametrize('x,d,length,wall,gas', [
    ((-2., 0., 0.), (1., 0., 0.), 4., .4, 2.),
    ((-2., 1.1, 0.), (1., 0., 0.), 4., 2*np.sqrt(1.2**2 - 1.1**2), 0.),
    ((-1.15, 0., 0.), (1., 0., 0.), .1, .1, 0.),
    ((1.05, 0., 0.), (1., 0., 0.), .1, .1, 0.),
    ((1.1, 0., -3.), (0., 0., 1.), 6., 4., 0.),
    ((1.1, 0., 3.), (0., 0., -1.), 6., 4., 0.),
])
def test_wall_loss_tail_and_msc_consume_complete_intervals(monkeypatch, x, d, length, wall, gas):
    g, steps, final, calls, tails, moments, xcols = one_chord(monkeypatch, x, d, length)
    f = g.metal_t / (g.wall_rmax - g.radius)
    for mat, fraction in ((materials.MYLAR, 1-f), (materials.ALUMINIUM_6061, f)):
        np.testing.assert_allclose(calls[mat.name], [wall*fraction], rtol=0, atol=1e-13)
        np.testing.assert_array_equal(tails[mat.name], calls[mat.name])
    assert (steps.empty if gas == 0 else len(steps) == 1)
    if gas:
        np.testing.assert_allclose(steps.len, [gas], rtol=0, atol=1e-13)
    else:
        assert fs.st.STRAW_GAS.name not in calls
    wallcol = WallIndex(g, g.wall_rmax, fs.ChordIndex).columns(np.array([x]), np.array([d]), np.array([length]))
    invx = (1-f)/eloss.radiation_length_mm(materials.MYLAR) + f/eloss.radiation_length_mm(materials.ALUMINIUM_6061)
    xx = wall*invx + gas/eloss.radiation_length_mm(fs.st.STRAW_GAS)
    np.testing.assert_allclose(xcols[0][1], [xx], rtol=1e-13, atol=1e-15)
    gm = gas/eloss.radiation_length_mm(fs.st.STRAW_GAS)
    np.testing.assert_allclose(moments[0][0], [(wallcol[1][0]*invx + gm*length/2)/xx], rtol=1e-13)
    np.testing.assert_allclose(moments[0][1], [(wallcol[2][0]*invx + gm*(length/2)**2)/xx], rtol=1e-13)
    assert final.event.iloc[0] == 2**54 + 1 and final.status.iloc[0] == 2


def test_enabled_consumption_not_legacy_wall_or_appended_duplicate(monkeypatch):
    original = fs._straw_pieces

    def poisoned(*args, **kwargs):
        sp = original(*args, **kwargs)
        return (*sp[:4], np.full_like(sp[4], 123.), *sp[5:])

    monkeypatch.setattr(fs, '_straw_pieces', poisoned)
    g, steps, _, calls, _, _, _ = one_chord(monkeypatch, (-2., 0., 0.))
    assert calls[materials.MYLAR.name][0] == pytest.approx(.36)
    assert calls[materials.ALUMINIUM_6061.name][0] == pytest.approx(.04)
    assert len(steps) == 1 and steps.len.iloc[0] == 2.


@pytest.mark.parametrize('x', [(-2., 0., 0.), (-2., 1.1, 0.)])
def test_default_and_explicit_false_do_not_build_wall_ledger(monkeypatch, x):
    def forbidden(*args, **kwargs):
        raise AssertionError('disabled wall ledger must not be built')

    monkeypatch.setattr(WallIndex, '__init__', forbidden)
    g = geom()
    opts = dict(chord_mm=4., max_chords=1, envelope=fs.Envelope(g, axis_xy=(0., 0.)),
                origin=np.zeros(3))
    ra, rb = np.random.default_rng(4), np.random.default_rng(4)
    a = fs.transport(ra, start(x), g, None, **opts)
    b = fs.transport(rb, start(x), g, None, complete_outer_wall=False, **opts)
    for left, right in zip(a, b):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert ra.bit_generator.state == rb.bit_generator.state


def test_material_free_chords_do_not_consume_rng_or_inflate_gas_steps():
    g = geom()
    opts = dict(chord_mm=4., max_chords=3, envelope=fs.Envelope(g, pad_r=20., axis_xy=(0., 0.)),
                origin=np.zeros(3))
    ra, rb = np.random.default_rng(5), np.random.default_rng(5)
    a = fs.transport(ra, start((-2., 10., 0.)), g, None, **opts)
    b = fs.transport(rb, start((-2., 10., 0.)), g, None, complete_outer_wall=True, **opts)
    for left, right in zip(a, b):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert a[0].empty and ra.bit_generator.state == rb.bit_generator.state


@pytest.mark.parametrize('wire', [False, True])
def test_gas_wire_support_columns_unchanged(monkeypatch, wire):
    class Support:
        def _support(self, x, c, length, moments):
            ln = np.full(len(x), .2)
            return {materials.COPPER.name: ln}, {materials.COPPER.name: (ln, ln*.5, ln*.25)}

    a = one_chord(monkeypatch, (-2., 0., 0.), enabled=False, wire=wire, support=Support())
    b = one_chord(monkeypatch, (-2., 0., 0.), wire=wire, support=Support())
    for mat in (fs.st.STRAW_GAS, materials.TUNGSTEN, materials.GOLD, materials.COPPER):
        if mat.name in a[3]:
            np.testing.assert_array_equal(a[3][mat.name], b[3][mat.name])
        else:
            assert mat.name not in b[3]
    pd.testing.assert_frame_equal(a[1], b[1], check_exact=True)


def test_wall_only_loss_changes_state_without_gas_step():
    g = geom()
    def loss(rng, mat, p, ln, mass):
        return ln
    opts = dict(chord_mm=4., max_chords=1, envelope=fs.Envelope(g, axis_xy=(0., 0.)),
                origin=np.zeros(3), loss=loss, msc=False)
    a = fs.transport(np.random.default_rng(6), start((-2., 1.1, 0.)), g, None, **opts)
    b = fs.transport(np.random.default_rng(6), start((-2., 1.1, 0.)), g, None,
                     complete_outer_wall=True, **opts)
    assert a[0].empty and b[0].empty
    assert b[1].px.iloc[0] < a[1].px.iloc[0]


@pytest.mark.parametrize('pdg,loss', [(13, None), (11, fs.electron_loss)])
def test_enabled_seed_repeatability(pdg, loss):
    g = geom()
    opts = dict(chord_mm=.7, max_chords=6, envelope=fs.Envelope(g, axis_xy=(0., 0.)),
                origin=np.zeros(3), loss=loss, complete_outer_wall=True)
    ra, rb = np.random.default_rng(7), np.random.default_rng(7)
    a = fs.transport(ra, start((-2., 0., 0.), p=105., pdg=pdg), g, np.array([0., 0., 1.]), **opts)
    b = fs.transport(rb, start((-2., 0., 0.), p=105., pdg=pdg), g, np.array([0., 0., 1.]), **opts)
    for left, right in zip(a, b):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert ra.bit_generator.state == rb.bit_generator.state


def test_mixed_lanes_feed_actual_consumers_and_dead_lane_is_not_scattered(monkeypatch):
    g = geom()
    initial = pd.concat([start((-2., 0., 0.), p=2000.), start((-2., 1.1, 0.), p=2100.),
                         start((-2., 10., 0.), p=2200.), start((-2., 10., 0.), p=2300.),
                         start((-2., 1.1, 0.), p=2400.)], ignore_index=True)
    initial['event'] = np.arange(5, dtype=np.uint64) + 2**54
    material, highland, tails = {}, [], {}
    original_columns = fs._columns

    def columns(n, parts):
        result = original_columns(n, parts)
        material.update({name: ln.copy() for name, (m, ln) in result.items()})
        return result

    def loss(rng, mat, p, ln, mass):
        return np.where((p == 2400.) & (mat is materials.MYLAR), 10000., 0.)

    def scatter(p, before, after, mass):
        highland.append(after-before)
        return np.zeros_like(p)

    def kink(rng, variance, w1, w2):
        return np.zeros((len(w1), 2)), np.zeros((len(w1), 2)), np.ones(len(w1))

    def tail(rng, parts, idx, p, pm, a, k):
        tails.update({m.name: ln.copy() for m, ln in parts})
        return np.zeros((len(p), 2))

    class Support:
        def _support(self, x, c, length, moments):
            ln = np.array([0., 0., 0., .2, 0.])
            return {materials.COPPER.name: ln}, {materials.COPPER.name: (ln, ln*.5, ln*.25)}

    monkeypatch.setattr(fs, '_columns', columns)
    monkeypatch.setattr(fs.tr, 'highland_var_increment', scatter)
    monkeypatch.setattr(fs.st, '_kink_core', kink)
    monkeypatch.setattr(fs.st, '_group_tail', tail)
    steps, final = fs.transport(np.random.default_rng(8), initial, g, None, Support(), chord_mm=4.,
                                max_chords=1, loss=loss, complete_outer_wall=True, origin=np.zeros(3),
                                envelope=fs.Envelope(g, axis_xy=(0., 0.)))
    graze = 2*np.sqrt(1.2**2 - 1.1**2)
    wall = np.array([.4, graze, 0., 0., graze])
    np.testing.assert_allclose(material[materials.MYLAR.name], wall*.9, rtol=0, atol=1e-14)
    np.testing.assert_allclose(material[materials.ALUMINIUM_6061.name], wall*.1, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(material[fs.st.STRAW_GAS.name], [2., 0., 0., 0., 0.])
    np.testing.assert_allclose(material[materials.TUNGSTEN.name], [.0246, 0., 0., 0., 0.], atol=1e-14)
    np.testing.assert_allclose(material[materials.GOLD.name], [.0004, 0., 0., 0., 0.], atol=1e-14)
    expected_x = sum(material[m.name]/eloss.radiation_length_mm(m) for m in
                     (materials.MYLAR, materials.ALUMINIUM_6061, fs.st.STRAW_GAS, materials.TUNGSTEN, materials.GOLD))
    expected_x[3] += .2/eloss.radiation_length_mm(materials.COPPER)
    np.testing.assert_allclose(highland[0], expected_x[[0, 1, 3]], rtol=1e-13, atol=1e-15)
    for name, ln in material.items():
        np.testing.assert_array_equal(tails[name], ln[[0, 1, 3]])
    np.testing.assert_array_equal(tails[materials.COPPER.name], [0., 0., .2])
    np.testing.assert_array_equal(final.status, [2, 2, 2, 2, 1])
    assert len(steps) == 1 and steps.event.iloc[0] == 2**54


def test_msc_disabled_has_exact_wall_debit_and_never_calls_scattering(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('MSC disabled')

    monkeypatch.setattr(fs.tr, 'highland_var_increment', forbidden)
    monkeypatch.setattr(fs.st, '_kink_core', forbidden)
    monkeypatch.setattr(fs.st, '_group_tail', forbidden)
    g = geom()
    steps, final = fs.transport(np.random.default_rng(9), start((-2., 1.1, 0.)), g, None,
                                chord_mm=4., max_chords=1, loss=lambda rng, mat, p, ln, mass: ln,
                                msc=False, complete_outer_wall=True, origin=np.zeros(3),
                                envelope=fs.Envelope(g, axis_xy=(0., 0.)))
    expected_loss = 2*np.sqrt(1.2**2 - 1.1**2)
    actual_loss = np.hypot(2000., eloss.MMU) - np.hypot(final.px.iloc[0], eloss.MMU)
    assert actual_loss == pytest.approx(expected_loss, abs=1e-12)
    assert steps.empty


@pytest.mark.parametrize('chord_mm', [.2, .7, 4.])
def test_transport_split_preserves_wall_without_duplicate_or_zero_gas_steps(chord_mm):
    consumed = {materials.MYLAR.name: 0., materials.ALUMINIUM_6061.name: 0.}

    def loss(rng, mat, p, ln, mass):
        if mat.name in consumed:
            consumed[mat.name] += float(ln.sum())
        return np.zeros_like(p)

    g = geom()
    steps, final = fs.transport(np.random.default_rng(10), start((-2., 0., 0.)), g, None,
                                chord_mm=chord_mm, max_path_mm=4., max_chords=25,
                                msc=False, wire=False, loss=loss, complete_outer_wall=True,
                                origin=np.zeros(3), envelope=fs.Envelope(g, axis_xy=(0., 0.)))
    np.testing.assert_allclose(list(consumed.values()), np.array([.9, .1])*.4, rtol=0, atol=2e-14)
    assert final.status.iloc[0] == 2 and final.x.iloc[0] == pytest.approx(2.)
    assert len(steps) == 1 and steps.len.iloc[0] == pytest.approx(2.)


def test_cap_only_passage_has_no_wall_material_draws():
    g = geom()
    consumed = []
    def loss(rng, mat, p, ln, mass):
        consumed.append(mat.name)
        return np.zeros_like(p)

    steps, final = fs.transport(np.random.default_rng(11), start((.5, 0., -3.), (0., 0., 1.)), g, None,
                                chord_mm=6., max_chords=1, loss=loss, complete_outer_wall=True, msc=False,
                                wire=False, origin=np.zeros(3), envelope=fs.Envelope(g, axis_xy=(0., 0.)))
    assert consumed == [fs.st.STRAW_GAS.name]
    assert len(steps) == 1 and steps.len.iloc[0] == 4.
