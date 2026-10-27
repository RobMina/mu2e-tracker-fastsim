"""EventNtuple writer: schema against the stored stock schema, empty inputs, round trip, replay, geometry."""
import json

import awkward as ak
import numpy as np
import pytest
import uproot

from trkfastsim.reco import ntcompare, ntschema, ntuple, tracktable

STOCK = ntuple.DEFAULT_SCHEMA


def _table(lines, n_events=3, ev=None, good=None, nhit=12, xings=True, momfit=True):
    """Synthetic track table: straight lines x = (a0, b0, a1, b1, t0, mom), straws along x placed on the line at
    equal steps in y with a 1 mm offset in z (DOCA 1 mm), one hit per straw."""
    x = np.asarray(lines, np.float64).reshape(-1, 6)
    T = len(x)
    ev = np.zeros(T, np.int64) if ev is None else np.asarray(ev, np.int64)
    good = np.ones(T, np.int32) if good is None else np.asarray(good, np.int32)
    p, u, t0, mom, v = ntuple.line(x) if T else (np.zeros((0, 3)),) * 5
    s = np.linspace(-600, 600, nhit)
    mid = (p[:, None] + u[:, None] * s[None, :, None]) if T else np.zeros((0, nhit, 3))
    mid = mid + np.array([0.0, 0.0, 1.0])
    wdir = np.zeros((T, nhit, 3))
    wdir[..., 0] = 1.0
    plane = np.arange(nhit) % 36
    sid = (plane << 10) | (2 << 7) | 10
    tb = {"events": {"run": np.full(n_events, 1400), "subrun": np.arange(n_events), "event": np.arange(n_events) + 100,
                     "pbt": np.zeros(n_events, np.float32)},
          "ev": ev, "x": x, "cov": np.tile(np.eye(6) * 1e-4, (T, 1, 1)), "chi2": np.full(T, 10.0),
          "ndof": np.full(T, 31), "status": np.where(good == 1, 1, -1).astype(np.int32), "goodfit": good,
          "momfit": np.full(T, int(momfit), np.int32),
          "mid": mid, "wdir": wdir, "hlen": np.full((T, nhit), 500.0), "esign": np.ones((T, nhit)),
          "vsig": np.full((T, nhit), 200.0), "tmeas": np.full((T, nhit), 30.0), "totdt": np.full((T, nhit), 15.0),
          "uvar": np.full((T, nhit), 900.0), "wdist": np.zeros((T, nhit)), "edep": np.full((T, nhit), 1e-3),
          "sh": np.tile(np.arange(nhit), (T, 1)), "hvalid": np.ones((T, nhit), bool), "sid": np.tile(sid, (T, 1)),
          "flag": np.full((T, nhit), (1 << 1) | (1 << 2) | (1 << 3)), "et0": np.full((T, nhit), 30.0),
          "et1": np.full((T, nhit), 31.0), "tot0": np.full((T, nhit), 20.0), "tot1": np.full((T, nhit), 21.0),
          "eend": np.zeros((T, nhit), np.int64), "state": np.tile(np.where(np.arange(nhit) % 4 == 0, -2, 1), (T, 1)).astype(np.int32),
          "drift": {"dt": np.array([0.0, 50.0]), "rdrift": np.array([0.0, 2.5]), "dvel": np.array([0.05, 0.05]),
                    "sderr": np.array([0.1, 0.1]), "uderr": np.array([0.2, 0.2])},
          "meta": {"version": tracktable.VERSION, "fit_type": "test", "stock_conditioned": False}}
    tb["active"] = tb["state"] > -2
    tb["lr"] = np.where(tb["state"] == 1, 1.0, 0.0)
    if xings:
        M = 4
        tb["xings"] = {"xvalid": np.ones((T, M), bool), "sid": tb["sid"][:, :M], "active": np.ones((T, M), bool),
                       "wall": np.full((T, M), 0.03), "gas": np.full((T, M), 3.0), "wire": np.zeros((T, M)),
                       "radlen": np.full((T, M), 1e-4), "doca": np.full((T, M), 1.0), "docavar": np.full((T, M), 0.01),
                       "dirdot": np.zeros((T, M))}
    else:
        tb["xings"] = None
    return tb


VERT = [0.0, 0.0, 0.0, 0.0, 50.0, 2000.0]          # vertical line through the origin, downward
SLANT = [100.0, -300.0, 0.2, 0.3, 80.0, 1950.0]


def _read(path):
    t = uproot.open(path)[ntuple.TREE]
    return t, {b: t[b].array() for b in t.keys(recursive=False)}


def test_schema_matches_stored_stock(tmp_path):
    out = tmp_path / "fast.root"
    ntuple.write(_table([VERT, SLANT], ev=[0, 2]), str(out))
    stock = json.load(open(STOCK))
    fast = ntschema.inventory(str(out), ntuple.BRANCHES)
    t = uproot.open(out)[ntuple.TREE]
    for b in ntuple.BRANCHES:
        sig_s = ntschema.signature(stock["branches"][b]["form"])
        arr = t[b].array()
        sig_f = ntschema.signature((ntuple.without_alias(arr) if b == "trk" else arr).layout.form.to_dict())
        assert sig_f == sig_s, b
        assert fast["branches"][b]["arrays_fields"] == stock["branches"][b]["arrays_fields"], b
    # the only addition is the trailing split-path alias, which mirrors the stock trk leaves under bare names
    fs = ntschema.signature(fast["branches"]["trk"]["form"])["record"]
    assert [f for f, _ in fs[:-1]] == [f for f, _ in ntschema.signature(stock["branches"]["trk"]["form"])["record"]]
    assert fs[-1][0] == ntuple.SPLIT_ALIAS and [f for f, _ in fs[-1][1]["record"]] == [f[4:] for f, _ in fs[:-1]]
    # without the alias the layout is the bare stock one
    out2 = tmp_path / "bare.root"
    ntuple.write(_table([VERT]), str(out2), split_alias=False)
    bare = ntschema.inventory(str(out2), ("trk",))
    assert ntschema.signature(bare["branches"]["trk"]["form"]) == ntschema.signature(stock["branches"]["trk"]["form"])


@pytest.mark.parametrize("n_events", [0, 4])
def test_empty_inputs(tmp_path, n_events):
    out = tmp_path / "empty.root"
    prov = ntuple.write(_table([], n_events=n_events, xings=False), str(out))
    t, a = _read(out)
    assert t.num_entries == n_events
    assert prov["counts"]["tracks_written"] == 0
    assert ak.sum(ak.num(a["trk"]["trk.status"], axis=1)) == 0
    assert set(a) == {"evtinfo", "trk", "trksegs", "trkhits", "trksegpars_kl"}       # no trkmats without crossings
    sig = ntschema.signature(json.load(open(STOCK))["branches"]["trksegs"]["form"])
    assert ntschema.signature(a["trksegs"].layout.form.to_dict()) == sig


def test_round_trip_values_and_sentinels(tmp_path):
    out = tmp_path / "rt.root"
    tb = _table([VERT, SLANT, VERT], ev=[0, 2, 2], good=[1, 1, 0])
    prov = ntuple.write(tb, str(out))
    t, a = _read(out)
    assert np.array_equal(ak.to_numpy(a["evtinfo"]["event"]), [100, 101, 102])
    assert np.array_equal(t["evtinfo/run"].array(library="np"), [1400] * 3)
    assert ak.to_list(ak.num(a["trk"]["trk.status"], axis=1)) == [1, 0, 1]             # failed fit not written
    assert prov["counts"]["not_written_failed_fits"] == 1
    trk = a["trk"]
    assert ak.to_list(ak.flatten(trk["trk.nhits"])) == [12, 12]
    assert ak.to_list(ak.flatten(trk["trk.nactive"])) == [9, 9]
    assert ak.to_list(ak.flatten(trk["trk.fitalg"])) == [3, 3]
    assert ak.to_list(ak.flatten(trk["trk.nseg"])) == [ntuple.UNSET_I] * 2               # sentinel
    # vertical line: enters and leaves through TT_Outer at y = +-ROUT
    seg = a["trksegs"][0][0]
    assert ak.to_list(seg["sid"]) == [4, 4]
    assert np.allclose(ak.to_numpy(seg["pos"]["fCoordinates"]["fY"]), [ntuple.ROUT, -ntuple.ROUT], atol=1e-3)
    assert np.all(ak.to_numpy(seg["time"])[1] > ak.to_numpy(seg["time"])[0])
    assert np.allclose(ak.to_numpy(seg["mom"]["fCoordinates"]["fY"]), -2000.0)
    h = a["trkhits"][0][0]
    assert np.allclose(np.abs(ak.to_numpy(h["rdoca"])), 1.0, atol=1e-5)
    assert np.all(ak.to_numpy(h["udoca"]) == ntuple.UNSET_F)                              # sentinel
    assert np.all(np.diff(ak.to_numpy(h["ptoca"])) > 0)                                   # along the track
    assert ak.to_list(h["etime"][0]) == [30.0, 31.0]
    pars = a["trksegpars_kl"][0][0]
    assert np.allclose(ak.to_numpy(pars["theta"]), np.pi / 2, atol=1e-6)
    assert len(a["trkmats"][0][0]) == 4 and all(ak.to_list(a["trkmats"][0][0]["hashit"]))
    assert prov["coverage"]["trkhits"]["udoca"][0] == "sentinel"
    assert prov["coverage"]["trk"]["trk.nhits"][0] == "filled"
    assert ntuple.read_provenance(str(out))["content_sha256"] == prov["content_sha256"]


def test_all_fits_keeps_failed(tmp_path):
    tb = _table([VERT, SLANT], ev=[1, 1], good=[1, 0])
    arrays, prov = ntuple.build(tb, only_good=False)
    assert ak.to_list(ak.num(arrays["trk"]["trk.goodfit"], axis=1)) == [0, 2, 0]
    assert ak.to_list(arrays["trk"]["trk.goodfit"][1]) == [1, 0]
    assert ak.to_list(arrays["trk"]["trk.status"][1]) == [1, -1]


def test_replay_is_deterministic(tmp_path):
    tb = _table([SLANT, VERT], ev=[0, 1])
    p1 = ntuple.write(tb, str(tmp_path / "a.root"))
    p2 = ntuple.write(_table([SLANT, VERT], ev=[0, 1]), str(tmp_path / "b.root"))
    assert p1["content_sha256"] == p2["content_sha256"]
    _, a = _read(tmp_path / "a.root")
    _, b = _read(tmp_path / "b.root")
    for k in a:
        assert ak.array_equal(a[k], b[k], equal_nan=True), k


def test_fixed_momentum_errors_are_sentinels(tmp_path):
    tb = _table([SLANT], momfit=False, xings=False)
    tb["cov"][:, 5, :] = np.nan
    tb["cov"][:, :, 5] = np.nan
    arrays, prov = ntuple.build(tb)
    assert np.all(ak.to_numpy(ak.flatten(arrays["trksegs"]["momerr"], axis=None)) == ntuple.UNSET_F)
    assert np.all(ak.to_numpy(ak.flatten(arrays["trksegpars_kl"]["momerr"], axis=None)) == ntuple.UNSET_F)
    assert np.all(ak.to_numpy(ak.flatten(arrays["trksegpars_kl"]["thetaerr"], axis=None)) > 0)
    assert prov["coverage"]["trksegs"]["momerr"][0] == "sentinel"


def test_coverage_classes_follow_fit_type():
    tb = _table([SLANT])
    tb["meta"]["fit_type"] = "F1"
    _, prov = ntuple.build(tb)
    assert prov["coverage"]["trk"]["trk.radlen"][0] == "approx"
    assert prov["coverage"]["trkmats"]["radlen"][0] == "approx"
    assert prov["coverage"]["trksegs"]["mom.fCoordinates.fX"][0] == "approx"
    tb["meta"]["fit_type"] = "stock"
    assert ntuple.build(tb)[1]["coverage"]["trk"]["trk.radlen"][0] == "filled"


def test_tracker_crossings():
    # along z, off axis: crosses the TT_Front then the TT_Back disk, never the TT_Outer cylinder
    x = np.array([[100.0, 0.0, 0.0, 1e4, 0.0, 2000.0]])
    S, I = ntuple.tracker_crossings(x)
    p, u, *_ = ntuple.line(x)
    ok = np.isfinite(S[0])
    assert list(I[0][ok]) == [0, 2]
    assert np.allclose(p[0, 2] + u[0, 2] * S[0][ok], [ntuple.ZFRONT, ntuple.ZBACK])
    # steep line leaving the cylinder near the back end, then crossing the back disk outside r = 850 (stock "4-4-2")
    x = np.array([[0.0, 1600.0, 0.0, 0.0435, 0.0, 2000.0]])
    S, I = ntuple.tracker_crossings(x)
    o = np.argsort(np.where(np.isfinite(S[0]), S[0], np.inf))
    assert list(I[0][o][np.isfinite(S[0][o])]) == [4, 4, 2]
    miss = np.array([[2000.0, 0.0, 0.0, 0.0, 0.0, 2000.0]])                       # vertical line outside r = 950
    assert not np.isfinite(ntuple.tracker_crossings(miss)[0]).any()


def test_kl_params_match_linefit():
    from trkfastsim.reco import linefit
    x = np.array([SLANT, VERT], np.float64)
    x[:, 5] = linefit.MOM
    kp = ntuple.kl_params(x)
    ref = linefit.to_kinematic_line(x[:, :5])
    for i, k in enumerate(("d0", "phi0", "z0", "theta", "t0")):
        assert np.allclose(kp[:, i], ref[k], atol=1e-9), k


def test_reader_access_forms(tmp_path):
    out = tmp_path / "r.root"
    ntuple.write(_table([SLANT, VERT], ev=[0, 0]), str(out))
    t = uproot.open(out)[ntuple.TREE]
    assert ak.to_list(t["evtinfo/run"].array()) == [1400] * 3
    assert t.arrays(["trk", "trksegs"]).fields == ["trk", "trksegs"]
    trk = t["trk"].array()
    assert "trk.nactive" in trk.fields
    # split-branch paths as the analyses use them (crvextrap.io.load_quality, trackline, crvdtcal)
    for leaf in ("status", "nactive", "fitcon", "chisq", "ndof", "goodfit", "nplanes"):
        a = t[f"trk/trk.{leaf}"].array()
        assert str(a.type) == str(trk[f"trk.{leaf}"].type), leaf
        assert ak.array_equal(a, trk[f"trk.{leaf}"]), leaf
    assert ak.to_list(ak.num(t["trk/trk.nactive"].array(), axis=1)) == [2, 0, 0]
    # documented uproot 5.7 RNTuple limit: tree.arrays([...]) returns empty records (pyutils Importer)
    assert "trk.nactive" not in t.arrays(["trk"])["trk"].fields
    assert t.arrays(["evtinfo"])["evtinfo"].fields == []
    bare = tmp_path / "bare.root"
    ntuple.write(_table([SLANT]), str(bare), split_alias=False)
    with pytest.raises(uproot.KeyInFileError):
        uproot.open(bare)[ntuple.TREE]["trk/trk.status"]


def test_crvextrap_reader_unmodified(tmp_path):
    XIO = pytest.importorskip("crvextrap.io")
    out = tmp_path / "x.root"
    ntuple.write(_table([SLANT, VERT, SLANT], ev=[0, 0, 2]), str(out))
    keys = XIO.event_keys(XIO.open_tree(str(out)))
    assert keys.tolist() == [[1400, 0, 100], [1400, 1, 101], [1400, 2, 102]]
    q = XIO.load_quality([str(out)])
    assert ak.to_list(ak.num(q["status"], axis=1)) == [2, 0, 1]
    fx = XIO.flat_xings(XIO.load_branch([str(out)], "trksegs"))
    assert fx.ntracks == 3 and set(fx.sid.tolist()) <= {0, 2, 4}


def test_ntcompare_identity(tmp_path):
    out = tmp_path / "c.root"
    ntuple.write(_table([SLANT, VERT], ev=[0, 2]), str(out))
    A = ntcompare.load(str(out))
    res = ntcompare.compare(A, A)
    assert res["track_match"]["matched"] == 2 and res["track_match"]["A_unmatched"] == 0
    assert res["matched_trk"]["nactive"]["identical_frac"] == 1.0
    assert res["matched_ref_segment"]["4"]["dx"]["q"][2] == 0.0


def test_stored_schemas_agree_on_consumed_branches():
    """The two stored stock schemas (EventNtuple v06_13_02 fork, v06_15_00) agree on every written branch except the
    two trkmats leaves v06_15_00 adds; the writer follows whichever schema it is given."""
    import os
    d = os.path.dirname(STOCK)
    a = json.load(open(os.path.join(d, "eventntuple_v06_13_02_ssmc.json")))
    b = json.load(open(os.path.join(d, "eventntuple_v06_15_00.json")))
    for br in ntuple.BRANCHES:
        la = set(ntschema.leaves(ntschema.signature(a["branches"][br]["form"])))
        lb = set(ntschema.leaves(ntschema.signature(b["branches"][br]["form"])))
        assert la <= lb and {p for p, _, _ in lb - la} <= {"dead", "phi"}, br
    arrays, _ = ntuple.build(_table([SLANT]), schema=os.path.join(d, "eventntuple_v06_15_00.json"))
    assert {"dead", "phi"} <= set(arrays["trkmats"].fields)
