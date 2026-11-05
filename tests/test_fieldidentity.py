"""Exact saved-primary links, fixed proxy controls, and vectorized native ntuple layout."""
import json

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from trkfastsim.validation import fieldidentity as fi


def inputs(n=12):
    keys = pd.DataFrame({"run": np.full(n, 1430, dtype=np.uint64),
                         "subrun": np.zeros(n, dtype=np.uint64),
                         "event": np.arange(n, dtype=np.uint64)})
    particles = keys.assign(sp=pd.array(np.ones(n, dtype=np.uint64), dtype="UInt64"))
    steps = particles.assign(t=0.0, px=2500.0, py=0.0, pz=0.0, plane=2)
    frames = {}
    for arm in fi.ARMS:
        frames[arm] = keys.assign(ntrk=1, ntracks_total=1, status=1.0, nactive=30.0, nhits=35.0,
                                 ndof=25.0, chisq=26.0, chi2ndf=26 / 25, fitcon=0.5, sid=0,
                                 p_reco=np.linspace(2450, 2550, n), p_true=np.nan,
                                 mc_id=pd.array(np.ones(n, dtype=np.uint64), dtype="UInt64"),
                                 mc_rank0_count=1, mc_missing=False, mc_ambiguous=False,
                                 mc_invalid_id=False, mc_unavailable=False)
    return particles, {k: steps.copy() for k in fi.FILES}, frames


def extra_particle(particles, steps, event, sp, momentum, time):
    p = particles[particles.event == event].iloc[:1].copy()
    p["sp"] = pd.array([sp], dtype="UInt64")
    particles = pd.concat([particles, p], ignore_index=True)
    for name, s in steps.items():
        steps[name] = pd.concat([s, p.assign(t=time, px=momentum, py=0.0, pz=0.0, plane=9)], ignore_index=True)
    return particles, steps


def test_single_particle_proxy_unchanged_and_fixed_bins():
    p, s, f = inputs()
    d = fi.assemble(p, s, f)
    assert len(d) == 12 and d.same_id_proxy_pair.all()
    assert not d.multiprimary.any()
    for arm in fi.ARMS:
        pd.testing.assert_series_equal(d[f"{arm}_p_first_own"], d[f"{arm}_p_own_id"], check_names=False)
        assert str(d[f"{arm}_mc_id"].dtype) == "UInt64"
    r = fi.report(d)
    assert [(b["p_lo"], b["p_hi"]) for b in r["bins"]] == list(zip(fi.fieldreco.P_EDGES[:-1], fi.fieldreco.P_EDGES[1:]))
    b = r["two_to_five_GeV"]
    a = b["original_proxy_all_pairs"]
    assert a == b["original_proxy_same_id_subset"] == b["own_particle_proxy_same_id_subset"]
    assert a["bootstrap_draws"] == 500
    assert a["metrics"] == ["relative_median", "relative_IQR_sigma"]
    assert a["difference"] == [0.0, 0.0] and a["paired_error"] == [0.0, 0.0]
    assert b["width_attribution"]["association_change_same_subset"] == 0
    assert b["original_proxy_single_entry_particle_events"] == a
    assert r["bins"][0]["own_particle_proxy_same_id_subset"]["difference"] == [None, None]
    assert "NaN" not in json.dumps(r, allow_nan=False)


def test_multiparticle_wrong_event_first_and_crossarm_identity():
    p, s, f = inputs()
    for event in range(12):
        p, s = extra_particle(p, s, event, 2, 4000.0, 10.0)
    # The reco track belongs to the later primary, not the event-earliest straw.
    for arm in fi.ARMS:
        f[arm]["mc_id"] = pd.array([2] * 12, dtype="UInt64")
    f["fast"]["p_reco"] += np.linspace(-20, 20, 12)
    # Both are legitimate particles but the first good tracks select different primaries in event zero.
    f["fast"].loc[0, "mc_id"] = 1
    d = fi.assemble(p, s, f)
    assert d.multiprimary.all() and d.g4_event_first_id_mismatch.all()
    assert d.crossarm_selected_id_mismatch.sum() == 1
    assert not d.loc[0, "same_id_proxy_pair"]
    assert d.loc[0, "excluded_reason"] == "crossarm_selected_id_mismatch"
    assert d.g4_p_first_own.eq(2500).all() and d.g4_p_own_id.eq(4000).all()
    b = fi.report(d)["two_to_five_GeV"]
    assert b["denominator_events"] == 12  # bins remain old-event-first G4, not own primary
    assert b["original_proxy_all_pairs"]["events"] == 12
    assert b["original_proxy_same_id_subset"]["events"] == 11
    assert b["own_particle_proxy_same_id_subset"]["events"] == 11
    assert b["width_attribution"]["association_change_same_subset"] != 0


def test_all_denominator_rows_including_no_reco_and_foreign_ids():
    p, s, f = inputs()
    f["g4"] = f["g4"][f["g4"].event != 0].copy()
    f["fast"].loc[1, "ntrk"] = 0
    f["fast"].loc[1, "mc_id"] = pd.NA
    f["fast"].loc[2, "mc_id"] = 99
    # Foreign ID is never mapped to the only available particle.
    d = fi.assemble(p, s, f)
    assert len(d) == 12
    assert d.loc[0, "g4_ntuple_missing"]
    assert d.loc[1, "fast_no_good_track"]
    assert d.loc[2, "fast_foreign_id"] and pd.isna(d.loc[2, "fast_matched_id"])
    assert pd.isna(d.loc[2, "fast_p_own_id"])
    # The old table proxy is preserved even when the ntuple is absent.
    assert d.loc[0, "g4_p_first_own"] == 2500
    assert d.loc[0, "original_proxy_pair"] == False
    counts = fi.report(d)["excluded_reason_counts"]
    assert sum(counts.values()) == 12 and counts["included"] == 9


def test_exact_uint64_keys_above_float_precision_and_no_event_leakage():
    p, s, f = inputs(6)
    big = 2 ** 53 + 17
    ids = np.array([big, big + 1, 2 ** 63 + 7, 4, 5, 6], dtype=np.uint64)
    events = np.arange(big, big + 6, dtype=np.uint64)
    p["event"], p["sp"] = events, pd.array(ids, dtype="UInt64")
    for arm in s:
        s[arm]["event"], s[arm]["sp"] = events, pd.array(ids, dtype="UInt64")
    for arm in f:
        f[arm]["event"], f[arm]["mc_id"] = events, pd.array(ids, dtype="UInt64")
    # A real ID in a DIFFERENT event must still be foreign, never a global-ID match.
    f["fast"].loc[0, "mc_id"] = big + 1
    d = fi.assemble(p, s, f)
    assert list(d.event) == list(events)
    assert int(d.loc[1, "g4_matched_id"]) == big + 1
    assert int(d.loc[2, "g4_matched_id"]) == 2 ** 63 + 7
    assert d.loc[0, "fast_foreign_id"]
    assert d.same_id_proxy_pair.sum() == 5
    f["g4"]["mc_id"] = f["g4"].mc_id.astype(float)
    with pytest.raises(ValueError, match="noninteger"):
        fi.assemble(p, s, f)


def test_no_own_steps_and_baseline_exact_preservation():
    p, s, f = inputs()
    s["fast"] = s["fast"][s["fast"].event != 0]
    d = fi.assemble(p, s, f)
    assert d.loc[0, "fast_no_own_steps"] and len(d) == 12
    assert not d.loc[0, "same_id_proxy_pair"]
    cols = fi.KEY + ["p_first", "first_plane"] + [f"{a}_{c}" for a in fi.ARMS for c in (
        "ntrk", "nactive", "nhits", "ndof", "chisq", "fitcon", "chi2ndf", "sid", "p_reco", "p_true", "p_first_own")]
    baseline = d[cols].copy()
    assert fi.check_baseline(d, baseline)["unchanged"]
    baseline.loc[1, "fast_p_first_own"] += 1e-10
    with pytest.raises(ValueError, match="changed"):
        fi.check_baseline(d, baseline)


class Branch:
    def __init__(self, data):
        self.data = data

    def array(self, library="ak"):
        assert library == "ak"
        return self.data


class Tree(dict):
    num_entries = 8
    object_path = "/EventNtuple/ntuple;9"


class Root(dict):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


def synthetic_tree():
    """Real awkward native shapes behind uproot's branch API; deliberately no trk parent array."""
    tree = Tree()
    for k in fi.KEY:
        tree[f"evtinfo/{k}"] = Branch(ak.Array(np.arange(2 ** 53 + 11, 2 ** 53 + 19, dtype=np.uint64)))
    status = ak.Array([[0, 1, 1], [], [1], [1], [1], [1], [-1], [1]])
    for k in fi.fieldreco.TRK:
        tree[f"trk/trk.{k}"] = Branch(status if k == "status" else ak.ones_like(status) * 20)
    seg = {"sid": 1, "sindex": 0, "inbounds": True, "time": 10.0,
           "mom": {"fCoordinates": {"fX": 3000.0, "fY": 0.0, "fZ": 0.0}}}
    earlier = {**seg, "sid": 4, "time": 2.0}
    outside = {**seg, "sid": 0, "time": 1.0, "inbounds": False}
    tree["trksegs"] = Branch(ak.Array([[[seg], [seg, earlier, outside], [seg]], [], [[seg]], [[seg]], [[seg]],
                                         [[seg]], [[seg]], [[seg]]]))
    def mc(id, valid=1, rank=0):
        return {"id": id, "valid": valid, "rank": rank, "pdg": 13, "nhits": 20, "nactive": 18}
    tree["trkmcsim"] = Branch(ak.Array([[[mc(9)], [mc(2 ** 53 + 17)], [mc(88)]], [],
        [[mc(2, valid=0), mc(3, rank=1)]], [[mc(4), mc(5)]], [[mc(0)]], [[mc(99)]], [[mc(7)]],
        [[mc(12), mc(12)]]]))
    return tree


def test_vectorized_selected_first_good_missing_ambiguous_and_no_guesses(monkeypatch):
    root = Root({fi.fieldreco.TREE: synthetic_tree()})
    monkeypatch.setattr(fi.uproot, "open", lambda path: root)
    monkeypatch.setattr(ak, "to_list", lambda *a, **k: pytest.fail("per-event Python conversion"))
    d = fi.read_arm("synthetic.root")
    assert root.closed
    assert list(d.ntrk) == [2, 0, 1, 1, 1, 1, 0, 1]
    assert list(d.ntracks_total) == [3, 0, 1, 1, 1, 1, 1, 1]
    assert list(d.selected_track_index) == [1, -1, 0, 0, 0, 0, -1, 0]
    assert list(d.ntracks_not_good) == [1, 0, 0, 0, 0, 0, 1, 0]
    assert d.loc[0, "sid"] == 4 and d.loc[0, "p_reco"] == 3000
    assert int(d.loc[0, "mc_id"]) == 2 ** 53 + 17
    assert d.loc[2, "mc_missing"] and d.loc[3, "mc_ambiguous"]
    assert d.loc[4, "mc_invalid_id"] and int(d.loc[4, "mc_raw_rank0_id"]) == 0
    assert int(d.loc[5, "mc_id"]) == 99
    assert pd.isna(d.loc[6, "mc_id"]) and not d.loc[6, "mc_missing"]
    assert d.loc[7, "mc_ambiguous"]  # duplicate rank0 records do not certify a unique contributor
    assert d.mc_id.isna().sum() == 6
    assert int(d.loc[0, "event"]) == 2 ** 53 + 11


def test_vectorized_non_native_mc_id_is_missing(monkeypatch):
    tree = synthetic_tree()
    mc = tree["trkmcsim"].data
    tree["trkmcsim"] = Branch(ak.with_field(mc, ak.values_astype(mc.id, np.float64), "id"))
    monkeypatch.setattr(fi.uproot, "open", lambda path: Root({fi.fieldreco.TREE: tree}))
    d = fi.read_arm("float.root")
    assert d.mc_id.isna().all() and d.mc_invalid_id.sum() == 3


def test_run_hashes_highest_cycles_baseline_and_fresh_output(tmp_path, monkeypatch):
    p, s, frames = inputs()
    for arm, filename in fi.FILES.items():
        with uproot.recreate(tmp_path / filename) as root:
            part = p.rename(columns={"sp": "key"})
            # Native TTree leaves, as in the saved injector files.
            root.mktree("dump/particles", {c: v.to_numpy(dtype=np.uint64) for c, v in part.items()})
            root.mktree("dump/sgsteps", {c: v.to_numpy(dtype=np.uint64 if c in fi.TRACK else None) for c, v in s[arm].items()})
        # Second cycle must be selected, not first.
        with uproot.update(tmp_path / filename) as root:
            root.mktree("dump/particles", {c: v.to_numpy(dtype=np.uint64) for c, v in part.items()})
    for arm in fi.ARMS:
        with uproot.recreate(tmp_path / f"{arm}.root") as root:
            root.mktree(fi.fieldreco.TREE, {"evtinfo/event": np.arange(12, dtype=np.uint64)})
        with uproot.update(tmp_path / f"{arm}.root") as root:
            root.mktree(fi.fieldreco.TREE, {"evtinfo/event": np.arange(12, dtype=np.uint64)})
    (tmp_path / "fieldcompare.json").write_text('{"first_pass": {"tracks": 12}}')
    monkeypatch.setattr(fi, "read_arm", lambda path: frames[path.stem].copy())
    args = (tmp_path / "g4.root", tmp_path / "fast.root", tmp_path, tmp_path / "out")
    r = fi.run(*args)
    assert r["events"] == 12
    assert r["inputs"]["g4_all"]["particles"].endswith(";2")
    assert r["inputs"]["g4_ntuple"]["tree"].endswith(";2")
    assert all(v["sha256_before"] == v["sha256_after"] for v in r["inputs"].values())
    assert json.loads((tmp_path / "out/fieldidentity.json").read_text()) == r
    saved = pd.read_csv(tmp_path / "out/events.csv.gz", dtype={k: "uint64" for k in fi.KEY})
    assert len(saved) == 12
    legacy_columns = fi.KEY + ["p_first", "first_plane"] + [f"{a}_{c}" for a in fi.ARMS for c in (
        "ntrk", "nactive", "nhits", "ndof", "chisq", "fitcon", "chi2ndf", "sid", "p_reco", "p_true", "p_first_own")]
    legacy = fi.assemble(p, s, frames)[legacy_columns]
    legacy.to_csv(tmp_path / "baseline.csv.gz", index=False)
    checked = fi.run(*args[:3], tmp_path / "checked", baseline_events=tmp_path / "baseline.csv.gz")
    assert checked["baseline_check"]["unchanged"]
    assert checked["inputs"]["baseline_events"]["sha256_before"] == checked["inputs"]["baseline_events"]["sha256_after"]
    # All particle records, not just identity keys, must agree across the three saved arms.
    with uproot.update(tmp_path / fi.FILES["fast"]) as root:
        changed = p.rename(columns={"sp": "key"}).copy()
        changed.loc[0, "key"] = 99
        root.mktree("dump/particles", {c: v.to_numpy(dtype=np.uint64) for c, v in changed.items()})
    with pytest.raises(AssertionError):
        fi.run(*args[:3], tmp_path / "different-particles")
    assert not (tmp_path / "different-particles").exists()
    with pytest.raises(FileExistsError, match="fresh"):
        fi.run(*args)


def test_changed_input_fails_before_output(tmp_path, monkeypatch):
    # Input hashes are checked after all computation, not just written as a receipt.
    p, s, frames = inputs()
    for arm, filename in fi.FILES.items():
        with uproot.recreate(tmp_path / filename) as root:
            root["dump/particles"] = {c: v.to_numpy(dtype=np.uint64) for c, v in p.rename(columns={"sp": "key"}).items()}
            root["dump/sgsteps"] = {c: v.to_numpy(dtype=np.uint64 if c in fi.TRACK else None) for c, v in s[arm].items()}
    for arm in fi.ARMS:
        with uproot.recreate(tmp_path / f"{arm}.root") as root:
            root[fi.fieldreco.TREE] = {"event": np.arange(12)}
    receipt = tmp_path / "fieldcompare.json"
    receipt.write_text("{}")
    def read(path):
        receipt.write_text(receipt.read_text() + " ")
        return frames[path.stem]
    monkeypatch.setattr(fi, "read_arm", read)
    with pytest.raises(ValueError, match="closed input changed"):
        fi.run(tmp_path / "g4.root", tmp_path / "fast.root", tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_native_unsigned_mc_id_above_signed_range(monkeypatch):
    tree = synthetic_tree()
    mc = tree["trkmcsim"].data
    unsigned = ak.values_astype(mc.id, np.uint64) + np.uint64(2 ** 63)
    tree["trkmcsim"] = Branch(ak.with_field(mc, unsigned, "id"))
    monkeypatch.setattr(fi.uproot, "open", lambda path: Root({fi.fieldreco.TREE: tree}))
    d = fi.read_arm("uint64.root")
    assert int(d.loc[0, "mc_id"]) == 2 ** 63 + 2 ** 53 + 17
    assert int(d.loc[5, "mc_id"]) == 2 ** 63 + 99
    assert not d.loc[0, "mc_invalid_id"]


def test_unavailable_contributors_never_fall_back_to_only_primary(monkeypatch):
    tree = synthetic_tree()
    del tree["trkmcsim"]
    monkeypatch.setattr(fi.uproot, "open", lambda path: Root({fi.fieldreco.TREE: tree}))
    d = fi.read_arm("no-mc.root")
    assert d.mc_id.isna().all() and d.mc_unavailable.all()
    assert d.mc_missing.sum() == 6


def test_stable_own_first_step_ties_and_foreign_step_fail_closed():
    p, s, frames = inputs()
    first = s["fast"].iloc[[0]].copy()
    first["px"] = 2600.0
    # Same time but original table order must select the first inserted record.
    s["fast"] = pd.concat([first, s["fast"]], ignore_index=True)
    d = fi.assemble(p, s, frames)
    assert d.loc[0, "fast_p_own_id"] == 2600.0
    s["g4_all"].loc[0, "sp"] = 99
    with pytest.raises(ValueError, match="outside particle table"):
        fi.assemble(p, s, frames)


def test_missing_and_ambiguous_links_preserve_denominator_and_exclusion_counts():
    p, s, frames = inputs()
    frames["fast"].loc[0, "mc_id"] = pd.NA
    frames["fast"].loc[0, "mc_missing"] = True
    frames["fast"].loc[0, "mc_rank0_count"] = 0
    frames["g4"].loc[1, "mc_id"] = pd.NA
    frames["g4"].loc[1, "mc_ambiguous"] = True
    frames["g4"].loc[1, "mc_rank0_count"] = 2
    d = fi.assemble(p, s, frames)
    assert len(d) == 12 and d.original_proxy_pair.sum() == 12
    assert d.same_id_proxy_pair.sum() == 10
    assert d.loc[0, "excluded_reason"] == "fast_mc_missing"
    assert d.loc[1, "excluded_reason"] == "g4_mc_ambiguous"
    assert pd.isna(d.loc[0, "fast_matched_id"]) and pd.isna(d.loc[1, "g4_matched_id"])
    r = fi.report(d)
    assert r["flag_counts"]["fast_mc_missing"] == 1
    assert r["flag_counts"]["g4_mc_ambiguous"] == 1


def test_baseline_checks_exact_saved_decimal_payload_for_native_float32():
    import io
    p, s, f = inputs()
    for arm in s:
        s[arm]["px"] = np.linspace(2500.123, 2501.123, 12, dtype=np.float32)
        s[arm][["py", "pz"]] = s[arm][["py", "pz"]].astype(np.float32)
    d = fi.assemble(p, s, f)
    cols = fi.KEY + ["p_first", "first_plane"] + [f"{a}_{c}" for a in fi.ARMS for c in (
        "ntrk", "nactive", "nhits", "ndof", "chisq", "fitcon", "chi2ndf", "sid", "p_reco", "p_true", "p_first_own")]
    baseline = pd.read_csv(io.StringIO(d[cols].to_csv(index=False)),
                           dtype={k: 'uint64' for k in fi.KEY}, float_precision='round_trip')
    assert not np.array_equal(d.p_first.to_numpy().astype(np.float64), baseline.p_first.to_numpy())
    assert fi.check_baseline(d, baseline)['unchanged']
    baseline.loc[0, 'p_first'] = np.nextafter(baseline.loc[0, 'p_first'], np.inf)
    with pytest.raises(ValueError, match='changed'):
        fi.check_baseline(d, baseline)
