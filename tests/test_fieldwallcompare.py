"""Saved-only full-denominator wall pilot contracts; no transport or reconstruction jobs."""
import io
import json

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from trkfastsim.validation import fieldwallcompare as fw


def inputs(n=12):
    keys = pd.DataFrame(dict(run=np.full(n, 2**63 + 7, dtype=np.uint64),
                             subrun=np.zeros(n, dtype=np.uint64),
                             event=np.arange(2**53 + 17, 2**53 + 17 + n, dtype=np.uint64)))
    p = keys.assign(sp=np.arange(2**63 + 51, 2**63 + 51 + n, dtype=np.uint64))
    s = p.assign(t=np.float32(0), px=np.linspace(2500.123, 2501.456, n, dtype=np.float32),
                 py=np.float32(1.234), pz=np.float32(2.345), plane=2, panel=1, straw=7)
    frames = {a: keys.assign(ntrk=1, ntracks_total=1, status=1., nactive=30., nhits=35., ndof=25.,
                            chisq=26., chi2ndf=26/25, fitcon=.5, sid=0, p_reco=np.linspace(2450, 2550, n),
                            p_true=np.nan, nstraw=28., active_hit_rows=30.) for a in fw.ARMS}
    steps = {a: s.copy() for a in (*fw.FILES, "candidate")}
    baseline = keys.assign(p_first=fw._first(s).p_first_own.to_numpy(), first_plane=2)
    for a, name in (("g4", "g4"), ("legacy", "fast")):
        baseline[f"{name}_p_first_own"] = fw._first(s).p_first_own.to_numpy()
        for c in ("ntrk", "nactive", "nhits", "ndof", "chisq", "fitcon", "chi2ndf", "sid", "p_reco", "p_true"):
            baseline[f"{name}_{c}"] = frames[a][c].to_numpy()
    baseline = pd.read_csv(io.StringIO(baseline.to_csv(index=False)))
    return p, steps, frames, baseline


def test_full_fixed_denominator_high_uint64_and_missing_ntuple():
    p, s, f, b = inputs()
    f["candidate"] = f["candidate"].iloc[1:].copy()
    f["candidate"].loc[f["candidate"].index[0], "ntrk"] = 0
    d = fw.assemble(p, s, f, b)
    assert d.event.to_numpy().dtype == np.dtype("uint64")
    assert int(d.event.iloc[0]) == 2**53 + 17 and int(d.run.iloc[0]) == 2**63 + 7
    assert len(d) == len(b) == 12 and not d.candidate_ntuple_present.iloc[0]
    r = fw.report(d)
    eff = r["whole"]["comparisons"]["candidate_vs_g4"]["efficiency"]
    assert eff["denominator_events"] == 12
    assert eff["candidate_good"] == 10 and eff["reference_good"] == 12
    assert eff["lost"] == 2 and eff["gained"] == 0
    assert eff["difference"]["mean"] == pytest.approx(-2/12)
    assert eff["difference"]["sem"] > 0
    assert r["ntuple_exposure"]["candidate"]["baseline_missing"] == 1
    assert fw.check_baseline(d, b)["unchanged"]


def test_foreign_and_outside_baseline_are_counted_before_joins():
    p, s, f, b = inputs()
    extra = p.iloc[[0]].copy()
    extra["event"] += np.uint64(100)
    p = pd.concat([p, extra], ignore_index=True)
    for a in f:
        out = f[a].iloc[[0]].copy()
        out["event"] += np.uint64(100)
        foreign = out.copy()
        foreign["event"] += np.uint64(100)
        f[a] = pd.concat([f[a], out, foreign], ignore_index=True)
    d = fw.assemble(p, s, f, b)
    r = fw.report(d)
    assert r["events"] == 12
    ex = r["ntuple_exposure"]["candidate"]
    assert ex["selected_ntuple_rows"] == 14 and ex["good_events"] == 14
    assert ex["outside_baseline_events"] == 2 and ex["outside_baseline_good_events"] == 2
    assert ex["foreign_entry_events"] == 1 and ex["foreign_entry_good_events"] == 1
    assert ex["foreign_entry_keys"][0]["event"] == 2**53 + 217
    ent = r["entry_exposure"]
    assert ent["particles"] == 13 and ent["entry_events"] == 13
    assert ent["entry_events_outside_baseline"] == 1
    assert ent["steps"]["candidate"]["zero_gas_particles"] == 1
    assert ent["steps"]["candidate"]["zero_gas_particles_outside_baseline"] == 1
    assert ent["steps"]["candidate"]["zero_gas_entry_events_outside_baseline"] == 1
    assert ent["reconstruction"]["candidate_vs_g4"]["denominator_events"] == 13
    assert "NaN" not in json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("kind", ["particles", "frames", "baseline", "float_event", "float_sp", "foreign_step", "clock_missing"])
def test_identity_contract_fails_closed(kind):
    p, s, f, b = inputs()
    if kind == "particles":
        p = pd.concat([p, p.iloc[[0]]], ignore_index=True)
    elif kind == "frames":
        f["g4"] = pd.concat([f["g4"], f["g4"].iloc[[0]]], ignore_index=True)
    elif kind == "baseline":
        b = pd.concat([b, b.iloc[[0]]], ignore_index=True)
    elif kind == "float_event":
        f["candidate"]["event"] = f["candidate"].event.astype(float)
    elif kind == "float_sp":
        s["candidate"]["sp"] = s["candidate"].sp.astype(float)
    elif kind == "foreign_step":
        s["candidate"].loc[0, "sp"] = np.uint64(99)
    else:
        s["g4_clock"] = s["g4_clock"].iloc[1:]
    with pytest.raises(ValueError):
        fw.assemble(p, s, f, b)


def test_float32_proxy_original_sort_and_default_parser_contract():
    p, s, f, b = inputs()
    d = fw.assemble(p, s, f, b)
    x = s["candidate"].sort_values(fw.KEY + ["t"]).groupby(fw.KEY, sort=False).first().reset_index()
    expected = np.sqrt(x.px ** 2 + x.py ** 2 + x.pz ** 2)
    np.testing.assert_array_equal(d.candidate_p_first_own, expected)
    double = np.sqrt(x.px.astype(float)**2 + x.py.astype(float)**2 + x.pz.astype(float)**2)
    assert not np.array_equal(expected, double)
    assert not np.array_equal(d.g4_p_first_own.to_numpy().astype(float), b.g4_p_first_own)
    checked = fw.check_baseline(d, b)
    assert checked["unchanged"] and "NOT raw-bit" in checked["comparison_representation"]
    for col in ("g4_p_reco", "fast_p_first_own", "p_first"):
        changed = b.copy()
        changed.loc[0, col] = np.nextafter(changed.loc[0, col], np.inf)
        with pytest.raises(ValueError, match="changed"):
            fw.check_baseline(d, changed)
    late = s["candidate"].iloc[[0]].copy()
    late["t"], late["px"] = np.float32(-1), np.float32(2700)
    s["candidate"] = pd.concat([s["candidate"], late], ignore_index=True)
    assert fw.assemble(p, s, f, b).candidate_p_first_own.iloc[0] > 2699


def test_bootstrap_fixed_bins_pair_cohorts_and_three_way_diagnostic():
    p, s, f, b = inputs(20)
    f["candidate"]["p_reco"] += np.linspace(-2, 2, 20)
    f["candidate"].loc[0, "sid"] = 4
    f["legacy"].loc[1, "p_reco"] = np.nan
    f["candidate"].loc[2, "ntrk"] = 0
    d = fw.assemble(p, s, f, b)
    r = fw.report(d)
    comp = r["whole"]["comparisons"]
    assert comp["candidate_vs_g4"]["momentum_pair_events"] == 18
    assert comp["legacy_vs_g4"]["momentum_pair_events"] == 19
    assert comp["candidate_vs_legacy"]["momentum_pair_events"] == 17
    common = r["whole"]["three_way_common_diagnostic"]
    assert common["events"] == 17 and "not an exclusion" in common["label"]
    assert {v["core"]["events"] for v in common["comparisons"].values()} == {17}
    core = comp["candidate_vs_g4"]["momentum"]["core"]
    mask = fw._mask(d, ("candidate", "g4"))
    expected = fw.cereco.paired_core(fw._relative(d, "candidate")[mask], fw._relative(d, "g4")[mask], draws=500, seed=20261103)
    assert core["paired_error"] == expected["paired_error"]
    assert core["bootstrap_draws"] == 500 and core["bootstrap_seed"] == 20261103
    assert r == fw.report(d)
    assert [(v["reference_first_straw_p_lo"], v["reference_first_straw_p_hi"]) for v in r["bins"]] == [
        (300., 1000.), (1000., 2000.), (2000., 5000.), (5000., None)]
    assert r["bins"][2]["denominator_events"] == 20
    assert r["bins"][0]["comparisons"]["candidate_vs_g4"]["momentum"]["core"]["difference"] == [None, None]


def test_step_rows_distinct_straws_all_particles_zero_and_tail_boundaries():
    p, s, f, b = inputs()
    first = s["candidate"].iloc[[0]].copy()
    different = first.copy()
    different["straw"] += 1
    s["candidate"] = pd.concat([s["candidate"].iloc[:-1], first, different], ignore_index=True)
    summary = fw.step_summary(p, s["candidate"])
    assert len(summary) == 12 and summary.sp.dtype == np.dtype("uint64")
    assert tuple(summary.loc[0, ["step_rows", "distinct_straws"]]) == (3, 2)
    assert tuple(summary.loc[11, ["step_rows", "distinct_straws"]]) == (0, 0)
    d = fw.assemble(p, s, f, b)
    d["candidate_p_first_own"] = 1000.
    d["candidate_p_reco"] = [1010., 1050., 1100., 1101.] + [1000.] * 8
    d["g4_p_first_own"], d["g4_p_reco"] = 1000., 1000.
    r = fw.report(d)["whole"]["comparisons"]["candidate_vs_g4"]
    tails = r["momentum"]["descriptive_tails"]
    assert tails["0.01"]["candidate"]["mean"] == 3/12
    assert tails["0.05"]["candidate"]["mean"] == 2/12
    assert tails["0.1"]["candidate"]["mean"] == 1/12
    assert r["distinct_active_straws_both_good"]["candidate"]["mean"] == 28
    assert r["active_hits_both_good"]["candidate"]["mean"] == 30


def test_particle_exposure_is_not_event_exposure_and_no_proxy_fallback():
    p, s, f, b = inputs()
    extra = p.iloc[[0]].copy()
    extra["sp"] += np.uint64(100)
    p = pd.concat([p, extra], ignore_index=True)
    later = s["candidate"].iloc[[0]].copy()
    later["sp"] += np.uint64(100)
    later["t"], later["px"] = np.float32(10), np.float32(4000)
    s["candidate"] = pd.concat([s["candidate"].iloc[1:], later], ignore_index=True)
    for arm in f:
        f[arm].loc[1, "sid"] = -1
    s["candidate"] = s["candidate"][s["candidate"].event != p.event.iloc[2]]
    d = fw.assemble(p, s, f, b)
    assert d.candidate_p_first_own.iloc[0] > 3999
    assert np.isnan(d.candidate_p_first_own.iloc[2])
    r = fw.report(d)
    assert r["events"] == 12 and r["entry_exposure"]["particles"] == 13
    assert r["entry_exposure"]["entry_events"] == 12
    comp = r["whole"]["comparisons"]["candidate_vs_g4"]
    assert comp["efficiency"]["candidate_good"] == 12
    assert comp["momentum_pair_events"] == 10
    counts = r["entry_exposure"]["steps"]["candidate"]
    assert counts["zero_gas_particles"] == 2 and counts["zero_gas_entry_events"] == 1
    summary = d.attrs["particle_steps"]
    assert summary.candidate_step_rows.iloc[0] == 0 and summary.candidate_step_rows.iloc[-1] == 1


class Branch:
    def __init__(self, data):
        self.data = data

    def array(self, library="ak"):
        return self.data


class Root(dict):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


class Tree(dict):
    num_entries = 3
    object_path = "/EventNtuple/ntuple;9"


def test_vectorized_first_good_earliest_inbounds_distinct_own_hits(monkeypatch):
    tree = Tree()
    for k in fw.KEY:
        tree[f"evtinfo/{k}"] = Branch(ak.Array(np.arange(2**63 + 17, 2**63 + 20, dtype=np.uint64)))
    status = ak.Array([[0, 1, 1], [], [-1]])
    for k in fw.fieldreco.TRK:
        tree[f"trk/trk.{k}"] = Branch(status if k == "status" else ak.ones_like(status) * (3 if k == "nactive" else 20))
    seg = dict(sid=1, time=10., inbounds=True, mom={"fCoordinates": dict(fX=3000., fY=0., fZ=0.)})
    early = dict(seg, sid=4, time=2.)
    outside = dict(seg, sid=0, time=1., inbounds=False)
    invalid = dict(seg, sid=8, time=0.)
    tree["trksegs"] = Branch(ak.Array([[[seg], [seg, early, outside, invalid], [seg]], [], [[seg]]]))
    hit = dict(plane=1, panel=2, straw=3, state=1)
    other = dict(hit, panel=3)
    inactive = dict(hit, straw=9, state=-2)
    tree["trkhits"] = Branch(ak.Array([[[other], [hit, hit, other, inactive], [other]], [], [[hit]]]))
    root = Root({fw.fieldreco.TREE: tree})
    monkeypatch.setattr(fw.uproot, "open", lambda path: root)
    monkeypatch.setattr(ak, "to_list", lambda *a, **k: pytest.fail("per-event conversion"))
    d = fw.read_arm("saved.root")
    assert root.closed and d.ntrk.tolist() == [2, 0, 0]
    assert d.sid.tolist() == [4, -1, -1] and d.p_reco.iloc[0] == 3000
    assert d.nstraw.iloc[0] == 2 and d.active_hit_rows.iloc[0] == 3
    assert d.nstraw.iloc[1:].isna().all() and d.p_true.isna().all()
    assert int(d.event.iloc[0]) == 2**63 + 17
    tree.num_entries = 4
    with pytest.raises(ValueError, match="entry/frame"):
        fw.read_arm("bad-count.root")


def saved_inputs(tmp_path, monkeypatch):
    p, s, f, b = inputs()
    monkeypatch.setattr(fw, "BASELINE_EVENTS", len(b))
    monkeypatch.setattr(fw, "ENTRY_PARTICLES", len(p))
    for arm in (*fw.FILES, "candidate"):
        path = tmp_path / (fw.FILES[arm] if arm in fw.FILES else "candidate.root")
        for mode in (uproot.recreate, uproot.update):
            with mode(path) as root:
                root.mktree("dump/particles", {c: v.to_numpy() for c, v in p.rename(columns={"sp": "key"}).items()})
                root.mktree("dump/sgsteps", {c: v.to_numpy() for c, v in s[arm].items()})
    for arm in fw.ARMS:
        path = tmp_path / f"{arm}_ntuple.root"
        for mode in (uproot.recreate, uproot.update):
            with mode(path) as root:
                root.mktree(fw.fieldreco.TREE, {"evtinfo/event": f[arm].event.to_numpy()})
    (tmp_path / "fieldcompare.json").write_text("{}")
    b.to_csv(tmp_path / "baseline.csv.gz", index=False)
    monkeypatch.setattr(fw, "read_arm", lambda path: f[path.stem.replace("_ntuple", "")].copy())
    args = dict(candidate_table=tmp_path / "candidate.root", legacy_dir=tmp_path,
                candidate_ntuple=tmp_path / "candidate_ntuple.root", legacy_ntuple=tmp_path / "legacy_ntuple.root",
                g4_ntuple=tmp_path / "g4_ntuple.root", baseline_events=tmp_path / "baseline.csv.gz", outdir=tmp_path / "out")
    return args, p, s, f, b


def test_run_closed_hashes_cycles_exact_baseline_exclusive_output(tmp_path, monkeypatch):
    args, p, s, f, b = saved_inputs(tmp_path, monkeypatch)
    r = fw.run(**args)
    assert r["status"] == "OK" and r["baseline_check"]["unchanged"]
    assert all(x["sha256_before"] == x["sha256_after"] for x in r["inputs"].values())
    assert r["inputs"]["candidate"]["particles"].endswith(";2")
    assert r["inputs"]["g4_ntuple"]["tree"].endswith(";2")
    assert r == json.loads((args["outdir"] / "fieldwallcompare.json").read_text())
    events = pd.read_csv(args["outdir"] / "events.csv.gz")
    assert len(events) == 12 and "candidate_ntrk" in events and "legacy_ntrk" in events and "g4_ntrk" in events
    for column in b:
        name = column.replace("fast_", "legacy_", 1) if column.startswith("fast_") else column
        pd.testing.assert_series_equal(events[name], b[column], check_names=False, check_exact=True)
    particle_steps = pd.read_csv(args["outdir"] / "particle_steps.csv.gz")
    assert len(particle_steps) == len(p) and particle_steps.candidate_step_rows.eq(1).all()
    original = (args["outdir"] / "fieldwallcompare.json").read_bytes()
    with pytest.raises(FileExistsError):
        fw.run(**args)
    assert original == (args["outdir"] / "fieldwallcompare.json").read_bytes()


@pytest.mark.parametrize("failure", ["hash", "ordered_particles", "count", "baseline", "fixed_denominator"])
def test_run_fails_closed_with_report(tmp_path, monkeypatch, failure):
    args, p, s, f, b = saved_inputs(tmp_path, monkeypatch)
    if failure == "hash":
        def read(path):
            receipt = tmp_path / "fieldcompare.json"
            receipt.write_text(receipt.read_text() + " ")
            return f[path.stem.replace("_ntuple", "")].copy()
        monkeypatch.setattr(fw, "read_arm", read)
    elif failure == "ordered_particles":
        with uproot.update(args["candidate_table"]) as root:
            root.mktree("dump/particles", {c: v.to_numpy()[::-1] for c, v in p.rename(columns={"sp": "key"}).items()})
    elif failure == "count":
        f["candidate"] = f["candidate"].iloc[:-1]
    elif failure == "baseline":
        b.loc[0, "g4_p_reco"] += 1
        b.to_csv(args["baseline_events"], index=False)
    else:
        monkeypatch.setattr(fw, "BASELINE_EVENTS", 40991)
    with pytest.raises(ValueError):
        fw.run(**args)
    r = json.loads((args["outdir"] / "fieldwallcompare.json").read_text())
    assert r["status"] == "FAILED" and r["error"]
    assert not (args["outdir"] / "events.csv.gz").exists()
    if failure == "hash":
        item = r["inputs"]["fieldcompare"]
        assert item["sha256_before"] != item["sha256_after"]


def test_kinkal_active_straws_include_both_drift_signs_and_null_not_only_right():
    hits=ak.Array([[dict(plane=0,panel=0,straw=i,state=s) for i,s in
                    enumerate([-3,-2,-1,0,1])],[]])
    straw,rows=fw._active_straws(hits,np.array([True,False]))
    assert straw[0]==3 and rows[0]==3
    assert np.isnan(straw[1]) and np.isnan(rows[1])
    right,_=fw._selected_straws(hits,np.array([True,False]),hits.state>0)
    assert right[0]==1


def test_active_straw_dedupe_cross_states_and_events():
    hit=dict(plane=1,panel=2,straw=3,state=-1)
    rows=ak.Array([[hit,dict(hit,state=0),dict(hit,state=1),dict(hit,state=-2)],
                   [hit,dict(hit,straw=4,state=0)]])
    unique,counts=fw._active_straws(rows,np.array([True,True]))
    np.testing.assert_array_equal(unique,[1.,2.])
    np.testing.assert_array_equal(counts,[3.,2.])


def test_reader_active_nactive_guard_and_right_only_columns(monkeypatch):
    tree=Tree()
    tree.num_entries=1
    for key in fw.KEY:
        tree[f'evtinfo/{key}']=Branch(ak.Array(np.array([1430],dtype=np.uint64)))
    for key in fw.fieldreco.TRK:
        tree[f'trk/trk.{key}']=Branch(ak.Array([[1 if key=='status' else 3]]))
    seg=dict(sid=0,time=1.,inbounds=True,mom={'fCoordinates':dict(fX=3000.,fY=0.,fZ=0.)})
    tree['trksegs']=Branch(ak.Array([[[seg]]]))
    hit=dict(plane=1,panel=1,straw=1,state=-1)
    tree['trkhits']=Branch(ak.Array([[[hit,dict(hit,state=0),dict(hit,state=1),dict(hit,state=-2)]]]))
    monkeypatch.setattr(fw.uproot,'open',lambda path:Root({fw.fieldreco.TREE:tree}))
    out=fw.read_arm('saved')
    assert out.nactive.iloc[0]==out.active_hit_rows.iloc[0]==3
    assert out.nstraw.iloc[0]==1
    assert out.state_positive_nstraw.iloc[0]==out.state_positive_hit_rows.iloc[0]==1
    tree['trk/trk.nactive']=Branch(ak.Array([[2]]))
    with pytest.raises(ValueError,match='persisted nactive'):
        fw.read_arm('bad')
