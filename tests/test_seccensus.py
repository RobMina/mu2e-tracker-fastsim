"""seccensus: synthetic breakdowns, collision-free occupancy, primary selection and strict file pairing."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim import geometry as G
from trkfastsim.analyses import seccensus as SC


def _geom():
    n = 36 * 6 * 96
    up = np.repeat(np.arange(36 * 6), 96)
    mid = np.tile(G.DETECTOR_ORIGIN_EXTRACTED, (n, 1))
    return G.StrawGeometry(plane=up // 6, panel=up % 6, straw=np.tile(np.arange(96), 216), mid=mid,
                           dir=np.tile([1.0, 0, 0], (n, 1)), halflen=np.full(n, 500.0), radius=2.5085,
                           panel_R=np.tile(np.eye(3), (216, 1, 1)), panel_t=np.zeros((216, 3)),
                           plane_R=np.tile(np.eye(3), (36, 1, 1)), plane_t=np.zeros((36, 3)))


def _steps():
    # muon in plane 10 (panel 2, straws 40-41); secondaries: 2 steps in muon straw 40, one in plane 10 panel 2 straw 42
    # (same panel within 3), one in plane 10 panel 4 (same plane), one in plane 12 (2 planes), two in plane 20 straw 5
    rows = [(10, 2, 40, 13, 1), (10, 2, 41, 13, 1), (10, 2, 40, 11, 0), (10, 2, 40, 11, 0), (10, 2, 42, 11, 0),
            (10, 4, 7, 11, 0), (12, 0, 3, 11, 0), (20, 1, 5, 11, 0), (20, 1, 5, 22, 0)]
    d = []
    for i, (pl, pa, st, pdg, prim) in enumerate(rows):
        d.append({"run": 1400, "subrun": 1, "event": 1, "sid": int(G.straw_id(pl, pa, st)), "plane": pl, "panel": pa,
                  "straw": st, "upanel": pl * 6 + pa, "eion": 1e-3 * (i + 1), "len": 2.0, "t": float(i),
                  "x0": -100.0, "y0": 1.0, "z0": 0.0, "x1": 100.0, "y1": 1.0, "z1": 0.0, "px": 0.0, "py": -1000.0,
                  "pz": 0.0, "pdg": pdg, "isprim": prim, "ismcprim": prim})
    s = pd.DataFrame(d)
    s["mu"] = (s.isprim == 1) & (s.pdg.abs() == 13)
    return s


def test_breakdowns_on_a_synthetic_event():
    s = SC._classify(_steps(), _geom())
    sec = s[~s.mu].sort_values("t")
    assert sec["loc"].tolist() == [0, 0, 1, 2, 3, 4, 4]
    assert sec.dplmin.tolist() == [-1, -1, 0, 0, 2, 10, 10]
    c, _, tot = SC._counts(s, 0, False)
    pd_ = dict(zip(c["plane_distance"]["bin"], c["plane_distance"]["n"]))
    assert pd_ == {0: 2.0, 2: 1.0, 6: 2.0}                     # 0 planes x2, 2 planes, 10+ planes x2
    m = dict(zip(c["sec_steps_per_straw"]["bin"], c["sec_steps_per_straw"]["n"]))
    assert m == {1: 1.0, 10: 3.0, 11: 1.0}                     # muon straw with 2; others: 3 straws with 1, one with 2
    aw = dict(zip(c["along_wire_by_location"]["bin"], c["along_wire_by_location"]["n"]))
    assert sum(aw.values()) == 7 and set(b // 10 for b in aw) == {0, 1, 2, 3, 4}
    assert tot["sec_steps"] == 7 and tot["mu_steps"] == 2


def test_occ_uses_full_event_and_straw_keys():
    # The old ev*32768+sid loses run bits and aliases a plane-32 straw with
    # plane 0 in the next event. Event/subrun bit fields also used to overlap.
    s = pd.DataFrame([
        (1, 1, 1, 32768, 192, True),
        (1, 1, 1, 32768, 192, False),  # genuine muon-straw secondary: exclude
        (1, 1, 1, 32769, 192, False),
        (1, 1, 1, 32769, 192, False),  # duplicate straw: count once
        (3, 1, 1, 32768, 192, False),
        (1, 1, 2, 0, 0, False),
        (1, 1 + 2**24, 1, 32768, 192, False),
        (1, 1, 1 + 2**24, 32768, 192, False),
        (1 + 2**16, 1, 1, 32768, 192, False),
    ], columns=SC.T.KEY + ["sid", "upanel", "mu"])
    accs = {}
    SC._occ(s, accs)
    assert accs["all"]["N"] == 7
    assert accs["all"]["nev"] == 6
    assert accs["all"]["QN"] == 9
    assert accs["muon_straws"]["N"] == 1
    assert accs["secondary_only"]["N"] == 6
    assert accs["secondary_only"]["nev"] == 6
    assert accs["secondary_only"]["QN"] == 6
    assert accs["secondary_only"]["C"][0] == 1
    assert accs["secondary_only"]["C"][192] == 5
    # File-local factorized IDs may be reused on separate accumulation calls.
    SC._occ(s, accs)
    assert accs["all"]["N"] == 14
    assert accs["all"]["QN"] == 18
    assert accs["secondary_only"]["nev"] == 12


def test_occ_empty_input():
    accs = {}
    SC._occ(_steps().iloc[:0], accs)
    assert set(accs) == {"all", "muon_straws", "secondary_only"}
    assert all(a["N"] == 0 and a["nev"] == 0 for a in accs.values())


def test_selected_keys_any_primary_muon_but_not_combined_particles():
    def steps(event, sp, n, pdg=13, isprim=1, ismcprim=0):
        s = pd.concat([_steps().iloc[[0]]] * n, ignore_index=True)
        return s.assign(event=event, sp=sp, pdg=pdg, isprim=isprim, ismcprim=ismcprim)

    s = pd.concat([
        steps(1, 1, 1, ismcprim=1), steps(1, 2, 12, pdg=-13),
        steps(2, 1, 6), steps(2, 2, 6),
        steps(3, 1, 12, pdg=11, ismcprim=1),
        steps(4, 1, 12, isprim=0, ismcprim=1),
    ], ignore_index=True)
    assert SC._selected_keys(s).to_dict("records") == [{"run": 1400, "subrun": 1, "event": 1}]


def test_momentum_bin_uses_first_primary_muon_for_whole_event():
    s = _steps()
    s.loc[0, ["ismcprim", "py", "t"]] = [0, -100, 0]
    s.loc[1, ["ismcprim", "py", "t"]] = [1, -1000, 1]
    s.loc[2, "t"] = -1  # an earlier secondary must not define the muon bin
    classified = SC._classify(s, _geom())
    assert classified.pb.tolist() == [0] * len(s)
    counts, den, _ = SC._counts(classified, 0, False)
    assert counts["muon_p_first_straw"]["bin"].tolist() == [0]
    assert den["muon_p_first_straw"]["bin"].tolist() == [0]
    s["ismcprim"] = 0  # also works without an MC-primary muon
    s.loc[0, "py"] = -500
    assert SC._classify(s, _geom()).pb.tolist() == [1] * len(s)
    s.loc[1, "t"] = 0  # stable input-order tie break, not MC-primary status
    assert SC._classify(s, _geom()).pb.tolist() == [1] * len(s)


def _subruns(monkeypatch, coverage):
    def read(path, tree, columns):
        assert tree == "subruns" and columns == ["run", "subrun"]
        return pd.DataFrame(coverage[path], columns=columns)
    monkeypatch.setattr(SC.T, "read", read)


def test_pair_files_uses_run_and_subrun_and_keeps_merged_fast_groups(monkeypatch):
    _subruns(monkeypatch, {"f1": [(1, 1), (1, 2)], "f2": [(2, 1)],
                          "r1": [(1, 1)], "r2": [(1, 2)], "r3": [(2, 1)]})
    assert SC._paired_files(["f1", "f2"], ["r3", "r2", "r1"]) == [
        ("f1", ["r1", "r2"], 0), ("f2", ["r3"], 1)]


def test_pair_files_accepts_merged_reference_within_one_fast_group(monkeypatch):
    _subruns(monkeypatch, {"f": [(1, 1), (1, 2)], "r": [(1, 2), (1, 1)]})
    assert SC._paired_files(["f"], ["r"]) == [("f", ["r"], 0)]


@pytest.mark.parametrize("coverage,fast,ref,reason", [
    ({"f": [(1, 1)], "r": [(2, 1)]}, ["f"], ["r"], "unequal"),
    ({"f": [(1, 1), (1, 2)], "r": [(1, 1)]}, ["f"], ["r"], "unequal"),
    ({"f": [(1, 1)], "r": [(1, 1), (1, 2)]}, ["f"], ["r"], "unequal"),
    ({"f1": [(1, 1)], "f2": [(1, 1)], "r": [(1, 1)]}, ["f1", "f2"], ["r"], "overlapping"),
    ({"f": [(1, 1)], "r1": [(1, 1)], "r2": [(1, 1)]}, ["f"], ["r1", "r2"], "overlapping"),
    ({"f": [(1, 1)], "r": [(1, 1)]}, ["f", "f"], ["r"], "overlapping"),
    ({"f1": [(1, 1)], "f2": [(1, 2)], "r": [(1, 1), (1, 2)]}, ["f1", "f2"], ["r"], "straddles"),
    ({"f": [], "r": [(1, 1)]}, ["f"], ["r"], "missing"),
    ({"f": [(1, 1)], "r": [(1, None)]}, ["f"], ["r"], "missing"),
    ({}, [], [], "both arms"),
])
def test_pair_files_rejects_ambiguous_or_incomplete_coverage(monkeypatch, coverage, fast, ref, reason):
    _subruns(monkeypatch, coverage)
    with pytest.raises(ValueError, match=reason):
        SC._paired_files(fast, ref)


@pytest.mark.parametrize("bins", [["e-", "mu", "missing"], [2, 0, 99]])
def test_by_group_sums_duplicates_and_reindexes(bins):
    df = pd.DataFrame({"bin": [bins[0], bins[0], bins[1], "ignored"],
                       "group": [2, 2, SC.NG - 1, 0], "n": [1., 3., 5., 100.]})
    expected = np.zeros((3, SC.NG))
    expected[0, 2] = 4
    expected[1, -1] = 5
    np.testing.assert_array_equal(SC._by_group(df, "n", bins), expected)
    np.testing.assert_array_equal(SC._by_group(None, "n", bins), np.zeros_like(expected))
    np.testing.assert_array_equal(SC._by_group(df.iloc[:0], "n", bins), np.zeros_like(expected))
    assert SC._by_group(df, "n", []).shape == (0, SC.NG)


def test_pair_files_rejects_reference_reuse_even_when_group_modulo_matches(monkeypatch):
    fast = [f"f{i}" for i in range(SC.NG + 1)]
    ref = ["merged"] + [f"r{i}" for i in range(1, SC.NG)]
    coverage = {f: [(1, i)] for i, f in enumerate(fast)}
    coverage.update({f"r{i}": [(1, i)] for i in range(1, SC.NG)})
    coverage["merged"] = [(1, 0), (1, SC.NG)]
    _subruns(monkeypatch, coverage)
    with pytest.raises(ValueError, match="straddles"):
        SC._paired_files(fast, ref)


@pytest.mark.parametrize("ngroups", [1, 2])
def test_run_reports_inputs_selection_and_paired_intersection(monkeypatch, tmp_path, capsys, ngroups):
    import json

    fast, ref = ["f1", "f2"][:ngroups], ["r1", "r2"][:ngroups]
    _subruns(monkeypatch, {"f1": [(1, 1)], "r1": [(1, 1)], "f2": [(2, 1)], "r2": [(2, 1)]})
    monkeypatch.setattr(SC.T, "expand", lambda inputs: list(inputs))
    monkeypatch.setattr(SC.T, "livetime", lambda files: float(len(files)))
    monkeypatch.setattr(SC.geometry, "from_gdml", lambda _: _geom())

    def load(path, g4):
        # Both arms select one shared and one disjoint event per file, with no ismcprim muon.
        s = pd.concat([_steps(), pd.concat([_steps().iloc[[0]]] * 10)], ignore_index=True)
        s = s.assign(run=int(path[-1]), sp=np.where(s.mu, 1, 2), ismcprim=0,
                     ccode=31, spx=0., spy=0., spz=0.)
        return pd.concat([s, s.assign(event=2 if g4 else 3)], ignore_index=True)

    monkeypatch.setattr(SC, "_load", load)
    # The occupancy estimator has separate tests; avoid low-statistics diagnostic warnings here.
    monkeypatch.setattr(SC.O, "compare", lambda *args, **kwargs: {
        "chi2ndf_poisson": 0., "chi2ndf_clustered": 0., "overdispersion_median": 0.,
        "max_abs_pull": 0., "panels_beyond_5sigma": 0})
    SC.run(fast, str(tmp_path), compare=ref, gdml="unused", paired=True)
    result = json.loads((tmp_path / "seccensus.json").read_text())
    assert result["input_files"] == {"fast": fast, "g4": ref}
    assert result["file_groups"] == [{"fast": "f1", "g4": ["r1"], "group": 0},
                                     {"fast": "f2", "g4": ["r2"], "group": 1}][:ngroups]
    assert "any one primary muon" in result["selection"]["event_filter"]
    assert "earliest primary-muon straw step" in result["selection"]["momentum_bin"]
    assert "intersection" in result["selection"]["event_keys"]
    assert "(run, subrun)" in result["selection"]["file_pairing"]
    assert result["selected_events"] == {"fast": 2 * ngroups, "g4": 2 * ngroups, "both": ngroups}
    assert result["totals"]["fast"]["events"] == result["totals"]["g4"]["events"] == ngroups
    if ngroups == 1:
        assert result["eion_per_straw"]["g4_half_vs_half_band"] is None
        assert not result["eion_per_straw"]["event_clustered_g4_floor"]["available"]
    output = capsys.readouterr().out
    assert '"input_files"' in output and '"selection"' in output


def test_run_rejects_bad_pairing_before_loading_geometry(monkeypatch, tmp_path):
    _subruns(monkeypatch, {"f": [(1, 1)], "r": [(2, 1)]})
    monkeypatch.setattr(SC.T, "expand", lambda inputs: list(inputs))
    monkeypatch.setattr(SC.geometry, "from_gdml", lambda _: pytest.fail("must validate coverage first"))
    with pytest.raises(ValueError, match="unequal"):
        SC.run(["f"], str(tmp_path), compare=["r"], paired=True)


def test_jackknife_does_not_count_empty_file_groups():
    f, r = np.zeros(20), np.zeros(20)
    f[0], r[0] = 12., 10.
    value, error = SC._jack(f, r, r, r)
    assert np.isclose(value, .2)
    assert np.isnan(error)
    f[1], r[1] = 8., 11.
    a = SC._jack(f, r, r, r)
    b = SC._jack(f[:2], r[:2], r[:2], r[:2])
    assert np.allclose(a, b)
