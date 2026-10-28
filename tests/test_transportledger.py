"""Segment representation tests; no production transport or G4 closure claims."""
import copy

import numpy as np
import pytest

from mu2efastsim import eloss, materials, transport
from trkfastsim import transportledger as ledger


@pytest.fixture
def meta():
    return ledger.provenance(
        geometry_sha256="0" * 64, tracker_commit="1" * 40, core_commit="2" * 40,
        inputs=[{"name": "synthetic-fixture", "sha256": "3" * 64}], frame="synthetic Cartesian",
        media=[ledger.medium(), ledger.medium(materials.ALUMINIUM, cut_MeV=1e9, loss_basis="unrestricted"),
               ledger.medium(materials.COPPER, cut_MeV=0.1, loss_basis="restricted")],
        settings={"mode": "fixture", "secondary_library": "off", "stop_policy": "supplied by fixture"})


def rows():
    return dict(run=np.ones(6, dtype=np.int64), subrun=np.ones(6, dtype=np.int64),
                event=np.ones(6, dtype=np.int64), sp=np.array([5, 5, 5, 5, 6, 7]),
                segment=np.array([0, 1, 2, 3, 0, 0]), medium=np.array([0, 1, -1, 2, 1, 0]),
                kind=np.array(["physical", "physical", "displacement", "physical", "physical", "physical"]),
                terminal=np.array(["none", "none", "none", "exhausted", "none", "exhausted"], dtype="U16"),
                path_mm=np.array([10., 2., 0., 1., 0.01, 0.]),
                ke_in_MeV=np.array([10., 10., 7., 7., 0.02, 0.]),
                ke_out_MeV=np.array([10., 7., 7., 0., 0.001, 0.]),
                sampled_loss_MeV=np.array([0., 3., 0., 8., 0.019, 0.]),
                applied_loss_MeV=np.array([0., 3., 0., 7., 0.019, 0.]),
                deposited_MeV=np.array([0., np.nan, 0., 5., 0.019, 0.]),
                escaping_MeV=np.array([0., np.nan, 0., 2., 0., 0.]),
                dt_ns=np.array([1., np.nan, 0., 2., 0.1, 0.]),
                sample_status=np.array(["known"] * 6),
                deposit_status=np.array(["known", "unknown", "known", "known", "known", "known"]),
                escape_status=np.array(["known", "unmodeled", "known", "known", "known", "known"]),
                dt_status=np.array(["known", "unknown", "known", "known", "known", "known"]),
                entry_mm=np.array([[0, 0, 0], [0, 10, 0], [0, 12, 0], [1, 12, 0], [0, 0, 0], [0, 0, 0.]]),
                exit_mm=np.array([[0, 10, 0], [0, 12, 0], [1, 12, 0], [1, 13, 0], [0, .01, 0], [0, 0, 0.]]))


def test_multimuon_sequential_materials_near_stop_and_stop(meta):
    h = ledger.SegmentHistory(rows(), meta)
    assert len(h) == 6
    np.testing.assert_array_equal(h["sp"], [5, 5, 5, 5, 6, 7])
    assert h["ke_out_MeV"][4] > 0
    assert h["ke_out_MeV"][3] == h["ke_out_MeV"][5] == 0
    a = h.accounting()
    np.testing.assert_allclose(a["primary_residual_MeV"], 0, atol=1e-14)
    np.testing.assert_array_equal(a["partition_complete"], [True, False, True, True, True, True])
    assert np.isnan(a["partition_residual_MeV"][1])
    np.testing.assert_allclose(a["partition_residual_MeV"][[0, 2, 3, 4, 5]], 0)
    assert h["sampled_loss_MeV"][3] > h["applied_loss_MeV"][3]


def test_empty_roundtrip(meta, tmp_path):
    h = ledger.SegmentHistory.empty(meta)
    assert len(h) == 0 and h["entry_mm"].shape == (0, 3)
    assert h.accounting()["partition_complete"].shape == (0,)
    h.save(tmp_path / "empty.npz")
    assert len(ledger.SegmentHistory.load(tmp_path / "empty.npz")) == 0


def test_roundtrip_immutable_no_overwrite(meta, tmp_path):
    c = rows()
    h = ledger.SegmentHistory(c, meta)
    c["path_mm"][0] = 123.
    meta["settings"]["mode"] = "mutated"
    assert h["path_mm"][0] == 10
    assert h.metadata["settings"]["mode"] == "fixture"
    with pytest.raises(ValueError):
        h["path_mm"][0] = 20
    with pytest.raises(ValueError):
        h["path_mm"].flags.writeable = True
    path = tmp_path / "history.npz"
    h.save(path)
    reread = ledger.SegmentHistory.load(path)
    assert reread.metadata == h.metadata
    for field in ledger.FIELDS:
        np.testing.assert_equal(reread[field], h[field])
    with pytest.raises(FileExistsError):
        h.save(path)


def test_sampling_and_capture_do_not_change_core_random_stream(meta):
    a, b = np.random.default_rng(19), np.random.default_rng(19)
    p, length = np.full(256, 300.), np.full(256, 0.1)
    expected = eloss.sample_loss(a, materials.ALUMINIUM, p, length, 1e9)
    sampled = eloss.sample_loss(b, materials.ALUMINIUM, p, length, 1e9)
    after = transport._p_after(p, sampled)
    kinetic = np.hypot(p, eloss.MMU) - eloss.MMU
    kinetic_after = np.hypot(after, eloss.MMU) - eloss.MMU
    n = len(p)
    c = {name: np.repeat(value[1:2], n, axis=0) for name, value in rows().items()}
    c.update(sp=np.arange(n), segment=np.zeros(n, dtype=np.int64), path_mm=length,
             entry_mm=np.zeros((n, 3)), exit_mm=np.column_stack((length, np.zeros((n, 2)))),
             ke_in_MeV=kinetic, ke_out_MeV=kinetic_after, sampled_loss_MeV=sampled,
             applied_loss_MeV=kinetic - kinetic_after)
    state = copy.deepcopy(b.bit_generator.state)
    captured = ledger.capture(c, meta, enabled=True)
    assert b.bit_generator.state == state
    assert ledger.capture(object(), object()) is None
    np.testing.assert_array_equal(captured["sampled_loss_MeV"], expected)
    np.testing.assert_array_equal(a.random(n), b.random(n))
    np.testing.assert_allclose(captured["applied_loss_MeV"], sampled, atol=1e-12)


@pytest.mark.parametrize("field,index,value,message", [
    ("ke_out_MeV", 1, 8., "debit"),
    ("path_mm", 2, 1., "zero physical"),
    ("dt_ns", 2, 0.1, "zero physical"),
    ("dt_ns", 0, 0., "positive physical"),
    ("escaping_MeV", 1, 0., "must be NaN"),
    ("escape_status", 1, "known", "known escaping"),
    ("deposit_status", 1, "missing", "invalid deposit"),
    ("deposited_MeV", 3, 4., "partition"),
    ("medium", 0, 20, "valid medium"),
    ("medium", 2, 0, "no physical medium"),
    ("terminal", 5, "none", "absorbing"),
    ("terminal", 1, "model_stop", "no later segment"),
    ("segment", 3, 2, "contiguous"),
    ("segment", 4, 1, "contiguous"),
    ("sp", 4, 4, "ordered"),
    ("run", 0, -1, "nonnegative"),
    ("sampled_loss_MeV", 0, 1., "known zero"),
    ("sampled_loss_MeV", 3, -1., "nonnegative"),
    ("path_mm", 0, 9., "shorter than"),
    ("path_mm", 5, 1., "stopped lanes"),
    ("path_mm", 0, np.inf, "finite"),
])
def test_bad_rows_are_rejected(meta, field, index, value, message):
    c = rows()
    c[field][index] = value
    with pytest.raises(ValueError, match=message):
        ledger.SegmentHistory(c, meta)


def test_history_continuity_uses_full_identity(meta):
    c = rows()
    c["ke_in_MeV"][1] += 1
    c["ke_out_MeV"][1] += 1
    with pytest.raises(ValueError, match="discontinuous kinetic"):
        ledger.SegmentHistory(c, meta)
    c = rows()
    c["entry_mm"][3, 0] += .1
    c["exit_mm"][3, 0] += .1
    with pytest.raises(ValueError, match="discontinuous positions"):
        ledger.SegmentHistory(c, meta)
    c = rows()
    c["sp"][4:] = 5
    c["event"][4:] = [2, 3]
    ledger.SegmentHistory(c, meta)
    c["event"][4:] = 1
    c["subrun"][4:] = [2, 3]
    ledger.SegmentHistory(c, meta)
    c["subrun"][4:] = 1
    c["run"][4:] = [2, 3]
    ledger.SegmentHistory(c, meta)


def test_zero_length_material_and_model_stop_residual(meta):
    c = rows()
    c["medium"][5] = 1
    c["terminal"][5] = "model_stop"
    c["ke_in_MeV"][5] = c["ke_out_MeV"][5] = .1
    h = ledger.SegmentHistory(c, meta)
    assert h["ke_out_MeV"][-1] == .1
    assert h["deposited_MeV"][-1] == 0


@pytest.mark.parametrize("field,value", [("geometry_sha256", "short"), ("core_commit", "main"),
                                           ("frame", ""), ("units", {}), ("schema", "v9"),
                                           ("materials_cuts_sha256", "a" * 64)])
def test_bad_provenance(meta, field, value):
    meta[field] = value
    with pytest.raises(ValueError):
        ledger.SegmentHistory.empty(meta)


def test_shape_integer_and_schema_validation(meta):
    for field, value in (("entry_mm", np.zeros((6, 2))), ("event", np.ones(6)), ("run", np.ones((6, 1), int))):
        c = rows()
        c[field] = value
        with pytest.raises(ValueError):
            ledger.SegmentHistory(c, meta)
    c = rows()
    del c["dt_status"]
    with pytest.raises(ValueError, match="schema"):
        ledger.SegmentHistory(c, meta)


def test_material_cut_identity(meta):
    meta["media"][1]["cut_MeV"] = 1.
    with pytest.raises(ValueError, match="identity mismatch"):
        ledger.SegmentHistory.empty(meta)
    with pytest.raises(ValueError):
        ledger.medium(cut_MeV=1.)
    with pytest.raises(ValueError):
        ledger.medium(materials.ALUMINIUM, cut_MeV=0., loss_basis="restricted")


@pytest.mark.parametrize("value,status", [("deposited_MeV", "deposit_status"), ("escaping_MeV", "escape_status")])
def test_partial_partition_cannot_exceed_primary_debit(meta, value, status):
    c = rows()
    c[value][1], c[status][1] = 1000., "known"
    with pytest.raises(ValueError, match="subtotal"):
        ledger.SegmentHistory(c, meta)


def test_public_backing_storage_is_immutable(meta):
    h = ledger.SegmentHistory(rows(), meta)
    backing = h["path_mm"]
    while isinstance(backing, np.ndarray):
        with pytest.raises(ValueError):
            backing.flags.writeable = True
        backing = backing.base
    assert isinstance(backing, bytes)
    assert h["path_mm"][0] == 10.


@pytest.mark.parametrize("field,value", [("rho", -1.), ("I_eV", 0.), ("fractions", {"Al": 2.}),
                                           ("sternheimer", [1.]), ("cuts_e_MeV", {"tracker": -1.})])
def test_incomplete_or_invalid_material_description(meta, field, value):
    meta["media"][1]["material"][field] = value
    with pytest.raises(ValueError):
        ledger.provenance(**{k: meta[k] for k in ("geometry_sha256", "tracker_commit", "core_commit",
                                                 "inputs", "media", "frame", "settings")})
    meta["media"][1]["material"] = {"name": "Al"}
    with pytest.raises(ValueError, match="complete core Material"):
        ledger.provenance(**{k: meta[k] for k in ("geometry_sha256", "tracker_commit", "core_commit",
                                                 "inputs", "media", "frame", "settings")})
