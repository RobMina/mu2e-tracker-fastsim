"""Independent nominal-grid construction and diagnostic-only panel inference."""
import numpy as np
import pytest

from trkfastsim.reco import nativegeometry as G


def geometry(panels=4, *, rounded=False, alignment=None, pitch_u=0., layer_u=0.,
             straw_scale=3., translation_scale=1.e4):
    """Build true straw ends directly from nominal coordinates, not wire corrections."""
    rng = np.random.default_rng(49813)
    rotations, _ = np.linalg.qr(rng.normal(size=(panels, 3, 3)))
    rotations[:, :, 2] *= np.linalg.det(rotations)[:, None]
    origins = rng.normal(size=(panels, 1, 3)) * translation_scale
    i = np.arange(96)
    pitch = np.linspace(4.9, 5.2, panels)
    nominal_mid = np.zeros((panels, 96, 3))
    nominal_mid[..., 0] = pitch_u * (i // 2) + layer_u * (i % 2)
    nominal_mid[..., 1] = pitch[:, None] * (i // 2) + 2.1 * (i % 2)
    nominal_mid[..., 2] = 2.7 * (i % 2)
    half_length = rng.uniform(60., 650., size=(panels, 96))
    offsets = rng.normal(size=(panels, 96, 8)) * 0.15
    offsets[..., 4:] *= straw_scale / 0.15
    if alignment is not None:
        offsets = np.asarray(alignment).reshape(panels, 96, 8).copy()
    nominal_cal = nominal_mid.copy()
    nominal_hv = nominal_mid.copy()
    nominal_cal[..., 0] += half_length
    nominal_hv[..., 0] -= half_length
    wire_cal = nominal_cal.copy()
    wire_hv = nominal_hv.copy()
    straw_cal = nominal_cal.copy()
    straw_hv = nominal_hv.copy()
    wire_cal[..., 1:] += offsets[..., :2]
    wire_hv[..., 1:] += offsets[..., 2:4]
    straw_cal[..., 1:] += offsets[..., 4:6]
    straw_hv[..., 1:] += offsets[..., 6:8]
    transform = lambda x: np.einsum("pij,psj->psi", rotations, x) + origins
    wc, wh = transform(wire_cal), transform(wire_hv)
    sc, sh = transform(straw_cal), transform(straw_hv)
    mid = 0.5 * wc + 0.5 * wh
    chord = wc - wh
    length = np.linalg.norm(chord, axis=-1)
    unit = chord / length[..., None]
    if rounded:
        dumped_half = (0.5 * length).astype(np.float32).astype(np.float64)
        wc = mid + dumped_half[..., None] * unit
        wh = mid - dumped_half[..., None] * unit
    flat_mid = mid.reshape(-1, 3)
    flat_unit = unit.reshape(-1, 3)
    cond = {f"mid{c}": flat_mid[:, k].copy() for k, c in enumerate("xyz")}
    cond.update({f"dir{c}": flat_unit[:, k].copy() for k, c in enumerate("xyz")})
    straw_unit = (sc - sh) / np.linalg.norm(sc - sh, axis=-1, keepdims=True)
    return (cond, wc.reshape(-1, 3), wh.reshape(-1, 3), offsets.reshape(-1, 8)), {
        "midpoint": (0.5 * sc + 0.5 * sh).reshape(-1, 3),
        "direction": straw_unit.reshape(-1, 3), "frames": rotations, "pitch": pitch,
    }


def displace(inputs, displacement):
    """Change the midpoint grid while preserving wire direction and endpoint closure."""
    cond, cal, hv, align = inputs
    cal += displacement
    hv += displacement
    for k, c in enumerate("xyz"):
        cond[f"mid{c}"] += displacement[:, k]


@pytest.mark.parametrize("panels,straw_scale", [(1, 0.1), (6, 30.)])
def test_known_rotations_true_straw_axes_and_metrics(panels, straw_scale):
    inputs, true = geometry(panels, straw_scale=straw_scale)
    result = G.straw_axes(*inputs)
    np.testing.assert_allclose(result["panel_frames"], true["frames"], atol=2e-10, rtol=0)
    np.testing.assert_allclose(result["midpoint"], true["midpoint"], atol=2e-9, rtol=0)
    np.testing.assert_allclose(result["direction"], true["direction"], atol=2e-11, rtol=0)
    np.testing.assert_allclose(result["panel_metrics"]["pitch"], true["pitch"], atol=2e-10, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(result["direction"], axis=-1), 1., atol=4e-16)
    assert result["metrics"]["panel_count"] == panels
    assert result["metrics"]["max_panel_residual"] < 1e-8
    assert result["metrics"]["max_rotation_orthogonality"] < 1e-7
    wire_mid = np.stack([inputs[0][f"mid{c}"] for c in "xyz"], axis=-1)
    wire_dir = np.stack([inputs[0][f"dir{c}"] for c in "xyz"], axis=-1)
    assert result["metrics"]["max_midpoint_correction"] == pytest.approx(
        np.linalg.norm(result["midpoint"] - wire_mid, axis=-1).max(), abs=5e-12)
    assert result["metrics"]["max_direction_correction"] == pytest.approx(
        np.linalg.norm(result["direction"] - wire_dir, axis=-1).max(), abs=5e-12)
    assert result["metadata"]["independently_native_exact"] is False
    assert "inferred" in result["metadata"]["frame_status"]
    assert "rounded halfLength" in result["metadata"]["direction_status"]
    a = inputs[3].reshape(panels, 96, 8)
    grid = np.broadcast_to(np.stack((np.ones(96), np.arange(96) // 2, np.arange(96) % 2), -1),
                           (panels, 96, 3))
    design = np.concatenate((grid, (a[..., :2] + a[..., 2:4]) / 2), axis=-1)
    np.testing.assert_allclose(design @ result["panel_coefficients"],
                               wire_mid.reshape(panels, 96, 3), atol=1e-8, rtol=0)


def test_large_wire_offsets_and_large_alignment_means():
    inputs, _ = geometry(4)
    alignment = inputs[3].copy() * 200
    alignment += np.tile([2000., -3000., 2000., -3000., -4000., 5000., -4000., 5000.],
                         (alignment.shape[0], 1))
    inputs, true = geometry(4, alignment=alignment, translation_scale=1e6)
    result = G.straw_axes(*inputs)
    np.testing.assert_allclose(result["panel_frames"], true["frames"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(result["midpoint"], true["midpoint"], atol=5e-8, rtol=0)
    np.testing.assert_allclose(result["direction"], true["direction"], atol=2e-11, rtol=0)
    assert result["metrics"]["max_endpoint_correction"] > 9000


def test_identical_straw_wire_alignment_is_identity():
    inputs, _ = geometry(2)
    inputs[3][:, 4:] = inputs[3][:, :4]
    result = G.straw_axes(*inputs)
    np.testing.assert_array_equal(result["midpoint"],
                                  np.stack([inputs[0][f"mid{c}"] for c in "xyz"], axis=-1))
    np.testing.assert_allclose(result["direction"],
                               np.stack([inputs[0][f"dir{c}"] for c in "xyz"], axis=-1), atol=2e-16)
    assert result["metrics"]["max_endpoint_correction"] == 0
    assert result["metrics"]["max_direction_correction"] == 0


def test_negative_pitch_rejected():
    inputs, true = geometry(1)
    displace(inputs, -2 * true["pitch"][0] * (np.arange(96) // 2)[:, None] * true["frames"][0, :, 1])
    with pytest.raises(G.GeometryValidationError, match="constant pitch"):
        G.straw_axes(*inputs)


def test_rounded_wire_lengths_are_not_claimed_exact():
    inputs, true = geometry(3, rounded=True, straw_scale=12.)
    result = G.straw_axes(*inputs)
    np.testing.assert_allclose(result["midpoint"], true["midpoint"], atol=2e-9, rtol=0)
    error = np.linalg.norm(result["direction"] - true["direction"], axis=-1)
    assert 1e-10 < error.max() < 2e-8
    assert not result["metadata"]["independently_native_exact"]


def test_batched_svd_inputs_unchanged_and_panel_independence(monkeypatch):
    inputs, _ = geometry(5)
    cond, cal, hv, align = inputs
    before = [x.copy() for x in (cal, hv, align, *cond.values())]
    shapes = []
    svd = np.linalg.svd

    def recorded_svd(a, **kwargs):
        shapes.append(a.shape)
        return svd(a, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", recorded_svd)
    together = G.straw_axes(*inputs)
    assert shapes == [(5, 96, 5)]
    for panel in range(5):
        section = slice(panel * 96, (panel + 1) * 96)
        separate = G.straw_axes({k: v[section] for k, v in cond.items()},
                                cal[section], hv[section], align[section])
        np.testing.assert_allclose(together["midpoint"][section], separate["midpoint"], atol=1e-11)
        np.testing.assert_allclose(together["direction"][section], separate["direction"], atol=1e-13)
    for actual, expected in zip((cal, hv, align, *cond.values()), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("which", ["cal", "hv", "align", "mid", "dir"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected(which, bad):
    inputs, _ = geometry(1)
    cond, cal, hv, align = inputs
    target = {"cal": cal, "hv": hv, "align": align, "mid": cond["midx"], "dir": cond["dirx"]}[which]
    target.flat[0] = bad
    with pytest.raises(G.GeometryValidationError, match="non-finite"):
        G.straw_axes(*inputs)


@pytest.mark.parametrize("rows", [0, 95, 97, 191])
def test_partial_empty_panels(rows):
    inputs, _ = geometry(2)
    cond, cal, hv, align = inputs
    with pytest.raises(G.GeometryValidationError, match="complete nonempty panels"):
        G.straw_axes({k: v[:rows] for k, v in cond.items()}, cal[:rows], hv[:rows], align[:rows])


@pytest.mark.parametrize("which", ["cal", "hv", "align", "mid", "dir", "missing", "none"])
def test_malformed_inputs(which):
    inputs, _ = geometry(1)
    cond, cal, hv, align = inputs
    if which == "cal":
        cal = cal[:, :2]
    elif which == "hv":
        hv = hv[:-1]
    elif which == "align":
        align = np.vstack((align, align[:1]))
    elif which == "mid":
        cond["midx"] = cond["midx"][:, None]
    elif which == "dir":
        cond["dirz"] = cond["dirz"][:-1]
    elif which == "missing":
        del cond["midy"]
    else:
        align = None
    with pytest.raises(G.GeometryValidationError):
        G.straw_axes(cond, cal, hv, align)


@pytest.mark.parametrize("kind", ["zero", "constant", "correlated", "pitch_confounded", "ill_conditioned"])
def test_missing_alignment_rank_and_condition_fail(kind):
    inputs, _ = geometry(2)
    a = inputs[3].copy().reshape(2, 96, 8)
    if kind == "zero":
        a[1, :, :4] = 0
    elif kind == "constant":
        a[1, :, :4] = 1
    elif kind == "correlated":
        a[1, :, 1] = a[1, :, 0]
        a[1, :, 3] = a[1, :, 2]
    elif kind == "pitch_confounded":
        a[1, :, 0] = np.arange(96) // 2
        a[1, :, 2] = np.arange(96) // 2
    else:
        a[1, :, 1] = a[1, :, 0] + 1e-10 * a[1, :, 5]
        a[1, :, 3] = a[1, :, 2] + 1e-10 * a[1, :, 7]
    broken, _ = geometry(2, alignment=a)
    with pytest.raises(G.GeometryValidationError, match="(degenerate|condition).+\\[1\\]"):
        G.straw_axes(*broken)


def test_nonrigid_midpoint_grid_identifies_only_failing_panel():
    inputs, true = geometry(3)
    shift = np.zeros_like(inputs[1])
    shift[97] = 0.01 * true["frames"][1, :, 2]
    displace(inputs, shift)
    with pytest.raises(G.GeometryValidationError, match=r"nonrigid.+\[1\]"):
        G.straw_axes(*inputs)


def test_nonorthonormal_frame_not_silently_repaired():
    inputs, true = geometry(1)
    mean_dv = 0.5 * (inputs[3][:, 0] + inputs[3][:, 2])
    displace(inputs, 0.01 * mean_dv[:, None] * true["frames"][0, :, 1])
    with pytest.raises(G.GeometryValidationError, match="not orthonormal"):
        G.straw_axes(*inputs)


@pytest.mark.parametrize("kwargs,match", [({"pitch_u": .01}, "constant pitch"),
                                          ({"layer_u": .01}, "odd/even layer")])
def test_nominal_grid_coefficients_validated(kwargs, match):
    inputs, _ = geometry(1, **kwargs)
    with pytest.raises(G.GeometryValidationError, match=match):
        G.straw_axes(*inputs)


@pytest.mark.parametrize("kind,match", [("zero", "zero or non-finite length"),
                                       ("reversed", "wire orientation"),
                                       ("midpoint", "conditions/endpoint midpoint"),
                                       ("direction", "conditions/endpoint direction")])
def test_wire_closure_and_orientation(kind, match):
    inputs, _ = geometry(1)
    cond, cal, hv, align = inputs
    if kind == "zero":
        cal[0] = hv[0]
    elif kind == "reversed":
        cal, hv = hv.copy(), cal.copy()
        for c in "xyz":
            cond[f"dir{c}"] *= -1
    elif kind == "midpoint":
        cond["midx"][0] += .01
    else:
        cond["dirx"][0] += .01
    with pytest.raises(G.GeometryValidationError, match=match):
        G.straw_axes(cond, cal, hv, align)


@pytest.mark.parametrize("kwargs", [{"midpoint_atol": 0}, {"rotation_atol": -1},
                                    {"pitch_atol": np.nan}, {"direction_atol": np.inf},
                                    {"max_condition": 1}])
def test_invalid_validation_limits(kwargs):
    inputs, _ = geometry(1)
    with pytest.raises(G.GeometryValidationError, match="validation limits"):
        G.straw_axes(*inputs, **kwargs)


def test_thresholds_are_explicit_and_not_adapted_to_residuals():
    inputs, _ = geometry(1)
    shift = np.zeros_like(inputs[1])
    shift[0, 0] = 1e-5
    displace(inputs, shift)
    with pytest.raises(G.GeometryValidationError, match="nonrigid"):
        G.straw_axes(*inputs)
    assert G.straw_axes(*geometry(1)[0])["metadata"]["validation_limits"]["midpoint_atol"] == 1e-8


def test_fitter_geometry_loader_uses_native_float_alignment_and_shuffled_ids(tmp_path):
    import uproot
    from trkfastsim.reco import strawaxis, vr_f1
    original, _ = geometry(1)
    rounded = original[3].astype(np.float32).astype(np.float64)
    inputs, _ = geometry(1, alignment=rounded, rounded=True)
    cond, wc, wh, _ = inputs
    order = np.random.default_rng(4).permutation(96)
    root = tmp_path / "geometry.root"
    fields = {"sid": np.arange(96)[order][None]}
    fields.update({f"wcal{c}": wc[order, k][None] for k, c in enumerate("xyz")})
    fields.update({f"whv{c}": wh[order, k][None] for k, c in enumerate("xyz")})
    with uproot.recreate(root) as f:
        f.mktree("trkdump/conditions", fields)
    table = tmp_path / "alignment.csv"
    np.savetxt(table, np.column_stack((np.arange(96), np.arange(96), original[3])),
               delimiter=",", fmt=["%d", "%d"] + ["%.18e"] * 8)
    expected = G.straw_axes(cond, wc, wh, rounded)
    mid, direction = vr_f1._straw_axes(root, {"straw": cond}, table, "panel")
    np.testing.assert_array_equal(mid, expected["midpoint"])
    np.testing.assert_array_equal(direction, expected["direction"])
    approximate = strawaxis.straw_axes(cond, wc, wh, original[3])
    implicit = vr_f1._straw_axes(root, {"straw": cond}, table)
    explicit = vr_f1._straw_axes(root, {"straw": cond}, table, "approximate")
    for actual, default, reference in zip(explicit, implicit, approximate):
        np.testing.assert_array_equal(actual, default)
        np.testing.assert_array_equal(actual, reference)
    fields["sid"][0, 0] = fields["sid"][0, 1]
    with uproot.recreate(root) as f:
        f.mktree("trkdump/conditions", fields)
    with pytest.raises(ValueError, match="exactly one geometry row"):
        vr_f1._straw_axes(root, {"straw": cond}, table, "panel")
