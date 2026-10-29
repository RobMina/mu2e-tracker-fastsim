"""Synthetic initial CAD contracts; no fits, native capture, builds or remote jobs."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import f1_initial_cad_audit as D


CAD = [400, 800000, 50, 50, "TOT:LongVal", "Null:Inactive", "", 0]
MAXVAR = 800000.**2
GEOMETRY = "Offline/Mu2eG4/geom/geom_common_extracted_v04.txt"
FIELD = "Offline/Mu2eG4/geom/bfgeom_no_field.txt"


def test_cad_native_strict_boundaries_all_cuts_and_unusable():
    # Include both signs, exact boundaries and their immediate float64 neighbours.
    docas = [-401., -400., np.nextafter(-400., 0.), 0.,
             np.nextafter(400., 0.), 400., 401.]
    variances = [-1., 0., np.nextafter(0., 1.), np.nextafter(MAXVAR, 0.),
                 MAXVAR, np.nextafter(MAXVAR, np.inf)]
    radii = [-1., 0., np.nextafter(50., 0.), 50., np.nextafter(50., np.inf)]
    operands = list(product(docas, variances, radii, [False, True]))
    doca, variance, radius, usable = map(np.array, zip(*operands))
    result = D.cad_initial(doca, variance, radius, usable, CAD)
    # Independent scalar CADSHU branch oracle: minr == maxr rules out LR states.
    eligible = np.array([r < 50. and abs(d) < 400. and 0. < v < MAXVAR
                         for d, v, r, _ in operands])
    expected = np.array([-3 if not u else 0 if good else -2
                         for (*_, u), good in zip(operands, eligible)])
    np.testing.assert_array_equal(result["state"], expected)
    np.testing.assert_array_equal(result["active"], expected == 0)
    np.testing.assert_array_equal(result["updated"], usable)
    np.testing.assert_array_equal(result["algorithm"], np.where(usable, 1, -1))
    flags = result["constrain_tot"].astype(int) + 512*result["constrain_long"].astype(int)
    np.testing.assert_array_equal(flags, np.where(usable, 513, 0))
    assert not result["constrain_absdrift"].any()
    assert not result["frozen"].any()
    for name, expected_cut, margin in (
        ("drift_upper", radius < 50., 50.-radius),
        ("doca_upper", np.abs(doca) < 400., 400.-np.abs(doca)),
        ("variance_positive", variance > 0., variance),
        ("variance_upper", variance < MAXVAR, MAXVAR-variance),
    ):
        np.testing.assert_array_equal(result["cuts"][name], expected_cut)
        np.testing.assert_array_equal(result["margins"][name], margin)


@pytest.mark.parametrize("index,value", [(0, 401), (1, 800001), (2, 49), (3, 51),
                                          (4, "TOT"), (5, "Null"), (6, "Null"), (7, 1)])
def test_cad_only_declared_initial_policy(index, value):
    config = CAD.copy()
    config[index] = value
    with pytest.raises(ValueError, match="unsupported initial CAD config"):
        D.cad_initial([0.], [1.], [1.], [True], config)


@pytest.mark.parametrize("operand", range(4))
def test_cad_shape_and_mask_guards(operand):
    args = [np.array([0., 1.]), np.ones(2), np.ones(2), np.ones(2, dtype=bool)]
    malformed = list(args)
    malformed[operand] = malformed[operand][:1]
    with pytest.raises(ValueError, match="CAD shapes differ"):
        D.cad_initial(*malformed, CAD)
    args[operand] = np.ma.array(args[operand], mask=[False, True])
    with pytest.raises(ValueError, match="missing"):
        D.cad_initial(*args, CAD)


@pytest.mark.parametrize("operand", range(3))
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_cad_rejects_nonfinite_even_when_unusable(operand, value):
    args = [[0.], [1.], [1.], [False]]
    args[operand] = [value]
    with pytest.raises(ValueError, match="nonfinite CAD operand"):
        D.cad_initial(*args, CAD)


@pytest.mark.parametrize("usable", [[-1], [2], [1.], [np.nan], ["True"]])
def test_cad_usable_is_boolean_or_binary_integer(usable):
    with pytest.raises(ValueError, match="invalid CA usable flag"):
        D.cad_initial([0.], [1.], [1.], usable, CAD)


@pytest.fixture
def gaussian_operands():
    rng = np.random.default_rng(233771)
    factor = rng.normal(size=(3, 6, 6))
    cov = factor@factor.swapaxes(-1, -2) + np.eye(6)*2.
    q = np.array([[2., .2, -30., 1.1, 100., 105.],
                  [-3., -.4, 50., 2., 80., 90.], [0., .7, 1., .9, 60., 110.]])
    return q, cov


def ordered_matvec(matrix, vector):
    out = matrix[..., :, 0]*vector[..., 0, None]
    for j in range(1, 6):
        out = out + matrix[..., :, j]*vector[..., j, None]
    return out


@pytest.mark.parametrize("arm", D.ARMS)
def test_unbiased_zero_weight_parameters_covariance_and_stats(gaussian_operands, monkeypatch, arm):
    q, cov = gaussian_operands
    saved_q, saved_cov = q.copy(), cov.copy()
    monkeypatch.setattr(D.ALG, "INVERSE_STATS", dict(n_cholesky_success=17, n_cholesky_fallback=9))
    before = D.ALG.INVERSE_STATS.copy()
    outq, outcov, stats = D.unbiased_roundtrip(q, cov, arm)
    np.testing.assert_allclose(outq, q, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(outcov, cov, rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(q, saved_q)
    np.testing.assert_array_equal(cov, saved_cov)
    assert not np.shares_memory(outq, q) and not np.shares_memory(outcov, cov)
    assert D.ALG.INVERSE_STATS == before
    if arm == "ideal_zero_weight_control":
        np.testing.assert_array_equal(outq, q)
        np.testing.assert_array_equal(outcov, cov)
        assert stats["n_inversions"] == 0
        assert "no native roundtrip" in stats["scope"]
    else:
        assert stats["n_inversions"] == 2*len(q)
        assert "not native exact intermediate" in stats["scope"]
        assert stats["n_cholesky_success"] == (2*len(q) if arm == "chol_factor_roundtrip" else 0)
        assert stats["n_cholesky_fallback"] == 0
        if arm == "chol_factor_roundtrip":
            np.testing.assert_array_equal(outcov, outcov.swapaxes(-1, -2))
        else:
            weight = np.linalg.inv(cov)
            oracle_cov = np.linalg.inv(weight)
            np.testing.assert_array_equal(outcov, oracle_cov)
            np.testing.assert_array_equal(outq, ordered_matvec(oracle_cov, ordered_matvec(weight, q)))


def test_unbiased_chol_passes_unmodified_zero_weight_operands(gaussian_operands, monkeypatch):
    q, cov = gaussian_operands
    real_inverse = D.ALG._inverse_chol_factor
    inverse_inputs, inverse_outputs = [], []
    def inverse(matrix):
        inverse_inputs.append(matrix.copy())
        result = real_inverse(matrix)
        inverse_outputs.append(result.copy())
        return result
    monkeypatch.setattr(D.ALG, "_inverse_chol_factor", inverse)
    outq, outcov, _ = D.unbiased_roundtrip(q, cov, "chol_factor_roundtrip")
    assert len(inverse_inputs) == len(inverse_outputs) == 2
    np.testing.assert_array_equal(inverse_inputs[0], cov)
    # No cached measurement weight, symmetrizing average, jitter or other correction.
    np.testing.assert_array_equal(inverse_inputs[1], inverse_outputs[0])
    np.testing.assert_array_equal(outcov, inverse_outputs[1])
    np.testing.assert_array_equal(outq, ordered_matvec(outcov, ordered_matvec(inverse_outputs[0], q)))
    for result in inverse_outputs:
        lower_mirror = np.tril(result) + np.tril(result, -1).swapaxes(-1, -2)
        np.testing.assert_array_equal(result, lower_mirror)


def test_unbiased_ideal_control_does_not_invert_or_regularize(gaussian_operands, monkeypatch):
    q, cov = gaussian_operands
    cov[:] = 0.
    def forbidden(*args, **kwargs):
        pytest.fail("ideal zero-weight identity must not invert or regularize")
    monkeypatch.setattr(D.ALG, "_inverse_chol_factor", forbidden)
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    outq, outcov, stats = D.unbiased_roundtrip(q, cov, "ideal_zero_weight_control")
    np.testing.assert_array_equal(outq, q)
    np.testing.assert_array_equal(outcov, cov)
    assert stats["n_inversions"] == 0


@pytest.mark.parametrize("arm", D.ARMS[:2])
def test_unbiased_singular_lane_fails_whole_batch_and_restores_stats(gaussian_operands, monkeypatch, arm):
    q, cov = gaussian_operands
    cov[1] = 0.
    saved = cov.copy()
    monkeypatch.setattr(D.ALG, "INVERSE_STATS", dict(n_cholesky_success=11, n_cholesky_fallback=7))
    before = D.ALG.INVERSE_STATS.copy()
    with pytest.raises(ValueError, match="singular.*no regularization"):
        D.unbiased_roundtrip(q, cov, arm)
    np.testing.assert_array_equal(cov, saved)
    assert D.ALG.INVERSE_STATS == before


def test_unbiased_cholesky_fallback_preserves_lower_storage_and_counts(gaussian_operands, monkeypatch):
    q, cov = gaussian_operands
    # Invertible indefinite lane takes unregularized LU, not jitter or eigenvalue repair.
    cov[1] = np.diag([-2., 3., 4., 5., 6., 7.])
    cov[1, 0, 1] = cov[1, 1, 0] = .25
    monkeypatch.setattr(D.ALG, "INVERSE_STATS", dict(n_cholesky_success=19, n_cholesky_fallback=4))
    before = D.ALG.INVERSE_STATS.copy()
    outq, outcov, stats = D.unbiased_roundtrip(q, cov, "chol_factor_roundtrip")
    np.testing.assert_allclose(outq, q, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(outcov, cov, atol=2e-13, rtol=2e-13)
    np.testing.assert_array_equal(outcov, outcov.swapaxes(-1, -2))
    assert outcov[1, 0, 0] < 0
    assert stats["n_cholesky_success"] == 4 and stats["n_cholesky_fallback"] == 2
    assert D.ALG.INVERSE_STATS == before


def test_unbiased_restores_stats_if_second_inversion_raises(gaussian_operands, monkeypatch):
    real_inverse = D.ALG._inverse_chol_factor
    before = D.ALG.INVERSE_STATS.copy()
    calls = []
    def inverse(matrix):
        calls.append(matrix.copy())
        result = real_inverse(matrix)
        if len(calls) == 2:
            raise RuntimeError("synthetic second inverse failure")
        return result
    monkeypatch.setattr(D.ALG, "_inverse_chol_factor", inverse)
    with pytest.raises(RuntimeError, match="second inverse failure"):
        D.unbiased_roundtrip(*gaussian_operands, "chol_factor_roundtrip")
    assert len(calls) == 2 and D.ALG.INVERSE_STATS == before


@pytest.mark.parametrize("arm", D.ARMS)
@pytest.mark.parametrize("mutation,message", [
    ("qshape", "unbiased operand shape"), ("cshape", "unbiased operand shape"),
    ("qnan", "invalid seed parameters"), ("zero_momentum", "invalid seed parameters"),
    ("negative_momentum", "invalid seed parameters"), ("cinf", "nonfinite covariance"),
    ("asymmetric", "nonsymmetric covariance"), ("qmask", "missing"), ("cmask", "missing"),
])
def test_unbiased_operand_guards(gaussian_operands, arm, mutation, message):
    q, cov = gaussian_operands
    if mutation == "qshape":
        q = q[0]
    elif mutation == "cshape":
        cov = cov[:1]
    elif mutation == "qnan":
        q[1, 0] = np.nan
    elif mutation == "zero_momentum":
        q[1, -1] = 0.
    elif mutation == "negative_momentum":
        q[1, -1] = -1.
    elif mutation == "cinf":
        cov[1, 0, 0] = np.inf
    elif mutation == "asymmetric":
        cov[1, 0, 1] = np.nextafter(cov[1, 0, 1], np.inf)
    elif mutation == "qmask":
        q = np.ma.array(q, mask=np.zeros(q.shape, dtype=bool))
        q.mask[1, 0] = True
    else:
        cov = np.ma.array(cov, mask=np.zeros(cov.shape, dtype=bool))
        cov.mask[1, 0, 0] = True
    with pytest.raises(ValueError, match=message):
        D.unbiased_roundtrip(q, cov, arm)


def test_unbiased_rejects_undeclared_arm(gaussian_operands):
    with pytest.raises(ValueError, match="undeclared unbiased arithmetic arm"):
        D.unbiased_roundtrip(*gaussian_operands, "regularized")


@pytest.fixture
def covariance_tables():
    hits, pieces, seeds = [], [], []
    # Repeated target hits, shuffled tables and collisions in every partial identity.
    identities = [(7, 2, 9, 0), (7, 2, 9, 1), (7, 3, 9, 0), (8, 2, 9, 0), (7, 2, 10, 0)]
    for i, values in enumerate(identities):
        identity = dict(zip(D.KEYS, values))
        cov = np.eye(6)*(i+1.)
        cov[0, 1] = cov[1, 0] = .125*(i+1)
        payload = dict(covariance=cov.ravel().tolist(), bnom=[0., 0., .001])
        seeds.append(dict(identity, **payload))
        pieces.append(dict(identity, iteration_ordinal=0, owner=0, piece_index=0, **payload))
        # Poisoned candidate/later rows must never provide initial old covariance.
        for iteration, owner in ((0, 1), (1, 0)):
            pieces.append(dict(identity, iteration_ordinal=iteration, owner=owner, piece_index=0,
                               covariance=(cov*10).ravel().tolist(), bnom=[1., 0., 0.]))
        for effect in (2, 5):
            hits.append(dict(identity, iteration_ordinal=0, effect_index=effect,
                             reference_covariance=cov.ravel().tolist(), reference_bnom=[0., 0., .001]))
    return ak.Array(hits)[::-1], ak.Array(pieces)[::-1], ak.Array(seeds)[::-1]


def replace_record(table, row, field, value):
    records = ak.to_list(table)
    records[row][field] = value
    return ak.Array(records)


def old_row(pieces):
    return int(np.flatnonzero((D.column(pieces, "iteration_ordinal") == 0)
                              & (D.column(pieces, "owner") == 0))[0])


def test_reference_covariance_full_key_join_repeated_targets_and_initial_old_only(covariance_tables):
    hits, pieces, seeds = covariance_tables
    actual = D.reference_covariance(*covariance_tables)
    expected = np.asarray(ak.to_list(hits.reference_covariance)).reshape(-1, 6, 6)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(D.reference_covariance(hits, pieces[::-1], seeds[::-1]), expected)


@pytest.mark.parametrize("source", ["effects", "old", "seed"])
@pytest.mark.parametrize("field", ["covariance", "bnom"])
def test_reference_covariance_exact_source_consistency_and_seed_bnom(covariance_tables, source, field):
    tables = list(covariance_tables)
    index = {"effects": 0, "old": 1, "seed": 2}[source]
    row = old_row(tables[index]) if source == "old" else 0
    name = ("reference_" if source == "effects" else "")+field
    value = ak.to_list(tables[index][name][row])
    value[0] = np.nextafter(value[0], np.inf)
    tables[index] = replace_record(tables[index], row, name, value)
    with pytest.raises(ValueError, match="seed/old/reference covariance differs" if field == "covariance"
                       else "captured nominal field differs from seed convention"):
        D.reference_covariance(*tables)


@pytest.mark.parametrize("source", ["effects", "old", "seed"])
@pytest.mark.parametrize("field", ["covariance", "bnom"])
def test_reference_covariance_rejects_masked_source_lane(covariance_tables, source, field):
    tables = list(covariance_tables)
    index = {"effects": 0, "old": 1, "seed": 2}[source]
    row = old_row(tables[index]) if source == "old" else 0
    name = ("reference_" if source == "effects" else "")+field
    values = ak.to_list(tables[index][name][row])
    values[0] = None
    tables[index] = replace_record(tables[index], row, name, values)
    with pytest.raises(ValueError, match="missing|mask"):
        D.reference_covariance(*tables)


@pytest.mark.parametrize("source", ["effects", "old", "seed"])
def test_reference_covariance_rejects_nonfinite_nominal_field(covariance_tables, source):
    tables = list(covariance_tables)
    index = {"effects": 0, "old": 1, "seed": 2}[source]
    row = old_row(tables[index]) if source == "old" else 0
    name = "reference_bnom" if source == "effects" else "bnom"
    tables[index] = replace_record(tables[index], row, name, [0., np.nan, 0.])
    with pytest.raises(ValueError, match="captured nominal field differs from seed convention"):
        D.reference_covariance(*tables)


@pytest.mark.parametrize("source", ["old", "seed"])
@pytest.mark.parametrize("failure", ["missing", "duplicate"])
def test_reference_covariance_requires_unique_complete_sources(covariance_tables, source, failure):
    tables = list(covariance_tables)
    index = 1 if source == "old" else 2
    row = old_row(tables[index]) if source == "old" else 0
    if failure == "duplicate":
        tables[index] = ak.concatenate([tables[index], tables[index][row:row+1]])
    else:
        tables[index] = tables[index][np.arange(len(tables[index])) != row]
    with pytest.raises(ValueError, match="duplicate|missing"):
        D.reference_covariance(*tables)


@pytest.mark.parametrize("mutation,message", [("nonfinite", "nonfinite covariance"),
                                               ("asymmetric", "nonsymmetric reference covariance")])
def test_reference_covariance_rejects_invalid_effect_covariance(covariance_tables, mutation, message):
    hits, pieces, seeds = covariance_tables
    value = ak.to_list(hits.reference_covariance[0])
    value[1] = np.nan if mutation == "nonfinite" else np.nextafter(value[1], np.inf)
    with pytest.raises(ValueError, match=message):
        D.reference_covariance(replace_record(hits, 0, "reference_covariance", value), pieces, seeds)


@pytest.fixture
def geometry_files(tmp_path):
    area = tmp_path/"native"
    paths = [area/GEOMETRY, area/"geometry/tracker.txt", area/"geometry/radii.txt", area/FIELD]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    paths[0].write_text('/* #include "missing.txt" */\n#include "geometry/tracker.txt"\n')
    paths[1].write_text('// double tracker.wireRadius = 99;\n#include "geometry/radii.txt"\n')
    paths[2].write_text('double tracker.wireRadius = 1.25e-2;\ndouble tracker.strawOuterRadius = 2.5;\n')
    paths[3].write_text('vector<string> bfield.innerMaps = {};\nvector<string> bfield.outerMaps = {};\n'
                        'double bfield.scaleFactor = 0; // zero field required\n')
    fcl = tmp_path/"expanded.fcl"
    fcl.write_text(f'GeometryService : {{\n  inputFile : "{GEOMETRY}"\n  bFieldFile : "{FIELD}"\n}}\n'
                   'StrawHitClusterLevel : "none"\nStrawHitClusterLevel : "none"\n')
    return area, fcl, paths


def test_source_geometry_chained_includes_comments_zero_field_and_watch(geometry_files):
    area, fcl, paths = geometry_files
    watched = []
    actual = D.source_geometry(area, fcl, lambda path: watched.append(Path(path)))
    assert watched == paths
    assert actual["radii"] == dict(wireRadius=.0125, strawOuterRadius=2.5)
    assert actual["geometry_include_chain"] == list(map(str, paths[:3]))
    assert actual["bfield_path"] == str(paths[3]) and actual["zero_field"] is True
    assert "B=0" in actual["drift_angle_scope"]


@pytest.mark.parametrize("failure", ["geometry_missing", "geometry_duplicate", "geometry_wrong", "field_missing",
                                     "field_duplicate", "field_wrong", "cluster_missing", "cluster_wrong", "cluster_mixed"])
def test_source_geometry_requires_declared_geometry_zero_field_and_no_clusters(geometry_files, failure):
    area, fcl, _ = geometry_files
    text = fcl.read_text()
    if failure.endswith("missing"):
        prefix = {"geometry_missing": "inputFile", "field_missing": "bFieldFile",
                  "cluster_missing": "StrawHitClusterLevel"}[failure]
        text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(prefix))
    elif failure == "geometry_duplicate":
        text = text.replace('}\n', f'  inputFile : "{GEOMETRY}"\n}}\n', 1)
    elif failure == "field_duplicate":
        text = text.replace('}\n', f'  bFieldFile : "{FIELD}"\n}}\n', 1)
    elif failure == "geometry_wrong":
        text = text.replace(GEOMETRY, "other.txt")
    elif failure == "field_wrong":
        text = text.replace(FIELD, "field_nonzero.txt")
    elif failure == "cluster_wrong":
        text = text.replace('"none"', '"panel"')
    else:
        text += 'StrawHitClusterLevel : "panel"\n'
    fcl.write_text(text)
    with pytest.raises(ValueError, match="unsupported geometry/field|clustering unsupported"):
        D.source_geometry(area, fcl, lambda path: None)


@pytest.mark.parametrize("failure", ["missing", "nonliteral", "zero_wire", "negative_wire", "equal_radii", "reversed", "infinite"])
def test_source_geometry_radius_guards(geometry_files, failure):
    area, fcl, paths = geometry_files
    declarations = {
        "missing": "double tracker.wireRadius = .01;",
        "nonliteral": "double tracker.wireRadius = 1./80.; double tracker.strawOuterRadius = 2.5;",
        "zero_wire": "double tracker.wireRadius = 0; double tracker.strawOuterRadius = 2.5;",
        "negative_wire": "double tracker.wireRadius = -.01; double tracker.strawOuterRadius = 2.5;",
        "equal_radii": "double tracker.wireRadius = 2.5; double tracker.strawOuterRadius = 2.5;",
        "reversed": "double tracker.wireRadius = 3; double tracker.strawOuterRadius = 2.5;",
        "infinite": "double tracker.wireRadius = .01; double tracker.strawOuterRadius = 1e999;",
    }
    paths[2].write_text(declarations[failure])
    with pytest.raises(ValueError, match="missing tracker radii|nonliteral tracker radius|invalid tracker radii"):
        D.source_geometry(area, fcl, lambda path: None)


@pytest.mark.parametrize("failure", ["cycle", "escape", "absolute", "symlink"])
def test_source_geometry_safe_include_chain(geometry_files, tmp_path, failure):
    area, fcl, paths = geometry_files
    outside = tmp_path/"outside.txt"
    outside.write_text("double tracker.wireRadius = .01;")
    target = {"cycle": GEOMETRY, "escape": "../outside.txt", "absolute": str(outside),
              "symlink": "geometry/link.txt"}[failure]
    if failure == "symlink":
        (area/target).symlink_to(outside)
    paths[2].write_text(f'#include "{target}"\n')
    watched = []
    with pytest.raises(ValueError, match="unsafe/cyclic geometry include"):
        D.source_geometry(area, fcl, lambda path: watched.append(Path(path)))
    assert outside not in watched


@pytest.mark.parametrize("failure", ["inner_nonempty", "outer_nonempty", "inner_missing", "outer_duplicate",
                                     "scale_nonzero", "scale_missing", "scale_duplicate", "scale_nan"])
def test_source_geometry_magnetic_map_guards(geometry_files, failure):
    area, fcl, paths = geometry_files
    text = paths[3].read_text()
    if failure in ("inner_nonempty", "outer_nonempty"):
        name = failure.split("_")[0]+"Maps"
        text = text.replace(name+' = {}', name+' = {"nonzero.map"}')
    elif failure == "inner_missing":
        text = text.replace("bfield.innerMaps", "bfield.unused")
    elif failure == "outer_duplicate":
        text += "vector<string> bfield.outerMaps = {};\n"
    elif failure == "scale_missing":
        text = text.replace("bfield.scaleFactor", "bfield.unused")
    elif failure == "scale_duplicate":
        text += "double bfield.scaleFactor = 0;\n"
    else:
        text = text.replace("scaleFactor = 0", "scaleFactor = "+("nan" if failure == "scale_nan" else "1"))
    paths[3].write_text(text)
    with pytest.raises(ValueError, match="magnetic field maps|nonzero field scale"):
        D.source_geometry(area, fcl, lambda path: None)


def test_source_geometry_rejects_additional_conflicting_geometry(geometry_files):
    area, fcl, _ = geometry_files
    fcl.write_text(fcl.read_text().replace('}\n', '  inputFile : "unreviewed_geometry.txt"\n}\n', 1))
    with pytest.raises(ValueError, match="unsupported geometry/field configuration"):
        D.source_geometry(area, fcl, lambda path: None)


@pytest.fixture
def residual_receipt(tmp_path, monkeypatch):
    evidence = tmp_path/"synthetic_input.root"
    evidence.write_text("synthetic evidence only\n")
    manifest = {str(evidence): D.digest(evidence)}
    data = dict(completed=True, hashes_unchanged=True, before_sha256=manifest,
                end_sha256=manifest.copy(), rtol=D.R.RTOL, atol=D.R.ATOL,
                population=dict(n_hits=2538, n_targets=147, n_excluded_hits=0))
    path = tmp_path/"residual.json"
    def save():
        path.write_text(json.dumps(data))
        monkeypatch.setattr(D, "RESIDUAL_REPORT_SHA256", D.digest(path))
    save()
    return path, evidence, data, save


def test_receipt_production_pin_and_declared_arms():
    assert D.RESIDUAL_REPORT_SHA256 == "1539f808dd7aeab96042dac29023aed33e6688349686d6abb2c804a305bd3219"
    assert D.ARMS == ("chol_factor_roundtrip", "numpy_lu_roundtrip", "ideal_zero_weight_control")


def test_receipt_valid_tracks_declared_evidence_and_pinned_report(residual_receipt):
    path, evidence, data, _ = residual_receipt
    watched = {}
    assert D.receipt(path, watched) == data
    assert watched == {str(path.resolve()): D.RESIDUAL_REPORT_SHA256, str(evidence): D.digest(evidence)}
    watched["new"] = "copy only"
    assert "new" not in data["before_sha256"]


def test_receipt_checks_pinned_hash_before_json_and_retains_pin(residual_receipt):
    path, _, _, _ = residual_receipt
    path.write_text("not JSON")
    watched = {}
    with pytest.raises(ValueError, match="residual receipt hash differs"):
        D.receipt(path, watched)
    assert watched == {str(path.resolve()): D.RESIDUAL_REPORT_SHA256}


@pytest.mark.parametrize("field,value,message", [
    ("completed", False, "incomplete residual receipt"), ("completed", 1, "incomplete residual receipt"),
    ("hashes_unchanged", 1, "incomplete residual receipt"),
    ("before_sha256", {}, "missing residual manifest"), ("before_sha256", [], "missing residual manifest"),
    ("end_sha256", {}, "incomplete residual receipt"),
    ("population", dict(n_hits=2537, n_targets=147, n_excluded_hits=0), "receipt census differs"),
    ("population", dict(n_hits=2538, n_targets=146, n_excluded_hits=0), "receipt census differs"),
    ("population", dict(n_hits=2538, n_targets=147, n_excluded_hits=1), "receipt census differs"),
    ("population", dict(n_hits=2538., n_targets=147, n_excluded_hits=0), "invalid receipt census"),
    ("population", dict(n_hits=2538, n_targets=147, n_excluded_hits=False), "invalid receipt census"),
    ("population", {}, "invalid receipt census"),
    ("rtol", 1e-4, "receipt tolerances differ"), ("atol", 1e-6, "receipt tolerances differ"),
])
def test_receipt_rejects_bad_contract(residual_receipt, field, value, message):
    path, _, data, save = residual_receipt
    data[field] = value
    save()
    with pytest.raises(ValueError, match=message):
        D.receipt(path, {})


@pytest.mark.parametrize("sha", ["g"*64, "A"*64, "a"*63, 123, None])
def test_receipt_rejects_bad_digest_and_retains_declared_manifest(residual_receipt, sha):
    path, evidence, data, save = residual_receipt
    data["before_sha256"] = data["end_sha256"] = {str(evidence): sha}
    save()
    watched = {}
    with pytest.raises(ValueError, match="invalid receipt digest"):
        D.receipt(path, watched)
    assert watched[str(evidence)] == sha
    assert watched[str(path)] == D.RESIDUAL_REPORT_SHA256


@pytest.fixture
def default_environment(monkeypatch):
    for name in list(D.os.environ):
        if name.startswith("F1_"):
            monkeypatch.delenv(name)


def run_args(tmp_path, receipt=None):
    return SimpleNamespace(residual_report=receipt or tmp_path/"residual.json",
                           native_source=tmp_path/"native", out=tmp_path/"out.json")


def test_run_nondefault_environment_exclusive_failure_json_scope(tmp_path, monkeypatch, default_environment):
    args = run_args(tmp_path)
    monkeypatch.setenv("F1_SYNTHETIC_NONDEFAULT", "1")
    def forbidden(*args):
        pytest.fail("environment must be checked before receipt")
    monkeypatch.setattr(D, "receipt", forbidden)
    report = D.run(args)
    assert not report["completed"] and not report["hashes_unchanged"]
    assert "requires default F1 environment" in report["error"]
    assert "missing/changed" in report["integrity_error"]
    for name in ("full_closure", "full_numerical_validation", "native_unbiased_intermediates_compared",
                 "full_state_policy_closure"):
        assert report[name] is False
    assert report["declared_arms"] == list(D.ARMS)
    assert report["limits"] == D.LIMITS
    assert report["rtol"] == 1e-5 and report["atol"] == 1e-7
    assert report["before_sha256"] == report["end_sha256"] == {}
    assert json.loads(args.out.read_text()) == report
    saved = args.out.read_bytes()
    with pytest.raises(FileExistsError):
        D.run(args)
    assert args.out.read_bytes() == saved


@pytest.mark.parametrize("mutation", ["unchanged", "changed", "deleted"])
def test_run_failure_rehashes_every_known_input(tmp_path, monkeypatch, default_environment, mutation):
    args = run_args(tmp_path)
    evidence = tmp_path/"synthetic.root"
    evidence.write_text("original")
    original = D.digest(evidence)
    def receipt(path, watched):
        watched[str(evidence)] = original
        return {}
    monkeypatch.setattr(D, "receipt", receipt)
    real_digest = D.digest
    def fail_at_script(path):
        if Path(path).name == "f1_initial_cad_audit.py":
            if mutation == "changed":
                evidence.write_text("changed")
            elif mutation == "deleted":
                evidence.unlink()
            raise ValueError("synthetic source failure")
        return real_digest(path)
    monkeypatch.setattr(D, "digest", fail_at_script)
    report = D.run(args)
    assert report["completed"] is False and "synthetic source failure" in report["error"]
    assert report["before_sha256"] == {str(evidence): original}
    assert report["end_sha256"] == {str(evidence): None if mutation == "deleted" else real_digest(evidence)}
    assert report["hashes_unchanged"] is (mutation == "unchanged")
    assert ("integrity_error" in report) is (mutation != "unchanged")
    assert json.loads(args.out.read_text()) == report


def test_run_incomplete_receipt_retains_entire_failure_manifest(residual_receipt, tmp_path, default_environment):
    path, evidence, data, save = residual_receipt
    data["completed"] = False
    save()
    args = run_args(tmp_path, path)
    report = D.run(args)
    assert report["completed"] is False
    assert "incomplete residual receipt" in report["error"]
    expected = {str(path): D.RESIDUAL_REPORT_SHA256, str(evidence): D.digest(evidence)}
    assert report["before_sha256"] == report["end_sha256"] == expected
    assert report["hashes_unchanged"] is True
    assert "integrity_error" not in report
    assert json.loads(args.out.read_text()) == report


def test_run_missing_receipt_records_pin_and_null_end_hash(tmp_path, default_environment):
    args = run_args(tmp_path)
    report = D.run(args)
    assert report["completed"] is report["hashes_unchanged"] is False
    assert "FileNotFoundError" in report["error"]
    assert report["before_sha256"] == {str(args.residual_report): D.RESIDUAL_REPORT_SHA256}
    assert report["end_sha256"] == {str(args.residual_report): None}
    assert json.loads(args.out.read_text()) == report


@pytest.mark.parametrize("mutation", ["changed", "deleted", "receipt_changed"])
def test_run_receipt_failure_preserves_expected_hashes(residual_receipt, tmp_path, default_environment, mutation):
    path, evidence, data, _ = residual_receipt
    original = deepcopy(data["before_sha256"])
    if mutation == "changed":
        evidence.write_text("changed")
    elif mutation == "deleted":
        evidence.unlink()
    else:
        path.write_text("not JSON")
    args = run_args(tmp_path, path)
    report = D.run(args)
    expected = {str(path): D.RESIDUAL_REPORT_SHA256}
    if mutation != "receipt_changed":
        expected.update(original)
    assert report["before_sha256"] == expected
    assert report["end_sha256"] == {p: D.digest(p) if Path(p).exists() else None for p in expected}
    assert report["completed"] is report["hashes_unchanged"] is False
    assert "error" in report and "integrity_error" in report
    assert json.loads(args.out.read_text()) == report


def test_geometry_service_scope_ignores_global_constants(geometry_files):
    area, fcl, _ = geometry_files
    fcl.write_text(fcl.read_text()+'GlobalConstantsService: {\n inputFile: "constants.txt"\n}\n')
    assert D.source_geometry(area, fcl, lambda _: None)["zero_field"]


def test_geometry_service_scope_rejects_second_input_in_same_table(geometry_files):
    area, fcl, _ = geometry_files
    fcl.write_text(fcl.read_text().replace('}\n', ' inputFile: "wrong_geometry.txt"\n}\n', 1))
    with pytest.raises(ValueError, match="unsupported geometry/field"):
        D.source_geometry(area, fcl, lambda _: None)


def test_nominal_field_cannot_be_replaced_by_physical_zero(covariance_tables):
    hits, pieces, seeds = covariance_tables
    hits = ak.with_field(hits, ak.Array(np.zeros((len(hits), 3))), "reference_bnom")
    with pytest.raises(ValueError, match="nominal field differs"):
        D.reference_covariance(hits, pieces, seeds)


def test_lorentz_angle_retained_z_axis_sign_fold_and_no_renormalized_radial():
    state = dict(sensor_direction=np.array([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [.6, 0., .8]]),
                 delta=np.array([[0., 1., -1.], [0., -1., -1.], [0., -1., 1.], [0., 1., -1.]]))
    np.testing.assert_allclose(D.lorentz_angle(state), [np.pi/4., np.pi/4., np.pi/4., np.arctan2(.6, 1.)], rtol=0, atol=0)


@pytest.mark.parametrize("field", ["delta", "sensor_direction"])
def test_lorentz_angle_masks_and_nonfinite_fail(field):
    state = dict(sensor_direction=np.array([[1., 0., 0.]]), delta=np.array([[0., 1., -1.]]))
    state[field] = np.ma.array(state[field], mask=[[True, False, False]])
    with pytest.raises(ValueError, match="missing"):
        D.lorentz_angle(state)


def test_similarity_vectorized_quadratic_and_false_masks():
    h = np.arange(12, dtype=float).reshape(2, 6)/7.
    c = np.broadcast_to(np.eye(6), (2, 6, 6)).copy()
    np.testing.assert_allclose(D.similarity(h, c), np.einsum("ni,nij,nj->n", h, c, h), rtol=2e-16)
    np.testing.assert_array_equal(D.similarity(np.ma.array(h, mask=False), np.ma.array(c, mask=False)), D.similarity(h, c))


@pytest.mark.parametrize("operand", ["h", "c"])
@pytest.mark.parametrize("failure", ["mask", "shape", "nan", "inf"])
def test_similarity_rejects_malformed_operands(operand, failure):
    h, c = np.ones((2, 6)), np.broadcast_to(np.eye(6), (2, 6, 6)).copy()
    value = h if operand == "h" else c
    if failure == "shape":
        value = value[:1]
    elif failure == "mask":
        value = np.ma.array(value, mask=False)
        value.mask.flat[0] = True
    else:
        value.flat[0] = np.nan if failure == "nan" else np.inf
    if operand == "h":
        h = value
    else:
        c = value
    with pytest.raises(ValueError, match="missing|shape|nonfinite"):
        D.similarity(h, c)


def test_similarity_overflow_not_silent_zero():
    with pytest.raises(ValueError, match="nonfinite arithmetic"):
        D.similarity(np.full((2, 6), 1e308), np.broadcast_to(np.eye(6), (2, 6, 6)))
